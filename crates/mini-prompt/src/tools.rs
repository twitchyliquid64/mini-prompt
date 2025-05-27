use crate::data_model::{FinishReason, MessageRole};
use crate::{CallErr, ChatMessage, CompletionsResponse, Model, ModelCaller, Tool};

const MAX_TOOL_ITER: usize = 12;

pub type RawToolFunc = Box<dyn FnMut(String) -> ChatMessage + Send + Sync>;

// /// A collection of tools a model can use.
// pub trait Toolbox: Send {
//     fn tools(&self) -> Vec<Tool>;

//     fn tool_call(
//         &mut self,
//         name: &str,
//         args: String,
//     ) -> impl std::future::Future<Output = Result<ChatMessage, ()>> + Send;
// }

/// A model call which has access to tools.
///
/// This type implements [ModelCaller], but any tools provided during invocation will
/// be ignored in favor of the tools provided when creating the [ToolsSession].
pub struct ToolsSession<B: ModelCaller> {
    tools: Vec<(Tool, RawToolFunc)>,
    backend: B,
}

impl<B: ModelCaller> ToolsSession<B> {
    /// Constructs a new [ToolsSession] with the given backend and tools.
    pub fn new(b: B, tools: Vec<(Tool, RawToolFunc)>) -> Self {
        Self { tools, backend: b }
    }

    fn tool_call(&mut self, name: &String, args: String) -> Result<ChatMessage, CallErr> {
        for (d, f) in self.tools.iter_mut() {
            if Some(name) == d.function.name.as_ref() {
                return Ok((*f)(args));
            }
        }
        Err(format!("no such tool: {}", name).into())
    }
}

impl<B: ModelCaller> ModelCaller for ToolsSession<B> {
    fn get_model(&self) -> impl Model {
        self.backend.get_model()
    }

    async fn call(
        &mut self,
        mut messages: Vec<ChatMessage>,
        _tools: Vec<Tool>,
    ) -> Result<CompletionsResponse, CallErr> {
        let mut last_res: Option<CompletionsResponse> = None;
        for _ in 0..MAX_TOOL_ITER {
            let res = self
                .backend
                .call(
                    messages.clone(),
                    self.tools.iter().map(|(t, _)| t.clone()).collect(),
                )
                .await;

            let resp = match res {
                Err(CallErr::NoCompletions) => {
                    return if let Some(last_res) = last_res {
                        Ok(last_res)
                    } else {
                        Err(CallErr::NoCompletions)
                    }
                }
                Err(e) => {
                    return Err(e);
                }
                Ok(resp) => resp,
            };

            match resp.choices[0].finish_reason {
                FinishReason::Stop => {
                    println!("trace: {:?}", messages);
                    return Ok(resp);
                }
                FinishReason::ToolCalls => {
                    println!("tool call: {:?}", &resp.choices[0].message);
                    messages.push(resp.choices[0].message.clone());

                    for c in resp.choices[0].message.tool_calls.iter() {
                        let response_msg = self
                            .tool_call(&c.function.name, c.function.arguments.clone())
                            .map(|mut m| {
                                m.role = MessageRole::Tool;
                                m.tool_call_id = Some(c.id.clone());
                                // m.name = Some(c.function.name.clone());
                                m
                            })
                            .map_err(|e| -> CallErr {
                                format!("function call failed: {:?}", e).into()
                            })?;
                        messages.push(response_msg);
                    }
                }
                _ => unreachable!(),
            }

            last_res = Some(resp);
        }

        Err(format!("exceeded max tool iterations: {}", MAX_TOOL_ITER).into())
    }
}
