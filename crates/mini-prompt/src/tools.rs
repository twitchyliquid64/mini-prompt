//! Tool-calling scaffolding.
//!
//! See `examples/tool_call.rs` for an end-to-end example.

use crate::models::Model;
use crate::{
    CallBase, CallErr, CallResp, FinishReason, Message, ModelCaller, Role, ToolInfo, Turn,
};

const MAX_TOOL_ITER: usize = 12;

/// The type of a function usable in a [ToolsSession].
///
/// For example:
/// ```
/// # use mini_prompt::tools::RawToolFunc;
/// let my_tool: RawToolFunc = Box::new(move |_args| {
///     r#"{"status": "success", "message": "flubb completed successfully"}"#
///         .to_string()
/// });
/// ```
pub type RawToolFunc = Box<dyn FnMut(String) -> String + Send + Sync>;

// /// A collection of tools a model can use.
// pub trait Toolbox: Send {
//     fn tools(&self) -> Vec<Tool>;

//     fn tool_call(
//         &mut self,
//         name: &str,
//         args: String,
//     ) -> impl std::future::Future<Output = Result<OAIChatMessage, ()>> + Send;
// }

/// A model call which has access to tools.
///
/// This type implements [ModelCaller], but any tools provided during invocation will
/// be ignored in favor of the tools provided when creating the [ToolsSession].
pub struct ToolsSession<B: ModelCaller> {
    tools: Vec<(ToolInfo, RawToolFunc)>,
    backend: B,
}

impl<B: ModelCaller> ToolsSession<B> {
    /// Constructs a new [ToolsSession] with the given backend and tools.
    pub fn new(b: B, tools: Vec<(ToolInfo, RawToolFunc)>) -> Self {
        Self { tools, backend: b }
    }

    fn tool_call(&mut self, name: &String, args: String) -> Result<String, CallErr> {
        for (d, f) in self.tools.iter_mut() {
            if name == &d.name {
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

    async fn call(&mut self, params: CallBase, mut turns: Vec<Turn>) -> Result<CallResp, CallErr> {
        let params = CallBase {
            tools: self.tools.iter().map(|(td, _)| td.clone()).collect(),
            ..params
        };

        let mut last_res: Option<CallResp> = None;
        for _ in 0..MAX_TOOL_ITER {
            let res = self.backend.call(params.clone(), turns.clone()).await;

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

            match resp.finish_reason {
                FinishReason::Stop => {
                    // println!("trace: {:?}", turns);
                    return Ok(resp);
                }
                FinishReason::ToolCalls => {
                    // println!("tool call: {:?}", resp.content);
                    turns.push(resp.content.clone());

                    let mut tool_resp = Turn {
                        role: Role::Tool,
                        content: vec![],
                    };
                    for (id, name, args) in resp.content.content.iter().filter_map(|m| match m {
                        Message::ToolCall {
                            id,
                            name,
                            arguments,
                        } => Some((id, name, arguments)),
                        _ => None,
                    }) {
                        let response_msg =
                            self.tool_call(&name, args.clone())
                                .map(|m| Message::ToolResult {
                                    id: id.clone(),
                                    result: m,
                                })?;
                        tool_resp.content.push(response_msg);
                    }
                    turns.push(tool_resp);
                }
                _ => unreachable!(),
            }

            last_res = Some(resp);
        }

        Err(format!("exceeded max tool iterations: {}", MAX_TOOL_ITER).into())
    }
}
