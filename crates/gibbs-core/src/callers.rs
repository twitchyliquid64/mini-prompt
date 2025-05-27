use crate::data_model::{FinishReason, ToolChoice};
use crate::models::OpenrouterModel;
use crate::{CallErr, ChatMessage, CompletionsRequest, CompletionsResponse, Model, Tool};
use reqwest::Client;
use std::env;

/// A type which is able to make model calls.
pub trait ModelCaller: Send {
    /// Returns information about the model this backend is wired to.
    fn get_model(&self) -> impl Model;

    /// Implements one model call to complete a turn in an LLM conversation. The workhorse of this trait.
    fn call(
        &mut self,
        messages: Vec<ChatMessage>,
        tools: Vec<Tool>,
    ) -> impl std::future::Future<Output = Result<CompletionsResponse, CallErr>> + Send;

    /// Convenience method to prompt a model and get the response as a string.
    fn simple_call<S: Into<String> + Send>(
        &mut self,
        prompt: S,
    ) -> impl std::future::Future<Output = Result<String, CallErr>> {
        let prompt = self.get_model().make_prompt(prompt.into());
        async {
            let res = self.call(vec![prompt], vec![]).await?;
            match res.choices.into_iter().next().unwrap().message.content {
                Some(c) => Ok(c),
                None => Err("unexpected: no message content".into()),
            }
        }
    }
}

/// A [ModelCaller] that talks to a model accessible via Openrouter.
#[derive(Debug, Clone, Default)]
pub struct Openrouter<M: OpenrouterModel> {
    pub model: M,
    pub api_key: Option<String>,
}

impl<M: OpenrouterModel> ModelCaller for Openrouter<M> {
    fn get_model(&self) -> impl Model {
        M::default()
    }

    async fn call(
        &mut self,
        messages: Vec<ChatMessage>,
        tools: Vec<Tool>,
    ) -> Result<CompletionsResponse, CallErr> {
        let client = Client::new();
        let resp = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(
                self.api_key
                    .clone()
                    .unwrap_or_else(|| env::var("OR_KEY").unwrap()),
            )
            .json(&CompletionsRequest {
                model: M::MODEL_STR.into(),
                messages,
                tool_choice: if tools.is_empty() {
                    None
                } else {
                    Some(ToolChoice::Auto)
                },
                tools,
                ..Default::default()
            })
            .send()
            .await?;

        // println!("res: {:?}", resp);
        // panic!("body: {:?}", resp.text().await?);

        let mut res = resp.json::<CompletionsResponse>().await?;

        if res.model == "" {
            res.model = M::MODEL_STR.into();
        }

        if let Some("chat.completion") = res.object.as_ref().map(|s| s.as_str()) {
        } else if res.object.is_some() {
            return Err(format!("unexpected value for 'object': {}", res.object.unwrap()).into());
        }
        if res.choices.len() == 0 {
            return Err(CallErr::NoCompletions);
        }

        match res.choices[0].finish_reason {
            FinishReason::Stop | FinishReason::ToolCalls => Ok(res),
            // {
            //     next_messages.push(res.choices[0].message.clone());

            //     for c in res.choices[0].message.tool_calls.iter() {
            //         let response_msg = self
            //             .tool_call(&c.function.name, c.function.arguments.clone())
            //             .await
            //             .map_err(|e| -> Box<dyn std::error::Error> {
            //                 format!("function call failed: {:?}", e).into()
            //             })?;
            //         next_messages.push(response_msg);
            //     }

            //     self.call(next_messages)
            // }
            _ => Err(format!(
                "unexpected finish reason: {:?}",
                res.choices[0].finish_reason
            )
            .into()),
        }
    }
}
