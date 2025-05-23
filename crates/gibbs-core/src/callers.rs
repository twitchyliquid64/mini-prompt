use crate::data_model::{FinishReason, ToolChoice};
use crate::{ChatMessage, CompletionsRequest, CompletionsResponse, ModelCaller, ModelRef, Tool};
use reqwest::Client;
use std::env;

/// A [ModelCaller] that talks to a model accessible via Openrouter.
#[derive(Debug, Clone, Default)]
pub struct Openrouter {
    pub model: ModelRef,
    pub api_key: Option<String>,
}

impl ModelCaller for Openrouter {
    fn get_model(&self) -> &ModelRef {
        &self.model
    }

    async fn call(
        &mut self,
        messages: Vec<ChatMessage>,
        tools: Vec<Tool>,
    ) -> Result<CompletionsResponse, Box<dyn std::error::Error>> {
        let client = Client::new();
        let resp = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(
                self.api_key
                    .clone()
                    .unwrap_or_else(|| env::var("OR_KEY").unwrap()),
            )
            .json(&CompletionsRequest {
                model: self.model.openrouter_str().into(),
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
            res.model = self.model.openrouter_str().into();
        }

        if let Some("chat.completion") = res.object.as_ref().map(|s| s.as_str()) {
        } else if res.object.is_some() {
            return Err(format!("unexpected value for 'object': {}", res.object.unwrap()).into());
        }
        if res.choices.len() == 0 {
            return Err("unexpected: no completion choices returned".into());
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
