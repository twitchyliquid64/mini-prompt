use crate::data_model::{FinishReason, ToolChoice};
use crate::{ChatMessage, CompletionBackend, CompletionsRequest, CompletionsResponse, Model, Tool};
use reqwest::Client;
use std::env;

/// A [CompletionBackend] that talks to a model accessible via Openrouter.
#[derive(Debug, Clone)]
pub struct OpenrouterModel {
    pub model: Model,
    pub api_key: Option<String>,
}

impl CompletionBackend for OpenrouterModel {
    fn get_model(&self) -> &Model {
        &self.model
    }

    async fn call(
        &self,
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
        if res.choices[0].finish_reason != FinishReason::Stop {
            return Err(format!(
                "unexpected finish reason: {:?}",
                res.choices[0].finish_reason
            )
            .into());
        }

        Ok(res)
    }
}
