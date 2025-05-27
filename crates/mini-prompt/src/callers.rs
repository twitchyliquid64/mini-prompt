use crate::data_model::{
    AnthropicMsgRequest, AnthropicMsgResponse, ChatChoice, FinishReason, MessageRole, ToolChoice,
};
use crate::models::{AnthropicModel, OpenrouterModel};
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
///
/// If an API key is not provided, it will be read from the environment variable
/// `OPENROUTER_API_KEY` or `OR_KEY`.
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
            .bearer_auth(self.api_key.clone().unwrap_or_else(|| {
                env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| env::var("OR_KEY").unwrap())
            }))
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
            _ => Err(format!(
                "unexpected finish reason: {:?}",
                res.choices[0].finish_reason
            )
            .into()),
        }
    }
}

/// A [ModelCaller] that talks to a model via Anthropic's public messages API.
///
/// If an API key is not provided, it will be read from the environment variable
/// `ANTHROPIC_API_KEY`.
/// If max_tokens is not set, it defaults to 8k.
#[derive(Debug, Clone, Default)]
pub struct Anthropic<M: AnthropicModel> {
    pub model: M,
    pub max_tokens: Option<usize>,
    pub api_key: Option<String>,
}

impl<M: AnthropicModel> ModelCaller for Anthropic<M> {
    fn get_model(&self) -> impl Model {
        M::default()
    }

    async fn call(
        &mut self,
        mut messages: Vec<ChatMessage>,
        tools: Vec<Tool>,
    ) -> Result<CompletionsResponse, CallErr> {
        // Anthropic's messages API expects the system prompt as a request parameter rather
        // than an entry in the messages list. But anthropic's API doesn't let
        let system: Option<String> = match messages.get_mut(0) {
            Some(ChatMessage {
                role: role @ MessageRole::System,
                content,
                ..
            }) => {
                *role = MessageRole::User;
                content.clone()
            }
            _ => None,
        };

        let client = Client::new();
        let resp = client
            .post("https://api.anthropic.com/v1/messages")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .header(
                "x-api-key",
                self.api_key
                    .clone()
                    .unwrap_or_else(|| env::var("ANTHROPIC_API_KEY").unwrap()),
            )
            .json(&AnthropicMsgRequest {
                model: M::MODEL_STR.into(),
                max_tokens: self.max_tokens.unwrap_or(8192),
                messages,
                system,
                tool_choice: if tools.is_empty() {
                    None
                } else {
                    Some(ToolChoice::Auto)
                },
                tools: tools.into_iter().map(|t| t.into()).collect(),
                ..Default::default()
            })
            .send()
            .await?;

        // println!("res: {:?}", resp);
        // panic!("body: {:?}", resp.text().await?);

        let mut res = resp.json::<AnthropicMsgResponse>().await?;

        if res.model == "" {
            res.model = M::MODEL_STR.into();
        }

        if let Some("message") = res.object.as_ref().map(|s| s.as_str()) {
        } else if res.object.is_some() {
            return Err(format!("unexpected value for 'object': {}", res.object.unwrap()).into());
        }
        if let Some("assistant") = res.role.as_ref().map(|s| s.as_str()) {
        } else if res.object.is_some() {
            return Err(format!("unexpected value for 'role': {}", res.role.unwrap()).into());
        }

        match res.stop_reason {
            FinishReason::Stop | FinishReason::ToolCalls => {
                // Convert Anthropics format into whats expected by this API
                Ok(CompletionsResponse {
                    id: res.id,
                    created: None,
                    model: res.model,
                    object: Some("chat.completion".to_string()),
                    choices: vec![ChatChoice {
                        index: 0,
                        finish_reason: res.stop_reason,
                        message: ChatMessage::assistant(res.content[0].text_content().unwrap()), // HACK: need to support multiple messages - API change?
                    }],
                })
            }
            _ => Err(format!("unexpected finish reason: {:?}", res.stop_reason).into()),
        }
    }
}
