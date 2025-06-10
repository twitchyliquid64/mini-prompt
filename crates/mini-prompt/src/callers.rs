use crate::data_model::{
    AnthropicMessage, AnthropicMsgRequest, AnthropicMsgResponse, AnthropicToolChoice,
    OAICompletionsRequest, OAICompletionsResponse, OAIToolChoice,
};
use crate::models::{AnthropicModel, Model, OpenAIModel, OpenrouterModel};
use crate::{CallBase, CallErr, CallResp, FinishReason, Message, Turn};
use reqwest::Client;
use std::env;

/// A type which is able to make model calls.
pub trait ModelCaller: Send {
    /// Returns information about the model this caller is wired to.
    fn get_model(&self) -> impl Model;

    /// Performs a model call, returning the response from the model. This method
    /// is the workhorse of this trait.
    fn call(
        &mut self,
        params: CallBase,
        turns: Vec<Turn>,
    ) -> impl std::future::Future<Output = Result<CallResp, CallErr>> + Send;

    /// Convenience method to prompt a model and get the response as a string.
    fn simple_call<S: Into<String> + Send>(
        &mut self,
        prompt: S,
    ) -> impl std::future::Future<Output = Result<String, CallErr>> {
        let base_params = CallBase {
            instructions: prompt.into(),
            ..Default::default()
        };
        async {
            let res = self.call(base_params, vec![]).await?;

            match &res.content.content.get(0) {
                Some(Message::Text { text }) => Ok(text.clone()),
                _ => Err("unexpected: no message content".into()),
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

    async fn call(&mut self, params: CallBase, turns: Vec<Turn>) -> Result<CallResp, CallErr> {
        // Map `system` and `instructions` into one text stanza, as expected by
        // this API.
        let system_prompt = match (params.system != "", params.instructions != "") {
            (true, true) => Some((params.system + "\n\n" + &params.instructions).into()),
            (false, true) => Some(params.instructions.clone()),
            (true, false) => Some(params.system.clone()),
            (false, false) => None,
        }
        .map(|p| self.get_model().make_prompt(p));

        let mut messages = Vec::with_capacity(1 + turns.len());
        if let Some(system_prompt) = system_prompt {
            messages.push(system_prompt);
        }
        messages.extend(turns.into_iter().map(|t| t.into_oai_msgs()).flatten());

        let client = Client::new();
        let resp = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(self.api_key.clone().unwrap_or_else(|| {
                env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| env::var("OR_KEY").unwrap())
            }))
            .json(&OAICompletionsRequest {
                model: M::MODEL_STR.into(),
                temperature: params.temperature,
                provider: Some(crate::data_model::OpenrouterProvider {
                    // These providers kept returning other ppl's completions :O
                    ignore: vec!["Nebius".into(), "Kluster".into(), "DeepInfra".into()],
                }),
                messages,
                tool_choice: if params.tools.is_empty() {
                    None
                } else {
                    Some(OAIToolChoice::Auto)
                },
                tools: params.tools.into_iter().map(|td| td.into()).collect(),
                ..Default::default()
            })
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CallErr::RequestFailed(resp.status(), resp.text().await?));
        }

        let mut res = resp.json::<OAICompletionsResponse>().await?;

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

        let finish_reason = res.choices[0].finish_reason.clone();
        match finish_reason {
            FinishReason::Stop | FinishReason::ToolCalls => Ok(res.into()),
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

    async fn call(&mut self, params: CallBase, turns: Vec<Turn>) -> Result<CallResp, CallErr> {
        let mut messages = Vec::new();
        if !params.instructions.is_empty() {
            messages.push(AnthropicMessage::user_text(params.instructions));
        }
        messages.extend(turns.into_iter().map(|t| t.into_anthropic_msgs()).flatten());

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
                temperature: params.temperature,
                max_tokens: self.max_tokens.unwrap_or(8192),
                messages,
                system: if params.system == "" {
                    None
                } else {
                    Some(params.system)
                },
                tool_choice: if params.tools.is_empty() {
                    None
                } else {
                    Some(AnthropicToolChoice {
                        r#type: OAIToolChoice::Auto,
                        ..Default::default()
                    })
                },
                tools: params.tools.into_iter().map(|td| td.into()).collect(),
                ..Default::default()
            })
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CallErr::RequestFailed(resp.status(), resp.text().await?));
        }

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
                Ok(res.into())
            }
            _ => Err(format!("unexpected finish reason: {:?}", res.stop_reason).into()),
        }
    }
}

/// A [ModelCaller] that talks to a model accessible via the OpenAI chat completions API.
///
/// If an API key is not provided, it will be read from the environment variable
/// `OPENAI_API_KEY`.
#[derive(Debug, Clone, Default)]
pub struct Openai<M: OpenAIModel> {
    pub model: M,
    pub api_key: Option<String>,
}

impl<M: OpenAIModel> ModelCaller for Openai<M> {
    fn get_model(&self) -> impl Model {
        M::default()
    }

    async fn call(&mut self, params: CallBase, turns: Vec<Turn>) -> Result<CallResp, CallErr> {
        // Map `system` and `instructions` into one text stanza, as expected by
        // this API.
        let system_prompt = match (params.system != "", params.instructions != "") {
            (true, true) => Some((params.system + "\n\n" + &params.instructions).into()),
            (false, true) => Some(params.instructions.clone()),
            (true, false) => Some(params.system.clone()),
            (false, false) => None,
        }
        .map(|p| self.get_model().make_prompt(p));

        let mut messages = Vec::with_capacity(1 + turns.len());
        if let Some(system_prompt) = system_prompt {
            messages.push(system_prompt);
        }
        messages.extend(turns.into_iter().map(|t| t.into_oai_msgs()).flatten());

        let client = Client::new();
        let resp = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(
                self.api_key
                    .clone()
                    .unwrap_or_else(|| env::var("OPENAI_API_KEY").unwrap()),
            )
            .json(&OAICompletionsRequest {
                model: M::MODEL_STR.into(),
                temperature: params.temperature,
                provider: None,
                messages,
                tool_choice: if params.tools.is_empty() {
                    None
                } else {
                    Some(OAIToolChoice::Auto)
                },
                tools: params.tools.into_iter().map(|td| td.into()).collect(),
                ..Default::default()
            })
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(CallErr::RequestFailed(resp.status(), resp.text().await?));
        }

        let mut res = resp.json::<OAICompletionsResponse>().await?;

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

        let finish_reason = res.choices[0].finish_reason.clone();
        match finish_reason {
            FinishReason::Stop | FinishReason::ToolCalls => Ok(res.into()),
            _ => Err(format!(
                "unexpected finish reason: {:?}",
                res.choices[0].finish_reason
            )
            .into()),
        }
    }
}
