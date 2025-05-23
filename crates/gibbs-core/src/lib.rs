use std::collections::HashMap;

pub mod data_model;
use crate::data_model::{ChatMessage, CompletionsRequest, CompletionsResponse, Tool};
pub mod parse;

/// References a specific model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    Gemma27B3,
    Qwen235B3,
    Phi4,
    Gemini2Flash,
    Gemini25Flash,
    DevstralSmall,
}

impl Model {
    fn all() -> &'static [Model] {
        use Model::*;
        &[
            Gemma27B3,
            Qwen235B3,
            Phi4,
            Gemini2Flash,
            Gemini25Flash,
            DevstralSmall,
        ]
    }

    pub fn openrouter_str(&self) -> &'static str {
        use Model::*;
        match self {
            Gemma27B3 => &"google/gemma-3-27b-it",
            Qwen235B3 => &"qwen/qwen3-235b-a22b",
            Phi4 => &"microsoft/phi-4",
            Gemini2Flash => &"google/gemini-2.0-flash-001",
            Gemini25Flash => &"google/gemini-2.5-flash-preview-05-20",
            DevstralSmall => &"mistralai/devstral-small",
        }
    }

    pub fn make_prompt<S: Into<String>>(&self, prompt: S) -> ChatMessage {
        use Model::*;
        match self {
            Phi4 => ChatMessage::user(prompt),
            _ => ChatMessage::system(prompt),
        }
    }
}

impl<'a> TryFrom<&'a str> for Model {
    type Error = ();

    fn try_from(s: &'a str) -> Result<Model, Self::Error> {
        for candidate in Model::all().into_iter() {
            if candidate.openrouter_str() == s {
                return Ok(candidate.clone());
            }
        }
        Err(())
    }
}

pub mod backends;

pub trait CompletionBackend: Send {
    /// Returns information about the model this backend is wired to.
    fn get_model(&self) -> &Model;

    /// Implements one model call to complete a turn in an LLM conversation. The workhorse of this trait.
    fn call(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<Tool>,
    ) -> impl std::future::Future<Output = Result<CompletionsResponse, Box<dyn std::error::Error>>> + Send;

    /// Easy method to prompt a model and get the response as a string.
    fn simple_call<S: Into<String> + Send>(
        &self,
        prompt: S,
    ) -> impl std::future::Future<Output = Result<String, Box<dyn std::error::Error>>> {
        async {
            let res = self
                .call(vec![self.get_model().make_prompt(prompt)], vec![])
                .await?;
            match res.choices.into_iter().next().unwrap().message.content {
                Some(c) => Ok(c),
                None => Err("unexpected: no message content".into()),
            }
        }
    }
}

/// Returns the WAN IP from which this code is accessing the internet.
pub async fn wan_ip() -> Result<String, Box<dyn std::error::Error>> {
    let resp = reqwest::get("https://httpbin.org/ip")
        .await?
        .json::<HashMap<String, String>>()
        .await?;

    resp.get("origin")
        .cloned()
        .ok_or("Missing 'origin' key".into())
}

/// Makes a very simple, text-in-text-out model call.
pub async fn model_call<S: Into<String> + Send>(
    model: Model,
    prompt: S,
) -> Result<String, Box<dyn std::error::Error>> {
    backends::OpenrouterModel {
        model,
        api_key: None,
    }
    .simple_call(prompt)
    .await
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn smoke() {}
}
