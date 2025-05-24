//! Types representing the different LLMs which can be used.

use crate::ChatMessage;

/// Some specific LLM.
pub trait Model: Send + Default {
    /// Takes a 'system' prompt, formatting it into a message to be used in a model call.
    ///
    /// This is only needed because some models don't understand the system role.
    fn make_prompt(&self, prompt: String) -> ChatMessage;
}

/// An LLM which can be called via Openrouter.
pub trait OpenrouterModel: Model {
    const MODEL_STR: &'static str;
    const NO_SYS_PROMPT: bool;
}

/// The Gemma3 27b LLM.
#[derive(Default, Debug, Clone)]
pub struct Gemma27B3;

impl OpenrouterModel for Gemma27B3 {
    const MODEL_STR: &'static str = &"google/gemma-3-27b-it";
    const NO_SYS_PROMPT: bool = false;
}

/// The Qwen3 235b (22b active) LLM.
#[derive(Default, Debug, Clone)]
pub struct Qwen235B3;

impl OpenrouterModel for Qwen235B3 {
    const MODEL_STR: &'static str = &"qwen/qwen3-235b-a22b";
    const NO_SYS_PROMPT: bool = false;
}

/// The Phi4 LLM.
#[derive(Default, Debug, Clone)]
pub struct Phi4;

impl OpenrouterModel for Phi4 {
    const MODEL_STR: &'static str = &"microsoft/phi-4";
    const NO_SYS_PROMPT: bool = true;
}

/// The Gemini 2 flash LLM.
#[derive(Default, Debug, Clone)]
pub struct Gemini2Flash;

impl OpenrouterModel for Gemini2Flash {
    const MODEL_STR: &'static str = &"google/gemini-2.0-flash-001";
    const NO_SYS_PROMPT: bool = false;
}

/// The Gemini 2.5 flash LLM.
#[derive(Default, Debug, Clone)]
pub struct Gemini25Flash;

impl OpenrouterModel for Gemini25Flash {
    const MODEL_STR: &'static str = &"google/gemini-2.5-flash-preview-05-20";
    const NO_SYS_PROMPT: bool = false;
}

/// The Devstral Small LLM.
#[derive(Default, Debug, Clone)]
pub struct DevstralSmall;

impl OpenrouterModel for DevstralSmall {
    const MODEL_STR: &'static str = &"mistralai/devstral-small";
    const NO_SYS_PROMPT: bool = false;
}

impl<X: OpenrouterModel> Model for X {
    fn make_prompt(&self, prompt: String) -> ChatMessage {
        if X::NO_SYS_PROMPT {
            ChatMessage::user(prompt)
        } else {
            ChatMessage::system(prompt)
        }
    }
}
