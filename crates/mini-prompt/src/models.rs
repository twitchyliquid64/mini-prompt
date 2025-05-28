//! Types representing the different LLMs which can be used.

use crate::OAIChatMessage;

/// Some specific LLM.
pub trait Model: Send + Default {
    /// Takes a 'system' prompt, formatting it into a message to be used in a model call.
    ///
    /// This is only needed because some models don't understand the system role.
    fn make_prompt(&self, prompt: String) -> OAIChatMessage;
}

/// An LLM which can be called via Openrouter.
pub trait OpenrouterModel: Model {
    const MODEL_STR: &'static str;
    const NO_SYS_PROMPT: bool;
}

/// An LLM which can be called via the Anthropic public API.
pub trait AnthropicModel: Model {
    const MODEL_STR: &'static str;
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

/// OpenAI's GPT-4o-mini model.
#[derive(Default, Debug, Clone)]
pub struct GPT4oMini;

impl OpenrouterModel for GPT4oMini {
    const MODEL_STR: &'static str = &"openai/gpt-4o-mini";
    const NO_SYS_PROMPT: bool = false;
}

/// Deepseek v3 0324
#[derive(Default, Debug, Clone)]
pub struct Deepseek0324v3;

impl OpenrouterModel for Deepseek0324v3 {
    const MODEL_STR: &'static str = &"deepseek/deepseek-chat-v3-0324";
    const NO_SYS_PROMPT: bool = false;
}

/// Claude Sonnet 4
#[derive(Default, Debug, Clone)]
pub struct ClaudeSonnet4;

impl OpenrouterModel for ClaudeSonnet4 {
    const MODEL_STR: &'static str = &"anthropic/claude-sonnet-4";
    const NO_SYS_PROMPT: bool = false;
}

impl AnthropicModel for ClaudeSonnet4 {
    const MODEL_STR: &'static str = &"claude-sonnet-4-20250514";
}

/// Claude Haiku 3.5
#[derive(Default, Debug, Clone)]
pub struct ClaudeHaiku35;

impl AnthropicModel for ClaudeHaiku35 {
    const MODEL_STR: &'static str = &"claude-3-5-haiku-latest";
}

impl Model for ClaudeHaiku35 {
    fn make_prompt(&self, prompt: String) -> OAIChatMessage {
        OAIChatMessage::system(prompt)
    }
}

impl<X: OpenrouterModel> Model for X {
    fn make_prompt(&self, prompt: String) -> OAIChatMessage {
        if X::NO_SYS_PROMPT {
            OAIChatMessage::user(prompt)
        } else {
            OAIChatMessage::system(prompt)
        }
    }
}
