//! Lightweight abstractions for using LLMs via a providers API.
//!
//! Simple calls:
//! ```rust,no_run
//! # use mini_prompt::*;
//! let mut backend = callers::Openrouter::<models::Gemma27B3>::default();
//! # tokio::task::spawn(async move {
//! let resp =
//!     backend.simple_call("How much wood could a wood-chuck chop").await;
//! # });
//! ```
//!
//! If you are looking for more control over the input, you can use [call](ModelCaller::call) instead of [simple_call](ModelCaller::simple_call).
//!
//!
//!
//! With tools:
//! ```rust,no_run
//! # use mini_prompt::*;
//! let backend = callers::Anthropic::<models::ClaudeHaiku35>::default();
//! let mut session = ToolsSession::new(
//!             backend,
//!             vec![
//!                 (
//!                     ToolInfo::new("flubb", "Performs the flubb action.", None).into(),
//!                     Box::new(move |_args| {
//!                         r#"{"status": "success", "message": "flubb completed successfully"}"#
//!                             .to_string()
//!                     }),
//!                 ),
//!             ],
//!         );
//!
//! # tokio::task::spawn(async move {
//! let resp =
//!     session.simple_call("Go ahead and flubb for me").await;
//! # });
//! ```
//!
//! Structured output:
//! ```rust,no_run
//! # use mini_prompt::*;
//! # use mini_prompt::parse::*;
//! let mut backend = callers::Openrouter::<models::Gemma27B3>::default();
//! # tokio::task::spawn(async move {
//! let resp =
//!     backend.simple_call("Whats 2+2? output the final answer as JSON within triple backticks (A markdown code block with json as the language).").await;
//!
//! let json = markdown_codeblock(&resp.unwrap(), &MarkdownOptions::json()).unwrap();
//! let p: serde_json::Value = serde_json_lenient::from_str(&json).expect("json decode");
//! # });
//! ```

use serde::{Deserialize, Serialize};

pub mod data_model;
use crate::data_model::{AnthropicMessage, OAIChatMessage};

pub mod parse;

pub mod models;

pub mod callers;

pub use callers::ModelCaller;

pub mod tools;
pub use tools::ToolsSession;

/// Describes an error which occurred during a model call.
pub enum CallErr {
    /// The response lacked any completions, which can be non-erroneous for multi-turn contexts
    /// but is always anomalous in single-turn contexts.
    NoCompletions,
    /// A network or basic deserialization error occurred.
    API(reqwest::Error),
    /// A model call API returned a non-2xx status code.
    RequestFailed(reqwest::StatusCode, String),
    /// Any other error.
    Other(Box<dyn std::error::Error>),
    /// A tool call reported an error instead of returning a result.
    ToolFailed { name: String, err: String },
}

impl std::fmt::Debug for CallErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallErr::NoCompletions => write!(f, "NoCompletions"),
            CallErr::API(err) => f.debug_tuple("API").field(err).finish(),
            CallErr::RequestFailed(status_code, body) => f
                .debug_struct("RequestFailed")
                .field("status_code", status_code)
                .field("body", body)
                .finish(),
            CallErr::Other(err) => f.debug_tuple("Other").field(err).finish(),
            CallErr::ToolFailed { name, err } => f
                .debug_struct("ToolFailed")
                .field("name", name)
                .field("error", err)
                .finish(),
        }
    }
}

impl From<&str> for CallErr {
    fn from(inp: &str) -> Self {
        CallErr::Other(inp.into())
    }
}

impl From<String> for CallErr {
    fn from(inp: String) -> Self {
        CallErr::Other(inp.into())
    }
}

impl From<reqwest::Error> for CallErr {
    fn from(inp: reqwest::Error) -> Self {
        CallErr::API(inp)
    }
}

/// Describes the parameters and use of a tool made available to an LLM.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInfo {
    /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    name: String,
    /// A description of what the function does.
    description: String,
    /// The parameters the functions accepts, described as a JSON Schema object. See the guide for examples, and the JSON Schema reference for documentation about the format.
    /// To describe a function that accepts no parameters, provide the value {"type": "object", "properties": {}}.
    pub parameters: serde_json::Value,
}

impl ToolInfo {
    pub fn new<S: Into<String>>(
        name: S,
        description: S,
        parameters: Option<serde_json::Value>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: match parameters {
                Some(p) => p,
                None => serde_json::json!({"type": "object", "properties": {}}),
            },
        }
    }
}

impl From<ToolInfo> for data_model::OAITool {
    fn from(ti: ToolInfo) -> data_model::OAITool {
        data_model::OAITool {
            r#type: "function".into(),
            function: data_model::FunctionInfo {
                name: Some(ti.name),
                description: ti.description,
                parameters: ti.parameters,
            },
        }
    }
}

impl From<ToolInfo> for data_model::AnthropicTool {
    fn from(ti: ToolInfo) -> data_model::AnthropicTool {
        data_model::AnthropicTool {
            name: Some(ti.name),
            description: ti.description,
            input_schema: ti.parameters,
        }
    }
}

/// The basic parameters for a (possibly multi-turn) model call.
#[derive(Debug, Clone, PartialEq)]
pub struct CallBase {
    /// Short-form information describing the persona of the LLM.
    ///
    /// EG: "You are an expert software developer"
    pub system: String,
    /// Task-specific instructions for the LLM.
    pub instructions: String,
    /// Descriptions of tools that may be used.
    pub tools: Vec<ToolInfo>,

    pub temperature: Option<f32>,
    pub max_tokens: usize,
}

impl Default for CallBase {
    fn default() -> Self {
        Self {
            system: "".to_string(),
            instructions: "".to_string(),
            tools: vec![],

            temperature: None,
            max_tokens: 8192,
        }
    }
}

/// Describes a round of model input or output.
#[derive(Debug, Default, Clone)]
pub struct Turn {
    /// The source of the content: i.e. the user, the model (assistant), a tool.
    pub role: Role,
    /// Some data fed into or read from the model.
    pub content: Vec<Message>,
}

impl From<data_model::OAIChatMessage> for Turn {
    fn from(resp: data_model::OAIChatMessage) -> Self {
        let mut msgs = Vec::with_capacity(1 + resp.tool_calls.len());
        if let Some(text) = resp.content {
            msgs.push(Message::Text { text });
        }
        resp.tool_calls
            .into_iter()
            .for_each(|tc| msgs.push(tc.into()));

        Self {
            role: resp.role,
            content: msgs,
        }
    }
}

impl Turn {
    /// Converts our broad `Turn` type into the wire format expected by chat completions APIs.
    pub(crate) fn into_oai_msgs(self) -> Vec<OAIChatMessage> {
        use itertools::Itertools;
        self.content
            .into_iter()
            .map(|m| match self.role {
                Role::User => OAIChatMessage::user(match m {
                    Message::Text { text } => text,
                    _ => unreachable!(),
                }),
                Role::System => OAIChatMessage::system(match m {
                    Message::Text { text } => text,
                    _ => unreachable!(),
                }),
                Role::Assistant => match m {
                    Message::Text { text } => OAIChatMessage::assistant(text),
                    // These will be combined to one msg during coalesce()
                    Message::ToolCall {
                        id,
                        name,
                        arguments,
                    } => OAIChatMessage {
                        role: Role::Assistant,
                        content: None,
                        tool_calls: vec![crate::data_model::OAIToolCall {
                            id,
                            r#type: crate::data_model::ToolCallType::Function,
                            function: crate::data_model::FunctionCall { name, arguments },
                        }],
                        tool_call_id: None,
                        name: None,
                    },
                    _ => unreachable!(),
                },
                Role::Tool => match m {
                    Message::ToolResult { id, result } => OAIChatMessage {
                        tool_call_id: Some(id),
                        ..OAIChatMessage::tool(result)
                    },
                    _ => unreachable!(),
                },
            })
            // Combine tool call msgs with earlier Assistant msgs
            // if there was one, as they are expected together.
            .coalesce(|mut prev, next| {
                if prev.role == Role::Assistant
                    && next.role == Role::Assistant
                    && next.content.is_none()
                {
                    prev.tool_calls.extend(next.tool_calls);
                    Ok(prev)
                } else {
                    Err((prev, next))
                }
            })
            .collect()
    }

    /// Converts our broad `Turn` type into the wire format expected by Anthropic's messages API.
    pub(crate) fn into_anthropic_msgs(self) -> Vec<AnthropicMessage> {
        use itertools::Itertools;
        self.content
            .into_iter()
            .map(|m| match self.role {
                Role::User | Role::System => match m {
                    Message::Text { text } => AnthropicMessage::user_text(text),
                    _ => unreachable!(),
                },
                Role::Assistant => match m {
                    Message::Text { text } => AnthropicMessage::assistant_text(text),
                    // These will be combined to one msg during coalesce()
                    Message::ToolCall {
                        id,
                        name,
                        arguments,
                    } => AnthropicMessage::tool_use(
                        id,
                        name,
                        serde_json::from_str(&arguments).unwrap(),
                    ),
                    _ => unreachable!(),
                },
                Role::Tool => match m {
                    Message::ToolResult { id, result } => AnthropicMessage::tool_result(id, result),
                    _ => unreachable!(),
                },
            })
            // Combine tool call msgs with earlier Assistant msgs
            // if there was one, as they are expected together.
            .coalesce(|mut prev, next| {
                if prev.role == next.role {
                    prev.content.extend(next.content);
                    Ok(prev)
                } else {
                    Err((prev, next))
                }
            })
            .collect()
    }
}
/// The context of data in or out of the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Data in this turn represents user (non-LLM) input.
    #[default]
    User,
    /// Data in this turn represents the system prompt.
    System,
    /// Data in this turn was generated by the LLM.
    Assistant,
    /// Data in this turn represents the result of a tool call.
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
/// A unit of data written or read from the model.
pub enum Message {
    Text {
        /// Text tokens fed into or read from the model.
        text: String,
    },
    ToolCall {
        /// The identifier the model is using for this tool call.
        id: String,
        /// The name of the function the model wants to call.
        name: String,
        /// A (usually JSON-formatted) representation of the arguments the
        /// model is passing to this function call.
        arguments: String,
    },
    ToolResult {
        /// Corresponds to the ID set by the model in an earlier `ToolCall` message.
        id: String,
        /// A (usually JSON-formatted) representation of the result of the
        /// function call.
        result: String,
    },
}

impl Message {
    /// Creates a new `Text` message.
    pub fn text<T: Into<String>>(text: T) -> Self {
        Self::Text { text: text.into() }
    }
}

impl From<data_model::OAIToolCall> for Message {
    fn from(tc: data_model::OAIToolCall) -> Self {
        Self::ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
        }
    }
}

impl From<data_model::AnthropicCompletion> for Message {
    fn from(msg: data_model::AnthropicCompletion) -> Self {
        use data_model::AnthropicCompletion;
        match msg {
            AnthropicCompletion::Text { text } => Message::Text { text },
            AnthropicCompletion::ToolUse { id, name, input } => Message::ToolCall {
                id,
                name,
                arguments: input.to_string(),
            },
            AnthropicCompletion::ToolResult {
                tool_use_id,
                content,
            } => Message::ToolResult {
                id: tool_use_id,
                result: content,
            },
        }
    }
}

/// Describes the reason a model stopped providing tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    #[default]
    #[serde(alias = "end_turn")]
    Stop,
    #[serde(alias = "tool_use")]
    ToolCalls,
    #[serde(alias = "max_tokens")]
    Length,
    #[serde(alias = "refusal")]
    ContentFilter,
}

/// The response from the model for generating a single turn.
#[derive(Debug, Clone)]
pub struct CallResp {
    /// A provider-specific unique ID for this model call.
    pub id: String,
    /// Identifier for the model which was called.
    pub model: String,

    /// The reason the model stopped generating tokens.
    pub finish_reason: FinishReason,
    /// The tokens the model generated.
    pub content: Turn,
}

impl From<data_model::OAICompletionsResponse> for CallResp {
    fn from(resp: data_model::OAICompletionsResponse) -> Self {
        let finish_reason = resp.choices[0].finish_reason.clone();

        Self {
            id: resp.id,
            model: resp.model,
            finish_reason,
            content: resp.choices[0].message.clone().into(),
        }
    }
}

impl From<data_model::AnthropicMsgResponse> for CallResp {
    fn from(resp: data_model::AnthropicMsgResponse) -> Self {
        Self {
            id: resp.id,
            model: resp.model,
            finish_reason: resp.stop_reason,
            content: Turn {
                role: Role::Assistant,
                content: resp.content.into_iter().map(|m| m.into()).collect(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn smoke() {}
}
