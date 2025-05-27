pub mod data_model;
use crate::data_model::{ChatMessage, CompletionsRequest, CompletionsResponse, Tool};

pub mod parse;

pub mod models;
pub use models::Model;

pub mod callers;
pub use callers::ModelCaller;

pub mod tools;
pub use tools::ToolsSession;

pub enum CallErr {
    NoCompletions,
    API(reqwest::Error),
    Other(Box<dyn std::error::Error>),
}

impl std::fmt::Debug for CallErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallErr::NoCompletions => write!(f, "NoCompletions"),
            CallErr::API(err) => f.debug_tuple("API").field(err).finish(),
            CallErr::Other(err) => f.debug_tuple("Other").field(err).finish(),
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

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn smoke() {}
}
