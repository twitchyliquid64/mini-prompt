pub mod data_model;
use crate::data_model::{ChatMessage, CompletionsRequest, CompletionsResponse, Tool};

pub mod parse;

pub mod models;
pub use models::Model;

pub mod callers;
pub use callers::ModelCaller;

pub mod tools;

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn smoke() {}
}
