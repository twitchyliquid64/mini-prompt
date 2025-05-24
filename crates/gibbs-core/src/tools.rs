use crate::{ChatMessage, Tool};

/// A collection of tools a model can use.
pub trait Toolbox: Send {
    fn tools(&self) -> Vec<Tool>;

    fn tool_call(
        &mut self,
        name: &str,
        args: String,
    ) -> impl std::future::Future<Output = Result<ChatMessage, ()>> + Send;
}
