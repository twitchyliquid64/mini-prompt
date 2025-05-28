use mini_prompt::{callers, models, CallErr, ModelCaller, ToolInfo, ToolsSession};

use indoc::indoc;
use std::sync::{Arc, Mutex};

type M = models::Gemini25Flash;

#[tokio::main]
async fn main() -> Result<(), CallErr> {
    let flubb_count = Arc::new(Mutex::new(0usize));

    let resp = {
        let flubb_ref = flubb_count.clone();

        let backend = callers::Openrouter::<M>::default();
        let mut session = ToolsSession::new(
            backend,
            vec![
                (
                    ToolInfo::new("flubb", "Performs the flubb action.", None).into(),
                    Box::new(move |_args| {
                        (*(*flubb_ref).lock().unwrap()) += 1;
                        r#"{"status": "success", "message": "flubb completed successfully"}"#
                            .to_string()
                    }),
                ),
                (
                    ToolInfo::new("finish", "Finishes up; terminating the session.", None).into(),
                    Box::new(move |_args| "finished successfully.".to_string()),
                ),
            ],
        );

        session
            .simple_call(indoc! {
                "You are a concise AI assistant with access to a limited set of tools through which you can interact with the world.

                Use tool calling to flubb EXACTLY ONCE before finishing. Do not invoke the flubb tool more than once.

                DO NOT INVOKE THE FLUBB TOOL MORE THAN ONCE."
            })
            .await?
    };
    println!("{:?}", resp);
    println!("final flubb count: {}", *flubb_count.lock().unwrap());

    Ok(())
}
