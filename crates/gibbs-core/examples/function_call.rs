use gibbs_core::callers;
use gibbs_core::data_model::FunctionInfo;
use gibbs_core::{ModelCaller, ModelRef};

use indoc::indoc;

const MODEL: ModelRef = ModelRef::Gemini25Flash;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = callers::Openrouter {
        model: MODEL,
        ..Default::default()
    };

    let resp = backend
        .call(
            vec![MODEL.make_prompt(indoc! {
                "Use tools to flubb once before finishing."
            })],
            vec![
                FunctionInfo::new("flubb", "Performs the flubb action.", None).into(),
                FunctionInfo::new("finish", "Finishes up; terminating the session.", None).into(),
            ],
        )
        .await?;
    println!("{:?}", resp);

    Ok(())
}
