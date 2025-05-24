use gibbs_core::data_model::FunctionInfo;
use gibbs_core::{callers, models, Model, ModelCaller};

use indoc::indoc;

type M = models::Gemini25Flash;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = callers::Openrouter::<M>::default();

    let resp = backend
        .call(
            vec![M::default().make_prompt(
                indoc! {
                    "Use tools to flubb once before finishing."
                }
                .to_string(),
            )],
            vec![
                FunctionInfo::new("flubb", "Performs the flubb action.", None).into(),
                FunctionInfo::new("finish", "Finishes up; terminating the session.", None).into(),
            ],
        )
        .await?;
    println!("{:?}", resp);

    Ok(())
}
