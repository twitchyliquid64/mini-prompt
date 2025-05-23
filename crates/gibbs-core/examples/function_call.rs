use gibbs_core::backends;
use gibbs_core::data_model::FunctionInfo;
use gibbs_core::{CompletionBackend, Model};

use indoc::indoc;

const MODEL: Model = Model::Gemini25Flash;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = backends::OpenrouterModel {
        model: MODEL,
        api_key: None,
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
