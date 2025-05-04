use gibbs_core::parse::{markdown_codeblock, MarkdownOptions};
use gibbs_core::{model_call, Model};
use serde::Deserialize;

use indoc::indoc;

#[derive(Debug, Clone, Deserialize)]
pub struct Out {
    roots: Vec<f64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let resp =
        model_call(Model::Gemma27B3, indoc! {
            "Whats the roots of the equation y = 13x + 5x^2? Show your working, before reaching a final answer.
            
            At the end, output the final answer as JSON within triple backticks. Do not represent your working in the JSON."
        }).await?;
    println!("{}", resp);

    let json = markdown_codeblock(&resp, &MarkdownOptions::json()).unwrap();
    let p: Out = serde_json_lenient::from_str(&json)?;
    println!("json: {:?}", p.roots);

    Ok(())
}
