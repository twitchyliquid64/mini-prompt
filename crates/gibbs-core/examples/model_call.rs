use gibbs_core::{model_call, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let resp =
        model_call(Model::Gemma27B3, "Whats the roots of the equation y = 3x + 2x^2? Show your working. Output the final answer as JSON within triple backticks.").await?;
    println!("{}", resp);
    Ok(())
}
