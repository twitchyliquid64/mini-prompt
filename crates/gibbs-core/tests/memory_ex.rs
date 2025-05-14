use gibbs_core::parse::TagOptions;
use gibbs_core::{model_call, Model};
use indoc::indoc;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct NewMem {
    text: String,
    date: Option<String>,
}

#[tokio::test]
#[ignore]
async fn memory_add() {
    let resp =
        model_call(Model::Gemini2Flash, indoc! {
            "You are an AI assistant chatting with a user. You respond to the user based on their message and background knowledge you are provided with. There are a limited set of actions
            which can be invoked in addition to the response you provide to the user, and the details and context of those actions are below.

            ## Background knowledge

            The current date is 2025-01-02.

            ## Additional actions

            Additional actions are invoked by outputting the name of the action between tags, and the JSON-formatted arguments of the action within those tags.
            For example: <example_action>{\"say\": \"hi\"}</example_action>
            
            ### Additional action: add_memory

            The `add_memory` action creates a memory. If there is a date around which the memory will be relevant, provide it in the JSON, otherwise omit the `date` key.

            JSON format: {\"text\": \"The text of the memory.\", \"date\": \"2025-03-22\"}

            Do not add memories if one already exists; edit that memory instead.

            ## Example: Responding to the user without invoking any additional actions

            User says: What's the current date?
            Response: Its the 2nd of January, 2025.
            
            ## Example: Adding a memory

            User says: Make a note that I have a piano recital on the 22nd of January.
            Response: I've added that to my records.
            <add_memory>{\"text\": \"Piano recital\", \"date\": \"2025-01-22\"}</add_memory>

            ## Your turn

            User says: Record that Julian is hosting another sausage party on the 11th of Feb
            Response:
            "
        }).await.unwrap();
    println!("{}", resp);

    let new_mems: Vec<NewMem> = TagOptions::from("add_memory")
        .iter(resp.as_str())
        .map(|s| serde_json_lenient::from_str(s).unwrap())
        .collect();

    for m in &new_mems {
        println!("create memory: {:?}", m);
    }

    assert_eq!(new_mems.len(), 1);
    assert!(new_mems[0]
        .text
        .trim()
        .to_lowercase()
        .contains("sausage party"));
    assert_eq!(
        new_mems[0].date.clone().map(|s| s.trim().to_lowercase()),
        Some("2025-02-11".to_string())
    );
}
