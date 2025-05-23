use gibbs_core::parse::TagOptions;
use gibbs_core::{model_call, ModelRef};
use indoc::indoc;
use serde::Deserialize;

const MODEL: ModelRef = ModelRef::Gemini2Flash;

const BASE: &str = indoc! {
    "You are an AI assistant chatting with a user. You respond to the user based on their message and background knowledge you are provided with.
     - Memories are provided which detail important information, and which can be added to, edited, and deleted via additional actions.
     - There are a limited set of additional actions which can be invoked in addition to the response you provide to the user, and the details and context of those actions are below.

    ## Memories

     - [Date: 2025-04-11][ID: 412]: Moving day

    ## Additional actions

    Additional actions are invoked by outputting the name of the action between tags, and the JSON-formatted arguments of the action within those tags.
    For example: <example_action>{\"say\": \"hi\"}</example_action>
    
    ### Additional action: add_memory

    The `add_memory` action creates a memory. If there is a date around which the memory will be relevant, provide it in the JSON, otherwise omit the `date` key.

    JSON format: {\"text\": \"The text of the memory.\", \"date\": \"2025-03-22\"}

    Do not add memories if one already exists; edit that memory instead.

    ### Additional action: edit_memory

    The `edit_memory` action edits the text or date of an memory, referencing it by ID. The ID key must be included, and it must match the ID of the memory you are editing.
    Either the text or date keys may be omitted if they do not need to be changed.

    JSON format: {\"id\": 3, \"text\": \"The text of the memory.\", \"date\": \"2025-03-22\"}

    ### Additional action: delete_memory

    The `delete_memory` action deletes a memory, referencing it by ID.
    The body of the tag is a JSON-formatted array of ID's to be deleted.

    JSON format: [<ID>]

    ## Example: Responding to the user without invoking any additional actions

    User says: What's the current date?
    Response: Its the 2nd of January, 2025.
    
    ## Example: Adding a memory

    User says: Make a note that I have a piano recital on the 22nd of January.
    Response: I've added that to my records.
    <add_memory>{\"text\": \"Piano recital\", \"date\": \"2025-01-22\"}</add_memory>

    ## Example: Editing a memory

    Assuming a memory about a jazz rehearsal exists, with an ID of 5238. 

    User says: The jazz rehearsal was rescheduled to the 19th of June
    Response: I've updated my records.
    <edit_memory>{\"id\": 5238, \"date\": \"2025-06-19\"}</edit_memory>

    ## Example: Deleting a memory

    Assuming a memory about a jazz rehearsal exists, with an ID of 5238. 

    User says: The jazz rehearsal was cancelled, update your records
    Response: I've updated my records.
    <delete_memory>[5238]</delete_memory>

    ## Background knowledge

    The current date is 2025-01-02.
    "
};

#[derive(Debug, Clone, Deserialize)]
pub struct NewMem {
    text: String,
    date: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EditMem {
    id: usize,
    text: Option<String>,
    date: Option<String>,
}

#[tokio::test]
#[ignore]
async fn memory_add() {
    let resp = model_call(
        MODEL,
        String::from(BASE)
            + indoc! {
                "## Your turn

                User says: Record that Julian is hosting another sausage party on the 11th of Feb
                Response:
                "
            },
    )
    .await
    .unwrap();
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

#[tokio::test]
#[ignore]
async fn memory_edit() {
    let resp = model_call(
        MODEL,
        String::from(BASE)
            + indoc! {
                "## Your turn

                User says: My move has been postponed to the 23rd of July.
                Response:
                "
            },
    )
    .await
    .unwrap();
    println!("{}", resp);

    let edit_mems: Vec<EditMem> = TagOptions::from("edit_memory")
        .iter(resp.as_str())
        .map(|s| serde_json_lenient::from_str(s).unwrap())
        .collect();

    for m in &edit_mems {
        println!("edit memory: {:?}", m);
    }

    assert_eq!(edit_mems.len(), 1);
    assert_eq!(edit_mems[0].id, 412);
    assert_eq!(
        edit_mems[0].date.clone().map(|s| s.trim().to_lowercase()),
        Some("2025-07-23".to_string())
    );
    assert!(edit_mems[0].text.is_none() || edit_mems[0].text.as_ref().unwrap() == "Moving day");
}

#[tokio::test]
#[ignore]
async fn memory_delete() {
    let resp = model_call(
        MODEL,
        String::from(BASE)
            + indoc! {
                "## Your turn

                User says: My move has been cancelled, I'm saying in my current apartment.
                Response:
                "
            },
    )
    .await
    .unwrap();
    println!("{}", resp);

    let delete_mems: Vec<Vec<usize>> = TagOptions::from("delete_memory")
        .iter(resp.as_str())
        .map(|s| serde_json_lenient::from_str(s).unwrap())
        .collect();

    for m in &delete_mems {
        println!("delete memory: {:?}", m);
    }

    assert_eq!(delete_mems[0].len(), 1);
    assert_eq!(delete_mems[0][0], 412);
}
