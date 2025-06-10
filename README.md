# mini-prompt

[![Crates.io](https://img.shields.io/crates/v/mini-prompt.svg)](https://crates.io/crates/mini-prompt) [![mini_prompt](https://docs.rs/mini-prompt/badge.svg)](https://docs.rs/mini-prompt)

Lightweight abstractions for using LLMs via a providers API.

Simple calls:
```rust
let mut backend = callers::Openrouter::<models::Gemma27B3>::default();
let resp =
    backend.simple_call("How much wood could a wood-chuck chop").await;
```

If you are looking for more control over the input, you can use [call](ModelCaller::call) instead of [simple_call](ModelCaller::simple_call).



With tools:
```rust
let backend = callers::Anthropic::<models::ClaudeHaiku35>::default();
let mut session = ToolsSession::new(
            backend,
            vec![
                (
                    ToolInfo::new("flubb", "Performs the flubb action.", None).into(),
                    Box::new(move |_args| {
                        r#"{"status": "success", "message": "flubb completed successfully"}"#
                            .to_string()
                    }),
                ),
            ],
        );

let resp =
    session.simple_call("Go ahead and flubb for me").await;
```

Structured output:
```rust
let mut backend = callers::Openrouter::<models::Gemma27B3>::default();
let resp =
    backend.simple_call("Whats 2+2? output the final answer as JSON within triple backticks (A markdown code block with json as the language).").await;

let json = markdown_codeblock(&resp.unwrap(), &MarkdownOptions::json()).unwrap();
let p: serde_json::Value = serde_json_lenient::from_str(&json).expect("json decode");
```

License: MIT OR Apache-2.0
