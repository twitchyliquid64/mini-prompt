//! Helpers and utilities for extracting structured data from LLM output.

use markdown::mdast::{Code, Node};
use markdown::{to_mdast, ParseOptions};

#[allow(dead_code)]
#[derive(Debug, Clone)]
/// Describes how to extract a code section from a block of text.
pub struct MarkdownOptions<'a> {
    from_back: bool,
    require_lang: bool,
    lang: Option<&'a str>,
}

impl<'a> MarkdownOptions<'a> {
    pub fn json() -> Self {
        MarkdownOptions {
            from_back: true,
            require_lang: false,
            lang: Some(&"json"),
        }
    }

    pub fn python() -> Self {
        MarkdownOptions {
            from_back: true,
            require_lang: false,
            lang: Some(&"python"),
        }
    }

    pub fn leading(self) -> Self {
        MarkdownOptions {
            from_back: false,
            ..self
        }
    }
}

/// Extracts a leading or trailing markdown code block using the given opts as configuration.
pub fn markdown_codeblock(text: &str, opts: &MarkdownOptions) -> Option<String> {
    // TODO: Might want to deindent any global indentation if anything of the sort exists?

    let mut candidates = vec![to_mdast(text, &ParseOptions::default()).unwrap()];
    while let Some(node) = candidates.pop() {
        // println!("NEXT: {:?}", node);

        // Enqueue any nested markdown objects for consideration.
        match (node.children(), opts.from_back) {
            (Some(c), false) => {
                let mut c = c.iter().collect::<Vec<_>>();
                c.reverse();
                c.iter().for_each(|c| candidates.push((*c).clone()));
            }
            (Some(c), true) => {
                c.iter().for_each(|c| candidates.push((*c).clone()));
            }
            _ => {}
        }

        if let Node::Code(Code { value, lang, .. }) = node {
            match (lang, opts.lang, opts.require_lang) {
                (Some(lang), Some(want_lang), _) => {
                    if lang == want_lang {
                        return Some(value.as_str().into());
                    }
                }
                (None, _, false) => {
                    return Some(value.as_str().into());
                }
                _ => {}
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{markdown_codeblock, MarkdownOptions};

    #[test]
    fn find_markdown_json_trailing() {
        // Empty
        assert_eq!(
            markdown_codeblock(
                r"
            ",
                &MarkdownOptions::json()
            ),
            None
        );
        // Easy
        assert_eq!(
            markdown_codeblock(
                r#"
Some random earlier text.
```json
{"and": "blueberries"}
```
            "#,
                &MarkdownOptions::json()
            ),
            Some(r#"{"and": "blueberries"}"#.into()),
        );

        // Make sure order is respected
        assert_eq!(
            markdown_codeblock(
                r#"
```json
{"and": "swiggity swooty"}
```
Some random earlier text.
```json
{"and": "blueberries"}
```
            "#,
                &MarkdownOptions::json()
            ),
            Some(r#"{"and": "blueberries"}"#.into()),
        );
    }

    #[test]
    fn find_markdown_json_leading() {
        assert_eq!(
            markdown_codeblock(
                r#"
Merrpp
```json
{"and": "swiggity swooty"}
```
Some random earlier text.
```json
{"and": "blueberries"}
```
            "#,
                &MarkdownOptions::json().leading()
            ),
            Some(r#"{"and": "swiggity swooty"}"#.into()),
        );
    }
}
