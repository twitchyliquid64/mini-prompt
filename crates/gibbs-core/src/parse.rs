//! Helpers and utilities for extracting structured data from LLM output.

use markdown::mdast::{Code, Node};
use markdown::{to_mdast, ParseOptions};

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

#[allow(dead_code)]
#[derive(Debug, Clone)]
/// Describes how to extract a multiclass/classification answer.
///
/// Excepts the LLM to output in the form `<key>: <chosen class>` towards
/// the end of its output.
pub struct EnumOptions<'a> {
    key: &'a str,
    classes: &'a [&'a str],
}

impl<'a> From<&'a [&'a str]> for EnumOptions<'a> {
    fn from(classes: &'a [&'a str]) -> Self {
        Self {
            key: &"answer",
            classes,
        }
    }
}

/// Extracts a trailing multiclass answer using the given opts as configuration.
///
/// ```
/// use gibbs_core::parse::multiclass;
/// multiclass("uwu\nanswer: query", &["query", "action"][..].into());
/// ```
pub fn multiclass<'a>(text: &str, opts: &'a EnumOptions) -> Option<&'a str> {
    let mut lines: Vec<String> = text.split("\n").map(|s| s.trim().to_lowercase()).collect();
    lines.reverse();

    for line in lines.into_iter() {
        if line.starts_with(&opts.key.to_lowercase()) {
            if let Some(":") = line.get(opts.key.len()..opts.key.len() + 1) {
                if let Some(answer) = line.get(opts.key.len() + 1..).map(|a| a.trim()) {
                    for class in opts.classes.iter() {
                        if class == &answer {
                            return Some(class);
                        }
                    }
                }
            }
        }
    }

    None
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
/// Describes how to extract an answer between HTML/XML tags.
///
/// Excepts the LLM to output in the form `<key>answer</key>` towards
/// the end of its output.
pub struct TagOptions<'a> {
    key: &'a str,
}

impl<'a> Default for TagOptions<'a> {
    fn default() -> Self {
        Self { key: &"answer" }
    }
}

/// Extracts an answer between tags, using the given opts as configuration.
///
/// If a tagged answer is present, the answer is returned as well as any remaining
/// text after the answer.
pub fn tagged<'a, 'b>(mut text: &'a str, opts: &'b TagOptions<'b>) -> Option<(&'a str, &'a str)> {
    while text.len() > 0 {
        let l_bracket = text.find('<');
        let has_key = l_bracket
            .map(|i| text[i + 1..].starts_with(opts.key))
            .unwrap_or(false);
        let has_close = l_bracket
            .map(|i| text[i + 1 + opts.key.len()..].starts_with('>'))
            .unwrap_or(false);

        if has_key && has_close {
            let body = &text[l_bracket.unwrap() + opts.key.len() + 2..];

            let l_bracket = body.find("</");
            let has_key = l_bracket
                .map(|i| body[i + 2..].starts_with(opts.key))
                .unwrap_or(false);
            let has_close = l_bracket
                .map(|i| body[i + 2 + opts.key.len()..].starts_with('>'))
                .unwrap_or(false);

            if has_key && has_close {
                return Some((
                    &body[..l_bracket.unwrap()],
                    &body[l_bracket.unwrap() + 3 + opts.key.len()..],
                ));
            }
            text = body;
            continue;
        }
        break;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{markdown_codeblock, MarkdownOptions, TagOptions};
    use super::{multiclass, tagged};
    use indoc::indoc;

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

    #[test]
    fn parse_multiclass_simple() {
        assert_eq!(
            multiclass(
                indoc! {
                    "So based on the given input, the answer is:
                    answer: query
                    uwu"
                },
                &["query", "action"][..].into()
            ),
            Some("query")
        );
    }

    #[test]
    fn parse_multiclass_respects_last() {
        assert_eq!(
            multiclass(
                indoc! {
                    "So based on the given input, the answer is:
                    answer: query
                    answer: action"
                },
                &["query", "action"][..].into()
            ),
            Some("action")
        );
    }

    #[test]
    fn parse_multiclass_missing_space_bad_caps() {
        assert_eq!(
            multiclass(
                indoc! {
                    "So based on the given input, the answer is:
                    answer: query
                    answer:AcTiON"
                },
                &["query", "action"][..].into()
            ),
            Some("action")
        );
    }

    #[test]
    fn parse_tagged_simple() {
        assert_eq!(
            tagged(
                indoc! {
                    "So based on the given input, the answer is:
                    <answer>query</answer>"
                },
                &TagOptions::default()
            ),
            Some(("query", ""))
        );
    }

    #[test]
    fn parse_tagged_multiple() {
        let a1 = tagged(
            indoc! {
                "So based on the given input, the answer is:
                    <answer>swiggity</answer>
                    <answer>swooty</answer>
                    "
            },
            &TagOptions::default(),
        );
        assert_eq!(a1, Some(("swiggity", "\n<answer>swooty</answer>\n")));
        assert_eq!(
            tagged(a1.unwrap().1, &TagOptions::default()),
            Some(("swooty", "\n"))
        );
    }

    #[test]
    fn parse_tagged_missing() {
        assert_eq!(
            tagged(
                indoc! {
                    "Some text
                    Where the tag is not to be found"
                },
                &TagOptions::default()
            ),
            None
        );
        assert_eq!(
            tagged(
                indoc! {
                    "The tag isnt present, but a different one is:
                    <blueberries>query</blueberries>"
                },
                &TagOptions::default()
            ),
            None
        );
    }
}
