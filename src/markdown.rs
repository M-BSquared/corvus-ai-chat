//! What a model writes, taken apart for the bubble: paragraphs, headings,
//! lists, code blocks, and bold and code inside a line. Anything else stays
//! the text it is.

/// A run of text inside a line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Span {
    Plain(String),
    Bold(String),
    Code(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    Text(Vec<Span>),
    Heading(Vec<Span>),
    /// A list entry: how far in it stands, its mark and its text.
    Item {
        depth: usize,
        mark: String,
        spans: Vec<Span>,
    },
    Code(String),
}

pub fn blocks(text: &str) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::new();
    let mut para = String::new();
    let mut lines = text.lines().peekable();
    let flush = |para: &mut String, out: &mut Vec<Block>| {
        if !para.trim().is_empty() {
            out.push(Block::Text(spans(para.trim())));
        }
        para.clear();
    };
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            flush(&mut para, &mut out);
            let mut code = Vec::new();
            for l in lines.by_ref() {
                if l.trim_start().starts_with("```") {
                    break;
                }
                code.push(l);
            }
            out.push(Block::Code(code.join("\n")));
        } else if trimmed.starts_with('|') {
            // A table keeps its columns in a font of one width.
            flush(&mut para, &mut out);
            let mut rows = vec![trimmed.to_string()];
            while let Some(l) = lines.next_if(|l| l.trim_start().starts_with('|')) {
                rows.push(l.trim_start().to_string());
            }
            out.push(Block::Code(rows.join("\n")));
        } else if let Some(rest) = heading(trimmed) {
            flush(&mut para, &mut out);
            out.push(Block::Heading(spans(rest)));
        } else if let Some((mark, rest)) = item(trimmed) {
            flush(&mut para, &mut out);
            out.push(Block::Item {
                depth: ((line.len() - trimmed.len()) / 2).min(4),
                mark,
                spans: spans(rest),
            });
        } else if trimmed.is_empty() {
            flush(&mut para, &mut out);
        } else {
            if !para.is_empty() {
                para.push('\n');
            }
            para.push_str(line.trim_end());
        }
    }
    flush(&mut para, &mut out);
    out
}

fn heading(line: &str) -> Option<&str> {
    let rest = line.trim_start_matches('#');
    let depth = line.len() - rest.len();
    (1..=6).contains(&depth).then(|| rest.strip_prefix(' ')).flatten()
}

fn item(line: &str) -> Option<(String, &str)> {
    for mark in ["- ", "* ", "+ ", "\u{2022} "] {
        if let Some(rest) = line.strip_prefix(mark) {
            return Some(("\u{2022}".into(), rest));
        }
    }
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    if (1..=3).contains(&digits) {
        let (number, rest) = line.split_at(digits);
        let rest = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") "))?;
        return Some((format!("{number}."), rest));
    }
    None
}

pub fn spans(text: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut plain = String::new();
    let mut rest = text;
    while !rest.is_empty() {
        let marked = [("**", true), ("`", false)].into_iter().find_map(|(m, bold)| {
            let inner = rest.strip_prefix(m)?;
            let end = inner.find(m).filter(|e| *e > 0)?;
            let (body, after) = (&inner[..end], &inner[end + m.len()..]);
            // A mark that stands alone, as in 2 ** 3, is not one.
            (!body.starts_with(' ') && !body.contains('\n')).then_some((body, after, bold))
        });
        match marked {
            Some((body, after, bold)) => {
                if !plain.is_empty() {
                    out.push(Span::Plain(std::mem::take(&mut plain)));
                }
                out.push(if bold {
                    Span::Bold(body.into())
                } else {
                    Span::Code(body.into())
                });
                rest = after;
            }
            None => {
                let mut chars = rest.chars();
                plain.extend(chars.next());
                rest = chars.as_str();
            }
        }
    }
    if !plain.is_empty() {
        out.push(Span::Plain(plain));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(s: &str) -> Vec<Span> {
        vec![Span::Plain(s.into())]
    }

    #[test]
    fn a_reply_is_taken_apart() {
        let got = blocks("# Battery\nAll **fine**, see `BAT1`.\nSecond line.\n\n- one\n  - two\n3. three\n\n```\ncode **x**\n```\n| a | b |\n|---|---|");
        assert_eq!(
            got,
            [
                Block::Heading(plain("Battery")),
                Block::Text(vec![
                    Span::Plain("All ".into()),
                    Span::Bold("fine".into()),
                    Span::Plain(", see ".into()),
                    Span::Code("BAT1".into()),
                    Span::Plain(".\nSecond line.".into()),
                ]),
                Block::Item {
                    depth: 0,
                    mark: "\u{2022}".into(),
                    spans: plain("one")
                },
                Block::Item {
                    depth: 1,
                    mark: "\u{2022}".into(),
                    spans: plain("two")
                },
                Block::Item {
                    depth: 0,
                    mark: "3.".into(),
                    spans: plain("three")
                },
                Block::Code("code **x**".into()),
                Block::Code("| a | b |\n|---|---|".into()),
            ]
        );
    }

    #[test]
    fn marks_that_are_not_marks_stay_text() {
        for s in [
            "2 ** 3 ** 4",
            "a ` b",
            "**",
            "``",
            "#tag",
            "-5 m",
            "1.5 m",
            "** open",
            "**\u{fc}",
        ] {
            assert_eq!(blocks(s), [Block::Text(plain(s))], "{s}");
        }
        assert!(blocks(" \n\n").is_empty());
        assert_eq!(blocks("```\nnever closed"), [Block::Code("never closed".into())]);
    }
}
