//! The rules of the chat, with no window and no network in them: where a model
//! lives, what a request to it looks like in the two dialects, how its answer is
//! read, how Corvus' MCP tools become the model's tools, and how much of a long
//! conversation is sent again.

use serde_json::{json, Map, Value};

/// How many times a model may ask for tools in one turn before the turn is
/// ended, so a model that never stops asking cannot fly the aircraft forever.
pub const MAX_ROUNDS: usize = 24;
/// The most of one tool answer the model is given.
pub const TOOL_TEXT_MAX: usize = 16_000;
/// The most of one tool answer or argument list shown in the chat.
pub const SHOWN_MAX: usize = 600;
/// About how much of the conversation is sent with each request, in characters.
pub const HISTORY_CHARS: usize = 120_000;
/// The most entries the chat keeps.
pub const ITEMS_MAX: usize = 400;
/// The default for "Reply length" when the operator leaves it at 0 and the
/// dialect needs one.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

pub const DEFAULT_SYSTEM: &str = "You are the assistant built into Corvus GCS, a ground control station for PX4 and ArduPilot aircraft. You have the station's tools: use them, do not describe what could be done. Draw on the map when you talk about places, directions or routes. Read before you act: look at the telemetry and the flight mode first. Never arm, take off, change a parameter, start a mission or move the aircraft unless the operator clearly asked for exactly that, and say in one sentence what you are about to do before you do it. If a tool is refused or needs the operator's confirmation, tell the operator instead of looking for another way. Answer briefly and plainly, in the language the operator writes in.";

/// The operator's instructions, then what the station tells its clients.
pub fn system_with(own: &str, station: &str) -> String {
    let (own, station) = (own.trim(), station.trim());
    if station.is_empty() {
        own.to_string()
    } else if own.is_empty() {
        station.to_string()
    } else {
        format!("{own}\n\n{station}")
    }
}

/// The two ways of talking to a model that nearly every server and every local
/// runtime offers one of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// `POST /v1/chat/completions` (OpenAI, Ollama, LM Studio, llama.cpp, vLLM).
    OpenAi,
    /// `POST /v1/messages` (Anthropic and servers that copy it).
    Anthropic,
}

impl Provider {
    pub fn from_option(v: &str) -> Provider {
        if v == "anthropic" {
            Provider::Anthropic
        } else {
            Provider::OpenAi
        }
    }
}

/// Where a model server is: `scheme://host[:port][/base]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    pub tls: bool,
    pub host: String,
    pub port: u16,
    /// What comes before the dialect's own path, without a trailing slash.
    pub base: String,
}

impl Endpoint {
    /// Reads the address the operator typed. A missing scheme is http, which is
    /// what a model on this computer or the local network speaks.
    pub fn parse(text: &str) -> Result<Endpoint, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("No address for the model is set.".into());
        }
        let (tls, rest) = if let Some(r) = text.strip_prefix("https://") {
            (true, r)
        } else if let Some(r) = text.strip_prefix("http://") {
            (false, r)
        } else if text.contains("://") {
            return Err(format!("{text}: only http:// and https:// addresses work."));
        } else {
            (false, text)
        };
        if rest.contains(['?', '#', ' ']) {
            return Err(format!(
                "{text}: the address must not hold a query, a fragment or a space."
            ));
        }
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        };
        let default_port = if tls { 443 } else { 80 };
        let bad_port = |p: &str| format!("{text}: {p:?} is not a port.");
        let (host, port) = if let Some(inner) = authority.strip_prefix('[') {
            let (h, after) = inner.split_once(']').ok_or_else(|| format!("{text}: unclosed ["))?;
            let port = match after.strip_prefix(':') {
                Some(p) => p.parse::<u16>().map_err(|_| bad_port(p))?,
                None if after.is_empty() => default_port,
                None => return Err(format!("{text}: expected host:port.")),
            };
            (h.to_string(), port)
        } else {
            match authority.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), p.parse::<u16>().map_err(|_| bad_port(p))?),
                None => (authority.to_string(), default_port),
            }
        };
        if host.is_empty() {
            return Err(format!("{text}: no host."));
        }
        Ok(Endpoint {
            tls,
            host,
            port,
            base: path.trim_end_matches('/').to_string(),
        })
    }

    /// The `Host` header value.
    pub fn host_header(&self) -> String {
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        if self.port == if self.tls { 443 } else { 80 } {
            host
        } else {
            format!("{host}:{}", self.port)
        }
    }

    /// For showing: `host:port` of the model.
    pub fn label(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// The path of the dialect's chat call. A bare address gets the dialect's
    /// usual `/v1`, so `http://localhost:11434` is enough for Ollama.
    pub fn chat_path(&self, provider: Provider) -> String {
        let v1 = self.base.ends_with("/v1");
        match provider {
            Provider::OpenAi if self.base.is_empty() => "/v1/chat/completions".into(),
            Provider::OpenAi => format!("{}/chat/completions", self.base),
            Provider::Anthropic if v1 => format!("{}/messages", self.base),
            Provider::Anthropic => format!("{}/v1/messages", self.base),
        }
    }

    /// The path of the list of models.
    pub fn models_path(&self) -> String {
        if self.base.is_empty() {
            "/v1/models".into()
        } else {
            format!("{}/models", self.base)
        }
    }
}

/// One tool call a model asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub id: String,
    /// The name the model used, which is [`Tool::name`].
    pub name: String,
    pub args: Value,
}

/// One entry of the conversation as the model sees it.
#[derive(Debug, Clone, PartialEq)]
pub enum Msg {
    User(String),
    Assistant {
        text: String,
        calls: Vec<Call>,
    },
    Tool {
        id: String,
        name: String,
        text: String,
        error: bool,
    },
}

/// One of Corvus' MCP tools as the model is offered it.
#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    /// What the model calls it: letters, digits, `_` and `-`, 64 at most.
    pub name: String,
    /// What the MCP server calls it.
    pub mcp_name: String,
    pub description: String,
    pub schema: Value,
}

fn safe_name(raw: &str) -> String {
    let s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    if s.is_empty() {
        "tool".into()
    } else {
        s
    }
}

/// The tools of an MCP `tools/list` answer, with names a model accepts.
pub fn tools_from_mcp(list: &Value) -> Vec<Tool> {
    let mut out: Vec<Tool> = Vec::new();
    for t in list.get("tools").and_then(Value::as_array).into_iter().flatten() {
        let Some(mcp_name) = t.get("name").and_then(Value::as_str) else {
            continue;
        };
        let mut name = safe_name(mcp_name);
        let mut n = 2;
        while out.iter().any(|o| o.name == name) {
            let tail = format!("_{n}");
            name = format!(
                "{}{tail}",
                safe_name(&mcp_name.chars().take(64 - tail.len()).collect::<String>())
            );
            n += 1;
        }
        let mut schema = t.get("inputSchema").cloned().unwrap_or(Value::Null);
        if !schema.is_object() {
            schema = json!({"type": "object", "properties": {}});
        } else if schema.get("type").is_none() {
            schema["type"] = json!("object");
        }
        out.push(Tool {
            name,
            mcp_name: mcp_name.to_string(),
            description: t.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
            schema,
        });
    }
    out
}

/// The tools a small model is offered: reading, drawing and plain flying. The whole
/// list takes about eleven thousand tokens, more than many local models are run with.
const FEW: [&str; 24] = [
    "get_vehicle_state",
    "get_capabilities",
    "list_flight_modes",
    "download_mission",
    "get_parameters",
    "wait_for",
    "measure",
    "draw_on_map",
    "list_drawings",
    "remove_drawings",
    "show_on_map",
    "arm",
    "disarm",
    "takeoff",
    "land",
    "return_to_launch",
    "hold",
    "set_flight_mode",
    "fly_to",
    "move_by",
    "change_altitude",
    "set_speed",
    "set_heading",
    "orbit",
];

/// How many of the station's tools the model is offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Offer {
    All,
    Few,
}

impl Offer {
    /// The operator's choice; left to itself a model on plain http (this computer or
    /// the local network) gets the short list and a hosted one everything.
    pub fn from_option(v: &str, endpoint: &Endpoint) -> Offer {
        match v {
            "all" => Offer::All,
            "few" => Offer::Few,
            _ if endpoint.tls => Offer::All,
            _ => Offer::Few,
        }
    }

    /// `tools` as they are offered. A list with none of the short list's names in it
    /// is offered whole.
    pub fn of(self, tools: Vec<Tool>) -> Vec<Tool> {
        if self == Offer::All || !tools.iter().any(|t| FEW.contains(&t.mcp_name.as_str())) {
            return tools;
        }
        tools
            .into_iter()
            .filter(|t| FEW.contains(&t.mcp_name.as_str()))
            .collect()
    }
}

/// What an MCP `tools/call` answered, as text and whether it is a failure.
pub fn tool_text(result: &Value) -> (String, bool) {
    let error = result.get("isError") == Some(&Value::Bool(true));
    let mut parts: Vec<String> = result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|c| c.get("text").and_then(Value::as_str).map(str::to_string))
        .collect();
    if parts.is_empty() {
        if let Some(s) = result.get("structuredContent").filter(|s| !s.is_null()) {
            parts.push(s.to_string());
        }
    }
    let text = parts.join("\n");
    (if text.is_empty() { "(no answer)".into() } else { text }, error)
}

/// `text` cut to `max` characters with an ellipsis.
pub fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        let mut s: String = text.chars().take(max).collect();
        s.push('\u{2026}');
        s
    }
}

/// A call's arguments on one line, for the chat.
pub fn args_line(args: &Value) -> String {
    match args {
        Value::Object(m) if m.is_empty() => String::new(),
        Value::Null => String::new(),
        other => clip(&other.to_string(), 160),
    }
}

/// Reasoning some models print in front of the answer, between `<think>` tags.
pub fn strip_think(text: &str) -> String {
    let mut rest = text.trim_start();
    while let Some(after) = rest.strip_prefix("<think>") {
        match after.find("</think>") {
            Some(end) => rest = after[end + "</think>".len()..].trim_start(),
            None => return String::new(),
        }
    }
    rest.trim_end().to_string()
}

fn tool_json_openai(t: &Tool) -> Value {
    json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}})
}

/// The body of a chat completions request.
pub fn openai_body(model: &str, system: &str, msgs: &[Msg], tools: &[Tool], max_tokens: u32) -> Value {
    let mut messages = Vec::new();
    if !system.trim().is_empty() {
        messages.push(json!({"role": "system", "content": system}));
    }
    for m in msgs {
        match m {
            Msg::User(t) => messages.push(json!({"role": "user", "content": t})),
            Msg::Assistant { text, calls } => {
                let mut o = Map::new();
                o.insert("role".into(), json!("assistant"));
                o.insert(
                    "content".into(),
                    if text.is_empty() { Value::Null } else { json!(text) },
                );
                if !calls.is_empty() {
                    let list: Vec<Value> = calls
                        .iter()
                        .map(|c| {
                            json!({"id": c.id, "type": "function",
                                   "function": {"name": c.name, "arguments": c.args.to_string()}})
                        })
                        .collect();
                    o.insert("tool_calls".into(), Value::Array(list));
                }
                messages.push(Value::Object(o));
            }
            Msg::Tool { id, text, .. } => messages.push(json!({"role": "tool", "tool_call_id": id, "content": text})),
        }
    }
    let mut body = json!({"model": model, "messages": messages});
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools.iter().map(tool_json_openai).collect());
    }
    if max_tokens > 0 {
        body["max_tokens"] = json!(max_tokens);
    }
    body
}

/// The body of a messages request.
pub fn anthropic_body(model: &str, system: &str, msgs: &[Msg], tools: &[Tool], max_tokens: u32) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    for m in msgs {
        match m {
            Msg::User(t) => messages.push(json!({"role": "user", "content": t})),
            Msg::Assistant { text, calls } => {
                let mut blocks = Vec::new();
                if !text.is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for c in calls {
                    blocks.push(json!({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}));
                }
                if blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": "(no answer)"}));
                }
                messages.push(json!({"role": "assistant", "content": blocks}));
            }
            Msg::Tool { id, text, error, .. } => {
                let block = json!({"type": "tool_result", "tool_use_id": id, "content": text, "is_error": error});
                // Every answer to one round of calls travels in one user message.
                let joined = messages.last_mut().filter(|l| {
                    l["role"] == "user"
                        && l["content"]
                            .as_array()
                            .is_some_and(|a| a.iter().all(|b| b["type"] == "tool_result"))
                });
                match joined.and_then(|l| l["content"].as_array_mut()) {
                    Some(list) => list.push(block),
                    None => messages.push(json!({"role": "user", "content": [block]})),
                }
            }
        }
    }
    let mut body = json!({
        "model": model,
        "max_tokens": if max_tokens > 0 { max_tokens } else { DEFAULT_MAX_TOKENS },
        "messages": messages,
    });
    if !system.trim().is_empty() {
        body["system"] = json!(system);
    }
    if !tools.is_empty() {
        let list: Vec<Value> = tools
            .iter()
            .map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.schema}))
            .collect();
        body["tools"] = Value::Array(list);
    }
    body
}

/// What a model answered.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Reply {
    pub text: String,
    pub calls: Vec<Call>,
    /// The model ran out of room before it was done.
    pub cut: bool,
}

fn args_value(raw: &Value) -> Value {
    match raw {
        Value::String(s) if s.trim().is_empty() => json!({}),
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({"_unreadable_arguments": s})),
        Value::Null => json!({}),
        other => other.clone(),
    }
}

/// A chat completions answer.
pub fn parse_openai(v: &Value) -> Result<Reply, String> {
    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or("The model's answer holds no choice.")?;
    let message = choice.get("message").ok_or("The model's answer holds no message.")?;
    let text = match message.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    let mut calls = Vec::new();
    for (i, c) in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let f = c.get("function").unwrap_or(c);
        let Some(name) = f.get("name").and_then(Value::as_str) else {
            continue;
        };
        let id = c
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map_or_else(|| format!("call_{i}"), str::to_string);
        calls.push(Call {
            id,
            name: name.to_string(),
            args: args_value(f.get("arguments").unwrap_or(&Value::Null)),
        });
    }
    Ok(Reply {
        text: strip_think(&text),
        calls,
        cut: choice.get("finish_reason").and_then(Value::as_str) == Some("length"),
    })
}

/// A messages answer.
pub fn parse_anthropic(v: &Value) -> Result<Reply, String> {
    let blocks = v
        .get("content")
        .and_then(Value::as_array)
        .ok_or("The model's answer holds no content.")?;
    let mut text = String::new();
    let mut calls = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("tool_use") => {
                let Some(name) = b.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let id = b
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map_or_else(|| format!("call_{i}"), str::to_string);
                calls.push(Call {
                    id,
                    name: name.to_string(),
                    args: args_value(b.get("input").unwrap_or(&Value::Null)),
                });
            }
            _ => {}
        }
    }
    Ok(Reply {
        text: strip_think(&text),
        calls,
        cut: v.get("stop_reason").and_then(Value::as_str) == Some("max_tokens"),
    })
}

/// The reason in an error body of either dialect, or the status.
pub fn error_text(status: u16, body: &str) -> String {
    let from_json = serde_json::from_str::<Value>(body).ok().and_then(|v| {
        let e = v.get("error")?;
        e.get("message")
            .and_then(Value::as_str)
            .or_else(|| e.as_str())
            .map(str::to_string)
    });
    let why = from_json.unwrap_or_else(|| clip(body.trim(), 300));
    let hint = match status {
        401 | 403 => " Check the key.",
        404 => " Check the address and the model name.",
        429 => " The server asks to slow down.",
        _ => "",
    };
    if why.is_empty() {
        format!("The model's server answered {status}.{hint}")
    } else {
        format!("The model's server answered {status}: {why}.{hint}")
    }
}

/// The ids of a list of models.
pub fn model_ids(v: &Value) -> Vec<String> {
    let list = v.get("data").or_else(|| v.get("models")).and_then(Value::as_array);
    list.into_iter()
        .flatten()
        .filter_map(|m| m.get("id").or_else(|| m.get("name")).and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn size(m: &Msg) -> usize {
    match m {
        Msg::User(t) => t.len(),
        Msg::Assistant { text, calls } => {
            text.len() + calls.iter().map(|c| c.args.to_string().len() + 40).sum::<usize>()
        }
        Msg::Tool { text, .. } => text.len(),
    }
}

/// The end of the conversation that fits `budget` characters, starting at one
/// of the operator's own messages so no answer comes without its question.
pub fn window(msgs: &[Msg], budget: usize) -> &[Msg] {
    let mut used = 0;
    let mut start = msgs.len();
    for (i, m) in msgs.iter().enumerate().rev() {
        used += size(m);
        if used > budget && i + 1 < msgs.len() {
            break;
        }
        start = i;
    }
    match msgs[start..].iter().position(|m| matches!(m, Msg::User(_))) {
        Some(n) => &msgs[start + n..],
        None => &msgs[msgs.len().saturating_sub(1)..],
    }
}

/// Gives every call that never got an answer (the turn was stopped, or the
/// network went) a failed one right after it, because a model refuses a
/// conversation with a call left open.
pub fn repair(msgs: &mut Vec<Msg>) {
    let mut i = 0;
    while i < msgs.len() {
        if let Msg::Assistant { calls, .. } = &msgs[i] {
            let calls = calls.clone();
            let mut end = i + 1;
            while matches!(msgs.get(end), Some(Msg::Tool { .. })) {
                end += 1;
            }
            let answered: Vec<String> = msgs[i + 1..end]
                .iter()
                .filter_map(|m| match m {
                    Msg::Tool { id, .. } => Some(id.clone()),
                    _ => None,
                })
                .collect();
            for c in calls.iter().filter(|c| !answered.contains(&c.id)) {
                msgs.insert(
                    end,
                    Msg::Tool {
                        id: c.id.clone(),
                        name: c.name.clone(),
                        text: "The call did not run: the turn was stopped.".into(),
                        error: true,
                    },
                );
                end += 1;
            }
            i = end;
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_model_gets_the_short_list_unless_told_otherwise() {
        let local = Endpoint::parse("http://localhost:11434").unwrap();
        let hosted = Endpoint::parse("https://api.example.com").unwrap();
        assert_eq!(Offer::from_option("auto", &local), Offer::Few);
        assert_eq!(Offer::from_option("", &hosted), Offer::All);
        assert_eq!(Offer::from_option("all", &local), Offer::All);
        assert_eq!(Offer::from_option("few", &hosted), Offer::Few);
        let tools = |names: &[&str]| {
            let list: Vec<Value> = names.iter().map(|n| json!({"name": n})).collect();
            tools_from_mcp(&json!({ "tools": list }))
        };
        let short = Offer::Few.of(tools(&["hold", "tak_status", "draw_on_map"]));
        let names: Vec<&str> = short.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["hold", "draw_on_map"]);
        assert_eq!(Offer::All.of(tools(&["hold", "tak_status"])).len(), 2);
        assert_eq!(Offer::Few.of(tools(&["other"])).len(), 1, "never nothing at all");
    }

    fn ep(t: &str) -> Endpoint {
        Endpoint::parse(t).unwrap()
    }

    #[test]
    fn an_address_is_read_with_its_defaults() {
        assert_eq!(
            ep("http://localhost:1234/v1/"),
            Endpoint {
                tls: false,
                host: "localhost".into(),
                port: 1234,
                base: "/v1".into()
            }
        );
        assert_eq!(ep("localhost:11434").port, 11434);
        assert!(!ep("localhost:11434").tls);
        let c = ep("https://api.example.com");
        assert_eq!((c.tls, c.port, c.base.as_str()), (true, 443, ""));
        assert_eq!(ep("http://[::1]:8080").host, "::1");
        assert_eq!(ep("http://[::1]:8080").host_header(), "[::1]:8080");
        assert_eq!(c.host_header(), "api.example.com");
    }

    #[test]
    fn a_bad_address_says_why() {
        assert!(Endpoint::parse("").is_err());
        assert!(Endpoint::parse("ftp://x").unwrap_err().contains("only http"));
        assert!(Endpoint::parse("http://x:99999").unwrap_err().contains("not a port"));
        assert!(Endpoint::parse("http://x/v1?key=1").is_err());
        assert!(Endpoint::parse("http://:80").is_err());
    }

    #[test]
    fn the_chat_path_follows_the_dialect_and_the_base() {
        assert_eq!(ep("http://h:1").chat_path(Provider::OpenAi), "/v1/chat/completions");
        assert_eq!(ep("http://h:1/v1").chat_path(Provider::OpenAi), "/v1/chat/completions");
        assert_eq!(
            ep("http://h:1/api/openai").chat_path(Provider::OpenAi),
            "/api/openai/chat/completions"
        );
        assert_eq!(
            ep("https://api.anthropic.com").chat_path(Provider::Anthropic),
            "/v1/messages"
        );
        assert_eq!(ep("http://h:1/v1").chat_path(Provider::Anthropic), "/v1/messages");
        assert_eq!(ep("http://h:1/v1").models_path(), "/v1/models");
        assert_eq!(ep("http://h:1").models_path(), "/v1/models");
    }

    #[test]
    fn mcp_tools_get_names_a_model_accepts() {
        let list = json!({"tools": [
            {"name": "get_telemetry", "description": "d", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "plugin.x/y z", "inputSchema": {"properties": {"a": {"type": "string"}}}},
            {"name": "plugin_x_y_z"},
            {"description": "no name"},
        ]});
        let t = tools_from_mcp(&list);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].name, "get_telemetry");
        assert_eq!(t[1].name, "plugin_x_y_z");
        assert_eq!(t[1].mcp_name, "plugin.x/y z");
        assert_eq!(t[1].schema["type"], "object");
        assert_eq!(t[2].name, "plugin_x_y_z_2");
        assert_eq!(t[2].schema, json!({"type": "object", "properties": {}}));
    }

    #[test]
    fn a_tool_answer_becomes_text() {
        let ok = json!({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]});
        assert_eq!(tool_text(&ok), ("a\nb".to_string(), false));
        let bad = json!({"content": [{"type": "text", "text": "no"}], "isError": true});
        assert_eq!(tool_text(&bad), ("no".to_string(), true));
        assert_eq!(tool_text(&json!({})).0, "(no answer)");
        assert_eq!(tool_text(&json!({"structuredContent": {"a": 1}})).0, "{\"a\":1}");
    }

    fn convo() -> Vec<Msg> {
        vec![
            Msg::User("battery?".into()),
            Msg::Assistant {
                text: "Looking.".into(),
                calls: vec![
                    Call {
                        id: "c1".into(),
                        name: "get_telemetry".into(),
                        args: json!({"a": 1}),
                    },
                    Call {
                        id: "c2".into(),
                        name: "get_status".into(),
                        args: json!({}),
                    },
                ],
            },
            Msg::Tool {
                id: "c1".into(),
                name: "get_telemetry".into(),
                text: "82%".into(),
                error: false,
            },
            Msg::Tool {
                id: "c2".into(),
                name: "get_status".into(),
                text: "off".into(),
                error: true,
            },
            Msg::Assistant {
                text: "82 percent.".into(),
                calls: vec![],
            },
        ]
    }

    fn tool() -> Tool {
        Tool {
            name: "get_telemetry".into(),
            mcp_name: "get_telemetry".into(),
            description: "Read".into(),
            schema: json!({"type": "object"}),
        }
    }

    #[test]
    fn the_openai_body_carries_calls_as_text_arguments() {
        let b = openai_body("m", "sys", &convo(), &[tool()], 0);
        let m = b["messages"].as_array().unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"a\":1}");
        assert_eq!(m[2]["tool_calls"][1]["id"], "c2");
        assert_eq!(m[3], json!({"role": "tool", "tool_call_id": "c1", "content": "82%"}));
        assert_eq!(b["tools"][0]["function"]["name"], "get_telemetry");
        assert!(b.get("max_tokens").is_none());
        assert_eq!(openai_body("m", "", &[], &[], 50)["max_tokens"], 50);
        assert_eq!(
            openai_body("m", "", &[Msg::User("x".into())], &[], 0)["messages"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn the_anthropic_body_joins_the_answers_of_one_round() {
        let b = anthropic_body("m", "sys", &convo(), &[tool()], 0);
        assert_eq!(b["system"], "sys");
        assert_eq!(b["max_tokens"], DEFAULT_MAX_TOKENS);
        let m = b["messages"].as_array().unwrap();
        assert_eq!(m.len(), 4);
        assert_eq!(m[1]["content"][1]["type"], "tool_use");
        assert_eq!(m[1]["content"][1]["input"], json!({"a": 1}));
        let results = m[2]["content"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[1]["tool_use_id"], "c2");
        assert_eq!(results[1]["is_error"], true);
        assert_eq!(b["tools"][0]["input_schema"], json!({"type": "object"}));
    }

    #[test]
    fn an_openai_answer_is_read_with_calls_in_either_shape() {
        let v = json!({"choices": [{"finish_reason": "tool_calls", "message": {
        "content": null,
        "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "t", "arguments": "{\"x\": 2}"}},
            {"function": {"name": "u", "arguments": {"y": 3}}},
            {"id": "z", "function": {"name": "v", "arguments": ""}},
            {"id": "q", "function": {"name": "w", "arguments": "{broken"}},
        ]}}]});
        let r = parse_openai(&v).unwrap();
        assert_eq!(r.text, "");
        assert!(!r.cut);
        assert_eq!(r.calls[0].args, json!({"x": 2}));
        assert_eq!((r.calls[1].id.as_str(), &r.calls[1].args), ("call_1", &json!({"y": 3})));
        assert_eq!(r.calls[2].args, json!({}));
        assert_eq!(r.calls[3].args, json!({"_unreadable_arguments": "{broken"}));
        let cut = json!({"choices": [{"finish_reason": "length", "message": {"content": "<think>hm</think>\nDone"}}]});
        let r = parse_openai(&cut).unwrap();
        assert_eq!((r.text.as_str(), r.cut), ("Done", true));
        assert!(parse_openai(&json!({"choices": []})).is_err());
        let parts = json!({"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]});
        assert_eq!(parse_openai(&parts).unwrap().text, "ab");
    }

    #[test]
    fn an_anthropic_answer_is_read() {
        let v = json!({"stop_reason": "tool_use", "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "tu1", "name": "t", "input": {"x": 1}},
        ]});
        let r = parse_anthropic(&v).unwrap();
        assert_eq!(r.text, "Checking.");
        assert_eq!(
            r.calls,
            vec![Call {
                id: "tu1".into(),
                name: "t".into(),
                args: json!({"x": 1})
            }]
        );
        assert!(
            parse_anthropic(&json!({"stop_reason": "max_tokens", "content": []}))
                .unwrap()
                .cut
        );
        assert!(parse_anthropic(&json!({})).is_err());
    }

    #[test]
    fn think_blocks_are_dropped_only_in_front() {
        assert_eq!(strip_think("<think>a</think><think>b</think> Hi"), "Hi");
        assert_eq!(strip_think("<think>never closed"), "");
        assert_eq!(strip_think("a <think>b</think>"), "a <think>b</think>");
    }

    #[test]
    fn errors_name_the_reason_and_a_next_step() {
        let e = error_text(401, "{\"error\": {\"message\": \"bad key\"}}");
        assert!(e.contains("401") && e.contains("bad key") && e.contains("Check the key"));
        assert!(error_text(404, "not json").contains("not json"));
        assert!(error_text(500, "").contains("500"));
        assert!(error_text(400, "{\"error\": \"plain\"}").contains("plain"));
    }

    #[test]
    fn model_lists_of_both_shapes_are_read() {
        assert_eq!(model_ids(&json!({"data": [{"id": "a"}, {"id": "b"}]})), ["a", "b"]);
        assert_eq!(model_ids(&json!({"models": [{"name": "llama3"}]})), ["llama3"]);
        assert!(model_ids(&json!({})).is_empty());
    }

    #[test]
    fn the_window_keeps_the_end_and_starts_at_a_question() {
        let mut m = convo();
        m.push(Msg::User("and the gps?".into()));
        m.push(Msg::Assistant {
            text: "Fine.".into(),
            calls: vec![],
        });
        assert_eq!(window(&m, 1_000_000).len(), m.len());
        let small = window(&m, 20);
        assert!(matches!(small[0], Msg::User(_)));
        assert_eq!(small.len(), 2);
        let one = [Msg::User("x".repeat(500))];
        assert_eq!(window(&one, 10).len(), 1);
        assert!(window(&[], 10).is_empty());
    }

    #[test]
    fn a_call_without_an_answer_gets_a_failed_one() {
        let mut m = convo();
        m.remove(3);
        repair(&mut m);
        assert_eq!(m.len(), 5);
        assert!(matches!(&m[3], Msg::Tool { id, error: true, .. } if id == "c2"));
        let before = m.clone();
        repair(&mut m);
        assert_eq!(m, before);
        let mut open = vec![
            Msg::User("go".into()),
            Msg::Assistant {
                text: String::new(),
                calls: vec![Call {
                    id: "k".into(),
                    name: "n".into(),
                    args: json!({}),
                }],
            },
        ];
        repair(&mut open);
        assert!(matches!(open.last(), Some(Msg::Tool { id, .. }) if id == "k"));
    }

    #[test]
    fn long_text_is_clipped_with_an_ellipsis() {
        assert_eq!(clip("abc", 5), "abc");
        assert_eq!(clip("abcdef", 3), "abc\u{2026}");
        assert_eq!(args_line(&json!({})), "");
        assert_eq!(args_line(&json!({"a": 1})), "{\"a\":1}");
    }
}
