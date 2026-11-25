//! The work of one turn, on a thread of its own: ask the model, run the tools it
//! asks for through Corvus' MCP server, ask again, until it has said its piece.
//!
//! The plugin is an ordinary MCP client of the station it sits in. Corvus keeps
//! a token for it (`assistant` of `GET /api/mcp`, this computer only), so it
//! signs in by itself, also while the server for other programs is off. What it
//! may do is the operator's to say in Settings, Connections, MCP: a tool that is
//! off there is asked for in the window. Nothing here reaches the vehicle by
//! another road. A Corvus without that token is signed in to as any other client
//! does, with its server switched on.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use corvus_api::net::http::{self, Server};
use eframe::egui;
use serde_json::{json, Value};

use super::client::{self, Cancel};
use super::model::{
    self, anthropic_body, error_text, model_ids, openai_body, parse_anthropic, parse_openai, repair, tool_text, window,
    Call, Endpoint, Msg, Offer, Provider, Tool, HISTORY_CHARS, MAX_ROUNDS, TOOL_TEXT_MAX,
};

/// How the plugin stands with the station's MCP server.
#[derive(Debug, Clone, PartialEq)]
pub enum Mcp {
    /// Not looked at yet.
    Unknown,
    /// The operator switched it off in the plugin's gear.
    Declined,
    /// Off in Settings, Connections, MCP: AI Chat's own switch, or on a Corvus
    /// that has none the server itself.
    Off { own: bool },
    /// Signed in, with this many tools on offer.
    Ready(usize),
    /// Could not be reached, and why.
    Failed(String),
}

/// What a worker tells the window.
#[derive(Debug, Clone)]
pub enum Event {
    Mcp(Mcp),
    Assistant {
        text: String,
        calls: Vec<Call>,
        cut: bool,
    },
    ToolStart,
    ToolDone {
        id: String,
        name: String,
        text: String,
        error: bool,
    },
    /// The names of the models the server lists.
    Models(Vec<String>),
    Note(String),
    Failed(String),
    Finished,
}

/// What the operator set up.
#[derive(Debug, Clone)]
pub struct Config {
    pub provider: Provider,
    pub endpoint: Endpoint,
    pub key: String,
    pub model: String,
    pub system: String,
    pub use_mcp: bool,
    pub offer: Offer,
    pub max_tokens: u32,
    pub wait: Duration,
}

pub enum Job {
    /// One turn of conversation.
    Turn { history: Vec<Msg> },
    /// Looks at MCP and, with `models`, asks the model server for its models.
    Check { models: bool },
}

/// A turn or a check under way. Dropping it stops it and waits for the thread.
pub struct Worker {
    rx: Receiver<Event>,
    cancel: Cancel,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(ctx: &egui::Context, server: Server, cfg: Config, job: Job) -> Worker {
        let (tx, rx) = channel();
        let cancel = Cancel::default();
        let (c, ctx) = (cancel.clone(), ctx.clone());
        let handle = std::thread::Builder::new()
            .name("ai-chat".into())
            .spawn(move || {
                let out = Out { tx, ctx };
                run(&out, &server, &cfg, job, &c);
                out.send(Event::Finished);
            })
            .ok();
        Worker { rx, cancel, handle }
    }

    /// Asks the work to end; it does so within a moment.
    pub fn stop(&self) {
        self.cancel.cancel();
    }

    /// What has arrived since the last call.
    pub fn poll(&self) -> Vec<Event> {
        self.rx.try_iter().collect()
    }

    /// Whether the thread has ended and everything it said was taken.
    pub fn done(&self) -> bool {
        self.handle.as_ref().is_none_or(JoinHandle::is_finished)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

struct Out {
    tx: Sender<Event>,
    ctx: egui::Context,
}

impl Out {
    fn send(&self, e: Event) {
        let _ = self.tx.send(e);
        self.ctx.request_repaint();
    }
}

fn run(out: &Out, server: &Server, cfg: &Config, job: Job, cancel: &Cancel) {
    match job {
        Job::Check { models } => {
            let mcp = connect_mcp(server, cfg, cancel);
            out.send(Event::Mcp(mcp.state()));
            if let Some(m) = mcp.client {
                m.close(cancel);
            }
            if models && !cancel.is_set() {
                match list_models(cfg, cancel) {
                    Ok(ids) => out.send(Event::Models(ids)),
                    Err(e) => out.send(Event::Failed(e)),
                }
            }
        }
        Job::Turn { history } => {
            if let Err(e) = turn(out, server, cfg, history, cancel) {
                if !cancel.is_set() {
                    out.send(Event::Failed(e));
                }
            }
        }
    }
}

fn headers(cfg: &Config) -> Vec<(String, String)> {
    let mut h = vec![("Content-Type".to_string(), "application/json".to_string())];
    match cfg.provider {
        Provider::OpenAi => {
            h.push(("Accept".into(), "application/json".into()));
            if !cfg.key.is_empty() {
                h.push(("Authorization".into(), format!("Bearer {}", cfg.key)));
            }
        }
        Provider::Anthropic => {
            h.push(("Accept".into(), "application/json".into()));
            h.push(("anthropic-version".into(), "2023-06-01".into()));
            if !cfg.key.is_empty() {
                h.push(("x-api-key".into(), cfg.key.clone()));
            }
        }
    }
    h
}

fn as_refs(h: &[(String, String)]) -> Vec<(&str, &str)> {
    h.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
}

fn list_models(cfg: &Config, cancel: &Cancel) -> Result<Vec<String>, String> {
    let h = headers(cfg);
    let a = client::call(
        &cfg.endpoint,
        "GET",
        &cfg.endpoint.models_path(),
        &as_refs(&h),
        None,
        cancel,
        Duration::from_secs(20),
    )?;
    if a.status != 200 {
        return Err(error_text(a.status, &a.body));
    }
    let v: Value = serde_json::from_str(&a.body).map_err(|_| "The model server's list of models is not JSON.")?;
    Ok(model_ids(&v))
}

fn ask_model(cfg: &Config, msgs: &[Msg], tools: &[Tool], cancel: &Cancel) -> Result<model::Reply, String> {
    type Parse = fn(&Value) -> Result<model::Reply, String>;
    let (body, parse): (Value, Parse) = match cfg.provider {
        Provider::OpenAi => (
            openai_body(&cfg.model, &cfg.system, msgs, tools, cfg.max_tokens),
            parse_openai,
        ),
        Provider::Anthropic => (
            anthropic_body(&cfg.model, &cfg.system, msgs, tools, cfg.max_tokens),
            parse_anthropic,
        ),
    };
    let h = headers(cfg);
    let a = client::call(
        &cfg.endpoint,
        "POST",
        &cfg.endpoint.chat_path(cfg.provider),
        &as_refs(&h),
        Some(&body.to_string()),
        cancel,
        cfg.wait,
    )?;
    if a.status != 200 {
        return Err(error_text(a.status, &a.body));
    }
    let v: Value = serde_json::from_str(&a.body).map_err(|_| "The model's answer is not JSON.".to_string())?;
    parse(&v)
}

fn turn(out: &Out, server: &Server, cfg: &Config, mut history: Vec<Msg>, cancel: &Cancel) -> Result<(), String> {
    repair(&mut history);
    let mut mcp = connect_mcp(server, cfg, cancel);
    out.send(Event::Mcp(mcp.state()));
    let tools = mcp.tools.clone();
    let cfg = &Config {
        system: model::system_with(&cfg.system, &mcp.instructions),
        ..cfg.clone()
    };
    let result = rounds(out, cfg, &mut history, &mut mcp, &tools, cancel);
    if let Some(m) = mcp.client.take() {
        m.close(cancel);
    }
    result
}

fn rounds(
    out: &Out,
    cfg: &Config,
    history: &mut Vec<Msg>,
    mcp: &mut Connected,
    tools: &[Tool],
    cancel: &Cancel,
) -> Result<(), String> {
    for _ in 0..MAX_ROUNDS {
        if cancel.is_set() {
            return Ok(());
        }
        let reply = ask_model(cfg, window(history, HISTORY_CHARS), tools, cancel)?;
        history.push(Msg::Assistant {
            text: reply.text.clone(),
            calls: reply.calls.clone(),
        });
        out.send(Event::Assistant {
            text: reply.text,
            calls: reply.calls.clone(),
            cut: reply.cut,
        });
        if reply.calls.is_empty() {
            return Ok(());
        }
        for call in reply.calls {
            out.send(Event::ToolStart);
            let (text, error) = if cancel.is_set() {
                ("The call did not run: the turn was stopped.".to_string(), true)
            } else {
                run_tool(mcp, tools, &call, cancel)
            };
            let sent = model::clip(&text, TOOL_TEXT_MAX);
            out.send(Event::ToolDone {
                id: call.id.clone(),
                name: call.name.clone(),
                text: text.clone(),
                error,
            });
            history.push(Msg::Tool {
                id: call.id,
                name: call.name,
                text: sent,
                error,
            });
        }
    }
    out.send(Event::Note(format!(
        "Stopped after {MAX_ROUNDS} rounds of tool calls. Say go on to continue."
    )));
    Ok(())
}

fn run_tool(mcp: &mut Connected, tools: &[Tool], call: &Call, cancel: &Cancel) -> (String, bool) {
    let Some(tool) = tools.iter().find(|t| t.name == call.name) else {
        return (format!("There is no tool called {}.", call.name), true);
    };
    let Some(client) = mcp.client.as_mut() else {
        return ("The station's tools are not available.".into(), true);
    };
    let args = if call.args.is_object() {
        call.args.clone()
    } else {
        json!({})
    };
    match client.rpc(
        "tools/call",
        json!({"name": tool.mcp_name, "arguments": args}),
        cancel,
        Duration::from_secs(120),
    ) {
        Ok(r) => tool_text(&r),
        Err(e) => (e, true),
    }
}

/// The station's MCP server, signed in, and what it offers.
struct Connected {
    client: Option<McpClient>,
    tools: Vec<Tool>,
    /// What the server tells its clients about itself.
    instructions: String,
    failure: Option<Mcp>,
}

impl Connected {
    fn state(&self) -> Mcp {
        match (&self.failure, &self.client) {
            (Some(f), _) => f.clone(),
            (None, Some(_)) => Mcp::Ready(self.tools.len()),
            (None, None) => Mcp::Unknown,
        }
    }
}

fn connect_mcp(server: &Server, cfg: &Config, cancel: &Cancel) -> Connected {
    let none = |failure: Mcp| Connected {
        client: None,
        tools: Vec::new(),
        instructions: String::new(),
        failure: Some(failure),
    };
    if !cfg.use_mcp {
        return none(Mcp::Declined);
    }
    let info = match http::get_json(server, "/api/mcp") {
        Ok(v) => v,
        Err(e) => return none(Mcp::Failed(e)),
    };
    let own = info.get("assistant").filter(|a| a.is_object());
    let on = match own {
        Some(a) => a.get("on") == Some(&Value::Bool(true)),
        None => info.pointer("/settings/enabled") == Some(&Value::Bool(true)),
    };
    if !on {
        return none(Mcp::Off { own: own.is_some() });
    }
    let token = own.unwrap_or(&info).get("token").and_then(Value::as_str).unwrap_or("");
    if token.is_empty() {
        return none(Mcp::Failed(
            "The tools are only reachable on the computer Corvus GCS runs on.".into(),
        ));
    }
    let ep = Endpoint {
        tls: false,
        host: server.host.trim_start_matches('[').trim_end_matches(']').to_string(),
        port: server.port,
        base: String::new(),
    };
    let mut c = McpClient {
        ep,
        token: token.to_string(),
        session: None,
        id: 0,
    };
    let init = c.rpc(
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "Corvus AI Chat", "version": "1"}}),
        cancel,
        Duration::from_secs(15),
    );
    let instructions = match init {
        Ok(r) => r.get("instructions").and_then(Value::as_str).unwrap_or("").to_string(),
        Err(e) => return none(Mcp::Failed(e)),
    };
    let _ = c.notify("notifications/initialized", cancel);
    match c.rpc("tools/list", json!({}), cancel, Duration::from_secs(15)) {
        Ok(list) => {
            let tools = cfg.offer.of(model::tools_from_mcp(&list));
            Connected {
                client: Some(c),
                tools,
                instructions,
                failure: None,
            }
        }
        Err(e) => {
            c.close(cancel);
            none(Mcp::Failed(e))
        }
    }
}

/// A client of `POST /mcp`, one request after the other.
struct McpClient {
    ep: Endpoint,
    token: String,
    session: Option<String>,
    id: u64,
}

impl McpClient {
    fn post(&mut self, message: &Value, cancel: &Cancel, wait: Duration) -> Result<client::Answer, String> {
        let auth = format!("Bearer {}", self.token);
        let mut h = vec![
            ("Content-Type", "application/json"),
            ("Accept", "application/json, text/event-stream"),
            ("Authorization", auth.as_str()),
        ];
        if let Some(s) = &self.session {
            h.push(("Mcp-Session-Id", s.as_str()));
        }
        let a = client::call(&self.ep, "POST", "/mcp", &h, Some(&message.to_string()), cancel, wait)?;
        if let Some(s) = a.header("mcp-session-id") {
            self.session = Some(s.to_string());
        }
        Ok(a)
    }

    fn rpc(&mut self, method: &str, params: Value, cancel: &Cancel, wait: Duration) -> Result<Value, String> {
        self.id += 1;
        let a = self.post(
            &json!({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params}),
            cancel,
            wait,
        )?;
        let v: Value = serde_json::from_str(&a.body).unwrap_or(Value::Null);
        if let Some(why) = v.pointer("/error/message").and_then(Value::as_str) {
            return Err(why.to_string());
        }
        if a.status != 200 {
            return Err(format!("The station's MCP server answered {}.", a.status));
        }
        Ok(v.get("result").cloned().unwrap_or(Value::Null))
    }

    fn notify(&mut self, method: &str, cancel: &Cancel) -> Result<(), String> {
        self.post(
            &json!({"jsonrpc": "2.0", "method": method}),
            cancel,
            Duration::from_secs(5),
        )
        .map(|_| ())
    }

    /// Says goodbye, so the station's list of clients does not fill up.
    fn close(self, _cancel: &Cancel) {
        let Some(session) = &self.session else { return };
        let auth = format!("Bearer {}", self.token);
        let h = [("Authorization", auth.as_str()), ("Mcp-Session-Id", session.as_str())];
        // Never cancelled: a Stop pressed during the turn must not leave the session open.
        let _ = client::call(
            &self.ep,
            "DELETE",
            "/mcp",
            &h,
            None,
            &Cancel::default(),
            Duration::from_secs(5),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A stand in for the station's `/mcp` and a model, on one port: answers by what the
    /// request asks for and records every request it saw.
    fn stub(
        script: impl Fn(&str, &str, &Value) -> (u16, String, Vec<(String, String)>) + Send + 'static,
    ) -> (u16, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut data = Vec::new();
                let mut buf = [0u8; 8192];
                loop {
                    let n = s.read(&mut buf).unwrap_or(0);
                    data.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&data).to_string();
                    if let Some(i) = text.find("\r\n\r\n") {
                        let len = text[..i]
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                            })
                            .unwrap_or(0);
                        if data.len() >= i + 4 + len || n == 0 {
                            break;
                        }
                    } else if n == 0 {
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&data).to_string();
                let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
                let first = head.lines().next().unwrap_or("").to_string();
                let mut it = first.split_whitespace();
                let (method, path) = (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string());
                log.lock().unwrap().push(format!("{method} {path} {}", body));
                let json: Value = serde_json::from_str(body).unwrap_or(Value::Null);
                let (status, reply, extra) = script(&method, &path, &json);
                let mut out = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
                    reply.len()
                );
                for (k, v) in extra {
                    out.push_str(&format!("{k}: {v}\r\n"));
                }
                out.push_str("\r\n");
                out.push_str(&reply);
                let _ = s.write_all(out.as_bytes());
            }
        });
        (port, seen)
    }

    fn cfg(port: u16, provider: Provider) -> Config {
        Config {
            provider,
            endpoint: Endpoint::parse(&format!("http://127.0.0.1:{port}")).unwrap(),
            key: "k".into(),
            model: "m".into(),
            system: "s".into(),
            use_mcp: true,
            offer: Offer::All,
            max_tokens: 0,
            wait: Duration::from_secs(10),
        }
    }

    /// Runs a turn against one stub that plays both the model and the station.
    fn run_turn(port: u16, provider: Provider, history: Vec<Msg>) -> Vec<Event> {
        let ctx = egui::Context::default();
        let w = Worker::spawn(&ctx, Server::local(port), cfg(port, provider), Job::Turn { history });
        let mut events = Vec::new();
        let t = std::time::Instant::now();
        loop {
            events.extend(w.poll());
            if events.iter().any(|e| matches!(e, Event::Finished)) || t.elapsed() > Duration::from_secs(20) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        events
    }

    type Answer = (u16, String, Vec<(String, String)>);

    fn station_and_model(model_calls_tool: bool) -> impl Fn(&str, &str, &Value) -> Answer + Send + 'static {
        let asked = Arc::new(Mutex::new(0));
        move |method, path, body| match (method, path) {
            ("GET", "/api/mcp") => (
                200,
                json!({"settings": {"enabled": true}, "token": "tok"}).to_string(),
                vec![],
            ),
            ("POST", "/mcp") => {
                let id = body.get("id").cloned().unwrap_or(Value::Null);
                match body.get("method").and_then(Value::as_str) {
                    Some("initialize") => (200, json!({"jsonrpc": "2.0", "id": id, "result": {}}).to_string(), vec![("Mcp-Session-Id".into(), "sess".into())]),
                    Some("tools/list") => (200, json!({"jsonrpc": "2.0", "id": id, "result": {"tools": [{"name": "get_telemetry", "description": "d", "inputSchema": {"type": "object"}}]}}).to_string(), vec![]),
                    Some("tools/call") => (200, json!({"jsonrpc": "2.0", "id": id, "result": {"content": [{"type": "text", "text": "battery 82"}]}}).to_string(), vec![]),
                    _ => (202, String::new(), vec![]),
                }
            }
            ("DELETE", "/mcp") => (200, String::new(), vec![]),
            ("POST", "/v1/chat/completions") => {
                let mut n = asked.lock().unwrap();
                *n += 1;
                let reply = if model_calls_tool && *n == 1 {
                    json!({"choices": [{"message": {"content": "Looking.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_telemetry", "arguments": "{}"}}]}}]})
                } else {
                    json!({"choices": [{"message": {"content": "Battery is 82 percent."}}]})
                };
                (200, reply.to_string(), vec![])
            }
            ("POST", "/v1/messages") => {
                let mut n = asked.lock().unwrap();
                *n += 1;
                let reply = if model_calls_tool && *n == 1 {
                    json!({"content": [{"type": "text", "text": "Looking."}, {"type": "tool_use", "id": "t1", "name": "get_telemetry", "input": {}}]})
                } else {
                    json!({"content": [{"type": "text", "text": "Battery is 82 percent."}]})
                };
                (200, reply.to_string(), vec![])
            }
            _ => (404, "{}".into(), vec![]),
        }
    }

    #[test]
    fn a_turn_runs_the_tool_the_model_asks_for_and_signs_out() {
        for provider in [Provider::OpenAi, Provider::Anthropic] {
            let (port, seen) = stub(station_and_model(true));
            let events = run_turn(port, provider, vec![Msg::User("battery?".into())]);
            let kinds: Vec<&str> = events
                .iter()
                .map(|e| match e {
                    Event::Mcp(Mcp::Ready(1)) => "ready",
                    Event::Assistant { calls, .. } if !calls.is_empty() => "asks",
                    Event::Assistant { .. } => "says",
                    Event::ToolStart => "start",
                    Event::ToolDone { text, error: false, .. } if text == "battery 82" => "done",
                    Event::Finished => "finished",
                    other => panic!("unexpected {other:?}"),
                })
                .collect();
            assert_eq!(
                kinds,
                ["ready", "asks", "start", "done", "says", "finished"],
                "{provider:?}"
            );
            let seen = seen.lock().unwrap();
            let paths: Vec<String> = seen
                .iter()
                .map(|s| s.split_whitespace().take(2).collect::<Vec<_>>().join(" "))
                .collect();
            assert_eq!(paths.first().map(String::as_str), Some("GET /api/mcp"));
            assert_eq!(paths.last().map(String::as_str), Some("DELETE /mcp"));
            let call = seen.iter().find(|s| s.contains("tools/call")).unwrap();
            assert!(call.contains("\"name\":\"get_telemetry\""), "{call}");
        }
    }

    #[test]
    fn a_model_that_always_asks_is_ended_after_the_rounds_allowed() {
        let (port, _) = stub(|method, path, body| match (method, path) {
            ("GET", "/api/mcp") => (200, json!({"settings": {"enabled": false}}).to_string(), vec![]),
            ("POST", "/v1/chat/completions") => {
                let _ = body;
                (200, json!({"choices": [{"message": {"content": "", "tool_calls": [{"id": "c", "function": {"name": "nope", "arguments": "{}"}}]}}]}).to_string(), vec![])
            }
            _ => (404, "{}".into(), vec![]),
        });
        let events = run_turn(port, Provider::OpenAi, vec![Msg::User("x".into())]);
        let asked = events.iter().filter(|e| matches!(e, Event::Assistant { .. })).count();
        assert_eq!(asked, MAX_ROUNDS);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Note(n) if n.contains("rounds"))));
        assert!(events.iter().any(|e| matches!(e, Event::Mcp(Mcp::Off { own: false }))));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolDone { error: true, text, .. } if text.contains("no tool"))));
    }

    #[test]
    fn the_chat_signs_in_with_its_own_token_while_the_server_is_off() {
        let (port, seen) =
            stub(|method, path, body| {
                match (method, path) {
            ("GET", "/api/mcp") => (
                200,
                json!({"settings": {"enabled": false}, "token": "theirs", "assistant": {"on": true, "token": "mine"}})
                    .to_string(),
                vec![],
            ),
            ("POST", "/mcp") => {
                let id = body.get("id").cloned().unwrap_or(Value::Null);
                let result = match body.get("method").and_then(Value::as_str) {
                    Some("initialize") => json!({"instructions": "Draw on the map."}),
                    Some("tools/list") => json!({"tools": [{"name": "draw_on_map"}]}),
                    _ => return (202, String::new(), vec![]),
                };
                (200, json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(), vec![])
            }
            ("POST", "/v1/chat/completions") => (
                200,
                json!({"choices": [{"message": {"content": "Done."}}]}).to_string(),
                vec![],
            ),
            _ => (200, String::new(), vec![]),
        }
            });
        let events = run_turn(port, Provider::OpenAi, vec![Msg::User("x".into())]);
        assert!(
            events.iter().any(|e| matches!(e, Event::Mcp(Mcp::Ready(1)))),
            "{events:?}"
        );
        let seen = seen.lock().unwrap();
        let asked = seen.iter().find(|s| s.contains("/v1/chat/completions")).unwrap();
        assert!(
            asked.contains("Draw on the map."),
            "the model is told what the station says: {asked}"
        );
        assert!(asked.contains("draw_on_map"), "{asked}");
        drop(seen);

        let heads = Arc::new(Mutex::new(Vec::new()));
        let h = heads.clone();
        let (port, _) = stub(move |method, path, _| {
            h.lock().unwrap().push(format!("{method} {path}"));
            match (method, path) {
                ("GET", "/api/mcp") => (
                    200,
                    json!({"settings": {"enabled": true}, "token": "theirs", "assistant": {"on": false, "token": "mine"}})
                        .to_string(),
                    vec![],
                ),
                ("POST", "/v1/chat/completions") => (
                    200,
                    json!({"choices": [{"message": {"content": "Done."}}]}).to_string(),
                    vec![],
                ),
                _ => (200, String::new(), vec![]),
            }
        });
        let events = run_turn(port, Provider::OpenAi, vec![Msg::User("x".into())]);
        assert!(events.iter().any(|e| matches!(e, Event::Mcp(Mcp::Off { own: true }))));
        assert!(
            !heads.lock().unwrap().iter().any(|h| h == "POST /mcp"),
            "its own switch is off: it does not come in by the other door"
        );
    }

    #[test]
    fn a_refusal_from_the_model_server_reaches_the_operator() {
        let (port, _) = stub(|method, path, _| match (method, path) {
            ("GET", "/api/mcp") => (200, json!({"settings": {"enabled": false}}).to_string(), vec![]),
            _ => (401, json!({"error": {"message": "bad key"}}).to_string(), vec![]),
        });
        let events = run_turn(port, Provider::OpenAi, vec![Msg::User("x".into())]);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Failed(f) if f.contains("bad key"))));
    }

    #[test]
    fn a_check_lists_the_models() {
        let (port, _) = stub(|method, path, _| match (method, path) {
            ("GET", "/api/mcp") => (200, json!({"settings": {"enabled": false}}).to_string(), vec![]),
            ("GET", "/v1/models") => (200, json!({"data": [{"id": "qwen"}]}).to_string(), vec![]),
            _ => (404, "{}".into(), vec![]),
        });
        let ctx = egui::Context::default();
        let w = Worker::spawn(
            &ctx,
            Server::local(port),
            cfg(port, Provider::OpenAi),
            Job::Check { models: true },
        );
        let t = std::time::Instant::now();
        let mut events = Vec::new();
        while !events.iter().any(|e| matches!(e, Event::Finished)) && t.elapsed() < Duration::from_secs(10) {
            events.extend(w.poll());
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Models(m) if m == &["qwen".to_string()])));
    }

    #[test]
    fn stopping_ends_a_turn_the_model_never_answers() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let mut held = Vec::new();
            for s in listener.incoming().flatten() {
                held.push(s);
            }
        });
        let mut c = cfg(port, Provider::OpenAi);
        c.use_mcp = false;
        let ctx = egui::Context::default();
        let w = Worker::spawn(
            &ctx,
            Server::local(port),
            c,
            Job::Turn {
                history: vec![Msg::User("x".into())],
            },
        );
        std::thread::sleep(Duration::from_millis(500));
        let t = std::time::Instant::now();
        w.stop();
        drop(w);
        assert!(t.elapsed() < Duration::from_secs(3));
    }
}
