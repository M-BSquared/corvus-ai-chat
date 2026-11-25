//! The work of one turn, as a task the plugin moves along on every tick: ask the model, run the tools it
//! asks for through Corvus' MCP server, ask again, until it has said its piece.
//!
//! The plugin is an ordinary MCP client of the station it sits in. Corvus keeps
//! a token for it (`assistant` of `GET /api/mcp`, this computer only), so it
//! signs in by itself, also while the server for other programs is off. What it
//! may do is the operator's to say in Settings, Connections, MCP: a tool that is
//! off there is asked for in the window. Nothing here reaches the vehicle by
//! another road. A Corvus without that token is signed in to as any other client
//! does, with its server switched on.

use std::future::poll_fn;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use corvus_api::egui;
use corvus_api::net::fetch::{self, Answer};
use corvus_api::net::http::Server;
use corvus_api::net::station_get;
use corvus_api::task::Task;
use serde_json::{json, Value};

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

/// Whether the work in progress should end. Shared by every request of a turn.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// A turn or a check under way. Dropping it gives up whatever it waits for.
pub struct Worker {
    rx: Receiver<Event>,
    cancel: Cancel,
    task: Task,
}

impl Worker {
    pub fn spawn(ctx: &egui::Context, server: Server, cfg: Config, job: Job) -> Worker {
        let (tx, rx) = channel();
        let cancel = Cancel::default();
        let (stop, ctx) = (cancel.clone(), ctx.clone());
        let task = Task::new(async move {
            let link = Link { ctx, server, tx };
            run(&link, &cfg, job, &stop).await;
            link.send(Event::Finished);
        });
        let mut worker = Worker { rx, cancel, task };
        worker.task.poll();
        worker
    }

    /// Asks the work to end; it does so within a moment.
    pub fn stop(&self) {
        self.cancel.cancel();
    }

    /// Moves the work along and answers with what has happened since the last call.
    pub fn poll(&mut self) -> Vec<Event> {
        self.task.poll();
        self.rx.try_iter().collect()
    }

    /// Whether the work has ended.
    pub fn done(&self) -> bool {
        self.task.done()
    }
}

/// What the work talks through: the station it sits in and the window.
struct Link {
    ctx: egui::Context,
    server: Server,
    tx: Sender<Event>,
}

impl Link {
    fn send(&self, e: Event) {
        let _ = self.tx.send(e);
        self.ctx.request_repaint();
    }

    /// Sends one request to `url` and reads the whole answer, which is `wait` at most. `cancel`
    /// ends it within a moment.
    async fn call(
        &self,
        method: &str,
        url: String,
        headers: &[(&str, &str)],
        body: Option<String>,
        cancel: &Cancel,
        wait: Duration,
    ) -> Result<Answer, String> {
        if cancel.is_set() {
            return Err("Stopped.".into());
        }
        let request = fetch::Request {
            method: method.into(),
            url,
            headers: headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            body,
            wait,
        };
        let mut under_way = fetch::start(&self.ctx, &self.server, request);
        poll_fn(move |_| {
            if cancel.is_set() {
                return Poll::Ready(Err("Stopped.".to_string()));
            }
            under_way.take().map_or(Poll::Pending, Poll::Ready)
        })
        .await
    }
}

/// The whole address of `path` at the model server.
fn address(ep: &Endpoint, path: &str) -> String {
    let scheme = if ep.tls { "https" } else { "http" };
    format!("{scheme}://{}{path}", ep.host_header())
}

async fn run(out: &Link, cfg: &Config, job: Job, cancel: &Cancel) {
    match job {
        Job::Check { models } => {
            let mcp = connect_mcp(out, cfg, cancel).await;
            out.send(Event::Mcp(mcp.state()));
            if let Some(m) = mcp.client {
                m.close(out).await;
            }
            if models && !cancel.is_set() {
                match list_models(out, cfg, cancel).await {
                    Ok(ids) => out.send(Event::Models(ids)),
                    Err(e) => out.send(Event::Failed(e)),
                }
            }
        }
        Job::Turn { history } => {
            if let Err(e) = turn(out, cfg, history, cancel).await {
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

async fn list_models(out: &Link, cfg: &Config, cancel: &Cancel) -> Result<Vec<String>, String> {
    let h = headers(cfg);
    let url = address(&cfg.endpoint, &cfg.endpoint.models_path());
    let a = out
        .call("GET", url, &as_refs(&h), None, cancel, Duration::from_secs(20))
        .await?;
    if a.status != 200 {
        return Err(error_text(a.status, &a.body));
    }
    let v: Value = serde_json::from_str(&a.body).map_err(|_| "The model server's list of models is not JSON.")?;
    Ok(model_ids(&v))
}

async fn ask_model(
    out: &Link,
    cfg: &Config,
    msgs: &[Msg],
    tools: &[Tool],
    cancel: &Cancel,
) -> Result<model::Reply, String> {
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
    let url = address(&cfg.endpoint, &cfg.endpoint.chat_path(cfg.provider));
    let a = out
        .call("POST", url, &as_refs(&h), Some(body.to_string()), cancel, cfg.wait)
        .await?;
    if a.status != 200 {
        return Err(error_text(a.status, &a.body));
    }
    let v: Value = serde_json::from_str(&a.body).map_err(|_| "The model's answer is not JSON.".to_string())?;
    parse(&v)
}

async fn turn(out: &Link, cfg: &Config, mut history: Vec<Msg>, cancel: &Cancel) -> Result<(), String> {
    repair(&mut history);
    let mut mcp = connect_mcp(out, cfg, cancel).await;
    out.send(Event::Mcp(mcp.state()));
    let tools = mcp.tools.clone();
    let cfg = &Config {
        system: model::system_with(&cfg.system, &mcp.instructions),
        ..cfg.clone()
    };
    let result = rounds(out, cfg, &mut history, &mut mcp, &tools, cancel).await;
    if let Some(m) = mcp.client.take() {
        m.close(out).await;
    }
    result
}

async fn rounds(
    out: &Link,
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
        let reply = ask_model(out, cfg, window(history, HISTORY_CHARS), tools, cancel).await?;
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
                run_tool(out, mcp, tools, &call, cancel).await
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

async fn run_tool(out: &Link, mcp: &mut Connected, tools: &[Tool], call: &Call, cancel: &Cancel) -> (String, bool) {
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
    let params = json!({"name": tool.mcp_name, "arguments": args});
    match client
        .rpc(out, "tools/call", params, cancel, Duration::from_secs(120))
        .await
    {
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

async fn connect_mcp(out: &Link, cfg: &Config, cancel: &Cancel) -> Connected {
    let none = |failure: Mcp| Connected {
        client: None,
        tools: Vec::new(),
        instructions: String::new(),
        failure: Some(failure),
    };
    if !cfg.use_mcp {
        return none(Mcp::Declined);
    }
    let info = match station_get(&out.ctx, &out.server, "/api/mcp").wait().await {
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
    let mut c = McpClient {
        token: token.to_string(),
        session: None,
        id: 0,
    };
    let hello = json!({"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "Corvus AI Chat", "version": "1"}});
    let init = c.rpc(out, "initialize", hello, cancel, Duration::from_secs(15)).await;
    let instructions = match init {
        Ok(r) => r.get("instructions").and_then(Value::as_str).unwrap_or("").to_string(),
        Err(e) => return none(Mcp::Failed(e)),
    };
    let _ = c.notify(out, "notifications/initialized", cancel).await;
    match c
        .rpc(out, "tools/list", json!({}), cancel, Duration::from_secs(15))
        .await
    {
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
            c.close(out).await;
            none(Mcp::Failed(e))
        }
    }
}

/// A client of `POST /mcp`, one request after the other.
struct McpClient {
    token: String,
    session: Option<String>,
    id: u64,
}

impl McpClient {
    async fn post(&mut self, out: &Link, message: &Value, cancel: &Cancel, wait: Duration) -> Result<Answer, String> {
        let auth = format!("Bearer {}", self.token);
        let mut h = vec![
            ("Content-Type", "application/json"),
            ("Accept", "application/json, text/event-stream"),
            ("Authorization", auth.as_str()),
        ];
        if let Some(s) = &self.session {
            h.push(("Mcp-Session-Id", s.as_str()));
        }
        let a = out
            .call("POST", "/mcp".into(), &h, Some(message.to_string()), cancel, wait)
            .await?;
        if let Some(s) = a.header("mcp-session-id") {
            self.session = Some(s.to_string());
        }
        Ok(a)
    }

    async fn rpc(
        &mut self,
        out: &Link,
        method: &str,
        params: Value,
        cancel: &Cancel,
        wait: Duration,
    ) -> Result<Value, String> {
        self.id += 1;
        let message = json!({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params});
        let a = self.post(out, &message, cancel, wait).await?;
        let v: Value = serde_json::from_str(&a.body).unwrap_or(Value::Null);
        if let Some(why) = v.pointer("/error/message").and_then(Value::as_str) {
            return Err(why.to_string());
        }
        if a.status != 200 {
            return Err(format!("The station's MCP server answered {}.", a.status));
        }
        Ok(v.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn notify(&mut self, out: &Link, method: &str, cancel: &Cancel) -> Result<(), String> {
        let message = json!({"jsonrpc": "2.0", "method": method});
        self.post(out, &message, cancel, Duration::from_secs(5))
            .await
            .map(|_| ())
    }

    /// Says goodbye, so the station's list of clients does not fill up.
    async fn close(self, out: &Link) {
        let Some(session) = &self.session else { return };
        let auth = format!("Bearer {}", self.token);
        let h = [("Authorization", auth.as_str()), ("Mcp-Session-Id", session.as_str())];
        // Never cancelled: a Stop pressed during the turn must not leave the session open.
        let _ = out
            .call(
                "DELETE",
                "/mcp".into(),
                &h,
                None,
                &Cancel::default(),
                Duration::from_secs(5),
            )
            .await;
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
        let mut w = Worker::spawn(&ctx, Server::local(port), cfg(port, provider), Job::Turn { history });
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
        let mut w = Worker::spawn(
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
