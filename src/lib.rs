//! AI Chat: a chat with a language model the operator chooses, in a tab of its
//! own, that can use the whole station through Corvus' MCP server.
//!
//! The model is anything that speaks the chat completions dialect (OpenAI,
//! Ollama, LM Studio, llama.cpp, vLLM) or the messages dialect (Anthropic and
//! servers that copy it), on this computer, on the local network or hosted;
//! the address, the key and the model are set from the plugin's gear in
//! Settings. The station's tools are not built in here: the plugin signs in to
//! `/mcp` by itself, with the token Corvus keeps for it, and the operator's
//! switches and questions in Settings, Connections, MCP apply to it ([`agent`]).

mod agent;
pub mod markdown;
pub mod model;
mod view;

use std::time::Duration;

use corvus_api::egui;
use corvus_api::plugin::options::OptionSpec;
use corvus_api::plugin::{Host, Info, Plugin};
use serde_json::{json, Value};

use agent::{Config, Event, Job, Mcp, Worker};
use model::{clip, Call, Endpoint, Msg, Offer, Provider, DEFAULT_SYSTEM, ITEMS_MAX, TOOL_TEXT_MAX};

/// One line of the chat.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Msg(Msg),
    /// What the plugin itself has to say: a failure, a hint, the list of models.
    Note {
        text: String,
        error: bool,
    },
}

#[derive(Default)]
pub struct AiChat {
    pub items: Vec<Item>,
    pub input: String,
    pub mcp: Option<Mcp>,
    worker: Option<Worker>,
    /// Told to stop and not yet ended. Waiting for one here would hold the window.
    retired: Vec<Worker>,
    /// The tool calls whose answer is unfolded, by id.
    pub unfolded: std::collections::HashSet<String>,
    /// What is under way, for the view.
    pub busy: bool,
    pub checking: bool,
    /// The chat grew: the view scrolls to its end once.
    pub follow: bool,
    /// The operator pressed the button that switches MCP on.
    switching: Option<corvus_api::plugin::Reply>,
    probed: bool,
}

const OPT_PROVIDER: &str = "provider";
const OPT_URL: &str = "url";
const OPT_KEY: &str = "key";
const OPT_MODEL: &str = "model";
const OPT_SYSTEM: &str = "system";
const OPT_MCP: &str = "mcp";
const OPT_OFFER: &str = "offer";
const OPT_TOKENS: &str = "tokens";
const OPT_WAIT: &str = "wait";

impl Plugin for AiChat {
    fn info(&self) -> Info {
        Info {
            id: "ai-chat",
            name: "AI Chat",
            icon: "bot",
            description:
                "Chat with a language model of your choice, local or hosted, that can use the station through MCP",
            version: "2.0.0",
            order: 20,
            can_tab: true,
        }
    }

    fn options(&self) -> Vec<OptionSpec> {
        vec![
            OptionSpec::select(
                OPT_PROVIDER,
                "Kind of server",
                &[
                    ("openai", "OpenAI compatible (chat completions)"),
                    ("anthropic", "Anthropic compatible (messages)"),
                ],
                "openai",
            )
            .hint("Ollama, LM Studio, llama.cpp and vLLM speak the OpenAI kind. The Anthropic kind is for Claude and for servers that copy its messages interface."),
            OptionSpec::text(OPT_URL, "Address", "").hint(
                "Where the server is, for example http://localhost:11434 for Ollama, http://localhost:1234/v1 for LM Studio or https://api.anthropic.com. Without http:// or https:// it is http://.",
            ),
            OptionSpec::text(OPT_MODEL, "Model", "")
                .hint("The model's name as the server knows it. The refresh button in the chat lists what the server offers."),
            OptionSpec::text(OPT_KEY, "Key", "").secret().hint(
                "Leave empty for a server that needs none. It is saved in this plugin's settings on this computer, so leave it out of an exported settings file you share.",
            ),
            OptionSpec::toggle(OPT_MCP, "Use the station's tools", true).hint(
                "Lets the model read the aircraft, draw on the map and steer. Nothing has to be switched on for it. Corvus asks you before anything that changes the aircraft, unless you allowed it under Settings, Connections, MCP.",
            ),
            OptionSpec::select(
                OPT_OFFER,
                "Number of tools",
                &[
                    ("auto", "By itself: few for a local model"),
                    ("few", "Few: read, draw and fly"),
                    ("all", "All of them"),
                ],
                "auto",
            )
            .hint("All the tools take about 11000 tokens of the model's context and a capable model. A local model needs a context of 8000 tokens or more even for the few (num_ctx in Ollama)."),
            OptionSpec::text(OPT_SYSTEM, "Instructions", DEFAULT_SYSTEM)
                .hint("What the model is told before every conversation."),
            OptionSpec::number(
                OPT_TOKENS,
                "Reply length",
                Some(0.0),
                Some(64_000.0),
                Some(256.0),
                0.0,
            )
            .hint("The most tokens in one reply. 0 leaves it to the server (4096 for the Anthropic kind, which needs a number)."),
            OptionSpec::number(OPT_WAIT, "Time allowed", Some(10.0), Some(3600.0), Some(10.0), 300.0)
                .hint("Seconds to wait for one reply. A model on a modest computer can take minutes."),
        ]
    }

    fn open(&mut self, h: &mut Host) {
        self.follow = true;
        if !self.probed && self.worker.is_none() {
            self.probed = true;
            self.check(false, h);
        }
    }

    fn options_changed(&mut self, h: &mut Host) {
        self.mcp = None;
        self.probed = false;
        if self.worker.is_none() {
            self.probed = true;
            self.check(false, h);
        }
    }

    fn tick(&mut self, h: &mut Host) {
        // A turn that was cleared away still signs out.
        self.retired.retain_mut(|w| {
            w.poll();
            !w.done()
        });
        if let Some(w) = &mut self.worker {
            let events = w.poll();
            let done = w.done();
            let finished = events.iter().any(|e| matches!(e, Event::Finished));
            for e in events {
                self.apply(e);
            }
            if finished || done {
                self.worker = None;
                self.busy = false;
                self.checking = false;
                self.follow = true;
            }
        }
        if let Some(r) = &self.switching {
            if let Some(res) = r.take() {
                self.switching = None;
                match res {
                    Ok(_) => self.check(false, h),
                    Err(e) => self.note(e, true),
                }
            }
        }
    }

    fn show(&mut self, ui: &mut egui::Ui, area: egui::Rect, h: &mut Host) -> f32 {
        view::show(self, ui, area, h)
    }

    fn close(&mut self, _h: &mut Host) {}
}

impl AiChat {
    /// The model's setup from the gear, or what is missing.
    pub fn config(h: &Host) -> Result<Config, String> {
        let o = h.options();
        let text = |k: &str| o.get(k).and_then(Value::as_str).unwrap_or("").trim().to_string();
        let number = |k: &str, d: f64| o.get(k).and_then(Value::as_f64).filter(|n| n.is_finite()).unwrap_or(d);
        let endpoint = Endpoint::parse(&text(OPT_URL))?;
        let model = text(OPT_MODEL);
        if model.is_empty() {
            return Err("No model is set.".into());
        }
        let system = {
            let s = text(OPT_SYSTEM);
            if s.is_empty() {
                DEFAULT_SYSTEM.to_string()
            } else {
                s
            }
        };
        let offer = Offer::from_option(&text(OPT_OFFER), &endpoint);
        Ok(Config {
            offer,
            provider: Provider::from_option(&text(OPT_PROVIDER)),
            endpoint,
            key: text(OPT_KEY),
            model,
            system,
            use_mcp: o.get(OPT_MCP).and_then(Value::as_bool).unwrap_or(true),
            max_tokens: number(OPT_TOKENS, 0.0).clamp(0.0, 64_000.0) as u32,
            wait: Duration::from_secs(number(OPT_WAIT, 300.0).clamp(10.0, 3600.0) as u64),
        })
    }

    /// Whether a question can be asked, and if not what the operator has to do.
    pub fn ready(h: &Host) -> Result<(), String> {
        AiChat::config(h).map(|_| ())
    }

    fn note(&mut self, text: impl Into<String>, error: bool) {
        self.items.push(Item::Note {
            text: text.into(),
            error,
        });
        self.trim();
        self.follow = true;
    }

    fn trim(&mut self) {
        if self.items.len() > ITEMS_MAX {
            let cut = self.items.len() - ITEMS_MAX;
            let next = self.items[cut..]
                .iter()
                .position(|i| matches!(i, Item::Msg(Msg::User(_))))
                .map_or(self.items.len(), |n| cut + n);
            self.items.drain(..next);
        }
    }

    fn apply(&mut self, e: Event) {
        match e {
            Event::Mcp(m) => self.mcp = Some(m),
            Event::Assistant { text, calls, cut } => {
                let empty = text.trim().is_empty() && calls.is_empty();
                if !empty {
                    self.items.push(Item::Msg(Msg::Assistant { text, calls }));
                }
                if empty {
                    self.note("The model answered with nothing.", true);
                } else if cut {
                    self.note(
                        "The reply was cut short: the model ran out of room. Raise Reply length in the gear.",
                        false,
                    );
                }
                self.trim();
                self.follow = true;
            }
            Event::ToolStart => self.follow = true,
            Event::ToolDone { id, name, text, error } => {
                self.items.push(Item::Msg(Msg::Tool {
                    id,
                    name,
                    text: clip(&text, TOOL_TEXT_MAX),
                    error,
                }));
                self.follow = true;
            }
            Event::Models(list) => {
                let text = if list.is_empty() {
                    "The server lists no models.".to_string()
                } else {
                    let shown: Vec<&str> = list.iter().take(12).map(String::as_str).collect();
                    let more = if list.len() > 12 {
                        format!(" and {} more", list.len() - 12)
                    } else {
                        String::new()
                    };
                    format!("The server offers: {}{more}.", shown.join(", "))
                };
                self.note(text, false);
            }
            Event::Note(t) => self.note(t, false),
            Event::Failed(e) => self.note(e, true),
            Event::Finished => {}
        }
    }

    /// The conversation as the model sees it.
    fn history(&self) -> Vec<Msg> {
        let mut msgs: Vec<Msg> = self
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Msg(m) => Some(m.clone()),
                Item::Note { .. } => None,
            })
            .collect();
        model::repair(&mut msgs);
        msgs
    }

    /// Sends what is typed.
    pub fn send(&mut self, h: &mut Host) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.worker.is_some() {
            return;
        }
        let cfg = match AiChat::config(h) {
            Ok(c) => c,
            Err(e) => return self.note(e, true),
        };
        self.input.clear();
        self.items.push(Item::Msg(Msg::User(text)));
        self.trim();
        let history = self.history();
        self.worker = Some(Worker::spawn(h.ctx, h.server.clone(), cfg, Job::Turn { history }));
        self.busy = true;
        self.follow = true;
    }

    /// Asks `question`, as if it had been typed.
    pub fn ask(&mut self, question: &str, h: &mut Host) {
        self.input = question.to_string();
        self.send(h);
    }

    /// Looks at the station's tools and, with `models`, at what the server offers.
    pub fn check(&mut self, models: bool, h: &mut Host) {
        if self.worker.is_some() {
            return;
        }
        let cfg = match AiChat::config(h) {
            Ok(c) => c,
            Err(e) => {
                if models {
                    self.note(e, true);
                }
                return;
            }
        };
        self.worker = Some(Worker::spawn(h.ctx, h.server.clone(), cfg, Job::Check { models }));
        self.checking = models;
    }

    /// Ends what is under way. The answers so far stay in the chat.
    pub fn stop(&mut self) {
        if let Some(w) = &self.worker {
            w.stop();
        }
    }

    /// Starts again with an empty chat.
    pub fn clear(&mut self) {
        self.stop();
        self.retired.extend(self.worker.take());
        self.busy = false;
        self.checking = false;
        self.items.clear();
        self.unfolded.clear();
    }

    /// Switches the station's tools on, from the button next to their status.
    pub fn switch_mcp_on(&mut self, h: &mut Host) {
        if self.switching.is_none() {
            let patch = match self.mcp {
                Some(Mcp::Off { own: false }) => json!({"enabled": true}),
                _ => json!({"assistant": true}),
            };
            self.switching = Some(h.post("/api/mcp/settings", patch));
        }
    }

    /// The calls of the assistant entry at `i` with how each one stands.
    pub fn calls_of(&self, i: usize) -> Vec<(&Call, Option<(&str, bool)>)> {
        let Some(Item::Msg(Msg::Assistant { calls, .. })) = self.items.get(i) else {
            return Vec::new();
        };
        calls
            .iter()
            .map(|c| {
                let answer = self.items[i + 1..].iter().find_map(|it| match it {
                    Item::Msg(Msg::Tool { id, text, error, .. }) if *id == c.id => Some((text.as_str(), *error)),
                    _ => None,
                });
                (c, answer)
            })
            .collect()
    }
}

/// The language files of this plugin, for the translation layer.
pub const LANG: &[(&str, &str)] = &[
    ("de", include_str!("../lang/de.json")),
    ("fr", include_str!("../lang/fr.json")),
    ("dl", include_str!("../lang/dl.json")),
];

/// One instance of the plugin, for the registry.
pub fn plugin() -> Box<dyn Plugin> {
    Box::new(AiChat::default())
}

corvus_api::export_plugin!(plugin, LANG);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(id: &str) -> Call {
        Call {
            id: id.into(),
            name: "get_telemetry".into(),
            args: json!({}),
        }
    }

    #[test]
    fn the_default_instructions_fit_an_option() {
        assert!(DEFAULT_SYSTEM.chars().count() <= 1000);
    }

    #[test]
    fn the_options_are_all_usable() {
        let specs = corvus_api::plugin::options::clean(AiChat::default().options());
        assert_eq!(specs.len(), 9);
        assert!(specs.iter().find(|s| s.key == OPT_KEY).is_some_and(|s| s.secret));
    }

    #[test]
    fn the_calls_of_an_entry_say_how_each_one_stands() {
        let mut c = AiChat::default();
        c.items.push(Item::Msg(Msg::User("x".into())));
        c.items.push(Item::Msg(Msg::Assistant {
            text: String::new(),
            calls: vec![call("a"), call("b")],
        }));
        c.apply(Event::ToolDone {
            id: "a".into(),
            name: "get_telemetry".into(),
            text: "82".into(),
            error: false,
        });
        let calls = c.calls_of(1);
        assert_eq!(calls[0].1, Some(("82", false)));
        assert_eq!(calls[1].1, None);
        assert!(c.calls_of(0).is_empty());
    }

    #[test]
    fn an_empty_answer_is_said_and_a_long_chat_is_cut_at_a_question() {
        let mut c = AiChat::default();
        c.apply(Event::Assistant {
            text: " ".into(),
            calls: vec![],
            cut: false,
        });
        assert!(matches!(&c.items[0], Item::Note { error: true, .. }));
        c.items.clear();
        for i in 0..(ITEMS_MAX + 30) {
            c.items.push(Item::Msg(if i % 2 == 0 {
                Msg::User(format!("q{i}"))
            } else {
                Msg::Assistant {
                    text: format!("a{i}"),
                    calls: vec![],
                }
            }));
        }
        c.trim();
        assert!(c.items.len() <= ITEMS_MAX);
        assert!(matches!(c.items[0], Item::Msg(Msg::User(_))));
    }

    #[test]
    fn the_history_closes_calls_that_never_got_an_answer() {
        let mut c = AiChat::default();
        c.items.push(Item::Msg(Msg::User("go".into())));
        c.items.push(Item::Msg(Msg::Assistant {
            text: String::new(),
            calls: vec![call("k")],
        }));
        c.note("stopped", false);
        let h = c.history();
        assert_eq!(h.len(), 3);
        assert!(matches!(&h[2], Msg::Tool { id, error: true, .. } if id == "k"));
    }

    #[test]
    fn models_are_listed_briefly() {
        let mut c = AiChat::default();
        c.apply(Event::Models((0..15).map(|i| format!("m{i}")).collect()));
        let Item::Note { text, .. } = &c.items[0] else { panic!() };
        assert!(
            text.contains("m11") && !text.contains("m12") && text.contains("3 more"),
            "{text}"
        );
    }
}
