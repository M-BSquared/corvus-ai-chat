//! What the operator sees: who is being talked to, the conversation in
//! bubbles, and the bar to write in, which stays at the bottom.

use std::sync::Arc;

// Under other names: the key check of the program reads the language files of the
// application for every literal handed to `translate(`, and a plugin keeps its own.
use corvus_api::i18n::{tr as translate, tr_fmt as translate_fmt};
use corvus_api::plugin::Host;
use corvus_api::ui::button::icon_btn;
use corvus_api::ui::icons;
use corvus_api::ui::kit::{Button, Variant};
use corvus_api::ui::message::{self, Tone};
use corvus_api::ui::style::{self, Weight};
use eframe::egui::text::{LayoutJob, TextFormat};
use eframe::egui::{self, Color32, CornerRadius, Galley, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use super::agent::Mcp;
use super::markdown::{self, Block, Span};
use super::model::{args_line, clip, Msg, SHOWN_MAX};
use super::{AiChat, Item};

const GAP: f32 = 10.0;
const PAD_X: f32 = 13.0;
const PAD_Y: f32 = 9.0;
const TEXT: f32 = 13.0;
const AVATAR: f32 = 28.0;
/// The bubbles start here, right of the avatar.
const INDENT: f32 = AVATAR + 8.0;
const ROUND: u8 = 16;
const SHOWN_CHARS: usize = 30_000;
/// The bar grows with what is written, up to this many lines.
const ROWS: usize = 6;
const SEND: f32 = 26.0;

/// What an empty chat offers to ask.
const QUESTIONS: [(&str, &str); 4] = [
    ("plane-takeoff", "Is the aircraft ready to fly?"),
    ("battery-medium", "How are the battery and the GPS?"),
    ("triangle-alert", "Explain the last warnings."),
    ("map-pin", "Draw the way from the aircraft back home on the map."),
];

pub fn show(p: &mut AiChat, ui: &mut egui::Ui, area: Rect, h: &mut Host) -> f32 {
    let (x, w) = (area.left(), area.width());
    let height = area.height().max(260.0);
    let bottom = area.top() + height;
    let setup = AiChat::ready(h);

    let top = header(p, ui, x, area.top(), w, h, setup.is_ok());
    let bar = bar_height(p, ui, w);
    let bar_rect = Rect::from_min_max(Pos2::new(x, bottom - bar), Pos2::new(x + w, bottom));
    let talk = Rect::from_min_max(
        Pos2::new(x, top),
        Pos2::new(x + w + 8.0, (bar_rect.top() - GAP).max(top + 40.0)),
    );
    conversation(p, ui, talk, w, h, &setup);
    composer(p, ui, bar_rect, h, setup.is_ok());
    if p.busy {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));
    }
    height
}

// ---- text -----------------------------------------------------------------

/// Text laid out as it is, with no translation: a model's words are not keys.
fn wrap(ui: &egui::Ui, s: &str, size: f32, weight: Weight, color: Color32, width: f32, rows: usize) -> Arc<Galley> {
    laid(ui, s, style::font(size, weight), color, width, rows)
}

fn wrap_mono(ui: &egui::Ui, s: &str, size: f32, color: Color32, width: f32, rows: usize) -> Arc<Galley> {
    laid(ui, s, style::mono(size), color, width, rows)
}

fn laid(ui: &egui::Ui, s: &str, font: egui::FontId, color: Color32, width: f32, rows: usize) -> Arc<Galley> {
    let mut job = LayoutJob::default();
    job.append(s, 0.0, format(font, color));
    job.wrap.max_width = width.max(20.0);
    if rows > 0 {
        job.wrap.max_rows = rows;
        job.wrap.overflow_character = Some('\u{2026}');
    }
    ui.painter().layout_job(job)
}

fn format(font: egui::FontId, color: Color32) -> TextFormat {
    TextFormat {
        line_height: Some(font.size * 1.5),
        font_id: font,
        color,
        ..Default::default()
    }
}

fn spans_galley(ui: &egui::Ui, spans: &[Span], size: f32, bold: bool, ink: Color32, width: f32) -> Arc<Galley> {
    let mut job = LayoutJob::default();
    for s in spans {
        let (text, font, back) = match s {
            Span::Plain(t) if !bold => (t, style::font(size, Weight::W400), Color32::TRANSPARENT),
            Span::Plain(t) | Span::Bold(t) => (t, style::font(size, Weight::W600), Color32::TRANSPARENT),
            Span::Code(t) => (t, style::mono(size - 1.0), style::tint(0.07)),
        };
        job.append(
            text,
            0.0,
            TextFormat {
                background: back,
                line_height: Some(size * 1.5),
                ..format(font, ink)
            },
        );
    }
    job.wrap.max_width = width.max(20.0);
    ui.painter().layout_job(job)
}

/// Text that can be selected and copied.
fn put(ui: &mut egui::Ui, at: Pos2, g: Arc<Galley>) {
    let rect = Rect::from_min_size(at, g.size());
    if ui.is_rect_visible(rect) {
        ui.put(rect, egui::Label::new(g).selectable(true));
    }
}

// ---- header ---------------------------------------------------------------

fn avatar(ui: &egui::Ui, center: Pos2, size: f32) {
    ui.painter().circle_filled(center, size / 2.0, style::accent_dark());
    icons::paint(
        ui,
        "bot",
        Rect::from_center_size(center, Vec2::splat(size * 0.56)),
        style::accent_bright(),
    );
}

fn header(p: &mut AiChat, ui: &mut egui::Ui, x: f32, y: f32, w: f32, h: &mut Host, ready: bool) -> f32 {
    let size = 34.0;
    let center = Pos2::new(x + size / 2.0, y + size / 2.0);
    avatar(ui, center, size);

    let (color, say) = match &p.mcp {
        Some(Mcp::Ready(n)) => (
            style::healthy(),
            translate_fmt("{n} station tools", &[("n", &n.to_string())]),
        ),
        Some(Mcp::Declined) => (style::text_3(), translate("without the station's tools").into_owned()),
        Some(Mcp::Off { .. }) => (style::warning(), translate("the station's tools are off").into_owned()),
        Some(Mcp::Failed(_)) => (style::critical(), translate("station tools failed").into_owned()),
        Some(Mcp::Unknown) | None if ready => (
            style::text_3(),
            translate("looking at the station's tools").into_owned(),
        ),
        _ => (style::text_3(), translate("not set up").into_owned()),
    };
    let dot = center + Vec2::splat(size * 0.35);
    ui.painter().circle_filled(dot, 6.0, style::surface_1());
    ui.painter().circle_filled(dot, 4.0, color);

    let mut right = x + w;
    let by = y + (size - 28.0) / 2.0;
    if !p.items.is_empty() {
        right -= 28.0;
        let r = Rect::from_min_size(Pos2::new(right, by), Vec2::splat(28.0));
        if icon_btn(ui, r, "plus", "New chat", false, "ai-chat-clear") {
            p.clear();
        }
        right -= 4.0;
    }
    if ready && !p.busy && !p.checking {
        right -= 28.0;
        let r = Rect::from_min_size(Pos2::new(right, by), Vec2::splat(28.0));
        let tip = "Looks at the station's tools and lists the models the server offers.";
        if icon_btn(ui, r, "refresh-cw", tip, false, "ai-chat-check") {
            p.check(true, h);
        }
    }
    let left = x + size + 10.0;
    let room = right - 8.0 - left;
    let (title, place) = match AiChat::config(h) {
        Ok(c) => (c.model, format!("{} \u{b7} {say}", c.endpoint.label())),
        Err(_) => (translate("AI Chat").into_owned(), say),
    };
    let t = wrap(ui, &title, 13.5, Weight::W600, style::text_1(), room, 1);
    let s = wrap(ui, &place, 11.5, Weight::W400, style::text_3(), room, 1);
    let ty = center.y - (t.size().y + s.size().y) / 2.0;
    let th = t.size().y;
    ui.painter().galley(Pos2::new(left, ty), t, style::text_1());
    ui.painter().galley(Pos2::new(left, ty + th), s, style::text_3());

    let mut bottom = y + size + 10.0;
    ui.painter().hline(
        egui::Rangef::new(x, x + w),
        bottom,
        Stroke::new(1.0, style::border_soft()),
    );
    bottom += 1.0;
    match &p.mcp {
        Some(Mcp::Failed(e)) => bottom = message::show(ui, x, bottom + 6.0, w, &clip(e, 200), Tone::Err),
        Some(Mcp::Off { .. }) => bottom = mcp_off(p, ui, x, bottom + 8.0, w, h),
        _ => {}
    }
    bottom
}

/// The station's tools are off: why that matters, and the button that changes it.
fn mcp_off(p: &mut AiChat, ui: &mut egui::Ui, x: f32, y: f32, w: f32, h: &mut Host) -> f32 {
    let b = Button::new("Switch on", Variant::Secondary)
        .small()
        .title("Lets AI Chat use the station's tools. Corvus asks you before anything that changes the aircraft.");
    let (bw, bh) = (b.width(ui), b.height());
    let say = translate("The station's tools are off for AI Chat, so the model cannot see or steer the aircraft");
    let g = wrap(ui, &say, 11.5, Weight::W400, style::text_2(), w - bw - 34.0, 0);
    let rect = Rect::from_min_size(Pos2::new(x, y), Vec2::new(w, g.size().y.max(bh) + 14.0));
    ui.painter().rect_filled(rect, 10, style::surface_2());
    ui.painter().galley(
        Pos2::new(x + 12.0, rect.center().y - g.size().y / 2.0),
        g,
        style::text_2(),
    );
    let at = Pos2::new(rect.right() - bw - 10.0, rect.center().y - bh / 2.0);
    if b.show(ui, Rect::from_min_size(at, Vec2::new(bw, bh)), "ai-chat-mcp-on") {
        p.switch_mcp_on(h);
    }
    rect.bottom()
}

// ---- conversation ---------------------------------------------------------

fn conversation(p: &mut AiChat, ui: &mut egui::Ui, rect: Rect, w: f32, h: &mut Host, setup: &Result<(), String>) {
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
    child.set_clip_rect(rect.intersect(ui.clip_rect()));
    let scroll = egui::ScrollArea::vertical()
        .id_salt("ai-chat-talk")
        .auto_shrink([false, false])
        .stick_to_bottom(true);
    scroll.show(&mut child, |ui| {
        let top = ui.max_rect().top();
        let x = ui.max_rect().left();
        let y = if p.items.is_empty() && !p.busy {
            welcome(p, ui, x, top, w, rect.height(), h, setup)
        } else {
            let mut y = transcript(p, ui, x, top + GAP, w);
            if p.busy {
                y = typing(ui, x, y) + GAP;
            }
            y
        };
        ui.allocate_rect(
            Rect::from_min_max(Pos2::new(x, top), Pos2::new(x + w, y.max(top + 1.0))),
            Sense::hover(),
        );
        if std::mem::take(&mut p.follow) && !p.items.is_empty() {
            let end = Rect::from_min_size(Pos2::new(x, y - 1.0), Vec2::new(w, 1.0));
            ui.scroll_to_rect_animation(end, Some(egui::Align::BOTTOM), egui::style::ScrollAnimation::none());
        }
    });
}

/// An empty chat: the symbol, what it is for, and what could be asked.
#[allow(clippy::too_many_arguments)]
fn welcome(
    p: &mut AiChat,
    ui: &mut egui::Ui,
    x: f32,
    top: f32,
    w: f32,
    room: f32,
    h: &mut Host,
    setup: &Result<(), String>,
) -> f32 {
    let inner = (w - 24.0).min(420.0);
    let left = x + (w - inner) / 2.0;
    let mut y = top + ((room - 376.0) / 2.0).max(18.0);
    avatar(ui, Pos2::new(x + w / 2.0, y + 32.0), 64.0);
    y += 64.0 + 14.0;
    let center = |ui: &egui::Ui, g: Arc<Galley>, y: f32, ink: Color32| -> f32 {
        let size = g.size();
        ui.painter().galley(Pos2::new(x + (w - size.x) / 2.0, y), g, ink);
        y + size.y
    };
    let head = match setup {
        Ok(()) => translate("What can I help with?"),
        Err(_) => translate("No model set up yet"),
    };
    let g = wrap(ui, &head, 16.0, Weight::W600, style::text_1(), inner, 0);
    y = center(ui, g, y, style::text_1()) + 8.0;
    let text = match setup {
        Ok(()) => translate("Ask about the aircraft, the mission or the setup, or have something drawn on the map or flown. The model uses the station's tools by itself, and you are asked before anything that changes the aircraft."),
        Err(_) => translate("Open the gear of AI Chat in Settings, Plugins, and enter where the model is and its name. A model on this computer is enough."),
    };
    let mut job = LayoutJob::default();
    job.append(&text, 0.0, format(style::font(12.0, Weight::W400), style::text_3()));
    job.wrap.max_width = inner;
    job.halign = egui::Align::Center;
    let g = ui.painter().layout_job(job);
    let tall = g.size().y;
    ui.painter().galley(Pos2::new(x + w / 2.0, y), g, style::text_3());
    y += tall + 18.0;
    if let Err(why) = setup {
        return message::boxed(ui, left, y, inner, &translate(why), Tone::Warn) + GAP;
    }
    for (icon, question) in QUESTIONS {
        let asked = translate(question);
        let g = wrap(ui, &asked, 12.5, Weight::W500, style::text_1(), inner - 54.0, 0);
        let rect = Rect::from_min_size(Pos2::new(left, y), Vec2::new(inner, style::target(g.size().y + 20.0)));
        let resp = ui
            .interact(rect, ui.scope_id().with(("ai-chat-ask", question)), Sense::click())
            .on_hover_cursor(egui::CursorIcon::PointingHand);
        let fill = if resp.hovered() {
            style::surface_hover()
        } else {
            style::surface_2()
        };
        ui.painter().rect_filled(rect, 12, fill);
        ui.painter()
            .rect_stroke(rect, 12, Stroke::new(1.0, style::border_soft()), StrokeKind::Inside);
        let at = Pos2::new(left + 22.0, rect.center().y);
        icons::paint(ui, icon, Rect::from_center_size(at, Vec2::splat(15.0)), style::text_2());
        ui.painter().galley(
            Pos2::new(left + 42.0, rect.center().y - g.size().y / 2.0),
            g,
            style::text_1(),
        );
        if resp.clicked() {
            p.ask(&asked, h);
        }
        y = rect.bottom() + 8.0;
    }
    y
}

/// Three dots that breathe while the model thinks.
fn typing(ui: &egui::Ui, x: f32, y: f32) -> f32 {
    let rect = Rect::from_min_size(Pos2::new(x + INDENT, y), Vec2::new(58.0, 34.0));
    avatar(ui, Pos2::new(x + AVATAR / 2.0, rect.bottom() - AVATAR / 2.0), AVATAR);
    theirs(ui, rect, true);
    let t = ui.input(|i| i.time);
    for k in 0..3 {
        let phase = (t * 4.0 - f64::from(k) * 0.7).sin() as f32;
        let at = Pos2::new(
            rect.center().x + (k as f32 - 1.0) * 11.0,
            rect.center().y - phase.max(0.0) * 3.0,
        );
        ui.painter()
            .circle_filled(at, 3.2, style::text_3().gamma_multiply(0.55 + 0.45 * phase.max(0.0)));
    }
    rect.bottom()
}

fn transcript(p: &mut AiChat, ui: &mut egui::Ui, x: f32, mut y: f32, w: f32) -> f32 {
    let mut flip: Option<String> = None;
    for (i, item) in p.items.iter().enumerate() {
        match item {
            Item::Note { text, error } => {
                let t = translate(text);
                y = note(ui, x, y, w, &t, *error) + GAP;
            }
            Item::Msg(Msg::User(t)) => y = mine(ui, x, y, w, t) + GAP,
            Item::Msg(Msg::Assistant { text, .. }) => {
                let calls = p.calls_of(i);
                if !text.trim().is_empty() {
                    y = answer(ui, x, y, w, text, i) + if calls.is_empty() { GAP } else { 6.0 };
                }
                for (call, got) in calls {
                    let open = p.unfolded.contains(&call.id);
                    let (bottom, hit) = tool_row(
                        ui,
                        x + INDENT,
                        y,
                        w - INDENT,
                        &call.id,
                        &call.name,
                        &args_line(&call.args),
                        got,
                        p.busy,
                        open,
                    );
                    if hit {
                        flip = Some(call.id.clone());
                    }
                    y = bottom + 6.0;
                }
                y += 2.0;
            }
            Item::Msg(Msg::Tool { .. }) => {}
        }
    }
    if let Some(id) = flip {
        if !p.unfolded.remove(&id) {
            p.unfolded.insert(id);
        }
    }
    y
}

/// What the plugin itself has to say: a line in the middle.
fn note(ui: &egui::Ui, x: f32, y: f32, w: f32, s: &str, error: bool) -> f32 {
    let ink = if error { style::critical() } else { style::text_3() };
    let g = wrap(ui, s, 11.5, Weight::W400, ink, w - 60.0, 0);
    let size = Vec2::new(g.size().x + 44.0, g.size().y + 12.0);
    let rect = Rect::from_min_size(Pos2::new(x + (w - size.x) / 2.0, y), size);
    ui.painter().rect_filled(rect, 12, style::surface_2());
    let icon = if error { "circle-alert" } else { "info" };
    icons::paint(
        ui,
        icon,
        Rect::from_center_size(Pos2::new(rect.left() + 18.0, rect.center().y), Vec2::splat(13.0)),
        ink,
    );
    ui.painter().galley(rect.min + Vec2::new(32.0, 6.0), g, ink);
    rect.bottom()
}

/// The operator's words: at the right, in the accent.
fn mine(ui: &mut egui::Ui, x: f32, y: f32, w: f32, s: &str) -> f32 {
    let s = clip(s, SHOWN_CHARS);
    let max = (w - INDENT).min(w * 0.86).max(120.0);
    let ink = style::text_inverted();
    let g = wrap(ui, &s, TEXT, Weight::W400, ink, max - 2.0 * PAD_X, 0);
    let size = Vec2::new(g.size().x + 2.0 * PAD_X, g.size().y + 2.0 * PAD_Y);
    let rect = Rect::from_min_size(Pos2::new(x + w - size.x, y), size);
    let round = CornerRadius {
        se: 4,
        ..CornerRadius::same(ROUND)
    };
    ui.painter().rect_filled(rect, round, style::accent());
    put(ui, rect.min + Vec2::new(PAD_X, PAD_Y), g);
    rect.bottom()
}

/// The shape of a bubble of the model's.
fn theirs(ui: &egui::Ui, rect: Rect, tail: bool) {
    let round = CornerRadius {
        sw: if tail { 4 } else { ROUND },
        ..CornerRadius::same(ROUND)
    };
    ui.painter().rect_filled(rect, round, style::surface_2());
    ui.painter()
        .rect_stroke(rect, round, Stroke::new(1.0, style::border_soft()), StrokeKind::Inside);
}

/// The model's words: at the left, beside its symbol, set as it marked them up.
fn answer(ui: &mut egui::Ui, x: f32, y: f32, w: f32, s: &str, i: usize) -> f32 {
    let s = clip(s.trim(), SHOWN_CHARS);
    let room = w - INDENT - 2.0 * PAD_X;
    let ink = style::text_1();
    struct Piece {
        at: Vec2,
        galley: Arc<Galley>,
        /// A list mark to the left of it.
        mark: Option<Arc<Galley>>,
        code: bool,
    }
    let mut pieces: Vec<Piece> = Vec::new();
    let (mut wide, mut tall) = (0.0_f32, 0.0_f32);
    let blocks = markdown::blocks(&s);
    let mut was_item = false;
    for b in &blocks {
        let is_item = matches!(b, Block::Item { .. });
        if tall > 0.0 {
            tall += if is_item && was_item { 3.0 } else { 8.0 };
        }
        was_item = is_item;
        let mut mark = None;
        let (left, galley, code) = match b {
            Block::Text(t) => (0.0, spans_galley(ui, t, TEXT, false, ink, room), false),
            Block::Heading(t) => (0.0, spans_galley(ui, t, TEXT + 1.0, true, ink, room), false),
            Block::Item { depth, mark: m, spans } => {
                let left = 21.0 + 14.0 * *depth as f32;
                mark = Some(wrap(ui, m, TEXT, Weight::W400, style::text_3(), 30.0, 1));
                (left, spans_galley(ui, spans, TEXT, false, ink, room - left), false)
            }
            Block::Code(t) => (10.0, wrap_mono(ui, t, 11.5, ink, room - 20.0, 0), true),
        };
        let size = galley.size();
        let edge = if code { 8.0 } else { 0.0 };
        pieces.push(Piece {
            at: Vec2::new(left, tall + edge),
            galley,
            mark,
            code,
        });
        wide = wide.max(left + size.x + if code { 10.0 } else { 0.0 });
        tall += size.y + 2.0 * edge;
    }
    let rect = Rect::from_min_size(
        Pos2::new(x + INDENT, y),
        Vec2::new(wide.min(room) + 2.0 * PAD_X, tall + 2.0 * PAD_Y),
    );
    avatar(ui, Pos2::new(x + AVATAR / 2.0, rect.bottom() - AVATAR / 2.0), AVATAR);
    theirs(ui, rect, true);
    let origin = rect.min + Vec2::new(PAD_X, PAD_Y);
    for piece in pieces {
        let at = origin + piece.at;
        if piece.code {
            let back = Rect::from_min_max(
                Pos2::new(origin.x, at.y - 8.0),
                Pos2::new(rect.right() - PAD_X, at.y + piece.galley.size().y + 8.0),
            );
            ui.painter().rect_filled(back, 8, style::tint(0.06));
        }
        if let Some(m) = piece.mark {
            ui.painter().galley(Pos2::new(at.x - 18.0, at.y), m, style::text_3());
        }
        put(ui, at, piece.galley);
    }
    // Over the bubble, a button that copies all of it.
    let around = rect.expand2(Vec2::new(0.0, 8.0));
    if ui.rect_contains_pointer(around) {
        let b = Rect::from_min_size(Pos2::new(rect.right() - 30.0, rect.top() + 4.0), Vec2::splat(26.0));
        ui.painter().rect_filled(b, 8, style::surface_1());
        if icon_btn(ui, b, "copy", "Copy", false, ("ai-chat-copy", i)) {
            ui.ctx().copy_text(s.clone());
        }
    }
    rect.bottom()
}

/// A tool the model used: one line, and pressed, what it was given and what it answered.
#[allow(clippy::too_many_arguments)]
fn tool_row(
    ui: &mut egui::Ui,
    x: f32,
    y: f32,
    w: f32,
    id: &str,
    name: &str,
    args: &str,
    answer: Option<(&str, bool)>,
    busy: bool,
    open: bool,
) -> (f32, bool) {
    let (icon, ink) = match answer {
        Some((_, true)) => ("circle-x", style::critical()),
        Some((_, false)) => ("check", style::healthy()),
        None if busy => ("wrench", style::warning()),
        None => ("ban", style::text_3()),
    };
    let head = wrap_mono(ui, name, 11.0, style::text_2(), w - 64.0, 1);
    let row = style::target(head.size().y + 12.0);
    let width = if open { w } else { (head.size().x + 64.0).min(w) };
    let mut details = Vec::new();
    if open {
        if !args.is_empty() {
            details.push(wrap_mono(ui, args, 10.5, style::text_3(), w - 24.0, 0));
        }
        if let Some((t, e)) = answer {
            let ink = if e { style::critical() } else { style::text_2() };
            details.push(wrap_mono(ui, &clip(t, SHOWN_MAX), 10.5, ink, w - 24.0, 0));
        }
    }
    let more: f32 = details.iter().map(|g| g.size().y + 8.0).sum();
    let rect = Rect::from_min_size(Pos2::new(x, y), Vec2::new(width, row + more));
    let line = Rect::from_min_size(rect.min, Vec2::new(width, row));
    let resp = ui
        .interact(line, ui.scope_id().with(("ai-chat-tool", id)), Sense::click())
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    let fill = if resp.hovered() {
        style::surface_hover()
    } else {
        style::surface_2()
    };
    ui.painter().rect_filled(rect, 10, fill);
    ui.painter()
        .rect_stroke(rect, 10, Stroke::new(1.0, style::border_soft()), StrokeKind::Inside);
    let mid = line.center().y;
    icons::paint(
        ui,
        icon,
        Rect::from_center_size(Pos2::new(x + 17.0, mid), Vec2::splat(13.0)),
        ink,
    );
    ui.painter()
        .galley(Pos2::new(x + 32.0, mid - head.size().y / 2.0), head, style::text_2());
    icons::paint(
        ui,
        if open { "chevron-up" } else { "chevron-down" },
        Rect::from_center_size(Pos2::new(rect.right() - 16.0, mid), Vec2::splat(13.0)),
        style::text_3(),
    );
    let mut dy = y + row;
    for g in details {
        let gh = g.size().y;
        put(ui, Pos2::new(x + 12.0, dy), g);
        dy += gh + 8.0;
    }
    (rect.bottom(), resp.clicked())
}

// ---- the bar --------------------------------------------------------------

/// What is written, as tall as it is, and how tall the field shows it.
fn bar_lines(p: &AiChat, ui: &egui::Ui, w: f32) -> (f32, f32) {
    let g = laid(
        ui,
        &p.input,
        style::font(TEXT, Weight::W400),
        style::text_1(),
        w - SEND - 30.0,
        0,
    );
    let row = TEXT * 1.5;
    let full = g.size().y.max(row);
    (full, full.min(row * ROWS as f32))
}

fn bar_height(p: &AiChat, ui: &egui::Ui, w: f32) -> f32 {
    (bar_lines(p, ui, w).1 + 14.0).max(style::target(SEND) + 8.0)
}

fn composer(p: &mut AiChat, ui: &mut egui::Ui, rect: Rect, h: &mut Host, ready: bool) {
    let id = ui.scope_id().with("ai-chat-input");
    let focused = ui.memory(|m| m.has_focus(id));
    let round = 17;
    let painter = ui.painter().clone();
    painter.rect_filled(rect, round, style::surface_1());
    painter.rect_filled(rect, round, style::input(0.6));
    let line = if focused { style::accent() } else { style::border() };
    painter.rect_stroke(rect, round, Stroke::new(1.0, line), StrokeKind::Inside);
    if focused {
        painter.rect_stroke(
            rect.expand(2.0),
            round + 2,
            Stroke::new(2.0, style::accent_dim()),
            StrokeKind::Inside,
        );
    }

    let size = style::target(SEND);
    let (full, shown) = bar_lines(p, ui, rect.width());
    let field = Rect::from_min_max(
        Pos2::new(rect.left() + 16.0, rect.center().y - shown / 2.0),
        Pos2::new(rect.right() - size - 10.0, rect.center().y + shown / 2.0),
    );
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(field));
    child.set_clip_rect(field.expand(1.0).intersect(ui.clip_rect()));
    let hint = translate("Write a message");
    let resp = egui::ScrollArea::vertical()
        .id_salt("ai-chat-write")
        .auto_shrink([false, false])
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
        .show(&mut child, |ui| {
            let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, width: f32| {
                laid(
                    ui,
                    buf.as_str(),
                    style::font(TEXT, Weight::W400),
                    style::text_1(),
                    width,
                    0,
                )
            };
            ui.add(
                egui::TextEdit::multiline(&mut p.input)
                    .id(id)
                    .frame(egui::Frame::NONE)
                    .margin(egui::Margin::ZERO)
                    .desired_rows(1)
                    .desired_width(field.width())
                    .min_size(Vec2::new(0.0, full))
                    .layouter(&mut layouter)
                    .hint_text(
                        egui::RichText::new(hint)
                            .font(style::font(TEXT, Weight::W400))
                            .color(style::text_3()),
                    )
                    // Enter sends, Shift and Enter starts a line.
                    .return_key(egui::KeyboardShortcut::new(egui::Modifiers::SHIFT, egui::Key::Enter)),
            )
        })
        .inner;
    let enter = resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);

    let center = Pos2::new(rect.right() - 4.0 - size / 2.0, rect.bottom() - 4.0 - size / 2.0);
    let button = Rect::from_center_size(center, Vec2::splat(size));
    let able = p.busy || (ready && !p.input.trim().is_empty());
    let (icon, tip) = if p.busy { ("square", "Stop") } else { ("send", "Send") };
    let press = ui
        .interact(button, ui.scope_id().with("ai-chat-send"), Sense::click())
        .on_hover_text(translate(tip));
    let fill = match (p.busy, press.hovered() && able) {
        (true, _) => style::critical(),
        (false, true) => style::accent_hover(),
        (false, false) => style::accent(),
    };
    painter.circle_filled(center, SEND / 2.0, fill.gamma_multiply(if able { 1.0 } else { 0.35 }));
    icons::paint(
        ui,
        icon,
        Rect::from_center_size(center, Vec2::splat(if p.busy { 11.0 } else { 13.0 })),
        style::text_inverted(),
    );
    if press.clicked() && p.busy {
        p.stop();
    } else if (press.clicked() || enter) && able && !p.busy {
        p.send(h);
        resp.request_focus();
    }
}

#[cfg(test)]
mod tests {
    use corvus_api::net::http::Server;
    use corvus_api::plugin::bus::Bus;
    use corvus_api::plugin::shapes::ShapeSet;
    use corvus_api::plugin::{Outbox, Plugin};
    use corvus_api::ui::fonts;
    use serde_json::{json, Map, Value};

    use super::super::model::Call;
    use super::*;

    /// Draws `p` for some frames in a panel of `size`, and returns the height it says it took.
    fn draw(p: &mut AiChat, size: Vec2, opts: &Map<String, Value>) -> f32 {
        let ctx = egui::Context::default();
        fonts::install(&ctx);
        let (server, state) = (Server::local(9), json!({}));
        let (mut saved, mut out, mut bus) = (json!({}), Outbox::default(), Bus::default());
        let mut shapes = ShapeSet::new("ai-chat");
        let info = p.info();
        let mut took = 0.0;
        for frame in 0..4 {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)),
                time: Some(f64::from(frame) * 0.1),
                ..Default::default()
            };
            let mut shown = ctx.clone().run_ui(input, |ui| {
                let mut h = Host::new(
                    &ctx,
                    &server,
                    &state,
                    info,
                    &mut shapes,
                    &mut saved,
                    opts,
                    &mut out,
                    &mut bus,
                );
                // As the program puts it there: in an area that scrolls.
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let area = Rect::from_min_size(ui.max_rect().min, size - Vec2::splat(20.0));
                    took = show(p, ui, area, &mut h);
                });
            });
            shown.textures_delta.clear();
        }
        took
    }

    fn set_up() -> Map<String, Value> {
        let v = json!({"url": "http://127.0.0.1:9", "model": "m", "provider": "openai"});
        v.as_object().unwrap().clone()
    }

    fn chat() -> AiChat {
        let call = |id: &str| Call {
            id: id.into(),
            name: "get_telemetry".into(),
            args: json!({"a": 1}),
        };
        let mut p = AiChat::default();
        for _ in 0..8 {
            p.items.push(Item::Msg(Msg::User("Is it ready?\nreally?".into())));
            p.items.push(Item::Msg(Msg::Assistant {
                text: "# Head\n- one\n  - two\n\n```\ncode\n```\n| a |\nAll **good** `x`".into(),
                calls: vec![call("1"), call("2")],
            }));
            p.items.push(Item::Msg(Msg::Tool {
                id: "1".into(),
                name: "get_telemetry".into(),
                text: "{}".into(),
                error: false,
            }));
            p.items.push(Item::Note {
                text: "The server offers: m.".into(),
                error: true,
            });
        }
        p
    }

    #[test]
    fn the_view_takes_the_room_there_is_and_no_more() {
        for size in [
            Vec2::new(360.0, 700.0),
            Vec2::new(900.0, 500.0),
            Vec2::new(320.0, 300.0),
        ] {
            for opts in [set_up(), Map::new()] {
                for mut p in [AiChat::default(), chat()] {
                    for busy in [false, true] {
                        for mcp in [Mcp::Off { own: true }, Mcp::Ready(8), Mcp::Failed("no".into())] {
                            p.busy = busy;
                            p.follow = true;
                            p.mcp = Some(mcp);
                            p.unfolded.insert("1".into());
                            assert_eq!(draw(&mut p, size, &opts), size.y - 20.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_bar_grows_with_what_is_written_and_stops() {
        let mut p = AiChat::default();
        let size = Vec2::new(360.0, 700.0);
        for text in ["", "one", &"line\n".repeat(40), &"word ".repeat(400)] {
            p.input = text.to_string();
            assert_eq!(draw(&mut p, size, &set_up()), size.y - 20.0);
        }
        assert!(!p.input.is_empty());
    }

    #[test]
    fn every_symbol_is_one_the_program_has() {
        for name in [
            "bot",
            "send",
            "square",
            "plus",
            "refresh-cw",
            "copy",
            "check",
            "circle-x",
            "wrench",
            "ban",
        ]
        .into_iter()
        .chain(QUESTIONS.map(|(icon, _)| icon))
        .chain(["circle-alert", "info", "chevron-up", "chevron-down"])
        {
            assert!(icons::exists(name), "{name}");
        }
    }
}
