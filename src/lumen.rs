//! Illumination channel — the heart of wide. (DESIGN principles 5 & 7, summarized as "make the illumination channel first-class".)
//!
//! The evaluator doesn't just produce values. It streams costs, facts, and warnings into this channel as structured records.
//! Provenance, cost, error sets, and specialization will all eventually flow here. It is not a side effect of print.

use crate::span::Span;

#[derive(Clone, Copy, Debug)]
pub enum Level {
    Info, // INFO: — cost/fact (allocation, size, residence)
    Warn, // WARN: — risk (no check, division by zero)
}

#[derive(Clone, Debug)]
pub struct Lumen {
    pub level: Level,
    pub span: Span,
    pub msg: String,
}

#[derive(Default)]
pub struct Channel {
    pub records: Vec<Lumen>,
}

impl Channel {
    pub fn new() -> Self {
        Channel { records: Vec::new() }
    }

    pub fn info(&mut self, span: Span, msg: impl Into<String>) {
        self.records.push(Lumen { level: Level::Info, span, msg: msg.into() });
    }

    pub fn warn(&mut self, span: Span, msg: impl Into<String>) {
        self.records.push(Lumen { level: Level::Warn, span, msg: msg.into() });
    }
}
