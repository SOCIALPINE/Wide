//! Illumination channel — the heart of wide. (DESIGN principles 5 & 7, summarized as "make the illumination channel first-class".)
//!
//! The evaluator doesn't just produce values. It streams costs, facts, and warnings into this channel as structured records.
//! Provenance, cost, error sets, and specialization will all eventually flow here. It is not a side effect of print.
//!
//! Memory (v0.53): a hot loop can emit the same record millions of times — storing each one made
//! long programs balloon (measured: 300k loop iterations → 55 MB). Identical records (same span,
//! same message) now *aggregate* into one record with a repeat count, and the channel is hard-capped;
//! past the cap new unique records are dropped and the truncation itself is reported (honestly).

use std::collections::HashMap;

use crate::span::Span;

/// Hard cap on *unique* records — far above any real program's unique illumination sites, so hitting
/// it means something is generating unbounded distinct messages; we stop storing and say so.
const MAX_RECORDS: usize = 100_000;

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
    pub count: usize, // how many times this exact record was emitted (loops aggregate)
}

#[derive(Default)]
pub struct Channel {
    pub records: Vec<Lumen>,
    index: HashMap<(usize, usize, String), usize>, // (line, col, msg) → records index
    pub truncated: usize, // records dropped past MAX_RECORDS (0 = none)
}

impl Channel {
    pub fn new() -> Self {
        Channel::default()
    }

    pub fn info(&mut self, span: Span, msg: impl Into<String>) {
        self.emit(Level::Info, span, msg.into());
    }

    pub fn warn(&mut self, span: Span, msg: impl Into<String>) {
        self.emit(Level::Warn, span, msg.into());
    }

    fn emit(&mut self, level: Level, span: Span, msg: String) {
        let key = (span.line, span.col, msg);
        if let Some(&i) = self.index.get(&key) {
            self.records[i].count += 1;
            return;
        }
        if self.records.len() >= MAX_RECORDS {
            self.truncated += 1;
            return;
        }
        self.records.push(Lumen { level, span, msg: key.2.clone(), count: 1 });
        self.index.insert(key, self.records.len() - 1);
    }
}
