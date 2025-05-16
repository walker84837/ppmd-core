//! aa
#![forbid(clippy::let_underscore_drop)]
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]
#![warn(missing_docs)]

use std::collections::HashMap;
use std::convert::AsRef;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use thiserror::Error as ThisError;

const TOP: u32 = 1 << 24;
const BOT: u32 = 1 << 15;
const MAX_FREQ: u8 = 124;
const DEFAULT_ORDER: u8 = 5;

///
pub type PpmResult<T> = Result<T, PpmError>;

#[derive(ThisError, Debug)]
/// ...
pub enum PpmError {
    /// ...
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    /// ...
    #[error("Corrupt input data")]
    CorruptData,
    /// ...
    #[error("Invalid decoder state")]
    InvalidState,
    /// ...
    #[error("Model error: {0}")]
    ModelError(&'static str),
}

struct RangeEncoder<W: Write> {
    low: u32,
    range: u32,
    buffer: Vec<u8>,
    writer: W,
}

impl<W: Write> RangeEncoder<W> {
    fn new(writer: W) -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            buffer: Vec::with_capacity(4096),
            writer,
        }
    }

    fn encode(&mut self, cum_freq: u32, freq: u32, tot_freq: u32) -> PpmResult<()> {
        assert!(tot_freq > 0, "total frequency must be positive");
        assert!(freq > 0, "symbol frequency must be positive");
        assert!(cum_freq < tot_freq, "cumulative freq out of range");
        assert!(cum_freq + freq <= tot_freq, "freq interval exceeds total");

        self.range /= tot_freq;
        self.low = self.low.wrapping_add(cum_freq * self.range);
        self.range = self.range.wrapping_mul(freq);

        while (self.low ^ (self.low.wrapping_add(self.range))) < TOP || self.range < BOT {
            if self.range < BOT {
                self.range = (-(self.low as i32) as u32) & (BOT - 1);
            }
            self.buffer.push((self.low >> 24) as u8);
            self.low <<= 8;
            self.range <<= 8;
            if self.buffer.len() >= 4096 {
                self.writer.write_all(&self.buffer)?;
                self.buffer.clear();
            }
        }
        Ok(())
    }

    fn finish(mut self) -> PpmResult<W> {
        assert!(self.range > 0, "range became zero in finish");
        for _ in 0..4 {
            self.buffer.push((self.low >> 24) as u8);
            self.low <<= 8;
        }
        assert!(!self.buffer.is_empty(), "nothing to flush in finish");
        self.writer.write_all(&self.buffer)?;
        Ok(self.writer)
    }
}

pub struct RangeDecoder<R: Read> {
    low: u32,
    code: u32,
    range: u32,
    reader: R,
    buffer: [u8; 1],
}

impl<R: Read> RangeDecoder<R> {
    pub fn new(mut reader: R) -> PpmResult<Self> {
        let mut code = 0;
        for _ in 0..4 {
            let mut byte = [0];
            reader.read_exact(&mut byte)?;
            code = (code << 8) | u32::from(byte[0]);
        }
        Ok(Self {
            low: 0,
            code,
            range: u32::MAX,
            reader,
            buffer: [0],
        })
    }

    fn get_freq(&mut self, tot_freq: u32) -> PpmResult<u32> {
        assert!(tot_freq > 0, "total frequency must be positive");
        self.range /= tot_freq;
        let tmp = (self.code.wrapping_sub(self.low)) / self.range;
        if tmp >= tot_freq {
            return Err(PpmError::CorruptData);
        }
        Ok(tmp)
    }

    fn decode(&mut self, cum_freq: u32, freq: u32, tot_freq: u32) -> PpmResult<()> {
        assert!(freq > 0, "frequency must be positive");
        assert!(cum_freq < tot_freq, "cumulative freq out of range");
        assert!(cum_freq + freq <= tot_freq, "freq interval exceeds total");

        if cum_freq.wrapping_add(freq) > tot_freq {
            return Err(PpmError::CorruptData);
        }
        self.low = self.low.wrapping_add(cum_freq * self.range);
        self.range = self.range.wrapping_mul(freq);
        while (self.low ^ (self.low.wrapping_add(self.range))) < TOP || self.range < BOT {
            if self.range < BOT {
                self.range = (-(self.low as i32) as u32) & (BOT - 1);
            }
            self.reader.read_exact(&mut self.buffer)?;
            self.code = (self.code << 8) | u32::from(self.buffer[0]);
            self.low <<= 8;
            self.range <<= 8;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct State {
    symbol: u8,
    freq: u8,
}

#[derive(Clone, Debug)]
struct PpmContext {
    stats: Vec<State>,
    total_freq: u32,
}

impl PpmContext {
    fn new() -> Self {
        PpmContext {
            stats: Vec::new(),
            total_freq: 0,
        }
    }

    /// PPMII "information inheritance":
    /// copy each parent frequency as max(1, parent.freq/2)
    fn inherit_from(&mut self, parent: &PpmContext) {
        // parents stats should be non‐empty if called in contexts
        assert!(parent.stats.len() > 0, "parent context has no stats");

        self.stats.clear();
        for st in &parent.stats {
            let f = (st.freq / 2).max(1);
            assert!(f >= 1, "inherited frequency dropped below 1");
            self.stats.push(State {
                symbol: st.symbol,
                freq: f,
            });
        }
        self.total_freq = self.stats.iter().map(|s| s.freq as u32).sum();
        assert!(
            self.total_freq > 0,
            "total_freq must be positive after inherit"
        );
    }

    /// PPMD escape probabilities:
    ///   symbol fᵢ = 2·cᵢ − 1
    ///   escape = q  (number of distinct symbols)
    ///   tot    = 2·C
    fn get_cumulative(&self) -> (Vec<u8>, Vec<u32>, u32, u32) {
        let c: u32 = self.stats.iter().map(|s| s.freq as u32).sum();
        let q = self.stats.len() as u32;
        let tot = 2 * c;
        assert!(tot > 0, "total (2*C) must be positive");

        let mut syms = Vec::with_capacity(self.stats.len());
        let mut freqs = Vec::with_capacity(self.stats.len());
        for st in &self.stats {
            let f = (st.freq as u32) * 2 - 1;
            assert!(f > 0, "computed symbol frequency must be positive");
            syms.push(st.symbol);
            freqs.push(f);
        }
        (syms, freqs, q, tot)
    }

    /// Lazy exclusion: bump only the first (highest-order) context
    /// that actually contained the symbol.
    fn update_exclusion(&mut self, symbol: u8) {
        let before = self.total_freq;
        if let Some(st) = self.stats.iter_mut().find(|s| s.symbol == symbol) {
            let new_freq = st.freq.saturating_add(1).min(MAX_FREQ);
            assert!(new_freq >= st.freq, "freq must not decrease on bump");

            st.freq = new_freq;
            self.total_freq = self.stats.iter().map(|s| s.freq as u32).sum();
            assert!(
                self.total_freq >= before,
                "total_freq must not shrink after update"
            );
        }
    }
}

pub struct PpmModel {
    max_order: u8,
    contexts: HashMap<Vec<u8>, PpmContext>,
}

impl PpmModel {
    pub fn new(max_order: u8) -> PpmResult<Self> {
        assert!(
            max_order > 0 && max_order <= 16,
            "max_order out of valid range"
        );
        let mut m = PpmModel {
            max_order,
            contexts: HashMap::new(),
        };
        // build the order−1 root context with uniform order‑0 stats
        let mut root = PpmContext::new();
        for sym in 0u8..=255 {
            root.stats.push(State {
                symbol: sym,
                freq: 1,
            });
        }
        root.total_freq = 256;
        assert_eq!(root.stats.len(), 256, "root must contain all 256 symbols");
        m.contexts.insert(Vec::new(), root);
        Ok(m)
    }

    /// Encode the entire input → output
    pub fn encode<R: Read, W: Write>(&mut self, mut input: R, output: W) -> PpmResult<W> {
        let mut encoder = RangeEncoder::new(output);
        let mut history = Vec::new();
        let mut buf = [0u8; 1];

        while input.read(&mut buf)? > 0 {
            let sym = buf[0];
            self.encode_symbol(&mut encoder, &history, sym)?;
            self.update_model(&mut history, sym)?;
        }
        encoder.finish()
    }

    fn encode_symbol<W: Write>(
        &self,
        encoder: &mut RangeEncoder<W>,
        history: &[u8],
        symbol: u8,
    ) -> PpmResult<()> {
        assert!(history.len() <= self.max_order as usize, "history too long");

        // back‑off from highest order down to order−1:
        for order in (1..=self.max_order.min(history.len() as u8)).rev() {
            let key = history[history.len() - order as usize..].to_vec();
            if let Some(ctx) = self.contexts.get(&key) {
                let (syms, freqs, esc, tot) = ctx.get_cumulative();
                let mut cum = 0;
                // if symbol found in this context, emit it
                for (i, &s) in syms.iter().enumerate() {
                    if s == symbol {
                        return encoder.encode(cum, freqs[i], tot);
                    }
                    cum += freqs[i];
                }
                // otherwise emit escape
                encoder.encode(cum, esc, tot)?;
            }
        }
        // final fallback at order−1 root (uniform)
        let root = &self.contexts[&Vec::new()];
        let tot0 = (root.stats.len() as u32) + 1;
        if let Some(pos) = root.stats.iter().position(|s| s.symbol == symbol) {
            encoder.encode(pos as u32, 1, tot0)
        } else {
            // one final escape (should never really happen)
            encoder.encode(root.stats.len() as u32, 1, tot0)
        }
    }

    /// After emitting symbol, update ALL contexts up to max_order
    fn update_model(&mut self, history: &mut Vec<u8>, symbol: u8) -> PpmResult<()> {
        let before = self.contexts.len();
        // lazy‐exclusion update on the longest suffix that contained the symbol
        let mut bumped = false;
        for i in 0..history.len() {
            let key = history[i..].to_vec();
            if let Some(ctx) = self.contexts.get_mut(&key) {
                if !bumped {
                    ctx.update_exclusion(symbol);
                    bumped = ctx.stats.iter().any(|s| s.symbol == symbol);
                }
            }
        }
        // if it never existed, add to the root order‑0
        if !bumped {
            let root = self.contexts.get_mut(&Vec::new()).unwrap();
            root.stats.push(State { symbol, freq: 1 });
            root.total_freq += 1;
        }

        // Slide the history window
        history.push(symbol);
        if history.len() > self.max_order as usize {
            history.remove(0);
        }
        assert!(self.contexts.len() >= before, "contexts should not shrink");

        // Create or inherit every missing context suffix up to max_order
        let current_len = history.len();
        let max_ctx = self.max_order.min(current_len as u8) as usize;
        for order in 1..=max_ctx {
            let key = history[current_len - order..].to_vec();
            if !self.contexts.contains_key(&key) {
                // build from the (order−1) parent:
                let parent_key = if key.len() > 1 {
                    key[1..].to_vec()
                } else {
                    Vec::new()
                };
                let mut ctx = PpmContext::new();
                if let Some(parent) = self.contexts.get(&parent_key) {
                    ctx.inherit_from(parent);
                }
                self.contexts.insert(key, ctx);
            }
        }

        assert!(
            history.len() <= self.max_order as usize,
            "history exceeded max_order"
        );
        Ok(())
    }

    /// Decode one byte, write it into `out`, and update the model/history
    pub fn decode_symbol<R: Read>(
        &mut self,
        decoder: &mut RangeDecoder<R>,
        history: &mut Vec<u8>,
        out: &mut [u8],
    ) -> PpmResult<()> {
        assert!(out.len() == 1, "output buffer must be exactly one byte");
        // back‑off decode:
        for order in (1..=self.max_order.min(history.len() as u8)).rev() {
            let key = history[history.len() - order as usize..].to_vec();
            if let Some(ctx) = self.contexts.get(&key) {
                let (syms, freqs, esc, tot) = ctx.get_cumulative();
                let threshold = tot.saturating_sub(esc);
                let r = decoder.get_freq(tot)?;
                if r < threshold {
                    // actual symbol
                    let mut cum = 0;
                    for (i, &f) in freqs.iter().enumerate() {
                        if r < cum + f {
                            let sym = syms[i];
                            decoder.decode(cum, f, tot)?;
                            out[0] = sym;
                            self.update_model(history, sym)?;
                            return Ok(());
                        }
                        cum += f;
                    }
                    unreachable!();
                } else {
                    // escape
                    decoder.decode(threshold, esc, tot)?;
                }
            }
        }

        // root fallback:
        let root = &self.contexts[&Vec::new()];
        let tot0 = (root.stats.len() as u32) + 1;
        let r0 = decoder.get_freq(tot0)?;
        if r0 < root.stats.len() as u32 {
            let sym = root.stats[r0 as usize].symbol;
            decoder.decode(r0, 1, tot0)?;
            out[0] = sym;
            self.update_model(history, sym)?;
            Ok(())
        } else {
            // end of stream
            decoder.decode(root.stats.len() as u32, 1, tot0)?;
            Err(PpmError::CorruptData)
        }
    }
}

// File‐based convenience wrappers:

pub fn encode_file<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    max_order: Option<usize>,
) -> PpmResult<()> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();

    // Step 1: determine original input length
    let input_file = File::open(input_path)?;
    let input_len = input_file.metadata()?.len();

    // Step 2: open output and write length prefix
    let mut output = File::create(output_path)?;
    output.write_all(&input_len.to_le_bytes())?;

    // Step 3: re-open input for reading (we already used it for metadata)
    let mut input = File::open(input_path)?;

    // Step 4: encode using PPM
    let order = max_order.unwrap_or(DEFAULT_ORDER as usize);
    let mut model = PpmModel::new(order.try_into().unwrap())?;
    model.encode(&mut input, &mut output)?;

    Ok(())
}

/// Decompress `input_path` → `output_path`, using default order 5
pub fn decode_file<P: AsRef<Path>, Q: AsRef<Path>>(input_path: P, output_path: Q) -> PpmResult<()> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();

    let mut input = File::open(input_path)?;
    let mut len_buf = [0u8; 8];
    input.read_exact(&mut len_buf)?;
    let expected = u64::from_le_bytes(len_buf);

    let mut decoder = RangeDecoder::new(input)?;
    let mut model = PpmModel::new(DEFAULT_ORDER)?;
    let mut history = Vec::new();
    let mut writer = BufWriter::new(File::create(output_path)?);

    let mut buf = [0u8; 1];
    let mut actual = 0;
    while actual < expected {
        model.decode_symbol(&mut decoder, &mut history, &mut buf)?;
        writer.write_all(&buf)?;
        actual += 1;
    }

    Ok(())
}
