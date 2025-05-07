use std::collections::HashMap;
use std::convert::AsRef;
use std::io::{self, Read, Write};
use std::path::Path;
use thiserror::Error as ThisError;

const TOP: u32 = 1 << 24;
const BOT: u32 = 1 << 15;
const MAX_FREQ: u8 = 124;
const DEFAULT_ORDER: u8 = 5;

pub type PpmResult<T> = Result<T, PpmError>;

#[derive(ThisError, Debug)]
pub enum PpmError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("Corrupt input data")]
    CorruptData,
    #[error("Invalid decoder state")]
    InvalidState,
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
        // tot_freq is now guaranteed > 0
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
        for _ in 0..4 {
            self.buffer.push((self.low >> 24) as u8);
            self.low <<= 8;
        }
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

    pub fn get_freq(&mut self, tot_freq: u32) -> PpmResult<u32> {
        // tot_freq > 0
        self.range /= tot_freq;
        let tmp = (self.code.wrapping_sub(self.low)) / self.range;
        if tmp >= tot_freq {
            return Err(PpmError::CorruptData);
        }
        Ok(tmp)
    }

    pub fn decode(&mut self, cum_freq: u32, freq: u32, tot_freq: u32) -> PpmResult<()> {
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

    /// PPMII “information inheritance”:
    /// copy each parent frequency as max(1, parent.freq/2)
    fn inherit_from(&mut self, parent: &PpmContext) {
        self.stats.clear();
        for st in &parent.stats {
            self.stats.push(State {
                symbol: st.symbol,
                freq: (st.freq / 2).max(1),
            });
        }
        self.total_freq = self.stats.iter().map(|s| s.freq as u32).sum();
    }

    /// PPMD escape probabilities:
    ///   symbol fᵢ = 2·cᵢ − 1
    ///   escape = q  (number of distinct symbols)
    ///   tot    = 2·C
    fn get_cumulative(&self) -> (Vec<u8>, Vec<u32>, u32, u32) {
        let C: u32 = self.stats.iter().map(|s| s.freq as u32).sum();
        let q = self.stats.len() as u32;
        let tot = 2 * C;
        let mut syms = Vec::with_capacity(self.stats.len());
        let mut freqs = Vec::with_capacity(self.stats.len());
        for st in &self.stats {
            syms.push(st.symbol);
            freqs.push(2 * (st.freq as u32).saturating_sub(1));
        }
        (syms, freqs, q, tot)
    }

    /// Lazy exclusion: bump only the first (highest-order) context
    /// that actually contained the symbol.
    fn update_exclusion(&mut self, symbol: u8) {
        if let Some(st) = self.stats.iter_mut().find(|s| s.symbol == symbol) {
            st.freq = st.freq.saturating_add(1).min(MAX_FREQ);
            self.total_freq = self.stats.iter().map(|s| s.freq as u32).sum();
        }
    }
}

pub struct PpmModel {
    max_order: u8,
    contexts: HashMap<Vec<u8>, PpmContext>,
}

impl PpmModel {
    pub fn new(max_order: u8) -> PpmResult<Self> {
        if max_order == 0 || max_order > 16 {
            return Err(PpmError::ModelError("Invalid max order"));
        }
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
        // 1) Lazy‐exclusion update on the longest suffix that contained the symbol
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

        // 2) Slide the history window
        history.push(symbol);
        if history.len() > self.max_order as usize {
            history.remove(0);
        }

        // 3) **Create or inherit** every missing context suffix up to max_order
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

        Ok(())
    }

    /// Decode one byte, write it into `out`, and update the model/history
    pub fn decode_symbol<R: Read>(
        &mut self,
        decoder: &mut RangeDecoder<R>,
        history: &mut Vec<u8>,
        out: &mut [u8], // length 1
    ) -> PpmResult<()> {
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

/// Compress `input_path` → `output_path`, optional max_order (default 5)
pub fn encode_file<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    max_order: Option<u8>,
) -> PpmResult<()> {
    let input = std::fs::File::open(input_path)?;
    let output = std::fs::File::create(output_path)?;
    let mut model = PpmModel::new(max_order.unwrap_or(DEFAULT_ORDER))?;
    model.encode(input, output)?;
    Ok(())
}

/// Decompress `input_path` → `output_path`, using default order 5
pub fn decode_file<P: AsRef<Path>, Q: AsRef<Path>>(input_path: P, output_path: Q) -> PpmResult<()> {
    let input = std::fs::File::open(input_path)?;
    let output = std::fs::File::create(output_path)?;
    let mut decoder = RangeDecoder::new(input)?;
    let mut model = PpmModel::new(DEFAULT_ORDER)?;
    let mut history = Vec::new();
    let mut writer = std::io::BufWriter::new(output);
    let mut buf = [0u8; 1];

    loop {
        match model.decode_symbol(&mut decoder, &mut history, &mut buf) {
            Ok(()) => writer.write_all(&buf)?,
            Err(PpmError::CorruptData) => break,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
