use std::collections::HashMap;
use std::convert::AsRef;
use std::io::{self, Read, Write};
use std::path::Path;
use thiserror::Error as ThisError;

const TOP: u32 = 1 << 24;
const BOT: u32 = 1 << 15;
const MAX_FREQ: u32 = 124;
const DEFAULT_ORDER: u8 = 5;

pub type PpmResult<T> = Result<T, PpmError>;

#[derive(ThisError, Debug)]
pub enum PpmError {
    #[error("IO error: {0}")]
    IoError(io::Error),
    #[error("Corrupt input data")]
    CorruptData,
    #[error("Invalid decoder state")]
    InvalidState,
    #[error("Model error: {0}")]
    ModelError(&'static str),
}

impl From<io::Error> for PpmError {
    fn from(err: io::Error) -> Self {
        PpmError::IoError(err)
    }
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
        self.range /= tot_freq;
        self.low += cum_freq * self.range;
        self.range *= freq;

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
    fn new(mut reader: R) -> PpmResult<Self> {
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
        self.range /= tot_freq;
        let tmp = (self.code - self.low) / self.range;
        if tmp >= tot_freq {
            return Err(PpmError::CorruptData);
        }
        Ok(tmp)
    }

    fn decode(&mut self, cum_freq: u32, freq: u32, tot_freq: u32) -> PpmResult<()> {
        if cum_freq + freq > tot_freq {
            return Err(PpmError::CorruptData);
        }
        self.low += cum_freq * self.range;
        self.range *= freq;

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
    order: u8,
    stats: Vec<State>,
    suffix: Option<Box<PpmContext>>,
    total_freq: u32,
}

impl PpmContext {
    fn new(order: u8) -> Self {
        PpmContext {
            order,
            stats: Vec::new(),
            suffix: None,
            total_freq: 0,
        }
    }

    fn init(&mut self) {
        if self.order == 1 {
            for symbol in 0..=255 {
                self.stats.push(State {
                    symbol: symbol as u8,
                    freq: 1,
                });
            }
            self.total_freq = 256;
        }
    }

    fn find_symbol(&self, symbol: u8) -> Option<&State> {
        self.stats.iter().find(|s| s.symbol == symbol)
    }

    fn rescale(&mut self) -> PpmResult<()> {
        self.stats.retain(|s| s.freq > 0);
        self.total_freq = self.stats.iter().map(|s| u32::from(s.freq)).sum();
        if self.total_freq == 0 {
            Err(PpmError::ModelError("Context has zero total frequency"))
        } else {
            Ok(())
        }
    }
}

pub struct PpmModel {
    max_order: u8,
    contexts: HashMap<Vec<u8>, PpmContext>,
    current_order: u8,
}

impl PpmModel {
    pub fn new(max_order: u8) -> PpmResult<Self> {
        if max_order < 1 || max_order > 8 {
            return Err(PpmError::ModelError("Invalid max order"));
        }

        let mut model = PpmModel {
            max_order,
            contexts: HashMap::new(),
            current_order: 0,
        };

        let mut ctx = PpmContext::new(1);
        ctx.init();
        model.contexts.insert(Vec::new(), ctx);

        Ok(model)
    }

    pub fn encode<R: Read, W: Write>(&mut self, mut input: R, output: W) -> PpmResult<W> {
        let mut encoder = RangeEncoder::new(output);
        let mut buffer = [0; 1];
        let mut history = Vec::new();

        while input.read(&mut buffer)? > 0 {
            let symbol = buffer[0];
            let mut cum_freq = 0;
            let mut found = false;

            for order in (1..=self.current_order.min(history.len() as u8)).rev() {
                let ctx_key = history[history.len() - order as usize..].to_vec();
                if let Some(ctx) = self.contexts.get_mut(&ctx_key) {
                    if let Some(state) = ctx.find_symbol(symbol) {
                        let tot_freq = ctx.total_freq;
                        encoder.encode(cum_freq, state.freq as u32, tot_freq)?;
                        found = true;
                        break;
                    }
                    cum_freq += ctx.total_freq;
                }
            }

            if !found {
                let tot_freq = self.contexts[&Vec::new()].total_freq;
                encoder.encode(cum_freq, 1, tot_freq + 1)?;
            }

            self.update_model(&mut history, symbol)?;
        }

        encoder.finish()
    }

    pub fn decode_symbol<R: Read>(
        &mut self,
        decoder: &mut RangeDecoder<R>,
        buffer: &mut [u8],
    ) -> PpmResult<()> {
        let mut history = Vec::new();
        let mut cum_freq = 0;
        let mut found = false;
        let mut symbol = 0u8;

        for order in (1..=self.current_order.min(history.len() as u8)).rev() {
            let ctx_key = history[history.len() - order as usize..].to_vec();
            if let Some(ctx) = self.contexts.get(&ctx_key) {
                let tot_freq = ctx.total_freq;
                let current_threshold = cum_freq + tot_freq;
                let freq = decoder.get_freq(current_threshold)?;
                if freq < current_threshold {
                    let symbol_freq = freq - cum_freq;
                    let mut sum = 0;
                    for state in &ctx.stats {
                        if sum + state.freq as u32 > symbol_freq {
                            symbol = state.symbol;
                            decoder.decode(sum, state.freq as u32, tot_freq)?;
                            found = true;
                            break;
                        }
                        sum += state.freq as u32;
                    }
                    if found {
                        break;
                    } else {
                        return Err(PpmError::CorruptData);
                    }
                } else {
                    cum_freq += tot_freq;
                }
            }
        }

        if !found {
            let order0_ctx = self
                .contexts
                .get(&Vec::new())
                .ok_or(PpmError::CorruptData)?;
            let tot_freq = order0_ctx.total_freq + 1;
            let freq = decoder.get_freq(tot_freq)?;
            if freq >= order0_ctx.total_freq {
                return Err(PpmError::CorruptData);
            }
            let mut sum = 0;
            for state in &order0_ctx.stats {
                if sum + state.freq as u32 > freq {
                    symbol = state.symbol;
                    decoder.decode(sum, state.freq as u32, order0_ctx.total_freq)?;
                    found = true;
                    break;
                }
                sum += state.freq as u32;
            }
            if !found {
                return Err(PpmError::CorruptData);
            }
        }

        self.update_model(&mut history, symbol)?;
        buffer[0] = symbol;
        Ok(())
    }

    fn update_model(&mut self, history: &mut Vec<u8>, symbol: u8) -> PpmResult<()> {
        for i in 0..history.len() {
            let ctx_key = history[i..].to_vec();
            if let Some(ctx) = self.contexts.get_mut(&ctx_key) {
                if let Some(pos) = ctx.stats.iter_mut().position(|s| s.symbol == symbol) {
                    ctx.stats[pos].freq = (ctx.stats[pos].freq + 1).min(MAX_FREQ as u8);
                } else {
                    ctx.stats.push(State { symbol, freq: 1 });
                }
                ctx.rescale()?;
            }
        }

        history.push(symbol);
        if history.len() > self.max_order as usize {
            history.remove(0);
        }

        if !self.contexts.contains_key(history) {
            let order = history.len() as u8;
            let suffix_key = if order > 1 { &history[1..] } else { &[] };
            let suffix = self.contexts.get(suffix_key).cloned();
            let mut new_ctx = PpmContext::new(order);
            new_ctx.suffix = suffix.map(|ctx| Box::new(ctx));
            new_ctx.init();
            self.contexts.insert(history.clone(), new_ctx);
        }

        Ok(())
    }
}
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

pub fn decode_file<P: AsRef<Path>, Q: AsRef<Path>>(input_path: P, output_path: Q) -> PpmResult<()> {
    let input = std::fs::File::open(input_path)?;
    let output = std::fs::File::create(output_path)?;
    let mut decoder = RangeDecoder::new(input)?;
    let mut model = PpmModel::new(DEFAULT_ORDER)?;
    let mut output_writer = std::io::BufWriter::new(output);
    let mut buffer = [0u8; 1];

    loop {
        match model.decode_symbol(&mut decoder, &mut buffer) {
            Ok(()) => output_writer.write_all(&buffer)?,
            Err(PpmError::CorruptData) => break,
            Err(e) => return Err(e),
        }
    }

    Ok(())
}
