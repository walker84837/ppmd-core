# ppmd-core

[![Rust](https://github.com/walker84837/ppmd-core/actions/workflows/rust.yml/badge.svg)](https://github.com/walker84837/ppmd-core/actions/workflows/rust.yml)

A pure-Rust implementation of the PPMd (Prediction by Partial Matching, variant D) compressor with an underlying range coder.  
Designed for safety (no `unsafe`), and zero-dependency entropy coding in your Rust projects.

## Features

- **PPMd** order-N modeling (1 ≤ N ≤ 16) 
- Adaptive frequency tables with escape mechanism
- High-speed range encoder/decoder
- Configurable context order: trade CPU/memory vs. compression ratio
- Safe Rust only (`#![forbid(unsafe_code)]`)

## Quick Start

### File-based API

```rust
use ppmd_core::{encode_file, decode_file, PpmResult};

fn main() -> PpmResult<()> {
    encode_file("data/input.bin", "data/output.ppmd", None)?;
    encode_file("data/input.bin", "data/output_o8.ppmd", Some(8))?;
    decode_file("data/output.ppmd", "data/decoded.bin")?;
    Ok(())
}
```

### In-memory API

If you prefer streaming or in-memory buffers:

```rust
use ppmd_core::{PpmModel, PpmResult};
use std::io::{Cursor, Read, Write};

fn compress_bytes(input: &[u8], order: u8) -> PpmResult<Vec<u8>> {
    let mut model = PpmModel::new(order)?;
    let mut output = Vec::new();
    let mut writer = Cursor::new(&mut output);

    // (Optionally write length prefix yourself if needed)
    model.encode(&mut Cursor::new(input), &mut writer)?;
    Ok(output)
}

fn decompress_bytes(input: &[u8], original_len: usize) -> PpmResult<Vec<u8>> {
    use ppmd_core::RangeDecoder;
    use std::io::BufWriter;

    let mut reader = Cursor::new(input);
    let mut decoder = RangeDecoder::new(&mut reader)?;
    let mut model = PpmModel::new(5)?; // must match encoder’s order
    let mut history = Vec::new();
    let mut output = Vec::with_capacity(original_len);

    while output.len() < original_len {
        let mut byte = [0];
        model.decode_symbol(&mut decoder, &mut history, &mut byte)?;
        output.push(byte[0]);
    }
    Ok(output)
}
```

## Tuning and Performance

- **Order (context length)**
  * Low (1–3): very fast, minimal memory, poorer compression
  * Medium (4–8): balanced speed vs. ratio (DEFAULT = 5)
  * High (9–16): best ratio, more memory, slower
- **`MAX_FREQ`**:  Controls the maximum per-symbol count in any context (prevents overflow).

You can tweak `DEFAULT_ORDER` or call `encode_file(..., Some(order))` to experiment.

Benchmark (TODO) your own data with:

```bash
cargo bench
```

## License

MIT OR Apache-2.0 (See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) files.)
