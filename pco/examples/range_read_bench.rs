//! Benchmark for narrow range reads.
//!
//! The default paging config produces large (~262k) pages. Fetching a narrow
//! range used to mean allocating an output buffer for a whole page; this
//! example compares that against the range-read path, which only allocates
//! for the requested values plus a single batch of scratch.
//!
//! Run with:
//!   cargo run --release --example range_read_bench

use std::time::Instant;

use pco::standalone::{simple_compress, simple_decompress, simple_decompress_range};
use pco::{ChunkConfig, PagingSpec, FULL_BATCH_N};

const TOTAL_N: usize = 1 << 20;
const RANGE_N: usize = FULL_BATCH_N; // one batch (~2 KiB of i64)
const ITERS: usize = 200;

fn dataset() -> Vec<i64> {
  // mildly compressible, non-trivial distribution
  (0..TOTAL_N as i64)
    .map(|i| (i as f64 * 0.01).sin() * 1_000_000.0 + (i % 97) as f64)
    .map(|x| x as i64)
    .collect()
}

fn compress(nums: &[i64], paging_spec: PagingSpec) -> Vec<u8> {
  let config = ChunkConfig::default().with_paging_spec(paging_spec);
  simple_compress(nums, &config).unwrap()
}

// Old style: the public simple API only offered whole-file decompression, so
// reaching a narrow range meant materializing every number first.
fn old_style_read(compressed: &[u8], start: usize, take: usize) -> Vec<i64> {
  let all = simple_decompress::<i64>(compressed).unwrap();
  all[start..start + take].to_vec()
}

fn new_style_read(compressed: &[u8], start: usize, take: usize) -> Vec<i64> {
  let mut dst = vec![0_i64; take];
  simple_decompress_range(compressed, start, &mut dst).unwrap();
  dst
}

fn median(mut xs: Vec<u128>) -> u128 {
  xs.sort_unstable();
  xs[xs.len() / 2]
}

fn bench(name: &str, compressed: &[u8], old_style: bool, raw_bytes: usize) {
  // deterministic pseudo-random range locations, always batch aligned
  let starts: Vec<usize> = (0..ITERS)
    .map(|i| {
      let h = (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
      let max_start = (TOTAL_N - RANGE_N) / FULL_BATCH_N;
      ((h as usize) % max_start) * FULL_BATCH_N
    })
    .collect();

  // correctness + warmup
  for &start in &starts {
    let got = if old_style {
      old_style_read(compressed, start, RANGE_N)
    } else {
      new_style_read(compressed, start, RANGE_N)
    };
    let expect = &dataset()[start..start + RANGE_N];
    assert_eq!(got, expect);
  }

  let mut times = Vec::with_capacity(ITERS);
  for &start in &starts {
    let t0 = Instant::now();
    let got = if old_style {
      old_style_read(compressed, start, RANGE_N)
    } else {
      new_style_read(compressed, start, RANGE_N)
    };
    times.push(t0.elapsed().as_nanos());
    std::hint::black_box(got);
  }

  let output_bytes = if old_style {
    TOTAL_N * std::mem::size_of::<i64>()
  } else {
    RANGE_N * std::mem::size_of::<i64>()
  };

  println!(
    "{:<30} median {:>9} us   output peak {:>7} KiB   ratio {:.3}",
    name,
    median(times) / 1_000,
    output_bytes / 1024,
    raw_bytes as f64 / compressed.len() as f64,
  );
}

fn latency_vs_position(compressed: &[u8], label: &str) {
  let fractions = [0.05, 0.25, 0.5, 0.75, 0.95];
  print!("{:<30}", label);
  for f in fractions {
    let start = ((TOTAL_N as f64 * f) as usize / FULL_BATCH_N) * FULL_BATCH_N;
    let mut samples = Vec::new();
    for _ in 0..20 {
      let t0 = Instant::now();
      let got = new_style_read(compressed, start, RANGE_N);
      samples.push(t0.elapsed().as_nanos());
      std::hint::black_box(got);
    }
    print!(
      " {:>5}us@{}%",
      median(samples) / 1_000,
      (f * 100.0) as u32
    );
  }
  println!();
}

fn main() {
  let nums = dataset();
  let raw_bytes = nums.len() * std::mem::size_of::<i64>();

  let default_compressed = compress(&nums, PagingSpec::EqualPagesUpTo(1 << 18));
  let small_compressed = compress(&nums, PagingSpec::Pow2(11));
  let balanced_compressed = compress(&nums, PagingSpec::Pow2(14));
  let page_size_logs = [10usize, 12, 14, 16, 18];

  println!(
    "dataset: {} i64 values, {} KiB raw, range width {} values\n",
    TOTAL_N,
    raw_bytes / 1024,
    RANGE_N,
  );

  println!(
    "== page size sweep (random narrow range, {} iters) ==",
    ITERS
  );
  println!(
    "{:<30} {:>10}   {:>12}   {}",
    "config", "compressed", "ratio", "range-read latency / output peak"
  );
  for &log in &page_size_logs {
    let compressed = compress(&nums, PagingSpec::Pow2(log));
    let name = format!("pow2 page {}", 1usize << log);
    println!(
      "{:<30} {:>7} KiB   {:>12.3}",
      name,
      compressed.len() / 1024,
      raw_bytes as f64 / compressed.len() as f64,
    );
    bench(&name, &compressed, false, raw_bytes);
  }
  bench(
    "default equal pages 262144",
    &default_compressed,
    false,
    raw_bytes,
  );

  println!("\n== baseline: whole-file decompression ==");
  bench(
    "old: full decompress",
    &default_compressed,
    true,
    raw_bytes,
  );

  println!("\n== range-read latency vs target position ==");
  latency_vs_position(&default_compressed, "default pages 262144");
  latency_vs_position(&balanced_compressed, "pow2 pages 16384");
  latency_vs_position(&small_compressed, "pow2 pages 2048");
}
