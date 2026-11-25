use crate::chunk_config::{ChunkConfig, DeltaSpec};
use crate::errors::PcoResult;
use crate::standalone::{simple_compress, simple_decompress_range, FileDecompressor};
use crate::wrapped::{
  FileCompressor as WrappedFileCompressor, FileDecompressor as WrappedFileDecompressor,
};
use crate::{PagingSpec, FULL_BATCH_N};

fn config_with_pages(paging_spec: PagingSpec, delta_spec: DeltaSpec) -> ChunkConfig {
  ChunkConfig {
    paging_spec,
    delta_spec,
    ..Default::default()
  }
}

#[test]
fn test_pow2_paging_spec() {
  let spec = PagingSpec::Pow2(10);
  assert_eq!(
    spec.n_per_page(2500).unwrap(),
    vec![1024, 1024, 452]
  );
  assert_eq!(spec.n_per_page(1024).unwrap(), vec![1024]);
  assert_eq!(spec.n_per_page(1025).unwrap(), vec![1024, 1]);
  assert_eq!(
    spec.n_per_page(0).unwrap(),
    Vec::<usize>::new()
  );

  let spec = PagingSpec::Pow2(8);
  let pages = spec.n_per_page(1000).unwrap();
  assert!(pages[..pages.len() - 1]
    .iter()
    .all(|&n| n.is_multiple_of(FULL_BATCH_N)));

  assert!(PagingSpec::Pow2(200).n_per_page(10).is_err());
}

fn standalone_range(
  nums: &[i64],
  config: &ChunkConfig,
  start: usize,
  take: usize,
) -> PcoResult<Vec<i64>> {
  let compressed = simple_compress(nums, config)?;
  let mut dst = vec![0_i64; take];
  let progress = simple_decompress_range(compressed.as_slice(), start, &mut dst)?;
  assert_eq!(progress.n_processed, take);
  Ok(dst)
}

#[test]
fn test_standalone_range_no_delta() -> PcoResult<()> {
  let n = 3 * FULL_BATCH_N + 77;
  let nums: Vec<i64> = (0..n as i64)
    .map(|x| (x * 7919) % 100_000 - 50_000)
    .collect();
  // default large page: the narrow range sits deep inside one page
  let config = config_with_pages(
    PagingSpec::EqualPagesUpTo(1 << 18),
    DeltaSpec::NoOp,
  );

  let start = FULL_BATCH_N;
  let got = standalone_range(&nums, &config, start, FULL_BATCH_N)?;
  assert_eq!(got, nums[start..start + FULL_BATCH_N]);

  let start = 2 * FULL_BATCH_N;
  let got = standalone_range(&nums, &config, start, FULL_BATCH_N)?;
  assert_eq!(got, nums[start..start + FULL_BATCH_N]);

  let compressed = simple_compress(&nums, &config)?;
  let mut dst = vec![0_i64; 16];
  assert!(simple_decompress_range(compressed.as_slice(), 1, &mut dst).is_err());

  Ok(())
}

#[test]
fn test_standalone_range_pow2_pages_cross_chunk() -> PcoResult<()> {
  let n = 4 * FULL_BATCH_N + 31;
  let nums: Vec<i64> = (0..n as i64).map(|x| x * x - 3 * x + 11).collect();
  // 512-number pages/chunks with a 2nd-order delta; the range straddles the
  // boundary between two chunks.
  let config = config_with_pages(
    PagingSpec::Pow2(9),
    DeltaSpec::TryConsecutive(2),
  );

  let start = 2 * FULL_BATCH_N;
  let got = standalone_range(&nums, &config, start, FULL_BATCH_N)?;
  assert_eq!(got, nums[start..start + FULL_BATCH_N]);

  // range spanning three chunks
  let start = FULL_BATCH_N;
  let take = 2 * FULL_BATCH_N;
  let got = standalone_range(&nums, &config, start, take)?;
  assert_eq!(got, nums[start..start + take]);

  Ok(())
}

#[test]
fn test_standalone_range_to_end() -> PcoResult<()> {
  let n = 3 * FULL_BATCH_N + 40;
  let nums: Vec<i64> = (0..n as i64).map(|x| x ^ 0x5a5a).collect();
  let config = config_with_pages(PagingSpec::Pow2(9), DeltaSpec::NoOp);

  let start = 2 * FULL_BATCH_N;
  let compressed = simple_compress(&nums, &config)?;
  let mut tail = vec![0_i64; n - start];
  let progress = simple_decompress_range(compressed.as_slice(), start, &mut tail)?;
  assert!(progress.finished);
  assert_eq!(tail, nums[start..]);

  Ok(())
}

#[test]
fn test_standalone_range_unaligned_start_rejected() -> PcoResult<()> {
  let n = 2 * FULL_BATCH_N + 100;
  let nums: Vec<i64> = (0..n as i64).collect();
  let config = config_with_pages(PagingSpec::Pow2(9), DeltaSpec::NoOp);
  let compressed = simple_compress(&nums, &config)?;
  let mut dst = vec![0_i64; FULL_BATCH_N];
  // 300 is not a multiple of FULL_BATCH_N and not a chunk boundary
  assert!(simple_decompress_range(compressed.as_slice(), 300, &mut dst).is_err());
  Ok(())
}

#[test]
fn test_standalone_range_past_end_rejected() -> PcoResult<()> {
  let n = FULL_BATCH_N;
  let nums: Vec<i64> = (0..n as i64).collect();
  let config = config_with_pages(PagingSpec::Pow2(9), DeltaSpec::NoOp);
  let compressed = simple_compress(&nums, &config)?;
  let mut dst = vec![0_i64; FULL_BATCH_N];
  assert!(simple_decompress_range(compressed.as_slice(), n, &mut dst).is_err());
  Ok(())
}

#[test]
fn test_standalone_range_lookback_delta() -> PcoResult<()> {
  // repeating pattern gives lookback delta something to exploit
  let pattern: Vec<i64> = (0..64).map(|x| x * 13 - 400).collect();
  let n = 4 * FULL_BATCH_N;
  let nums: Vec<i64> = (0..n).map(|i| pattern[i % pattern.len()]).collect();
  let config = config_with_pages(PagingSpec::Pow2(9), DeltaSpec::TryLookback);

  let start = 3 * FULL_BATCH_N;
  let got = standalone_range(&nums, &config, start, FULL_BATCH_N)?;
  assert_eq!(got, nums[start..start + FULL_BATCH_N]);

  Ok(())
}

#[test]
fn test_standalone_chunk_read_range_single_chunk() -> PcoResult<()> {
  // one page per chunk, so the per-chunk read_range API applies directly
  let n = 3 * FULL_BATCH_N;
  let nums: Vec<i64> = (0..n as i64).map(|x| x.wrapping_mul(1000003)).collect();
  let config = config_with_pages(PagingSpec::Exact(vec![n]), DeltaSpec::NoOp);
  let compressed = simple_compress(&nums, &config)?;

  let (fd, src) = FileDecompressor::new(compressed.as_slice())?;
  let crate::standalone::DecompressorItem::Chunk(mut cd) = fd.chunk_decompressor::<i64, _>(src)?
  else {
    panic!("expected a chunk");
  };

  let start = FULL_BATCH_N;
  let mut dst = vec![0_i64; FULL_BATCH_N];
  cd.read_range(start, &mut dst)?;
  assert_eq!(dst, nums[start..start + FULL_BATCH_N]);

  Ok(())
}

fn wrapped_compressed(nums: &[u32], config: &ChunkConfig) -> PcoResult<Vec<u8>> {
  let fc = WrappedFileCompressor::default();
  let mut compressed = Vec::new();
  fc.write_header(&mut compressed)?;
  let mut cc = fc.chunk_compressor(nums, config)?;
  cc.write_meta(&mut compressed)?;
  cc.write_page(0, &mut compressed)?;
  Ok(compressed)
}

#[test]
fn test_wrapped_page_skip_and_read_range() -> PcoResult<()> {
  let n = 5 * FULL_BATCH_N;
  let nums: Vec<u32> = (0..n as u32).map(|x| x.wrapping_mul(2654435761)).collect();
  let config = config_with_pages(
    PagingSpec::EqualPagesUpTo(n),
    DeltaSpec::NoOp,
  );
  let compressed = wrapped_compressed(&nums, &config)?;

  let (fd, src) = WrappedFileDecompressor::new(compressed.as_slice())?;
  let (mut cd, page_src) = fd.chunk_decompressor::<u32, _>(src)?;
  let mut pd = cd.page_decompressor(page_src, n)?;

  pd.skip(2 * FULL_BATCH_N)?;
  let mut dst = vec![0_u32; FULL_BATCH_N];
  pd.read_range(0, &mut dst)?;
  let want_start = 2 * FULL_BATCH_N;
  assert_eq!(
    dst,
    nums[want_start..want_start + FULL_BATCH_N]
  );
  pd.skip(2 * FULL_BATCH_N)?;

  assert!(pd.skip(FULL_BATCH_N).is_err());

  Ok(())
}

#[test]
fn test_wrapped_read_range_within_page() -> PcoResult<()> {
  let n = 4 * FULL_BATCH_N;
  let nums: Vec<u32> = (0..n as u32).map(|x| (x / 7).wrapping_add(13)).collect();
  let config = config_with_pages(
    PagingSpec::EqualPagesUpTo(n),
    DeltaSpec::TryConsecutive(1),
  );
  let compressed = wrapped_compressed(&nums, &config)?;

  let (fd, src) = WrappedFileDecompressor::new(compressed.as_slice())?;
  let (mut cd, page_src) = fd.chunk_decompressor::<u32, _>(src)?;
  let mut pd = cd.page_decompressor(page_src, n)?;

  let skip = 3 * FULL_BATCH_N;
  let mut dst = vec![0_u32; FULL_BATCH_N];
  let progress = pd.read_range(skip, &mut dst)?;
  assert!(progress.finished);
  assert_eq!(dst, nums[skip..]);

  Ok(())
}
