#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
# define TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i) \
      pair_addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block); \
      pair_pixels = _mm512_loadu_si512(pair_addr); \
      even_pixel = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(pair_pixels, 0); \
      odd_pixel = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(pair_pixels, 1); \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel, odd_pixel); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel, odd_pixel); \
      part0 = _mm256_extractf128_si256(compressed_lo,0); \
      part2 = _mm256_extractf128_si256(compressed_lo,1); \
      part1 = _mm256_extractf128_si256(compressed_hi,0); \
      part3 =  _mm256_extractf128_si256(compressed_hi,1); \
      compact = _mm512_inserti32x4(LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), part0, 0); \
      compact = _mm512_inserti32x4(compact, part1, 1); \
      compact = _mm512_inserti32x4(compact, part2, 2); \
      compact = _mm512_inserti32x4(compact, part3, 3); \
      pair_addr = &LIBXSMM_VLA_ACCESS(6, tr_output, img, ofm1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm512_storeu_si512 (pair_addr, compact)

# define TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i) \
      pair_addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block); \
      even_pixel = _mm256_loadu_si256((const __m256i *) pair_addr); \
      odd_pixel = _mm256_xor_si256(odd_pixel, odd_pixel); \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel, odd_pixel); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel, odd_pixel); \
      part0 = _mm256_extractf128_si256(compressed_lo,0); \
      part2 = _mm256_extractf128_si256(compressed_lo,1); \
      part1 = _mm256_extractf128_si256(compressed_hi,0); \
      part3 =  _mm256_extractf128_si256(compressed_hi,1); \
      compact = _mm512_inserti32x4(LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), part0, 0); \
      compact = _mm512_inserti32x4(compact, part1, 1); \
      compact = _mm512_inserti32x4(compact, part2, 2); \
      compact = _mm512_inserti32x4(compact, part3, 3); \
      pair_addr = &LIBXSMM_VLA_ACCESS(6, tr_output, img, ofm1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm512_storeu_si512(pair_addr, compact)

{ /* scope for local variable declarations */
  element_output_type *pair_addr;
  int half_i;
  __m256i compressed_hi, compressed_lo, even_pixel, odd_pixel;
  __m128i part0, part1, part2, part3;
  __m512i compact, pair_pixels;

  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }

  if (handle->output_lp_padding != 0) {
    /* Zero out the "output padding pixel" */
    for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
      for (ij = 0; ij < handle->ofhp; ++ij) {
        ii = handle->ofwp-1;
        half_i = ii/2;
        TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
}
# undef TRANSPOSE_W_FULL_PAIR
# undef TRANSPOSE_W_HALF_PAIR
#else /* won't happen as this code only runs on AVX512 platforms */
  LIBXSMM_ASSERT(0);
#endif

