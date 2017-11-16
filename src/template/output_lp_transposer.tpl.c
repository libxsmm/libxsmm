#define TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i) \
      even_addr_lo = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      odd_addr_lo = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii+1, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      even_addr_hi = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 8, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      odd_addr_hi = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii+1, 8, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      even_pixel_lo = _mm256_loadu_si256((const union __m256i *) even_addr_lo); \
      odd_pixel_lo = _mm256_loadu_si256((const union __m256i *) odd_addr_lo); \
      even_pixel_hi = _mm256_loadu_si256((const union __m256i *) even_addr_hi); \
      odd_pixel_hi = _mm256_loadu_si256((const union __m256i *) odd_addr_hi); \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel_lo, odd_pixel_lo); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel_lo, odd_pixel_lo); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_lo,0), 0); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_hi,0), 1); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_lo,1), 0); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_hi,1), 1); \
      dst_lo = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      dst_hi = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1, ij, half_i, 8, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm256_storeu_si256((union __m256i *) dst_lo, compressed_lo_store); \
      _mm256_storeu_si256((union __m256i *) dst_hi, compressed_hi_store); \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel_hi, odd_pixel_hi); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel_hi, odd_pixel_hi); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_lo,0), 0); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_hi,0), 1); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_lo,1), 0); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_hi,1), 1); \
      dst_lo = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1+1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      dst_hi = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1+1, ij, half_i, 8, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm256_storeu_si256((union __m256i *) dst_lo, compressed_lo_store); \
      _mm256_storeu_si256((union __m256i *) dst_hi, compressed_hi_store);

#define TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i) \
      even_addr_lo = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      even_addr_hi = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 8, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      even_pixel_lo = _mm256_loadu_si256((const union __m256i *) even_addr_lo); \
      even_pixel_hi = _mm256_loadu_si256((const union __m256i *) even_addr_hi); \
      odd_pixel_lo = _mm256_xor_si256(odd_pixel_lo,odd_pixel_lo); \
      odd_pixel_hi = odd_pixel_lo; \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel_lo, odd_pixel_lo); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel_lo, odd_pixel_lo); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_lo,0), 0); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_hi,0), 1); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_lo,1), 0); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_hi,1), 1); \
      dst_lo = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      dst_hi = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1, ij, half_i, 8, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm256_storeu_si256((union __m256i *) dst_lo, compressed_lo_store); \
      _mm256_storeu_si256((union __m256i *) dst_hi, compressed_hi_store); \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel_hi, odd_pixel_hi); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel_hi, odd_pixel_hi); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_lo,0), 0); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_hi,0), 1); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_lo,1), 0); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_hi,1), 1); \
      dst_lo = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1+1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      dst_hi = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1+1, ij, half_i, 8, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm256_storeu_si256((union __m256i *) dst_lo, compressed_lo_store); \
      _mm256_storeu_si256((union __m256i *) dst_hi, compressed_hi_store);


element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
element_output_type *dst_lo, *dst_hi;
int half_i;
__m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;

for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
  for (ij = 0; ij < handle->ofhp; ++ij) {
    for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
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

#undef TRANSPOSE_W_FULL_PAIR
#undef TRANSPOSE_W_HALF_PAIR
