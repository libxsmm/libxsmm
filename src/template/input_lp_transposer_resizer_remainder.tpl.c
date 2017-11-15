int w_chunks = handle->ifwp_resized/16;
int w_remainder = handle->ifwp_resized%16;
int w_i, w;
int c_i;
int u = handle->desc.u;
element_input_type *base_addr;
const __m512i vgindex = _mm512_set_epi32(u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0);
const int gather_offsets[16] = {u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0};
const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
unsigned int int_mask = 0xffffffff;
for (c_i=0;c_i<16;c_i++) {
  if (gather_offsets[16-c_i-1] >= (w_remainder*64)*u) {
    int_mask = int_mask & ~(1 << c_i);
  }
}
const __mmask16 gmask = int_mask;
int mask_remainder = (w_remainder+1)/2;
unsigned int mask[8];
for (c_i=0; c_i<mask_remainder; c_i++) {
  mask[c_i] = (1<<31);
}
for (c_i=mask_remainder; c_i<8; c_i++) {
  mask[c_i] = 0;
}
__m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
int dst_i, dst_j, src_i, src_j;
for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
  for (dst_j=0; dst_j < handle->ifhp_resized; dst_j++) {
    src_j = dst_j * handle->desc.v;
    /* Handle full chunks  */
    for (w = 0; w < w_chunks; w++) {
      for (ifm2 = 0; ifm2 < 8; ++ifm2) {
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, src_j, w*16*u, ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
        __m512i gather_reg = _mm512_i32gather_epi32(vgindex, base_addr, 1);
        __m256i lo_reg= _mm512_extracti64x4_epi64(gather_reg,0);
        __m256i hi_reg= _mm512_extracti64x4_epi64(gather_reg,1);
        __m256i compressed_low  = _mm256_unpacklo_epi16(lo_reg, hi_reg);
        compressed_low =  _mm256_permutevar8x32_epi32(compressed_low, shuffler);
        __m256i compressed_high  = _mm256_unpackhi_epi16(lo_reg, hi_reg);
        compressed_high =  _mm256_permutevar8x32_epi32(compressed_high, shuffler);
        __m256i compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0);
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1);
        __m256i compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0);
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1);
        _mm256_storeu_si256((union __m256i *) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1*2, dst_j, ifm2*2, w*16, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), compressed_low_store);
        _mm256_storeu_si256((union __m256i *) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1*2, dst_j, ifm2*2+1, w*16, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), compressed_high_store);
      }
      for (ifm2 = 8; ifm2 < handle->ifmblock; ++ifm2) {
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, src_j, w*16*u, ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
        __m512i gather_reg = _mm512_i32gather_epi32(vgindex, base_addr, 1);
        __m256i lo_reg= _mm512_extracti64x4_epi64(gather_reg,0);
        __m256i hi_reg= _mm512_extracti64x4_epi64(gather_reg,1);
        __m256i compressed_low  = _mm256_unpacklo_epi16(lo_reg, hi_reg);
        compressed_low =  _mm256_permutevar8x32_epi32(compressed_low, shuffler);
        __m256i compressed_high  = _mm256_unpackhi_epi16(lo_reg, hi_reg);
        compressed_high =  _mm256_permutevar8x32_epi32(compressed_high, shuffler);
        __m256i compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0);
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1);
        __m256i compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0);
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1);
        _mm256_storeu_si256((union __m256i *) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1*2+1, dst_j, 2*ifm2-16, w*16, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), compressed_low_store);
        _mm256_storeu_si256((union __m256i *) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1*2+1, dst_j, 2*ifm2-15, w*16, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), compressed_high_store);
      }        
    }
    /* Handle remainder */
    for (ifm2 = 0; ifm2 < 8; ++ifm2) {
      base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, src_j, w_chunks*16*u, ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
      __m512i gather_reg = _mm512_mask_i32gather_epi32(gather_reg, gmask, vgindex, base_addr, 1);
      __m256i lo_reg= _mm512_extracti64x4_epi64(gather_reg,0);
      __m256i hi_reg= _mm512_extracti64x4_epi64(gather_reg,1);
      __m256i compressed_low   = _mm256_unpacklo_epi16(lo_reg, hi_reg);
      compressed_low =  _mm256_permutevar8x32_epi32(compressed_low, shuffler);
      __m256i compressed_high  = _mm256_unpackhi_epi16(lo_reg, hi_reg);
      compressed_high =  _mm256_permutevar8x32_epi32(compressed_high, shuffler);
      __m256i compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0);
      compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1);
      __m256i compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0);
      compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1);
      _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1*2, dst_j, ifm2*2, w_chunks*16, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), mask_reg, compressed_low_store);
      _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1*2, dst_j, ifm2*2+1, w_chunks*16, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), mask_reg, compressed_high_store);  
    }
    for (ifm2 = 8; ifm2 < handle->ifmblock; ++ifm2) {
      base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, src_j, w_chunks*16*u, ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
      __m512i gather_reg = _mm512_mask_i32gather_epi32(gather_reg, gmask, vgindex, base_addr, 1);
      __m256i lo_reg= _mm512_extracti64x4_epi64(gather_reg,0);
      __m256i hi_reg= _mm512_extracti64x4_epi64(gather_reg,1);
      __m256i compressed_low   = _mm256_unpacklo_epi16(lo_reg, hi_reg);
      compressed_low =  _mm256_permutevar8x32_epi32(compressed_low, shuffler);
      __m256i compressed_high  = _mm256_unpackhi_epi16(lo_reg, hi_reg);
      compressed_high =  _mm256_permutevar8x32_epi32(compressed_high, shuffler);
      __m256i compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0);
      compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1);
      __m256i compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0);
      compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1);
      _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1*2+1, dst_j, 2*ifm2-16, w_chunks*16, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), mask_reg, compressed_low_store);
      _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1*2+1, dst_j, 2*ifm2-15, w_chunks*16, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), mask_reg, compressed_high_store);  
    }          
  }
}

