/******************************************************************************
 ** Copyright (c) 2016-2017, Intel Corporation                                **
 ** All rights reserved.                                                      **
 **                                                                           **
 ** Redistribution and use in source and binary forms, with or without        **
 ** modification, are permitted provided that the following conditions        **
 ** are met:                                                                  **
 ** 1. Redistributions of source code must retain the above copyright         **
 **    notice, this list of conditions and the following disclaimer.          **
 ** 2. Redistributions in binary form must reproduce the above copyright      **
 **    notice, this list of conditions and the following disclaimer in the    **
 **    documentation and/or other materials provided with the distribution.   **
 ** 3. Neither the name of the copyright holder nor the names of its          **
 **    contributors may be used to endorse or promote products derived        **
 **    from this software without specific prior written permission.          **
 **                                                                           **
 ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
 ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
 ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
 ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
 ** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
 ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
 ** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
 ** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
 ** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
 ** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
 ** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
 ******************************************************************************/
/* Evangelos Georganas, John Pennycook, Jason Sewall (Intel Corp.)
******************************************************************************/
#define WEIGHT_INIT 0
#define UPDATE_KERNEL 1
#define WEIGHT_COPY 2
#define TRANSPOSE_EXEC 3
#define LIBXSMM_UPD_STREAMS_TRANSPOSE_IFMB_SHIFT 12

#if !defined(_OPENMP)
int ltid;
#endif
int block_j = 14;

handle->block_upd_ofm = 8;
handle->block_upd_ifm = 8;

if ( (handle->ofh == 14 && handle->desc.R != 3 ) ||  handle->ofh == 27 || (handle->ofh == 28 && handle->desc.R == 1) || handle->ofh == 48 || handle->ofh == 54 || handle->ofh == 56 || handle->ofh == 112 ) {
  block_j = 4;
}
while ( block_j % handle->upd_ofh_rb != 0 ) {
  block_j--;
}

if (block_j < handle->upd_ofh_rb ) {
  block_j = handle->upd_ofh_rb ;
}

block_j = handle->ofh ;

if ( handle->ofh == 56 ) {
  /* Pixel block is 196 Kbytes */
 handle->block_upd_ofm = handle->blocksofm;
 handle->block_upd_ifm = 1;

}

if ( handle->ofh == 28 ) {
  /* Pixel block is 49 Kbytes */
 handle->block_upd_ofm = 3;
 handle->block_upd_ifm = 3;
}

if ( handle->ofh == 14 || handle->ofh == 28 || handle->ofh == 56 ) {
  /* Pixel block is 12.25 Kbytes */
 handle->block_upd_ofm = 8;
 handle->block_upd_ifm = 16;
}

if ( handle->ofh == 7 ) {
  /* Pixel block is 3.06 Kbytes */
 handle->block_upd_ofm = 8;
 handle->block_upd_ifm = 16;
}


if ( handle->ofh == 28 || handle->ofh == 56 ) {
  /* Pixel block is 12.25 Kbytes */
 handle->block_upd_ofm = 32;
 handle->block_upd_ifm = 16;
}

#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
#else
for (ltid = 0; ltid < handle->desc.threads; ltid++)
#endif
{
#if defined(_OPENMP)
  int ltid = omp_get_thread_num();
#endif
  int img, ifmb, ofmb, ofm1, ifm1, num_ofw_strips, oi_, oj_, oi__, ii_, ij_, kh, kw, KW, ki, kj, local_entries, stride_w, stride_h ;
  int ojb;

  /* Here we assume that N % Threads == 0 */
  int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
  int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
  int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
  int n_code_segments;
  int mark_weight_init, mark_weight_copy;
  int *tmp_expanded_stream, tmp_stream_index;
  segment_t *encoded_code_segments;
  int expanded_size;
  int stretch_of_convs;
  int encoded_stream_index;
  int lookahead_index;

  /* Arrays of stream indices */
  int *compute_indices;
  char *kernel_variant;

  int padded_w, padded_h;
  if (handle->padding_flag == 1) {
    padded_h = handle->ifhp + 2 * handle->desc.pad_h;
    padded_w = handle->ifwp + 2 * handle->desc.pad_w + handle->qfma_input_pad;
  } else {
    padded_h = handle->ifhp;
    padded_w = handle->ifwp + handle->qfma_input_pad;
  }

  stride_w = handle->desc.v;
  stride_h = handle->desc.u;
  kh = handle->desc.R;
  kw = handle->desc.S;
  num_ofw_strips = 1; /* Internally always fully unroll W */
  local_entries = 0;
  if ( handle->ifmblock != 1  ) {
    KW = kw;
  } else {
    KW = 1;
  }

  n_code_segments = 0;
  tmp_stream_index = 0;

  /* Skip WEIGHT_INIT and WEIGHT_COPY when kernel uses NT stores */
  mark_weight_init = ( handle->ofh == 28 || handle->ofh == 56 ) ? 1 : 0;
  mark_weight_copy = ( handle->ofh == 28 || handle->ofh == 56 ) ? 1 : 0;

  /* Perform a dryrun to compute the memory requirements of the stream of indices */
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {

        for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {

              for (oj_ = ojb; oj_ < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj_ += handle->upd_ofh_rb) {
                for (oi__=0; oi__<num_ofw_strips; ++oi__) {

                  for (kj=0; kj < kh; ++kj) {
                    for (ki=0; ki < KW; ++ki) {
                      oi_=oi__*handle->upd_ofw_rb;
                      ii_ = oi_*stride_w;
                      ij_ = oj_*stride_h;
                      local_entries += 3;

                      /* For transpose: Find first (img,ifmb) in this stream. Occurs when ofmb == 0. */
                      if ( (ofmb == 0) && (ojb == 0) && (ofm1 == ofmb) && (ifm1 == ifmb) && (oj_ == ojb) && (oi__ == 0) && (kj == 0) && (ki == 0)) {
                        n_code_segments++;
                      }

                      if (mark_weight_init == 1) {
                        if ( (ki == 0) && (kj == 0) && (oi__ == 0) && (oj_ == ojb) && (ojb == 0) ) {
                          n_code_segments++;
                        }
                      }

                      if (mark_weight_copy == 1) {
                        if ( (ki+1 >= KW) && (kj+1 >= kh) && (oi__+1 >= num_ofw_strips) && (oj_+handle->upd_ofh_rb >= LIBXSMM_MIN(ojb+block_j,handle->ofh)) && (ojb+block_j >= handle->ofh) ) {
                          n_code_segments++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }


  /* Alocate auxiliary data structures for index jitting  */
  handle->n_entries_upd[ltid] = local_entries/3;
  compute_indices = (int*) libxsmm_aligned_malloc( (local_entries+3) * sizeof(int), 64);
  handle->compute_upd_indices_ptrs[ltid] = compute_indices;
  kernel_variant = (char*) libxsmm_aligned_malloc( (local_entries/3) * sizeof(char), 64);
  handle->kernel_upd_variant_ptrs[ltid] = kernel_variant;
  handle->n_upd_code_segments[ltid] = n_code_segments;
  expanded_size = local_entries/3 + n_code_segments;
  tmp_expanded_stream = (int*) malloc( expanded_size * sizeof(int) );
  tmp_stream_index = 0;
  if (n_code_segments) {
    encoded_code_segments = (segment_t*) libxsmm_aligned_malloc(n_code_segments * sizeof(segment_t), 2097152);
    handle->upd_code_segments[ltid] = encoded_code_segments;
  }

  /* Second run to compute actual indices */
  local_entries = 0;

  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {

        for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {

              for (oj_ = ojb; oj_ < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj_ += handle->upd_ofh_rb) {
                for (oi__=0; oi__<num_ofw_strips; ++oi__) {

                  for (kj=0; kj < kh; ++kj) {
                    for (ki=0; ki < KW; ++ki) {

                      oi_=oi__*handle->upd_ofw_rb;
                      ii_ = oi_*stride_w;
                      ij_ = oj_*stride_h;

                      /* For transpose: Find first (img,ifmb) in this stream. Occurs when ofmb == 0. */
                      if ( (ofmb == 0) && (ojb == 0) && (ofm1 == ofmb) && (ifm1 == ifmb) && (oj_ == ojb) && (oi__ == 0) && (kj == 0) && (ki == 0)) {
                        tmp_expanded_stream[tmp_stream_index] = TRANSPOSE_EXEC;
                        tmp_stream_index++;
                      }

                      if (mark_weight_init == 1) {
                        if ( (ki == 0) && (kj == 0) && (oi__ == 0) && (oj_ == ojb) && (ojb == 0) ) {
                          tmp_expanded_stream[tmp_stream_index] = WEIGHT_INIT;
                          tmp_stream_index++;
                        }
                      }

                      if (handle->trans_ofw_ifm == 1 ) {
                        compute_indices[local_entries] =  ( ( ( ( ( (img *  handle->blocksifm) +  ifm1) * padded_h )  +  (ij_+kj)) * handle->ifmblock) ) * padded_w  + (ii_ + ki) ;
                      } else {
                        compute_indices[local_entries] =  ( ( ( ( ( (img *  handle->blocksifm) +  ifm1) * padded_h )  +  (ij_+kj)) * padded_w)  + (ii_ + ki) ) *  handle->ifmblock;
                      }

                      /* use different weights format if we can skip init and copy */
                      if (mark_weight_init == 1 && mark_weight_copy == 1) {
                          compute_indices[local_entries+1] = ( ( (ofm1-ofmb) * handle->block_upd_ifm ) + (ifm1-ifmb) ) * handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock + kj * handle->desc.S *  handle->ifmblock *  handle->ofmblock + ki * handle->ifmblock *  handle->ofmblock;
                      } else {
                          compute_indices[local_entries+1] = ( ( (ofm1 *  handle->blocksifm ) + ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock + kj * handle->desc.S *  handle->ifmblock *  handle->ofmblock + ki * handle->ifmblock *  handle->ofmblock ) * handle->desc.threads;
                      }

                      compute_indices[local_entries+2] = ( ( ( ( ( (img *  handle->blocksofm) +  ofm1) *  handle->ofhp )  +  oj_ ) * handle->ofwp)  +  oi_ ) *  handle->ofmblock;
                      local_entries += 3;

                      tmp_expanded_stream[tmp_stream_index] = UPDATE_KERNEL;
                      tmp_stream_index++;

                      if (mark_weight_copy == 1) {
                        if ( (ki+1 >= KW) && (kj+1 >= kh) && (oi__+1 >= num_ofw_strips) && (oj_+handle->upd_ofh_rb >= LIBXSMM_MIN(ojb+block_j,handle->ofh)) && (ojb+block_j >= handle->ofh) ) {
                          tmp_expanded_stream[tmp_stream_index] = WEIGHT_COPY;
                          tmp_stream_index++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /* Process the expanded stream and encode the segments via run length encoding */
  if (n_code_segments) {
    stretch_of_convs = 0;
    encoded_stream_index = 0;
    tmp_stream_index = 0;
    lookahead_index = 1;

    while ( lookahead_index < expanded_size ) {
      while ( tmp_expanded_stream[lookahead_index] == UPDATE_KERNEL) {
        stretch_of_convs++;
        lookahead_index++;
        if ( lookahead_index >= expanded_size ) break;
      }
      encoded_code_segments[encoded_stream_index].segment_type = tmp_expanded_stream[tmp_stream_index];
      encoded_code_segments[encoded_stream_index].n_convs = stretch_of_convs;
      encoded_stream_index++;
      stretch_of_convs = 0;
      tmp_stream_index = lookahead_index;
      lookahead_index++;
    }

    /* Check if we have not written last segment entry -- in this case the stream ends with an action point */
    if ( encoded_stream_index < n_code_segments ) {
      encoded_code_segments[encoded_stream_index].segment_type = tmp_expanded_stream[tmp_stream_index];
      encoded_code_segments[encoded_stream_index].n_convs = stretch_of_convs;
    }

    /* Final pass over the segments to fill-in auxiliary indices... */
    encoded_stream_index = 0;
    for (img = my_img_start; img < my_img_end; img++) {
      for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
        for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {

          for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {

                for (oj_ = ojb; oj_ < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj_ += handle->upd_ofh_rb) {
                  for (oi__=0; oi__<num_ofw_strips; ++oi__) {

                    for (kj=0; kj < kh; ++kj) {
                      for (ki=0; ki < KW; ++ki) {
                        oi_=oi__*handle->upd_ofw_rb;
                        ii_ = oi_*stride_w;
                        ij_ = oj_*stride_h;

                        /* For transpose: Find first (img,ifmb) in this stream. Occurs when ofmb == 0. */
                        if ( (ofmb == 0) && (ojb == 0) && (ofm1 == ofmb) && (ifm1 == ifmb) && (oj_ == ojb) && (oi__ == 0) && (kj == 0) && (ki == 0)) {
                          assert(img < (1 << LIBXSMM_UPD_STREAMS_TRANSPOSE_IFMB_SHIFT));
                          assert(ifmb < (1 << (31-LIBXSMM_UPD_STREAMS_TRANSPOSE_IFMB_SHIFT)));
                          encoded_code_segments[encoded_stream_index].aux_index = img + (ifmb << LIBXSMM_UPD_STREAMS_TRANSPOSE_IFMB_SHIFT);
                          encoded_stream_index++;
                        }

                        if (mark_weight_init == 1) {
                          if ( (ki == 0) && (kj == 0) && (oi__ == 0) && (oj_ == ojb) && (ojb == 0) ) {
                            encoded_code_segments[encoded_stream_index].aux_index = ( ( (ofm1-ofmb) * handle->block_upd_ifm ) + (ifm1-ifmb) ) * handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock;
                            encoded_stream_index++;
                          }
                        }

                        if (mark_weight_copy == 1) {
                          if ( (ki+1 >= KW) && (kj+1 >= kh) && (oi__+1 >= num_ofw_strips) && (oj_+handle->upd_ofh_rb >= LIBXSMM_MIN(ojb+block_j,handle->ofh)) && (ojb+block_j >= handle->ofh) ) {
                            encoded_code_segments[encoded_stream_index].aux_index = (ofm1 * handle->blocksifm + ifm1) * handle->desc.R * handle->desc.S * handle->ifmblock;
                            encoded_stream_index++;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  free(tmp_expanded_stream);

  /* At the end of stream do not prefetch garbage */
  compute_indices[local_entries] = 0;
  compute_indices[local_entries+1] = 0;
  compute_indices[local_entries+2] = 0;

}

