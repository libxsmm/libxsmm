/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#define IMG_LOOP_INIT 0
#define OFM_LOOP_INIT 1
#define OFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3
#define IFM_LOOP_CLOSE_S 4

#define MIXED 0
#define KHWC 1
#define HWKC 2
#define CHWK 3
#define HWCK 4

int blockifm = 16;
int w_factor = 8;/*atoi(getenv("W"));*/
#if !defined(_OPENMP)
int ltid;
#endif
#if 0
int block_j;
# if 0
block_j = handle->ofh;
if ( handle->ofh == 14 || handle->ofh == 48 || handle->ofh == 54 || handle->ofh == 56 || handle->ofh == 112 ) {
  block_j = 4;
}

while ( block_j % handle->fwd_ofh_rb != 0 ) {
  block_j--;
}
# else
block_j = 4;
# endif
#endif
while (blockifm % handle->blocksifm_blocking != 0) {
  blockifm++;
}
handle->block_fwd_ofm = 8;
handle->block_fwd_ifm = blockifm;

/*handle->block_fwd_oj = block_j;*/
handle->block_fwd_oj = 4;/*atoi(getenv("H"));*/
/*handle->block_fwd_ofm = 8;*/
handle->block_fwd_ofm = 16;/*atoi(getenv("K"));*/
handle->block_fwd_ifm = blockifm;/*atoi(getenv("C"));*/

#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
#else
for (ltid = 0; ltid < handle->desc.threads; ltid++)
#endif
{
#if defined(_OPENMP)
  int ltid = omp_get_thread_num();
#endif
  int img, ofm1, ifm1, oj, oi, oib, ii_use, ij_use, oi_use, oj_use, local_entries = 0, ojb, ifmb, ofmb, my_img_start, my_img_end, my_ofm_start, my_ofm_end, my_h_start, my_h_end, my_w_start, my_w_end, block_w;
  int n_code_segments;
  int mark_ofm_init, mark_ofm_close;
  int *tmp_expanded_stream, tmp_stream_index;
  segment_t *encoded_code_segments = NULL;
  int expanded_size;
  int stretch_of_convs;
  int encoded_stream_index;
  int lookahead_index;
  int padded_h=0, padded_w=0;
  /* Arrays of stream indices */
  int *compute_indices;
  char *kernel_variant;
  /* calculate group sizes, we handle splits as additional images */
  int l_l1, l_l3, l_l1_gs, l_l2_ts, l_tidgroup;

  if (handle->desc.N == 1 && handle->fwd_ofh_rb == 1) {
    const int rows_per_thread = (handle->ofh+handle->desc.threads-1) / handle->desc.threads;
    w_factor = 8;
    block_w = w_factor*handle->fwd_ofw_rb;
    my_img_start = 0;
    my_img_end = handle->desc.N;
    my_w_start = 0;
    my_w_end = handle->ofw;
    my_ofm_start = 0;
    my_ofm_end = handle->blocksofm;
    my_h_start = LIBXSMM_MIN(ltid * rows_per_thread, handle->ofh);
    my_h_end =  LIBXSMM_MIN((ltid+1) * rows_per_thread, handle->ofh);
  } else {
    /* calculate group sizes, we handle splits as additional images */
    l_l1 = handle->desc.N * (handle->blocksofm);
    l_l3 = handle->ofh / handle->fwd_ofh_rb;
    /* number of threads need in the ofh loop (as we have l_l1 global parallel tasks) */
    l_l1_gs = handle->desc.threads / l_l1;
    /* number of elemens of ofh loop per thread */
    l_l2_ts = (l_l3 % l_l1_gs == 0) ? ((l_l3 / l_l1_gs)*handle->fwd_ofh_rb) : (((l_l3 / l_l1_gs) + 1)*handle->fwd_ofh_rb);
    /* get group id */
    l_tidgroup = ltid / l_l1_gs;
    /* compute img and ofm1 based on group */
    my_img_start = l_tidgroup / (handle->blocksofm);
    my_img_end = LIBXSMM_MIN(my_img_start + 1, handle->desc.N);
    my_h_start =  l_l2_ts * (ltid % l_l1_gs);
    my_h_end = ((my_h_start + l_l2_ts) <= handle->ofh) ? (my_h_start + l_l2_ts) : handle->ofh;
    my_w_start = 0;
    my_w_end = handle->ofw;
    my_ofm_start = l_tidgroup % (handle->blocksofm);
    my_ofm_end = my_ofm_start+1;
    block_w = handle->ofw;

  }

  if (handle->padding_flag == 1) {
    padded_h = handle->ifhp + 2 * handle->desc.pad_h;
    padded_w = handle->ifwp + 2 * handle->desc.pad_w;
  }

  mark_ofm_init = ((((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd == 0) ) || ( (handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) ) ? 1 : 0;
  mark_ofm_close = (((((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0)) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 0) ) ||
      ((((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0)) && (handle->use_fwd_for_bwd == 1) && (handle->use_nts_bwd == 0) ) ) ? 1 : 0;
  n_code_segments = 0;
  tmp_stream_index = 0;

  /* Perform a dryrun to compute the memory requirements of the stream of indices */
  for (img = my_img_start; img < my_img_end; img++) {
    if (handle->padding_flag == 1) {
      n_code_segments++;
    }
    for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
        for (ojb = my_h_start; ojb < my_h_end; ojb += handle->block_fwd_oj) {
          for (oib = my_w_start; oib < my_w_end; oib += block_w) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ifm1 += handle->blocksifm_blocking) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,my_h_end); oj += handle->fwd_ofh_rb) {
                  for (oi = oib; oi < LIBXSMM_MIN(oib+block_w,my_w_end); oi += handle->fwd_ofw_rb) {
                    local_entries += 3;

                    if (mark_ofm_init == 1) {
                      if (ifm1 == 0 /*&& oj == my_h_start && oi == my_w_start*/) {
                        n_code_segments++;
                      }
                    }

                    if (mark_ofm_close == 1) {
                      if (ifm1 == handle->blocksifm-1  && oj == my_h_end - handle->fwd_ofh_rb && oi == my_w_end - handle->fwd_ofw_rb) {
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

  handle->n_entries_fwd[ltid] = local_entries/3;

  /* Allocate auxiliary data structures for index jitting  */
  compute_indices = (int*)libxsmm_aligned_malloc(((size_t)local_entries+3) * sizeof(int), 64);
  handle->compute_fwd_indices_ptrs[ltid] = compute_indices;
  kernel_variant = (char*)(3 <= local_entries ? libxsmm_aligned_malloc((local_entries / 3) * sizeof(char), 64) : NULL);
  handle->kernel_fwd_variant_ptrs[ltid] = kernel_variant;
  handle->n_fwd_code_segments[ltid] = n_code_segments;
  expanded_size = local_entries/3 + n_code_segments;
  tmp_expanded_stream = (int*)(0 < expanded_size ? malloc(expanded_size * sizeof(int)) : 0);
  tmp_stream_index = 0;
  if (n_code_segments) {
    encoded_code_segments = (segment_t*) libxsmm_aligned_malloc(n_code_segments * sizeof(segment_t), 64);
    handle->fwd_code_segments[ltid] = encoded_code_segments;
    handle->ofh_fwd_start[ltid] = my_h_start;
    handle->ofh_fwd_end[ltid] = my_h_end;
  }
  local_entries = 0;

  /* Second run to compute actual indices */
  for (img = my_img_start; img < my_img_end; img++) {
    if (0 != tmp_expanded_stream && handle->padding_flag == 1) {
      tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_INIT;
      tmp_stream_index++;
    }
    for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
        for (ojb = my_h_start; ojb < my_h_end; ojb += handle->block_fwd_oj) {
          for (oib = my_w_start; oib < my_w_end; oib += block_w) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ifm1 += handle->blocksifm_blocking) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,my_h_end); oj += handle->fwd_ofh_rb) {
                  for (oi = oib; oi < LIBXSMM_MIN(oib+block_w,my_w_end); oi += handle->fwd_ofw_rb) {

                    if ( handle->use_fwd_for_bwd == 0 ) {
                      ij_use = oj * handle->desc.u;
                      ii_use = oi * handle->desc.v;
                      oi_use = oi;
                      oj_use = oj;
                    } else {
                      ij_use = oj;
                      ii_use = oi;
                      oi_use = oi * handle->desc.u;
                      oj_use = oj * handle->desc.v;
                    }

                    if (0 != tmp_expanded_stream && mark_ofm_init == 1 && ifm1 == 0 /*&& oj == my_h_start && oi == my_w_start*/) {
                      tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_INIT;
                      tmp_stream_index++;
                    }

                    if (handle->padding_flag == 1) {
                      compute_indices[local_entries] =  ( ( ( ifm1 *  padded_h  +  ij_use) * padded_w)  +  ii_use) *  handle->ifmblock * handle->fm_lp_block;
                    } else {
                      compute_indices[local_entries] =  ( ( ( ( ( (img *  handle->blocksifm) +  ifm1) *  handle->ifhp )  +  ij_use) * handle->ifwp)  +  ii_use  ) *  handle->ifmblock * handle->fm_lp_block;
                    }
                    compute_indices[local_entries+1] = ( (ofm1 *  handle->blocksifm )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock *  handle->fm_lp_block;
                    compute_indices[local_entries+2] = ( ( ( ( ( (img *  handle->blocksofm) +  ofm1) *  handle->ofhp )  +  oj_use) * handle->ofwp)  +  oi_use  ) *  handle->ofmblock;

                    /* Initialize kernel variant with the one that prefetches everything */
                    LIBXSMM_ASSERT(NULL != kernel_variant); /* TODO: proper error handling (malloc returned NULL eventually) */
                    kernel_variant[local_entries/3] = 2;
                    local_entries += 3;

                    if (0 != tmp_expanded_stream) {
                      tmp_expanded_stream[tmp_stream_index] = CONVOLUTION_KERNEL;
                      tmp_stream_index++;

                      if (mark_ofm_close == 1 && ifm1 == handle->blocksifm-1 && oj == (my_h_end - handle->fwd_ofh_rb) && oi == (my_w_end - handle->fwd_ofw_rb)) {
                        tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_CLOSE;
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

  /* Process the expanded stream and encode the segments via run length encoding */
  if (n_code_segments) {
    stretch_of_convs = 0;
    encoded_stream_index = 0;
    tmp_stream_index = 0;
    lookahead_index = 1;

    if (0 != tmp_expanded_stream) {
      while ( lookahead_index < expanded_size ) {
        while ( tmp_expanded_stream[lookahead_index] == CONVOLUTION_KERNEL) {
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
    }

    /* Final pass over the segments to fill-in auxiliary indices...  */
    encoded_stream_index = 0;
    for (img = my_img_start; img < my_img_end; img++) {
      if (handle->padding_flag == 1) {
        encoded_code_segments[encoded_stream_index].aux_index = img;
        encoded_stream_index++;
      }
      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
        for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
          for (ojb = my_h_start; ojb < my_h_end; ojb += handle->block_fwd_oj) {
            for (oib = my_w_start; oib < my_w_end; oib += block_w) {
              for ( ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ifm1 += handle->blocksifm_blocking) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,my_h_end); oj += handle->fwd_ofh_rb) {
                    for (oi = oib; oi < LIBXSMM_MIN(oib+block_w,my_w_end); oi += handle->fwd_ofw_rb) {
                      if ( handle->use_fwd_for_bwd == 0 ) {
                        ij_use = oj * handle->desc.u;
                        ii_use = oi * handle->desc.v;
                        oi_use = oi;
                        oj_use = oj;
                      } else {
                        ij_use = oj;
                        ii_use = oi;
                        oi_use = oi * handle->desc.u;
                        oj_use = oj * handle->desc.v;
                      }

                      if (mark_ofm_init == 1) {
                        if (ifm1 == 0 /*&& oj == my_h_start && oi == my_w_start*/) {
                          encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                          encoded_stream_index++;
                        }
                      }

                      if (mark_ofm_close == 1) {
                        if (ifm1 == handle->blocksifm-1  && oj == my_h_end - handle->fwd_ofh_rb && oi == my_w_end - handle->fwd_ofw_rb) {
                          encoded_code_segments[encoded_stream_index].aux_index = ofm1;
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

  free(tmp_expanded_stream);

  /* At the end of stream do not prefetch garbage */
  compute_indices[local_entries] = 0;
  compute_indices[local_entries+1] = 0;
  compute_indices[local_entries+2] = 0;
#if 0
  { /* Adjust the kernel variant  */
    const int total_calls = local_entries/3;
    int ii, cur_wt, cur_out, next_wt, next_out;
    for (ii = 0; ii < total_calls-1; ii++) {
      cur_wt = compute_indices[ii*3+1];
      next_wt = compute_indices[(ii+1)*3+1];
      cur_out = compute_indices[ii*3+2];
      next_out = compute_indices[(ii+1)*3+2];
      if ( cur_wt == next_wt ) {
        kernel_variant[ii] = 1;
      } else if ( cur_out == next_out ) {
        kernel_variant[ii] = 3;
      }
    }
  }
#endif
}

#undef IMG_LOOP_INIT
#undef OFM_LOOP_INIT
#undef OFM_LOOP_CLOSE
#undef CONVOLUTION_KERNEL
#undef IFM_LOOP_CLOSE_S
#undef MIXED
#undef KHWC
#undef HWKC
#undef CHWK
#undef HWCK

