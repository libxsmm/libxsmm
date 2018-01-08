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
/* Evangelos Georganas (Intel Corp.)
 ******************************************************************************/
#define IMG_LOOP_INIT 0
#define IFM_LOOP_INIT 1
#define IFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3
#define MIXED 0
#define HWKC 1

int block_j = 1;
#if !defined(_OPENMP)
int ltid;
#endif
int blockofm = 8;

if (handle->padding_flag == 1) {
  handle->block_bwd_ofm = handle->blocksofm;
} else {
  handle->block_bwd_ofm = 8;
}

if ( (handle->ifhp == 14 && handle->desc.R != 3 ) ||  handle->ifhp == 27 || (handle->ifhp == 28 && handle->desc.R == 1) || handle->ifhp == 48 || handle->ifhp == 54 || handle->ifhp == 56 || handle->ifhp == 112 ) {
  block_j = 4;
}
while ( block_j % handle->bwd_ofh_rb != 0 ) {
  block_j--;
}

block_j = handle->bwd_ofh_rb;
handle->block_bwd_oj = block_j;
int loop_order = MIXED;

/* Logic to use FWD generated convolution code ...  */
if ( handle->desc.R == 1 && handle->desc.S == 1 && handle->use_nts_bwd ==1 ) {
  block_j = 14;
  handle->block_bwd_ofm = handle->blocksofm_blocking;
  if (handle->desc.H >= 28 && handle->desc.R == 1) {
    loop_order = HWKC;
  }

  while (blockofm % handle->blocksofm_blocking != 0) {
    blockofm++;
  }
  handle->block_fwd_ofm = blockofm;

  if ((handle->ofh == 7 && handle->desc.u == 2) || (handle->ofh == 14 && handle->desc.R != 3 ) ||  handle->ofh == 27 || (handle->ofh == 28 && handle->desc.R == 1) || handle->ofh == 48 || handle->ofh == 54 || handle->ofh == 56 || handle->ofh == 112 ) {
    block_j = 4;
  }

  while ( block_j % handle->bwd_ofh_rb != 0 ) {
    block_j--;
  }
  handle->block_bwd_oj = block_j;
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
  int img, ofm1 = 0, ifm1 = 0, oj, oi, ij, ii, local_entries = 0, ojb, ifmb, ofmb;
  int cur_wt, next_wt, cur_out, next_out, padded_w = 0;
  int fmlpb = handle->fm_lp_block;
  int comp, kj = 0, ki = 0, aux;

  /* number of tasks for transpose that could be run in parallel */
  int transpose_work = handle->blocksofm * (handle->blocksifm * handle->fm_lp_block) * handle->desc.R * handle->desc.S;
  /* compute chunck size */
  int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
  int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

  /* Threading related variables */
  int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
  int threads_per_image = handle->desc.threads / handle->desc.N;
  int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
  int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
  int my_ifm_start = 0;
  int my_ifm_end = handle->blocksifm * fmlpb;
  int myIfmId;
  int nIfmBlocks;
  int total_calls;
  int n_code_segments;
  int mark_ifm_init, mark_ifm_close, mark_img_init;
  int *tmp_expanded_stream, tmp_stream_index;
  segment_t *encoded_code_segments = NULL;
  int expanded_size;
  int stretch_of_convs;
  int encoded_stream_index;
  int lookahead_index;

  /* Arrays of stream indices */
  int *compute_indices;
  int *trans_indices;
  char *kernel_variant;

  if (handle->padding_flag == 1) {
    padded_w = handle->ifwp + 2 * handle->desc.pad_w;
  }

  n_code_segments = 0;
  tmp_stream_index = 0;

  if ( imgpt <= 1 ) {
    my_img_start = LIBXSMM_MIN( ltid / threads_per_image, handle->desc.N);
    my_img_end = LIBXSMM_MIN( my_img_start + 1, handle->desc.N);
    myIfmId = ltid % threads_per_image;
    nIfmBlocks = (handle->blocksifm * fmlpb + threads_per_image - 1) / threads_per_image;
    my_ifm_start = LIBXSMM_MIN(myIfmId * nIfmBlocks, handle->blocksifm * fmlpb );
    my_ifm_end = LIBXSMM_MIN((myIfmId+1) * nIfmBlocks, handle->blocksifm * fmlpb);
  }

  mark_ifm_init = ((fmlpb != 1) || (handle->padding_flag == 1)) ? 1 : 0;
  mark_ifm_close = 1;
  mark_img_init = 1;

  /* Dryrun for transpose stream */
  handle->n_entries_trans_bwd[ltid] = transpose_thr_end - transpose_thr_begin;
  trans_indices = (int*) libxsmm_aligned_malloc( (transpose_thr_end - transpose_thr_begin + 1) * sizeof(int), 2097152);
  handle->transpose_bwd_indices_ptrs[ltid] = trans_indices;
  aux = 0;
  for (comp = transpose_thr_begin; comp < transpose_thr_end; ++comp) {
    ofm1 = comp / (handle->blocksifm * handle->desc.R * handle->desc.S);
    ifm1 = (comp % (handle->blocksifm * handle->desc.R * handle->desc.S)) / (handle->desc.R * handle->desc.S);
    kj = ((comp % (handle->blocksifm * handle->desc.R * handle->desc.S)) % (handle->desc.R * handle->desc.S)) / handle->desc.S;
    ki = ((comp % (handle->blocksifm * handle->desc.R * handle->desc.S)) % (handle->desc.R * handle->desc.S)) % handle->desc.S;
    trans_indices[aux] = ( ( ( ( ( ( ofm1 * handle->blocksifm) + ifm1) * handle->desc.R) + kj) * handle->desc.S) +  ki) * handle->ofmblock * handle->ifmblock * handle->fm_lp_block;
    aux++;
  }
  trans_indices[aux] = ( ( ( ( ( ( ofm1 * handle->blocksifm) + ifm1) * handle->desc.R) + kj) * handle->desc.S) +  ki) * handle->ofmblock * handle->ifmblock * handle->fm_lp_block;

  /* Perform a dryrun to compute the memory requirements of the stream of indices */
  for (img = my_img_start; img < my_img_end; img++) {
    if (mark_img_init == 1) {
      n_code_segments++;
    }
    for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += handle->block_bwd_ifm) {
      for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_bwd_ofm) {
        for (ojb = 0; ojb < handle->ofh; ojb += handle->block_bwd_oj) {
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_bwd_ifm, my_ifm_end); ifm1++ ) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_bwd_ofm, handle->blocksofm); ++ofm1) {
              for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_bwd_oj,handle->ofh); oj += handle->bwd_ofh_rb) {
                for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
                  local_entries += 3;

                  if (mark_ifm_init == 1) {
                    if (ofm1 == 0 && oj == 0 && oi == 0) {
                      n_code_segments++;
                    }
                  }

                  if (mark_ifm_close == 1) {
                    if (ofm1 == handle->blocksofm-1  && oj >=  handle->ofh - handle->bwd_ofh_rb && oi >=  handle->ofw - handle->bwd_ofw_rb) {
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

  handle->n_entries_bwd[ltid] = local_entries/3;

  /* Alocate auxiliary data structures for index jitting  */
  compute_indices = (int*) libxsmm_aligned_malloc( (local_entries+3) * sizeof(int), 2097152);
  handle->compute_bwd_indices_ptrs[ltid] = compute_indices;
  kernel_variant = (char*) libxsmm_aligned_malloc( (local_entries/3) * sizeof(char), 2097152);
  handle->kernel_bwd_variant_ptrs[ltid] = kernel_variant;
  handle->n_bwd_code_segments[ltid] = n_code_segments;
  expanded_size = local_entries/3 + n_code_segments;
  tmp_expanded_stream = (int*)(0 < expanded_size ? malloc(expanded_size * sizeof(int)) : 0);
  assert(0 != tmp_expanded_stream); /* TODO: should never happen */
#if !defined(NDEBUG)
  memset(tmp_expanded_stream, IMG_LOOP_INIT, expanded_size * sizeof(int));
#endif
  tmp_stream_index = 0;
  if (n_code_segments) {
    encoded_code_segments = (segment_t*) libxsmm_aligned_malloc(n_code_segments * sizeof(segment_t), 2097152);
    handle->bwd_code_segments[ltid] = encoded_code_segments;
  } else {
   encoded_code_segments = NULL;
  }
  local_entries = 0;

  /* Second run to compute actual indices */
  for (img = my_img_start; img < my_img_end; img++) {
    if (mark_img_init == 1) {
      tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_INIT;
      tmp_stream_index++;
    }
    for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += handle->block_bwd_ifm) {
      for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_bwd_ofm) {
        for (ojb = 0; ojb < handle->ofh; ojb += handle->block_bwd_oj) {
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_bwd_ifm, my_ifm_end); ifm1++ ) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_bwd_ofm, handle->blocksofm); ++ofm1) {
              for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_bwd_oj,handle->ofh); oj += handle->bwd_ofh_rb) {
                for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
                  ij = oj * handle->desc.u;
                  ii = oi * handle->desc.v;

                  if (mark_ifm_init == 1) {
                    if (ofm1 == 0 && oj == 0 && oi == 0) {
                      tmp_expanded_stream[tmp_stream_index] = IFM_LOOP_INIT;
                      tmp_stream_index++;
                    }
                  }

                  if ( handle->padding_flag == 1  ) {
                    compute_indices[local_entries] =  ij * padded_w * handle->ifmblock + ii * handle->ifmblock;
                  } else {
                    compute_indices[local_entries] =  ( ( ( ( ( (img * handle->blocksifm *  handle->fm_lp_block ) +  ifm1) *  handle->ifhp )  +  ij) * handle->ifwp)  +  ii  ) *  handle->ifmblock;
                  }
                  compute_indices[local_entries+1] = ( (ofm1 *  handle->blocksifm * handle->fm_lp_block)  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ofmblock *  handle->ifmblock *  handle->fm_lp_block;
                  compute_indices[local_entries+2] = ( ( ( ( ( (img *  handle->blocksofm) +  ofm1) *  handle->ofhp )  +  oj) * handle->ofwp)  +  oi  ) *  handle->ofmblock * handle->fm_lp_block;

                  /* Initialize kernel variant with the one that prefetches everything */
                  kernel_variant[local_entries/3] = 2;
                  local_entries += 3;

                  tmp_expanded_stream[tmp_stream_index] = CONVOLUTION_KERNEL;
                  tmp_stream_index++;

                  if (mark_ifm_close == 1) {
                    if (ofm1 == handle->blocksofm-1  && oj >=  handle->ofh - handle->bwd_ofh_rb && oi >=  handle->ofw - handle->bwd_ofw_rb) {
                      tmp_expanded_stream[tmp_stream_index] = IFM_LOOP_CLOSE;
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

  /* Process the expanded stream and encode the segments via run length encoding */
  if (n_code_segments) {
    stretch_of_convs = 0;
    encoded_stream_index = 0;
    tmp_stream_index = 0;
    lookahead_index = 1;

    while ( lookahead_index < expanded_size ) {
      while  ( tmp_expanded_stream[lookahead_index] == CONVOLUTION_KERNEL) {
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

    /* Final pass over the segments to fill-in auxiliary indices...  */
    encoded_stream_index = 0;
    for (img = my_img_start; img < my_img_end; img++) {
      if (mark_img_init == 1) {
        encoded_code_segments[encoded_stream_index].aux_index = img;
        encoded_stream_index++;
      }
      for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += handle->block_bwd_ifm) {
        for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_bwd_ofm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->block_bwd_oj) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_bwd_ifm, my_ifm_end); ifm1++ ) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_bwd_ofm, handle->blocksofm); ++ofm1) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_bwd_oj,handle->ofh); oj += handle->bwd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
                    ij = oj * handle->desc.u;
                    ii = oi * handle->desc.v;
                    if (mark_ifm_init == 1) {
                      if (ofm1 == 0 && oj == 0 && oi == 0) {
                        encoded_code_segments[encoded_stream_index].aux_index = ifm1;
                        encoded_stream_index++;
                      }
                    }

                    if (mark_ifm_close == 1) {
                      if (ofm1 == handle->blocksofm-1  && oj >=  handle->ofh - handle->bwd_ofh_rb && oi >=  handle->ofw - handle->bwd_ofw_rb) {
                        encoded_code_segments[encoded_stream_index].aux_index = ifm1;
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

  free(tmp_expanded_stream);

  /* At the end of stream do not prefetch garbage */
  compute_indices[local_entries] = 0;
  compute_indices[local_entries+1] = 0;
  compute_indices[local_entries+2] = 0;
  total_calls = local_entries/3;

  /* Adjust the kernel variant  */
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

