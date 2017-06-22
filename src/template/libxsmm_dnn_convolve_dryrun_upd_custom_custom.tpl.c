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
#if !defined(_OPENMP)
int ltid;
#endif

#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
#else
for (ltid = 0; ltid < handle->desc.threads; ltid++)
#endif
{
#if defined(_OPENMP)
  int ltid = omp_get_thread_num();
#endif
  int img, ofm1, ifm1, num_ofw_strips, num_ofh_strips, oi_, oj_, oi__, oj__,ii_, ij_, kh, kw, ofm1ifm1, ki, kj;

#if defined(LIBXSMM_WU_PER_THREAD_ALLOCATION)
  int i, j, ofm1ifm1img;
#endif

  /* number of tasks that could be run in parallel */
  const int work = handle->blocksifm*handle->blocksofm;
  /* compute chunck size */
  const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
  /* compute thr_begin and thr_end */
  const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM
  int ifm2;
  /* number of tasks that could be run in parallel */
  const int transpose_work = handle->desc.N*handle->blocksifm;
  /* compute chunck size */
  const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : (transpose_work / handle->desc.threads) + 1;
  /* compute thr_begin and thr_end */
  const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
  const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;
#endif

#ifdef LIBXSMM_WU_PER_THREAD_ALLOCATION
  /* number of tasks that could be run in parallel */
  const int img_parallel_work = handle->blocksifm*handle->blocksofm*handle->desc.N;
  /* compute chunck size */
  const int img_parallel_chunksize = (img_parallel_work % handle->desc.threads == 0) ? (img_parallel_work / handle->desc.threads) : (img_parallel_work / handle->desc.threads) + 1;
  /* compute thr_begin and thr_end */
  const int img_parallel_thr_begin = (ltid * img_parallel_chunksize < img_parallel_work) ? (ltid * img_parallel_chunksize) : img_parallel_work;
  const int img_parallel_thr_end = ((ltid + 1) * img_parallel_chunksize < img_parallel_work) ? ((ltid + 1) * img_parallel_chunksize) : img_parallel_work;
  /* number of tasks that could be run in parallel */
  const int reduce_work = handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock;
  /* compute chunck size */
  const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
  /* compute thr_begin and thr_end */
  const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
  const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;
#endif

#if defined(INPUT_PADDING)
  const int copywork = handle->desc.N*handle->blocksifm;
  const int copychunksize = (copywork % handle->desc.threads == 0) ? (copywork / handle->desc.threads) : (copywork / handle->desc.threads) + 1;
  const int copy_thr_begin = (ltid * copychunksize < copywork) ? (ltid * copychunksize) : copywork;
  const int copy_thr_end = ((ltid + 1) * copychunksize < copywork) ? ((ltid + 1) * copychunksize) : copywork;
#endif

  int total_calls;
  int n_code_segments;
  int mark_ifm_init, mark_ifm_close, mark_img_init;
  int *tmp_expanded_stream, tmp_stream_index;
  segment_t *encoded_code_segments;
  int expanded_size;
  int stretch_of_convs;
  int encoded_stream_index;
  int lookahead_index;

  /* Arrays of stream indices */
  int *compute_indices;
  int *trans_indices;
  char *kernel_variant;

  int padded_w, padded_h;
  if (handle->padding_flag == 1) {
    padded_h = handle->ifhp + 2 * handle->desc.pad_h;
    padded_w = handle->ifwp + 2 * handle->desc.pad_w;
  } else {
    padded_h = handle->ifhp;
    padded_w = handle->ifwp;
  }

  /* Dryrun for copy/padding */
  if (handle->padding_flag == 1) {
    /* Initialize in parallel scratch5 to zero */
    handle->n_entries_init_upd[ltid] = copy_thr_end - copy_thr_begin;
    init_indices = (int*) libxsmm_aligned_malloc( (copy_thr_end - copy_thr_begin + 1) * sizeof(int), 2097152);
    handle->init_upd_indices_ptrs[ltid] = init_indices;
    aux = 0;    
    for (imgifm1 = copy_thr_begin; imgifm1 < copy_thr_end; ++imgifm1) {
      img = imgifm1/handle->blocksifm;
      ifm1 = imgifm1%handle->blocksifm;
      init_indices[aux] = ifm1 * (handle->ifmblock * padded_w * padded_h) + img * (handle->ifmblock * padded_w * padded_h * handle->blocksifm);
      aux++;
    }

    if (trans_ofw_ifm == 0) {
      handle->n_entries_copy_upd[ltid] = copy_thr_end - copy_thr_begin;
      copy_indices = (int*) libxsmm_aligned_malloc( 2 * (copy_thr_end - copy_thr_begin + 1) * sizeof(int), 2097152);
      handle->copy_upd_indices_ptrs[ltid] = copy_indices;
      aux = 0;    
      for (imgifm1 = copy_thr_end-1; imgifm1 >= copy_thr_begin; imgifm1--) {
        img = imgifm1/handle->blocksifm;
        ifm1 = imgifm1%handle->blocksifm;
        copy_indices[aux] = ifm1 * (handle->ifmblock * handle->ifhp *  handle->ifwp) + img * (handle->ifmblock * handle->ifhp *  handle->ifwp * handle->blocksifm);
        copy_indices[aux+1] = handle->desc.pad_w * handle->ifmblock + handle->desc.pad_h * handle->ifmblock * padded_w + ifm1 * (handle->ifmblock * padded_w * padded_h) + img * (handle->ifmblock * padded_w * padded_h * handle->blocksifm);
        aux +=2;
      }
    }
  }

  if (handle->ifmblock == 1) { /* special case for ifmblock = 1 */

    if (fil) {
    
    
    }

  } else {
  
  
  }














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
  tmp_expanded_stream = (int*) malloc( expanded_size * sizeof(int) );
  tmp_stream_index = 0;
  if (n_code_segments) {
    encoded_code_segments = (segment_t*) libxsmm_aligned_malloc(n_code_segments * sizeof(segment_t), 2097152);
    handle->bwd_code_segments[ltid] = encoded_code_segments;
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

