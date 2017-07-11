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
  int img, ofm1, ifm1, num_ofw_strips, num_ofh_strips, oi_, oj_, oi__, oj__,ii_, ij_, kh, kw, ofm1ifm1, ki, kj, imgifm1, local_entries, stride_w, stride_h ;
  int i, j, ofm1ifm1img;

  /* number of tasks that could be run in parallel */
  const int work = handle->blocksifm*handle->blocksofm;
  /* compute chunck size */
  const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
  /* compute thr_begin and thr_end */
  const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
  int ifm2;
  /* number of tasks that could be run in parallel */
  const int transpose_work = handle->desc.N*handle->blocksifm;
  /* compute chunck size */
  const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : (transpose_work / handle->desc.threads) + 1;
  /* compute thr_begin and thr_end */
  const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
  const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

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

  const int copywork = handle->desc.N*handle->blocksifm;
  const int copychunksize = (copywork % handle->desc.threads == 0) ? (copywork / handle->desc.threads) : (copywork / handle->desc.threads) + 1;
  const int copy_thr_begin = (ltid * copychunksize < copywork) ? (ltid * copychunksize) : copywork;
  const int copy_thr_end = ((ltid + 1) * copychunksize < copywork) ? ((ltid + 1) * copychunksize) : copywork;

  int total_calls, aux;
  int n_code_segments = 0;
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
  int *init_indices, *copy_indices;

  int padded_w, padded_h;
  if (handle->padding_flag == 1) {
    padded_h = handle->ifhp + 2 * handle->desc.pad_h;
    padded_w = handle->ifwp + 2 * handle->desc.pad_w;
  } else {
    padded_h = handle->ifhp;
    padded_w = handle->ifwp;
  }

  stride_w = handle->desc.v;
  stride_h = handle->desc.u;
  kh = handle->desc.R;
  kw = handle->desc.S;

  /* Dryrun for copy/padding */
  /*
     if (handle->padding_flag == 1) {
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

     if (handle->trans_ofw_ifm == 0) {
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
     }*/

  num_ofw_strips = handle->ofw/handle->upd_ofw_rb;
  num_ofh_strips = handle->ofh/handle->upd_ofh_rb;
  local_entries = 0;

  if ( handle->use_thread_private_filter > 0 ) {
    for (ofm1ifm1img = img_parallel_thr_begin; ofm1ifm1img < img_parallel_thr_end; ++ofm1ifm1img) {
      ofm1 = ofm1ifm1img / (handle->blocksifm * handle->desc.N);
      imgifm1 = ofm1ifm1img % (handle->blocksifm * handle->desc.N);
      ifm1 = imgifm1 / handle->desc.N;
      img = imgifm1 % handle->desc.N;
      for (oi__=0; oi__<num_ofw_strips; ++oi__) {
        for (oj__=0; oj__<num_ofh_strips; ++oj__) {
          oi_=oi__*handle->upd_ofw_rb;
          oj_=oj__*handle->upd_ofh_rb;
          ii_ = oi_*stride_w;
          ij_ = oj_*stride_h;
          for (kj=0; kj < kh; ++kj) {
            if ( handle->ifmblock != 1  ) {
              for (ki=0; ki < kw; ++ki) {
                local_entries += 3;
              }
            } else {
              local_entries += 3;
            }
          }
        }
      }
    } 
  } else {
    for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
      ofm1 = ofm1ifm1/handle->blocksifm;
      ifm1 = ofm1ifm1%handle->blocksifm;
      for (img = 0; img < handle->desc.N; ++img) {
        for (oi__=0; oi__<num_ofw_strips; ++oi__) {
          for (oj__=0; oj__<num_ofh_strips; ++oj__) {
            oi_=oi__*handle->upd_ofw_rb;
            oj_=oj__*handle->upd_ofh_rb;
            ii_ = oi_*stride_w;
            ij_ = oj_*stride_h;
            for (kj=0; kj < kh; ++kj) {
              for (ki=0; ki < kw; ++ki) {
                local_entries += 3;
              }
            }
          }
        }
      }
    }
  }

  /* Alocate auxiliary data structures for index jitting  */
  handle->n_entries_upd[ltid] = local_entries/3;
  compute_indices = (int*) libxsmm_aligned_malloc( (local_entries+3) * sizeof(int), 2097152); 
  handle->compute_upd_indices_ptrs[ltid] = compute_indices;
  kernel_variant = (char*) libxsmm_aligned_malloc( (local_entries/3) * sizeof(char), 2097152); 
  handle->kernel_upd_variant_ptrs[ltid] = kernel_variant;
  handle->n_upd_code_segments[ltid] = n_code_segments;
  expanded_size = local_entries/3 + n_code_segments;

  /*tmp_expanded_stream = (int*) malloc( expanded_size * sizeof(int) );
    tmp_stream_index = 0;
    if (n_code_segments) {
    encoded_code_segments = (segment_t*) libxsmm_aligned_malloc(n_code_segments * sizeof(segment_t), 2097152);
    handle->upd_code_segments[ltid] = encoded_code_segments;
    }*/


  //printf("I am thread %d and I have %d local entries, trans is %d and per_thread is %d\n", ltid,  local_entries, handle->trans_ofw_ifm, handle->use_thread_private_filter );

  local_entries = 0;

  /* Second run to compute actual indices */

  if ( handle->use_thread_private_filter > 0 ) {
    for (ofm1ifm1img = img_parallel_thr_begin; ofm1ifm1img < img_parallel_thr_end; ++ofm1ifm1img) {
      ofm1 = ofm1ifm1img / (handle->blocksifm * handle->desc.N);
      imgifm1 = ofm1ifm1img % (handle->blocksifm * handle->desc.N);
      ifm1 = imgifm1 / handle->desc.N;
      img = imgifm1 % handle->desc.N;
      for (oi__=0; oi__<num_ofw_strips; ++oi__) {
        for (oj__=0; oj__<num_ofh_strips; ++oj__) {
          oi_=oi__*handle->upd_ofw_rb;
          oj_=oj__*handle->upd_ofh_rb;
          ii_ = oi_*stride_w;
          ij_ = oj_*stride_h;
          for (kj=0; kj < kh; ++kj) {
            if ( handle->ifmblock != 1  ) {
              for (ki=0; ki < kw; ++ki) {
                compute_indices[local_entries] =  ( ( ( ( ( (img *  handle->blocksifm) +  ifm1) * padded_h )  +  (ij_+kj)) * padded_w)  + (ii_ + ki) ) *  handle->ifmblock;
                compute_indices[local_entries+1] = ( (ofm1 *  handle->blocksifm )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock + kj * handle->desc.S *  handle->ifmblock *  handle->ofmblock + ki * handle->ifmblock *  handle->ofmblock;
                compute_indices[local_entries+2] = ( ( ( ( ( (img *  handle->blocksofm) +  ofm1) *  handle->ofhp )  +  oj_ ) * handle->ofwp)  +  oi_ ) *  handle->ofmblock;
                local_entries += 3;
              }
            } else {
              compute_indices[local_entries] =  ( ( ( ( ( (img *  handle->blocksifm) +  ifm1) * padded_h )  +  (ij_+kj)) * padded_w)  + ii_ ) *  handle->ifmblock;
              compute_indices[local_entries+1] = ( (ofm1 *  handle->blocksifm )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock + kj * handle->desc.S *  handle->ifmblock *  handle->ofmblock;
              compute_indices[local_entries+2] = ( ( ( ( ( (img *  handle->blocksofm) +  ofm1) *  handle->ofhp )  +  oj_ ) * handle->ofwp)  +  oi_ ) *  handle->ofmblock;
              local_entries += 3;
            }
          }
        }
      }
    } 
  } else {
    for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
      ofm1 = ofm1ifm1/handle->blocksifm;
      ifm1 = ofm1ifm1%handle->blocksifm;
      for (img = 0; img < handle->desc.N; ++img) {
        for (oi__=0; oi__<num_ofw_strips; ++oi__) {
          for (oj__=0; oj__<num_ofh_strips; ++oj__) {
            oi_=oi__*handle->upd_ofw_rb;
            oj_=oj__*handle->upd_ofh_rb;
            ii_ = oi_*stride_w;
            ij_ = oj_*stride_h;
            for (kj=0; kj < kh; ++kj) {
              for (ki=0; ki < kw; ++ki) {
                if (handle->trans_ofw_ifm == 1 ) {
                  compute_indices[local_entries] =  ( ( ( ( ( (img *  handle->blocksifm) +  ifm1) * padded_h )  +  (ij_+kj)) * handle->ifmblock) ) * padded_w  + (ii_ + ki) ;
                } else {
                  compute_indices[local_entries] =  ( ( ( ( ( (img *  handle->blocksifm) +  ifm1) * padded_h )  +  (ij_+kj)) * padded_w)  + (ii_ + ki) ) *  handle->ifmblock;
                }
                compute_indices[local_entries+1] = ( (ofm1 *  handle->blocksifm )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock + kj * handle->desc.S *  handle->ifmblock *  handle->ofmblock + ki * handle->ifmblock *  handle->ofmblock;
                compute_indices[local_entries+2] = ( ( ( ( ( (img *  handle->blocksofm) +  ofm1) *  handle->ofhp )  +  oj_ ) * handle->ofwp)  +  oi_ ) *  handle->ofmblock;
                local_entries += 3;
              }
            }
          }
        }
      }
    }
  }




  //free(tmp_expanded_stream);

  /* At the end of stream do not prefetch garbage */
  compute_indices[local_entries] = 0;
  compute_indices[local_entries+1] = 0;
  compute_indices[local_entries+2] = 0;
  total_calls = local_entries/3;

}

