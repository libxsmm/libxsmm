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

handle->block_upd_ofm = 4;
handle->block_upd_ifm = 4;

#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
#else
for (ltid = 0; ltid < handle->desc.threads; ltid++)
#endif
{
#if defined(_OPENMP)
  int ltid = omp_get_thread_num();
#endif
  int img, imgb, ifmb, ofmb, ofm1, ifm1, num_ofw_strips, num_ofh_strips, oi_, oj_, oi__, oj__,ii_, ij_, kh, kw, KW, ofm1ifm1, ki, kj, imgifm1, local_entries, stride_w, stride_h ;
  int i, j, ofm1ifm1img;

  /* Here we assume that N % Threads == 0 */
  int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
  int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
  int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);

  /* We always have to reduce in this case... */
  const int reduce_work = handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock;
  /* compute chunck size */
  const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
  /* compute thr_begin and thr_end */
  const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
  const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;

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
  num_ofw_strips = handle->ofw/handle->upd_ofw_rb;
  num_ofh_strips = handle->ofh/handle->upd_ofh_rb;
  local_entries = 0;
  if ( handle->ifmblock != 1  ) {
    KW = kw;
  } else {
    KW = 1;
  }

  /* Perform a dryrun to compute the memory requirements of the stream of indices */
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) { 
        for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
            for (oi__=0; oi__<num_ofw_strips; ++oi__) {
              for (oj__=0; oj__<num_ofh_strips; ++oj__) {
                for (kj=0; kj < kh; ++kj) {
                  for (ki=0; ki < KW; ++ki) {
                    oi_=oi__*handle->upd_ofw_rb;
                    oj_=oj__*handle->upd_ofh_rb;
                    ii_ = oi_*stride_w;
                    ij_ = oj_*stride_h;
                    local_entries += 3;
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
  compute_indices = (int*) libxsmm_aligned_malloc( (local_entries+3) * sizeof(int), 2097152); 
  handle->compute_upd_indices_ptrs[ltid] = compute_indices;
  kernel_variant = (char*) libxsmm_aligned_malloc( (local_entries/3) * sizeof(char), 2097152); 
  handle->kernel_upd_variant_ptrs[ltid] = kernel_variant;
  handle->n_upd_code_segments[ltid] = n_code_segments;
  expanded_size = local_entries/3 + n_code_segments;

  /* Second run to compute actual indices */
  local_entries = 0;
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
        for (oj__=0; oj__<num_ofh_strips; ++oj__) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
              for (kj=0; kj < kh; ++kj) {
                for (oi__=0; oi__<num_ofw_strips; ++oi__) {
                  for (ki=0; ki < KW; ++ki) {
                    oi_=oi__*handle->upd_ofw_rb;
                    oj_=oj__*handle->upd_ofh_rb;
                    ii_ = oi_*stride_w;
                    ij_ = oj_*stride_h;
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
    }
  }

  compute_indices[local_entries] = 0;
  compute_indices[local_entries+1] = 0;
  compute_indices[local_entries+2] = 0;
  total_calls = local_entries/3;

}

