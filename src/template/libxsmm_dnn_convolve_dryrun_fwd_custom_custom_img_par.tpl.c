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

int block_j = 14;
int h_chunk_size;
#if !defined(_OPENMP)
int ltid;
#endif
handle->block_fwd_ofm = 8;
handle->block_fwd_ifm = 8;

if ( handle->ofh == 14 || handle->ofh == 48 || handle->ofh == 54 || handle->ofh == 56 || handle->ofh == 112 ) {
  block_j = 4;
}

while ( block_j % handle->fwd_ofh_rb != 0 ) {
  block_j--;
}

handle->block_fwd_oj = block_j;

#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
#else
for (ltid = 0; ltid < handle->desc.threads; ltid++)
#endif
{
#if defined(_OPENMP)
  int ltid = omp_get_thread_num();
#endif
  int img, ofm1, ifm1, oj, oi, ij, ii, local_entries = 0, ojb, ifmb, ofmb, my_img_start, my_img_end, my_ofm_start, my_ofm_end, myOfmId, myHId, my_h_start, my_h_end, my_w_start, my_w_end;
  int total_calls;
  int n_code_segments;
  int mark_ofm_init, mark_ofm_close;
  int *tmp_expanded_stream, tmp_stream_index;
  int *encoded_code_segments;
  int expanded_size;
  int stretch_of_convs;
  int encoded_stream_index;
  int lookahead_index;
  int cur_wt, next_wt, cur_out, next_out, padded_h, padded_w;
  /* Arrays of stream indices */
  int *compute_indices;
  char *kernel_variant;
  /* Threading related variables */
  int threads_per_image = handle->desc.threads / handle->desc.N;
  while (threads_per_image % 4 != 0 && threads_per_image > 2) {
    threads_per_image--;
  }

  my_img_start = LIBXSMM_MIN( ltid / threads_per_image, handle->desc.N);
  my_img_end = LIBXSMM_MIN( my_img_start + 1, handle->desc.N);
  my_ofm_start = 0;
  my_ofm_end = handle->blocksofm;
  my_w_start = 0;
  my_w_end = handle->ofw;

  if (handle->padding_flag == 1) {
    padded_h = handle->ifhp + 2 * handle->desc.pad_h;
    padded_w = handle->ifwp + 2 * handle->desc.pad_w;
  }

  n_code_segments = 0;
  tmp_stream_index = 0;
  /* Threading strategie  */
  if (threads_per_image == 16 ) {
    myOfmId = (ltid/8) % 2;
    if (myOfmId == 0) {
      my_ofm_end = handle->blocksofm/2;
    } else {
      my_ofm_start = handle->blocksofm/2;
    }
    myHId = ltid % 8;
    h_chunk_size = (handle->ofh+7)/8;
    while ( h_chunk_size % handle->fwd_ofh_rb != 0 ) {
      h_chunk_size++;
    }
    my_h_start = LIBXSMM_MIN(myHId * h_chunk_size, handle->ofh);
    my_h_end = LIBXSMM_MIN((myHId + 1) * h_chunk_size, handle->ofh);

  } else if (threads_per_image == 8) {
    if (handle->blocksofm == 2) {
      myOfmId = (ltid/4) % 2;
      my_ofm_start = myOfmId;
      my_ofm_end = myOfmId + 1;
      myHId = ltid % 4;
      h_chunk_size = (handle->ofh+3)/4;
      while ( h_chunk_size % handle->fwd_ofh_rb != 0 ) {
        h_chunk_size++;
      }      
      my_h_start = LIBXSMM_MIN(myHId * h_chunk_size, handle->ofh);
      my_h_end = LIBXSMM_MIN((myHId + 1) * h_chunk_size, handle->ofh);
    } else if (handle->blocksofm == 4) {
      myOfmId = (ltid/2) % 4;
      my_ofm_start = myOfmId;
      my_ofm_end = myOfmId + 1;
      myHId = ltid % 2;
      h_chunk_size = (handle->ofh+1)/2;
      while ( h_chunk_size % handle->fwd_ofh_rb != 0 ) {
        h_chunk_size++;
      }
      my_h_start = LIBXSMM_MIN(myHId * h_chunk_size, handle->ofh);
      my_h_end = LIBXSMM_MIN((myHId + 1) * h_chunk_size, handle->ofh);
    } else {
      myHId = ltid % 8;
      h_chunk_size = (handle->ofh+7)/8;
      while ( h_chunk_size % handle->fwd_ofh_rb != 0 ) {
        h_chunk_size++;
      }
      my_h_start = LIBXSMM_MIN(myHId * h_chunk_size, handle->ofh);
      my_h_end = LIBXSMM_MIN((myHId + 1) * h_chunk_size, handle->ofh);
    }

  } else if (threads_per_image == 4 ) {
    if ( handle->blocksofm == 2 ) {
      myOfmId = (ltid/2) % 2;
      my_ofm_start = myOfmId;
      my_ofm_end = myOfmId + 1;
      myHId = ltid % 2;
      h_chunk_size = (handle->ofh+1)/2;
      while ( h_chunk_size % handle->fwd_ofh_rb != 0 ) {
        h_chunk_size++;
      }  
      my_h_start = LIBXSMM_MIN(myHId * h_chunk_size, handle->ofh);
      my_h_end = LIBXSMM_MIN((myHId + 1) * h_chunk_size, handle->ofh);
    } else {
      myHId = ltid % 4;
      h_chunk_size = (handle->ofh+3)/4;
      while ( h_chunk_size % handle->fwd_ofh_rb != 0 ) {
        h_chunk_size++;
      }
      my_h_start = LIBXSMM_MIN(myHId * h_chunk_size, handle->ofh);
      my_h_end = LIBXSMM_MIN((myHId + 1) * h_chunk_size, handle->ofh);
    } 

  } else if (threads_per_image == 2 ) { 
    myHId = ltid % 2;
    h_chunk_size = (handle->ofh+1)/2;
    while ( h_chunk_size % handle->fwd_ofh_rb != 0 ) {
      h_chunk_size++;
    }  
    my_h_start = LIBXSMM_MIN(myHId * h_chunk_size, handle->ofh);
    my_h_end = LIBXSMM_MIN((myHId + 1) * h_chunk_size, handle->ofh);

  } else {
    my_h_start = 0;
    my_h_end = handle->ofh;
  }

  mark_ofm_init = ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ? 1 : 0;
  mark_ofm_close = (handle->datatype != handle->datatype_itm) ? 1 : 0;

  /* Perform a dryrun to compute the memory requirements of the stream of indices */
  for (img = my_img_start; img < my_img_end; img++) {
    if (handle->padding_flag == 1) {
      n_code_segments++;
    }     
    for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) { 
        for (ojb = my_h_start; ojb < my_h_end; ojb += handle->block_fwd_oj) {
          for ( ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ++ifm1) {
              for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,my_h_end); oj += handle->fwd_ofh_rb) {
                for (oi = my_w_start; oi < my_w_end; oi += handle->fwd_ofw_rb) {
                  local_entries += 3;

                  if (mark_ofm_init == 1) {
                    if (ifm1 == 0 && oj == my_h_start && oi == my_w_start) {
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

  handle->n_entries_fwd[ltid] = local_entries/3;

  /* Alocate auxiliary data structures for index jitting  */
  compute_indices = (int*) libxsmm_aligned_malloc( (local_entries+3) * sizeof(int), 2097152); 
  handle->compute_fwd_indices_ptrs[ltid] = compute_indices;
  kernel_variant = (char*) libxsmm_aligned_malloc( (local_entries/3) * sizeof(char), 2097152); 
  handle->kernel_fwd_variant_ptrs[ltid] = kernel_variant;
  handle->n_fwd_code_segments[ltid] = n_code_segments;
  expanded_size = local_entries/3 + n_code_segments;
  tmp_expanded_stream = (int*) malloc( expanded_size * sizeof(int) );
  tmp_stream_index = 0;
  if (n_code_segments) {
    encoded_code_segments = (int*) libxsmm_aligned_malloc( 2 * n_code_segments * sizeof(int), 2097152);
    handle->fwd_code_segments[ltid] = encoded_code_segments;
    handle->img_start[ltid] = my_img_start;
  }
  local_entries = 0;

  /* Second run to compute actual indices */
  for (img = my_img_start; img < my_img_end; img++) {
    if (handle->padding_flag == 1) {
      tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_INIT;
      tmp_stream_index++;
    } 
    for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) { 
        for (ojb = my_h_start; ojb < my_h_end; ojb += handle->block_fwd_oj) {
          for ( ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ++ifm1) {
              for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,my_h_end); oj += handle->fwd_ofh_rb) {
                for (oi = my_w_start; oi < my_w_end; oi += handle->fwd_ofw_rb) {
                  ij = oj * handle->desc.u;
                  ii = oi * handle->desc.v;

                  if (mark_ofm_init == 1) {
                    if (ifm1 == 0 && oj == my_h_start && oi == my_w_start) {
                      tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_INIT;
                      tmp_stream_index++;
                    }
                  }

                  if (handle->padding_flag == 1) { 
                    compute_indices[local_entries] =  ( ( ( ifm1 *  padded_h  +  ij) * padded_w)  +  ii) *  handle->ifmblock * handle->fm_lp_block;
                  } else {
                    compute_indices[local_entries] =  ( ( ( ( ( (img *  handle->blocksifm) +  ifm1) *  handle->ifhp )  +  ij) * handle->ifwp)  +  ii  ) *  handle->ifmblock * handle->fm_lp_block;
                  }
                  compute_indices[local_entries+1] = ( (ofm1 *  handle->blocksifm )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock *  handle->fm_lp_block;
                  compute_indices[local_entries+2] = ( ( ( ( ( (img *  handle->blocksofm * handle->fm_lp_block ) +  ofm1) *  handle->ofhp )  +  oj) * handle->ofwp)  +  oi  ) *  handle->ofmblock;

                  /* Initialize kernel variant with the one that prefetches everything */
                  kernel_variant[local_entries/3] = 2;
                  local_entries += 3;

                  tmp_expanded_stream[tmp_stream_index] = CONVOLUTION_KERNEL;
                  tmp_stream_index++;

                  if (mark_ofm_close == 1) {
                    if (ifm1 == handle->blocksifm-1  && oj == my_h_end - handle->fwd_ofh_rb && oi == my_w_end - handle->fwd_ofw_rb) {
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
      encoded_code_segments[encoded_stream_index] = tmp_expanded_stream[tmp_stream_index];
      encoded_code_segments[encoded_stream_index+1] = stretch_of_convs;
      encoded_stream_index += 2;
      stretch_of_convs = 0;
      tmp_stream_index = lookahead_index;
      lookahead_index++;
    }

    /* Check if we have not written last segment entry -- in this case the stream ends with an action point */
    if ( encoded_stream_index/2 < n_code_segments ) {
      encoded_code_segments[encoded_stream_index] = tmp_expanded_stream[tmp_stream_index];
      encoded_code_segments[encoded_stream_index+1] = stretch_of_convs;
    }
  }

  free(tmp_expanded_stream);

  /* At the end of stream do not prefetch garbage */
  compute_indices[local_entries] = 0;
  compute_indices[local_entries+1] = 0;
  compute_indices[local_entries+2] = 0;
  total_calls = local_entries/3;

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
