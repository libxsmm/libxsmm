/******************************************************************************
** Copyright (c) 2018, Intel Corporation                                     **
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

#if !defined(_OPENMP)
int ltid;
#endif

int BLOCKSIFM_BLOCKING = handle->blocksifm_blocking;
int BLOCKSIFM = handle->blocksifm_lp;
int BLOCKSOFM = handle->blocksofm;
int loop_order = handle->loop_order;

#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
#else
for (ltid = 0; ltid < handle->desc.threads; ltid++)
#endif
{
#if defined(_OPENMP)
  int ltid = omp_get_thread_num();
#endif
  int img, ofm1, ifm1, oj, oi, local_entries = 0, ojb, ifmb, ofmb;
  int ii_use, ij_use, oi_use, oj_use;
  int padded_h = 0, padded_w = 0;

  /* Threading related variables */
  int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
  int threads_per_image = handle->desc.threads / handle->desc.N;
  int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
  int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
  int my_ofm_start = 0;
  int my_ofm_end = BLOCKSOFM;
  int myOfmId;
  int nOfmBlocks;
  int n_code_segments;
  int mark_ofm_init, mark_ofm_close, mark_img_init, mark_ifm_close = 0;
  int *tmp_expanded_stream, tmp_stream_index;
  segment_t *encoded_code_segments = NULL;
  int expanded_size;
  int stretch_of_convs;
  int encoded_stream_index;
  int lookahead_index;

  /* Arrays of stream indices */
  int *compute_indices, *bn_indices = 0;
  char *kernel_variant;

  const int use_accumulation_scratch = (BLOCKSIFM_BLOCKING == BLOCKSIFM) ? 0 : 1 ;

  if (handle->padding_flag == 1) {
    padded_h = handle->ifhp + 2 * handle->desc.pad_h;
    padded_w = handle->ifwp + 2 * handle->desc.pad_w;
  }

  n_code_segments = 0;
  tmp_stream_index = 0;

  if ( imgpt <= 1 ) {
    my_img_start = LIBXSMM_MIN( ltid / threads_per_image, handle->desc.N);
    my_img_end = LIBXSMM_MIN( my_img_start + 1, handle->desc.N);
    myOfmId = ltid % threads_per_image;
    nOfmBlocks = (BLOCKSOFM + threads_per_image -1) / threads_per_image;
    my_ofm_start = LIBXSMM_MIN(myOfmId * nOfmBlocks, BLOCKSOFM);
    my_ofm_end = LIBXSMM_MIN((myOfmId+1) * nOfmBlocks, BLOCKSOFM);
  }

  mark_ofm_init = 0; /* ((((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd == 0) ) || ( (handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) ) ? 1 : 0;*/
  /*mark_ofm_close = (((((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0)) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 0) ) ||
    ((((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0)) && (handle->use_fwd_for_bwd == 1) && (handle->use_nts_bwd == 0) ) ) ? 1 : 0;*/
  handle->use_accumulation_scratch = use_accumulation_scratch;
  mark_ofm_close = use_accumulation_scratch;
  mark_img_init = ( (handle->padding_flag == 1) || (mark_ofm_close == 1) || (mark_ifm_close == 1) ) ? 1 : 0;

  /* Perform a dryrun to compute the memory requirements of the stream of indices */
  /* THE MIXED loop ordering is used necessarily when we have NON-1x1 concolutions, the result is accumulated to a scratchpad for each ofm...  */
  if (loop_order == MIXED) {
    for (img = my_img_start; img < my_img_end; img++) {
      if (mark_img_init== 1) {
        n_code_segments++;
      }
      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
        for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
            for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
              for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {
                    local_entries += 3;

                    if (mark_ofm_init == 1) {
                      if (ifm1 == 0 && oj == 0 && oi == 0) {
                        n_code_segments++;
                      }
                    }

                    if (mark_ofm_close == 1) {
                      if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING  && oj >= handle->ofh - handle->fwd_ofh_rb && oi >= handle->ofw - handle->fwd_ofw_rb) {
                        n_code_segments++;
                      }
                    }


                    if (mark_ifm_close == 1) {
                      if ( ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING ) {
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

  if (loop_order == HWKC) {
    for (img = my_img_start; img < my_img_end; img++) {
      if (mark_img_init== 1) {
        n_code_segments++;
      }
      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
        for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
          for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
            for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {

                    local_entries += 3;

                    if (mark_ofm_init == 1) {
                      if (ifm1 == 0 && oj == 0 && oi == 0) {
                        n_code_segments++;
                      }
                    }

                    if (mark_ofm_close == 1) {
                      if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING  && oj >= handle->ofh - handle->fwd_ofh_rb && oi >= handle->ofw - handle->fwd_ofw_rb) {
                        n_code_segments++;
                      }
                    }


                    if (mark_ifm_close == 1) {
                      if ( ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING ) {
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

  /* BN offsets...  */
  if  (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 1) ) {
    bn_indices = (int*) libxsmm_aligned_malloc( (local_entries/3) * sizeof(int), 64);
    handle->bn_indices_ptrs[ltid] = bn_indices;
  }

  kernel_variant = (char*)(3 <= local_entries ? libxsmm_aligned_malloc((local_entries / 3) * sizeof(char), 64) : NULL);
  handle->kernel_fwd_variant_ptrs[ltid] = kernel_variant;
  handle->n_fwd_code_segments[ltid] = n_code_segments;
  expanded_size = local_entries/3 + n_code_segments;
  tmp_expanded_stream = (int*)(0 < expanded_size ? malloc(expanded_size * sizeof(int)) : 0);
  tmp_stream_index = 0;
  if (n_code_segments) {
    encoded_code_segments = (segment_t*) libxsmm_aligned_malloc(n_code_segments * sizeof(segment_t), 64);
    handle->fwd_code_segments[ltid] = encoded_code_segments;
  }
  local_entries = 0;

  /* Second run to compute actual indices */
  if (loop_order == MIXED) {
    for (img = my_img_start; img < my_img_end; img++) {
      if (0 != tmp_expanded_stream && mark_img_init== 1) {
        tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_INIT;
        tmp_stream_index++;
      }
      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
        /*for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) { */
        for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
            for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
              for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {

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

                    if (0 != tmp_expanded_stream && mark_ofm_init == 1 && ifm1 == 0 && oj == 0 && oi == 0) {
                      tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_INIT;
                      tmp_stream_index++;
                    }

                    if (handle->padding_flag == 1) {
                      compute_indices[local_entries] =  ( ( ( ifm1 *  padded_h  +  ij_use) * padded_w)  +  ii_use) *  handle->ifmblock * handle->fm_lp_block;
                    } else {
                      compute_indices[local_entries] =  ( ( ( ( ( (img *  BLOCKSIFM) +  ifm1) *  handle->ifhp )  +  ij_use) * handle->ifwp)  +  ii_use  ) *  handle->ifmblock * handle->fm_lp_block;
                    }
                    compute_indices[local_entries+1] = ( (ofm1 *  BLOCKSIFM )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock *  handle->fm_lp_block;

                    if (use_accumulation_scratch) {
                      compute_indices[local_entries+2] = ( ( (oj_use) * handle->ofw)  +  oi_use) *  handle->ofmblock;
                    } else {
                      compute_indices[local_entries+2] = ( ( ( ( ( (img *  BLOCKSOFM) +  ofm1) *  handle->ofhp )  +  oj_use) * handle->ofwp)  +  oi_use) *  handle->ofmblock;
                    }

                    /* Initialize kernel variant with the one that prefetches everything */
                    if (handle->n_variants == 2) {
                      if (handle->h_variants) {
                        if (oj + handle->fwd_ofh_rb <= handle->ofh) {
                          kernel_variant[local_entries/3] = 0;
                        } else {
                          kernel_variant[local_entries/3] = 1;
                        }
                      } else {
                        if (oi + handle->fwd_ofw_rb <= handle->ofw) {
                          kernel_variant[local_entries/3] = 0;
                        } else {
                          kernel_variant[local_entries/3] = 1;
                        }
                      }
                    }


                    if (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 1) ) {
                      bn_indices[local_entries/3] =  img * handle->ofmblock + ofm1 * handle->ofmblock * handle->desc.N;
                    }

                    local_entries += 3;

                    if (0 != tmp_expanded_stream) {
                      tmp_expanded_stream[tmp_stream_index] = CONVOLUTION_KERNEL;
                      tmp_stream_index++;
                      if (mark_ifm_close == 1 && ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING) {
                        tmp_expanded_stream[tmp_stream_index] = IFM_LOOP_CLOSE_S;
                        tmp_stream_index++;
                      }
                      if (mark_ofm_close == 1 && ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= (handle->ofh - handle->fwd_ofh_rb) && oi >= (handle->ofw - handle->fwd_ofw_rb)) {
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


    if (loop_order == HWKC) {
      for (img = my_img_start; img < my_img_end; img++) {
        if (0 != tmp_expanded_stream && mark_img_init== 1) {
          tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_INIT;
          tmp_stream_index++;
        }

        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
            for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
              for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                  for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                    for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {

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

                      if (0 != tmp_expanded_stream && mark_ofm_init == 1 && ifm1 == 0 && oj == 0 && oi == 0) {
                        tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_INIT;
                        tmp_stream_index++;
                      }

                      if (handle->padding_flag == 1) {
                        compute_indices[local_entries] =  ( ( ( ifm1 *  padded_h  +  ij_use) * padded_w)  +  ii_use) *  handle->ifmblock * handle->fm_lp_block;
                      } else {
                        compute_indices[local_entries] =  ( ( ( ( ( (img *  BLOCKSIFM) +  ifm1) *  handle->ifhp )  +  ij_use) * handle->ifwp)  +  ii_use  ) *  handle->ifmblock * handle->fm_lp_block;
                      }
                      compute_indices[local_entries+1] = ( (ofm1 *  BLOCKSIFM )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock *  handle->fm_lp_block;
                      compute_indices[local_entries+2] = ( ( ( ( ( (img *  BLOCKSOFM) +  ofm1) *  handle->ofhp )  +  oj_use) * handle->ofwp)  +  oi_use  ) *  handle->ofmblock;

                      /* Initialize kernel variant with the one that prefetches everything */
                      if (handle->n_variants == 2) {
                        if (handle->h_variants) {
                          if (oj + handle->fwd_ofh_rb <= handle->ofh) {
                            kernel_variant[local_entries/3] = 0;
                          } else {
                            kernel_variant[local_entries/3] = 1;
                          }
                        } else {
                          if (oi + handle->fwd_ofw_rb <= handle->ofw) {
                            kernel_variant[local_entries/3] = 0;
                          } else {
                            kernel_variant[local_entries/3] = 1;
                          }
                        }
                      }

                      if (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 1) ) {
                        bn_indices[local_entries/3] = img * handle->ofmblock + ofm1 * handle->ofmblock * handle->desc.N;
                      }

                      local_entries += 3;

                      if (0 != tmp_expanded_stream) {
                        tmp_expanded_stream[tmp_stream_index] = CONVOLUTION_KERNEL;
                        tmp_stream_index++;

                        if (mark_ifm_close == 1 && ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING) {
                          tmp_expanded_stream[tmp_stream_index] = IFM_LOOP_CLOSE_S;
                          tmp_stream_index++;
                        }

                        if (mark_ofm_close == 1 && ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= handle->ofh - handle->fwd_ofh_rb && oi >= handle->ofw - handle->fwd_ofw_rb) {
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
      }

      /* Final pass over the segments to fill-in auxiliary indices...  */
      encoded_stream_index = 0;

      if (loop_order == MIXED) {
        for (img = my_img_start; img < my_img_end; img++) {
          if (mark_img_init== 1) {
            encoded_code_segments[encoded_stream_index].aux_index = img;
            encoded_stream_index++;
          }
          for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
            /*for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) { */
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
              for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                    for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                      for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {
                        if (mark_ofm_init == 1) {
                          if (ifm1 == 0 && oj == 0 && oi == 0) {
                            encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                            encoded_stream_index++;
                          }
                        }

                        if (mark_ofm_close == 1) {
                          if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= handle->ofh - handle->fwd_ofh_rb && oi >= handle->ofw - handle->fwd_ofw_rb) {
                            encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                            encoded_stream_index++;
                          }
                        }

                        if (mark_ifm_close == 1) {
                          if ( ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING ) {
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

        if (loop_order == HWKC) {
          for (img = my_img_start; img < my_img_end; img++) {
            if (mark_img_init== 1) {
              encoded_code_segments[encoded_stream_index].aux_index = img;
              encoded_stream_index++;
            }
            for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
              for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                    for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                      for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                        for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {
                          if (mark_ofm_init == 1) {
                            if (ifm1 == 0 && oj == 0 && oi == 0) {
                              encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                              encoded_stream_index++;
                            }
                          }

                          if (mark_ofm_close == 1) {
                            if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= handle->ofh - handle->fwd_ofh_rb && oi >= handle->ofw - handle->fwd_ofw_rb) {
                              encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                              encoded_stream_index++;
                            }
                          }

                          if (mark_ifm_close == 1) {
                            if ( ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING ) {
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

