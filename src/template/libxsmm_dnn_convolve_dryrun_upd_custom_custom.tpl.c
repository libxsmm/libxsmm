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
#if !defined(_OPENMP)
int ltid;
#endif

/* FIXME assignments here */
int BLOCKSIFM = handle->blocksifm;
int BLOCKSOFM = handle->blocksofm;

int IFMBLOCK;
if (handle->use_lp_kernel) {
  IFMBLOCK = handle->ifmblock_hp;
} else {
  IFMBLOCK = handle->ifmblock;
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
  int img, imgb, ofm1, ifm1, num_ofw_strips, oi_, oj_, oi__, ii_, ij_, kh, kw, ki, kj, local_entries;
  int img_block_size;
  int n_code_segments = 0;

  char *kernel_variant;
  /* Arrays of stream indices */
  int *compute_indices;

  int my_ofm_start;
  int my_ofm_end;
  int my_ifm_start;
  int my_ifm_end;
  int group_size = handle->desc.threads/handle->weight_copies;
  int tile_id = ltid/group_size;
  int tiles = handle->weight_copies;
  int img_per_tile = handle->desc.N/tiles;
  int my_img_start = LIBXSMM_MIN( tile_id * img_per_tile, handle->desc.N);
  int my_img_end = LIBXSMM_MIN( (tile_id+1) * img_per_tile, handle->desc.N);


  int my_in_tile_id = ltid % group_size;
  int ifms_per_thread = (BLOCKSIFM+group_size-1)/group_size;
  int ofms_per_thread;
  int block_ofm, block_ifm;

  int KW, ofmb, ifmb, ojb;
  int stride_w = handle->desc.v;
  int stride_h = handle->desc.u;
  int padded_w, padded_h;

  if (handle->padding_flag == 1) {
    padded_h = handle->ifhp + 2 * handle->desc.pad_h;
    padded_w = handle->ifwp + 2 * handle->desc.pad_w + handle->qfma_input_pad;
  } else {
    if (handle->resize_input == 1) {
      padded_h = handle->ifhp_resized;
      padded_w = handle->ifwp_resized + handle->qfma_input_pad;
      stride_w = 1;
      stride_h = 1;
    } else {
      padded_h = handle->ifhp;
      padded_w = handle->ifwp + handle->qfma_input_pad;
    }
  }

  kh = handle->desc.R;
  kw = handle->desc.S;

  num_ofw_strips = 1;
  local_entries = 0;

  KW = kw;
  my_ifm_start = LIBXSMM_MIN( my_in_tile_id * ifms_per_thread, BLOCKSIFM  );
  my_ifm_end = LIBXSMM_MIN( (my_in_tile_id+1) * ifms_per_thread, BLOCKSIFM  );
  my_ofm_start = 0;
  my_ofm_end = BLOCKSOFM;
  img_block_size = handle->blocksimg_blocking;

  if (handle->reduce_weights == 0) {
    int n_ifm_teams, n_ofm_teams, my_ifm_id, my_ofm_id;
    /* Parallelize over the FMs in this case and avoid reduction */
    int team_div = (int) libxsmm_isqrt_u32(handle->desc.threads);
    while ( handle->desc.threads % team_div != 0  ) {
      team_div--;
    }
    n_ifm_teams = (BLOCKSIFM > BLOCKSOFM) ? handle->desc.threads / team_div : team_div;
    n_ofm_teams = (BLOCKSIFM > BLOCKSOFM) ? team_div : handle->desc.threads / team_div;
    ifms_per_thread = (BLOCKSIFM+n_ifm_teams-1)/n_ifm_teams;
    ofms_per_thread = (BLOCKSOFM+n_ofm_teams-1)/n_ofm_teams;
    my_ifm_id = ltid/n_ofm_teams;
    my_ofm_id = ltid%n_ofm_teams;
    my_ifm_start =  LIBXSMM_MIN(my_ifm_id * ifms_per_thread, BLOCKSIFM);
    my_ifm_end =  LIBXSMM_MIN((my_ifm_id+1) * ifms_per_thread, BLOCKSIFM);
    my_ofm_start =  LIBXSMM_MIN(my_ofm_id * ofms_per_thread, BLOCKSOFM);
    my_ofm_end =  LIBXSMM_MIN((my_ofm_id+1) * ofms_per_thread, BLOCKSOFM);
    my_img_start = 0;
    my_img_end = handle->desc.N;
    img_block_size = handle->desc.threads;
  }

  block_ofm = my_ofm_end-my_ofm_start+1;
  block_ifm = my_ifm_end-my_ifm_start+1;

  for (imgb = my_img_start; imgb < my_img_end; imgb += img_block_size) {
    for (img = imgb; img < LIBXSMM_MIN(imgb+img_block_size, my_img_end); img+=handle->blocksimg_blocking) {
      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
        for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
                for (oj_ = ojb; oj_ < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj_ += handle->upd_ofh_rb) {
                  for (oi__=0; oi__<num_ofw_strips; ++oi__) {
                    for (kj=0; kj < kh; ++kj) {
                      for (ki=0; ki < KW; ++ki) {
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
    }
  }


  /* Allocate auxiliary data structures for index jitting  */
  handle->n_entries_upd[ltid] = local_entries/3;
  compute_indices = (int*) libxsmm_aligned_malloc( (local_entries+3) * sizeof(int), 64);
  handle->compute_upd_indices_ptrs[ltid] = compute_indices;
  kernel_variant = (char*)(3 <= local_entries ? libxsmm_aligned_malloc((local_entries / 3) * sizeof(char), 64) : NULL);
  handle->kernel_upd_variant_ptrs[ltid] = kernel_variant;
  handle->n_upd_code_segments[ltid] = n_code_segments;
  local_entries = 0;

  /* Second run to compute actual indices */
  if ( !(((handle->desc.C == 512) && (handle->desc.K == 1024)) || ((handle->desc.C == 1024) &&  (handle->desc.K == 256)))  ) {
    for (imgb = my_img_start; imgb < my_img_end; imgb += img_block_size) {
      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
        for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
              for (img = imgb; img < LIBXSMM_MIN(imgb+img_block_size, my_img_end); img+=handle->blocksimg_blocking)  {
                for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
                  for (oj_ = ojb; oj_ < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj_ += handle->upd_ofh_rb) {
                    for (oi__=0; oi__<num_ofw_strips; ++oi__) {
                      for (kj=0; kj < kh; ++kj) {
                        for (ki=0; ki < KW; ++ki) {

                          oi_=oi__*handle->upd_ofw_rb;
                          ii_ = oi_*stride_w;
                          ij_ = oj_*stride_h;

                          if (handle->trans_ofw_ifm == 1 ) {
                            compute_indices[local_entries] = ((((((img *  BLOCKSIFM) + ifm1) * padded_h) + (ij_ + kj)) * IFMBLOCK)) * padded_w + (ii_ + ki);
                          } else {
                            compute_indices[local_entries] = ((((((img *  BLOCKSIFM) + ifm1) * padded_h) + (ij_ + kj)) * padded_w) + (ii_ + ki)) *  IFMBLOCK;
                          }
                          compute_indices[local_entries+1] = (((ofm1 *  BLOCKSIFM) + ifm1) * handle->desc.R * handle->desc.S *  IFMBLOCK *  handle->ofmblock + kj * handle->desc.S *  IFMBLOCK *  handle->ofmblock + ki * IFMBLOCK *  handle->ofmblock)  * handle->weight_copies;

                          compute_indices[local_entries+2] = ((((((img *  BLOCKSOFM) + ofm1) *  handle->ofhp) + oj_) * (handle->ofwp + handle->output_lp_padding)) + oi_) *  handle->ofmblock;
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
      }
    }
  } else {
    for (imgb = my_img_start; imgb < my_img_end; imgb += img_block_size) {
      for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
              for (img = imgb; img < LIBXSMM_MIN(imgb+img_block_size, my_img_end); img+=handle->blocksimg_blocking)  {
                for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
                  for (oj_ = ojb; oj_ < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj_ += handle->upd_ofh_rb) {
                    for (oi__=0; oi__<num_ofw_strips; ++oi__) {
                      for (kj=0; kj < kh; ++kj) {
                        for (ki=0; ki < KW; ++ki) {

                          oi_=oi__*handle->upd_ofw_rb;
                          ii_ = oi_*stride_w;
                          ij_ = oj_*stride_h;

                          if (handle->trans_ofw_ifm == 1 ) {
                            compute_indices[local_entries] = ((((((img *  BLOCKSIFM) + ifm1) * padded_h) + (ij_ + kj)) * IFMBLOCK)) * padded_w + (ii_ + ki);
                          } else {
                            compute_indices[local_entries] = ((((((img *  BLOCKSIFM) + ifm1) * padded_h) + (ij_ + kj)) * padded_w) + (ii_ + ki)) *  IFMBLOCK;
                          }
                          compute_indices[local_entries+1] = (((ofm1 *  BLOCKSIFM) + ifm1) * handle->desc.R * handle->desc.S *  IFMBLOCK *  handle->ofmblock + kj * handle->desc.S *  IFMBLOCK *  handle->ofmblock + ki * IFMBLOCK *  handle->ofmblock)  * handle->weight_copies;
                          compute_indices[local_entries+2] = ((((((img *  BLOCKSOFM) + ofm1) *  handle->ofhp) + oj_) * (handle->ofwp + handle->output_lp_padding)) + oi_) *  handle->ofmblock;
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
      }
    }
  }

  compute_indices[local_entries] = 0;
  compute_indices[local_entries+1] = 0;
  compute_indices[local_entries+2] = 0;
}

