/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

int img, my_img_start, my_img_end, ofmb, ifmb, ojb, ofm1, ifm1, ifm2, ofm2, oj, oi, ii, ij, kj, ki, ind, j_br, img_br, img_block_size = 1, my_ofm_start, my_ofm_end, my_ifm_start, my_ifm_end, block_ofm, block_ifm;
/* computing first logical thread */
const int ltid = tid - start_thread;
int LDA = handle->ofmblock;
int LDB = (handle->upd_pack_input == 1) ? handle->ifmblock : handle->desc.v * handle->ifmblock;
int LDC = handle->ofmblock;
int l_flags = LIBXSMM_GEMM_FLAGS('N', 'T');
element_output_type *const out = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, (const element_output_type*)out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, const element_input_type, input, (const element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, weight_global, (element_filter_type*)handle->grad_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
element_filter_type *weight_ptr = (element_filter_type*)handle->scratch7 + ltid * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S;
LIBXSMM_VLA_DECL(6, element_filter_type, weight_private, (element_filter_type*)weight_ptr, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
int prefetch_mode = (handle->desc.u == 2 || (handle->desc.R == 3 && handle->ofw == 7) ) ? libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE) : libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BL1);

/* Batch reduce related variables */
const element_output_type *A_ptrs[1024];
const element_input_type  *B_ptrs[1024];
unsigned long long n_blocks;

libxsmm_barrier_init(handle->barrier, ltid);

if (handle->upd_use_batchreduce == 0 && handle->upd_linearized_tasklist == 0) {
  /* Parallelize over minibatch */
  const int img_work = handle->desc.N;
  const int img_chunksize = (img_work % handle->desc.threads == 0) ? (img_work / handle->desc.threads) : (img_work / handle->desc.threads) + 1;
  const float beta = ((img_chunksize == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.0 : 1.0;
  gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb * handle->upd_ofh_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

  my_img_start = (ltid * img_chunksize < img_work) ? (ltid * img_chunksize) : img_work;
  my_img_end = ((ltid + 1) * img_chunksize < img_work) ? ((ltid + 1) * img_chunksize) : img_work;

  if (handle->avoid_init_weights == 0) {
    memset(weight_ptr, 0, handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S * sizeof(element_filter_type));
  }

  if (handle->upd_loop_order == 0) {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
        for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj+= handle->upd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->upd_ofw_rb) {
                    for (kj = 0; kj < handle->desc.R; ++kj) {
                      for (ki = 0; ki < handle->desc.S; ++ki) {
                        ii = oi * handle->desc.u + ki;
                        ij = oj * handle->desc.v + kj;
                        gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, weight_private, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
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
  if (handle->upd_loop_order == 1) {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
        for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj+= handle->upd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->upd_ofw_rb) {
                    for (kj = 0; kj < handle->desc.R; ++kj) {
                      for (ki = 0; ki < handle->desc.S; ++ki) {
                        ii = oi * handle->desc.u + ki;
                        ij = oj * handle->desc.v + kj;
                        gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, weight_private, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
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
  if (handle->upd_linearized_tasklist == 1) {
    /* Amount of work when using linearized view of tasks */
    const int work = handle->desc.R * handle->desc.S * handle->blocksofm * handle->blocksifm;
    const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
    const int work_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const int work_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    int work_item;
    int Cb = handle->blocksifm;
    int Kb = handle->blocksofm;
    int R = handle->desc.R;
    int S = handle->desc.S;

    if (handle->upd_avoid_rim_fmas == 0) {
      const int IFH = handle->ifhp/handle->desc.u;
      const int IFW = handle->ifwp/handle->desc.v;
      element_input_type *input_ptr_base = (handle->upd_pack_input == 1) ? (element_input_type*)handle->scratch1 + handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock * handle->desc.R * handle->desc.S : (element_input_type*)handle->reg_input->data;
      LIBXSMM_VLA_DECL(5, element_input_type, input_use, (element_input_type*)input_ptr_base, handle->blocksifm, IFH, IFW, handle->ifmblock);
      const float beta = ((handle->desc.N == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.0 : 1.0;
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb * handle->upd_ofh_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

      /* If requested, pack input to avoid strided accesses */
      if (handle->upd_pack_input == 1) {
        LIBXSMM_VLA_DECL(5, element_input_type, input_src, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        const int img_chunk = (handle->desc.N % handle->desc.threads == 0) ? handle->desc.N/handle->desc.threads : (handle->desc.N/handle->desc.threads) + 1;
        const int img_copy_start = LIBXSMM_MIN(ltid*img_chunk, handle->desc.N);
        const int img_copy_end = LIBXSMM_MIN((ltid+1)*img_chunk, handle->desc.N);

        for (img = img_copy_start; img < img_copy_end; img++) {
          for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
            for (oj = 0; oj < handle->ofh; oj++) {
              for (oi = 0; oi < handle->ofw; oi++) {
                ij = oj * handle->desc.u;
                ii = oi * handle->desc.v;
                LIBXSMM_PRAGMA_SIMD
                  for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                    LIBXSMM_VLA_ACCESS(5, input_use, img, ifm1, oj, oi, ifm2, handle->blocksifm, IFH, IFW, handle->ifmblock) = LIBXSMM_VLA_ACCESS(5, input_src, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                  }
              }
            }
          }
        }
        libxsmm_barrier_wait(handle->barrier, ltid);
      }

      /* Initialize weights to zero */
      if (!((handle->desc.N == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw))) {
        for (work_item = work_begin; work_item < work_end; work_item++) {
          ofm1 = work_item/(Cb*R*S);
          ifm1 = (work_item%(Cb*R*S))/(R*S);
          kj = ((work_item%(Cb*R*S))%(R*S))/S;
          ki = ((work_item%(Cb*R*S))%(R*S))%S;

          for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
            LIBXSMM_PRAGMA_SIMD
              for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
                LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) = (element_filter_type)0;
              }
          }
        }
      }

      for (img = 0; img < handle->desc.N; img++) {
        for (work_item = work_begin; work_item < work_end; work_item++) {
          ofm1 = work_item/(Cb*R*S);
          ifm1 = (work_item%(Cb*R*S))/(R*S);
          kj = ((work_item%(Cb*R*S))%(R*S))/S;
          ki = ((work_item%(Cb*R*S))%(R*S))%S;

          oi = 0;
          oj = 0;
          ii = oi + ki;
          ij = oj + kj;
          gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
              &LIBXSMM_VLA_ACCESS(5, input_use, img, ifm1, ij, ii, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
              &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
        }
      }
    } else {
      const float beta = ((handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.0 : 1.0;
      gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
      gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb-1, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

      for (work_item = work_begin; work_item < work_end; work_item++) {
        ofm1 = work_item/(Cb*R*S);
        ifm1 = (work_item%(Cb*R*S))/(R*S);
        kj = ((work_item%(Cb*R*S))%(R*S))/S;
        ki = ((work_item%(Cb*R*S))%(R*S))%S;
        oi = 0;
        oj = 0;
        ii = oi * handle->desc.u + ki;
        ij = oj * handle->desc.v + kj;
        img = 0;
        img_block_size = handle->desc.N;

        if (kj == 0) {
          ind = 0;
          for (img_br = 0; img_br < img_block_size; img_br++) {
            for (j_br = 1; j_br < handle->upd_ofh_rb; j_br++) {
              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              ind++;
            }
          }
          n_blocks = ind;
          br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
        } else if (ki == 0) {
          ind = 0;
          for (img_br = 0; img_br < img_block_size; img_br++) {
            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi + 1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br, ii + 1, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              ind++;
            }
          }
          n_blocks = ind;
          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
        } else if (oi == handle->ofw-handle->fwd_ofw_rb  && ki == handle->desc.S-1) {
          ind = 0;
          for (img_br = 0; img_br < img_block_size; img_br++) {
            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              ind++;
            }
          }
          n_blocks = ind;
          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
        } else {
          if (kj == handle->desc.R-1) {
            ind = 0;
            for (img_br = 0; img_br < img_block_size; img_br++) {
              for (j_br = 0; j_br < handle->upd_ofh_rb-1; j_br++) {
                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                ind++;
              }
            }
            n_blocks = ind;
            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
          } else {
            ind = 0;
            for (img_br = 0; img_br < img_block_size; img_br++) {
              for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                ind++;
              }
            }
            n_blocks = ind;
            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
          }
        }
      }
    }
  } else {
    /* Here we are using batch-reduce kernel and hybrid minibatch/FM parallelization */
    int group_size = (handle->desc.threads+handle->weight_copies-1)/handle->weight_copies;
    int tile_id = ltid/group_size;
    /* FIXME: Hardcoed logic for N=27  */
    if (handle->desc.threads == 27 && handle->desc.N == 27 && handle->ofw == 14 && handle->desc.R == 1 && handle->desc.u == 1) {
      if (ltid >=24) {
        group_size = 3;
      }
    }
    int tiles = handle->weight_copies;
    int img_per_tile = (handle->desc.N+tiles-1)/tiles;
    int my_in_tile_id = ltid % group_size;
    int ifms_per_thread = (handle->blocksifm+group_size-1)/group_size;
    int ofms_per_thread = (handle->blocksofm+group_size-1)/group_size;
    int my_R_start = 0;
    int my_R_end = handle->desc.R;
    const float beta = ((handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.0 : 1.0;
    gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
    element_filter_type *weight_ptr_group = (handle->weight_copies > 1) ? (element_filter_type*)handle->scratch7 + tile_id * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S : (element_filter_type*)handle->grad_filter->data;
    LIBXSMM_VLA_DECL(6, element_filter_type, weight_private_group, (element_filter_type*)weight_ptr_group, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
    my_img_start = LIBXSMM_MIN( tile_id * img_per_tile, handle->desc.N);
    my_img_end = LIBXSMM_MIN( (tile_id+1) * img_per_tile, handle->desc.N);
    my_ifm_start = LIBXSMM_MIN( my_in_tile_id * ifms_per_thread, handle->blocksifm  );
    my_ifm_end = LIBXSMM_MIN( (my_in_tile_id+1) * ifms_per_thread, handle->blocksifm  );
    my_ofm_start = 0;
    my_ofm_end = handle->blocksofm;
    /* FIXME: Hardcoed logic for N=27  */
    if (handle->desc.threads == 27 && handle->desc.N == 27 && handle->desc.C == 256 && handle->desc.K == 1024 && handle->ofh == 14 && handle->desc.u == 1) {
      my_ofm_start = LIBXSMM_MIN( my_in_tile_id * ofms_per_thread, handle->blocksofm  );
      my_ofm_end = LIBXSMM_MIN( (my_in_tile_id+1) * ofms_per_thread, handle->blocksofm  );
      my_ifm_start = 0;
      my_ifm_end = handle->blocksifm;
    }
    if (handle->desc.threads == 27 && handle->desc.N == 27 && handle->desc.R == 3 && handle->desc.S == 3 && handle->ofh == 14) {
      int r_per_tile = (handle->desc.R+group_size-1)/group_size;
      my_ifm_start = 0;
      my_ifm_end = handle->blocksifm;
      my_ofm_start = 0;
      my_ofm_end = handle->blocksofm;
      my_R_start = LIBXSMM_MIN( my_in_tile_id * r_per_tile, handle->desc.R );
      my_R_end = LIBXSMM_MIN( (my_in_tile_id+1) * r_per_tile, handle->desc.R );
    }
    block_ofm = my_ofm_end-my_ofm_start+1;
    block_ifm = my_ifm_end-my_ifm_start+1;
    //block_ofm = handle->block_upd_ofm;
    //block_ifm = handle->block_upd_ifm;
    img_block_size = my_img_end - my_img_start;

    if (handle->upd_loop_order == 0) {
      for (img = my_img_start; img < my_img_end; img += img_block_size) {
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
          for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj+= handle->upd_ofh_rb) {
                    for (oi = 0; oi < handle->ofw; oi += handle->upd_ofw_rb) {
                      for (kj = my_R_start; kj < my_R_end; ++kj) {
                        for (ki = 0; ki < handle->desc.S; ++ki) {
                          ii = oi * handle->desc.u + ki;
                          ij = oj * handle->desc.v + kj;
                          ind = 0;
                          for (img_br = 0; img_br < img_block_size; img_br++) {
                            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br * handle->desc.u, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                              ind++;
                            }
                          }
                          n_blocks = ind;
                          br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
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
      for (img = my_img_start; img < my_img_end; img += img_block_size) {
        for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
          for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
                for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj+= handle->upd_ofh_rb) {
                    for (oi = 0; oi < handle->ofw; oi += handle->upd_ofw_rb) {
                      for (kj = my_R_start; kj < my_R_end; ++kj) {
                        for (ki = 0; ki < handle->desc.S; ++ki) {
                          ii = oi * handle->desc.u + ki;
                          ij = oj * handle->desc.v + kj;
                          ind = 0;
                          for (img_br = 0; img_br < img_block_size; img_br++) {
                            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br * handle->desc.u, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                              ind++;
                            }
                          }
                          n_blocks = ind;
                          br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
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

if (handle->weight_copies > 1) {
  /* reduce work-related variables  */
  const int reduce_work = handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S * (handle->ofmblock/16) * handle->ifmblock;
  const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
  const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
  const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;

  /* Perform reduction here  */
  libxsmm_barrier_wait(handle->barrier, ltid);

  for ( ij = reduce_thr_begin; ij < reduce_thr_end; ij++ ) {
    element_filter_type *weight_ptr = (element_filter_type*) handle->grad_filter->data;
    __m512 weight_sum = _mm512_setzero_ps();
    for ( ii = 0; ii < handle->weight_copies; ii++ ) {
      element_filter_type *weight_ptr_src = (element_filter_type*)handle->scratch7 + ii * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S + ij * 16;
      weight_sum = _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(weight_ptr_src));
    }
    _mm512_store_ps(&weight_ptr[ij*16], weight_sum);
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);


