/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

int img, my_img_start, my_img_end, ofmb, ifmb, ojb, ofm1, ifm1, ifm2 = 0, ofm2 = 0, oj, oi, ii, ij, kj, ki, ind, j_br, img_br, img_block_size = 1, my_ofm_start, my_ofm_end, my_ifm_start, my_ifm_end, block_ofm, block_ifm;
/* computing first logical thread */
const int ltid = tid - start_thread;
libxsmm_blasint LDA = handle->blocksofm * handle->ofmblock;
libxsmm_blasint LDB = (handle->upd_pack_input == 1) ? handle->blocksifm * handle->ifmblock : handle->desc.v * handle->blocksifm * handle->ifmblock;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
libxsmm_blasint LDC = handle->ofmblock;
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
libxsmm_blasint LDC = handle->blocksofm * handle->ofmblock;
#endif
int l_flags = LIBXSMM_GEMM_FLAGS('N', 'T');
element_output_type *const out = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->blocksofm * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, (const element_output_type*)out, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
const int IFWP = (handle->upd_padding_copy == 1) ? handle->ifwp + 2*handle->desc.pad_w :  handle->ifwp;
const int IFHP = (handle->upd_padding_copy == 1) ? handle->ifhp + 2*handle->desc.pad_h :  handle->ifhp;
element_input_type *input_ptr_to_use = (handle->upd_padding_copy == 1) ? (element_input_type*) ((char*)handle->scratch + handle->upd_packing_padding_scratch_offset) : (element_input_type*)handle->reg_input->data;
LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) input_ptr_to_use, IFHP, IFWP, handle->blocksifm, handle->ifmblock);
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
LIBXSMM_VLA_DECL(6, element_filter_type, weight_global, (element_filter_type*)handle->grad_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
LIBXSMM_VLA_DECL(6, element_filter_type, weight_global, (element_filter_type*)handle->grad_filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
#endif
element_filter_type *weight_ptr = (handle->weight_copies == 1) ? (element_filter_type*)handle->grad_filter->data : (element_filter_type*) ((char*)handle->scratch + handle->upd_filter_scratch_offset) + ltid * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
LIBXSMM_VLA_DECL(6, element_filter_type, weight_private, (element_filter_type*)weight_ptr, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
LIBXSMM_VLA_DECL(6, element_filter_type, weight_private, (element_filter_type*)weight_ptr, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
#endif
int prefetch_mode = (handle->desc.u == 2 || (handle->desc.R == 3 && handle->ofw == 7) ) ? libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE) : libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BL1);

/* Batch reduce related variables */
const element_output_type *A_ptrs[1024];
const element_input_type  *B_ptrs[1024];
unsigned long long n_blocks;

int brgemm_pf_oob = 0;
const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
if ( 0 == env_brgemm_pf_oob ) {
} else {
  brgemm_pf_oob = atoi(env_brgemm_pf_oob);
}
if (brgemm_pf_oob > 0) {
  prefetch_mode = prefetch_mode | libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
}

libxsmm_barrier_init(handle->barrier, ltid);

/* physical pad input */
if (handle->upd_padding_copy == 1) {
  LIBXSMM_VLA_DECL(5, element_input_type, input_src, (element_input_type*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
  int imgpt = LIBXSMM_UPDIV(handle->desc.N, handle->desc.threads);
  my_img_start = LIBXSMM_MIN(ltid * imgpt, handle->desc.N);
  my_img_end = LIBXSMM_MIN((ltid+1) * imgpt, handle->desc.N);
  my_ifm_start = 0;
  my_ifm_end = handle->blocksifm;

  for (img = my_img_start; img < my_img_end; img++) {
    for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
      /* copy the inner part */
      for (ij = 0; ij < handle->ifhp+(2*handle->desc.pad_h); ij++) {
        for (ii = 0; ii < handle->ifwp+(2*handle->desc.pad_w); ii++) {
          if ( (ij >= handle->desc.pad_h) && (ii >= handle->desc.pad_w) && (ij < handle->ifhp+handle->desc.pad_h) && (ii < handle->ifwp+handle->desc.pad_w) ) {
            LIBXSMM_PRAGMA_SIMD
              for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                LIBXSMM_VLA_ACCESS(5,  input, img, ij, ii, ifm1, ifm2, IFHP, IFWP, handle->blocksifm, handle->ifmblock) =
                  LIBXSMM_VLA_ACCESS(5,  input_src, img, ij-handle->desc.pad_h, ii-handle->desc.pad_w, ifm1, ifm2, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
              }
          } else {
            LIBXSMM_PRAGMA_SIMD
              for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                LIBXSMM_VLA_ACCESS(5,  input, img, ij, ii, ifm1, ifm2, IFHP, IFWP, handle->blocksifm, handle->ifmblock) = (element_input_type)0;
              }
          }
        }
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, ltid);
}

if (handle->upd_use_batchreduce == 0 && handle->upd_linearized_tasklist == 0) {
  /* Parallelize over minibatch */
  const int img_work = handle->desc.N;
  const int img_chunksize = (img_work % handle->desc.threads == 0) ? (img_work / handle->desc.threads) : (img_work / handle->desc.threads) + 1;
  const float beta = ((img_chunksize == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.f : 1.f;
  gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb * handle->upd_ofh_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

  my_img_start = (ltid * img_chunksize < img_work) ? (ltid * img_chunksize) : img_work;
  my_img_end = ((ltid + 1) * img_chunksize < img_work) ? ((ltid + 1) * img_chunksize) : img_work;

  if (!((img_chunksize == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw))) {
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
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
                        gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS(5, input, img, ij, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, weight_private, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
                        gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS(5, input, img, ij, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, weight_private, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock) );
#endif
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
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
                        gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS(5, input, img, ij, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, weight_private, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
                        gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS(5, input, img, ij, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, weight_private, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock) );
#endif
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
#if 0
    int Kb = handle->blocksofm;
#endif
    int R = handle->desc.R;
    int S = handle->desc.S;

    if (handle->upd_avoid_rim_fmas == 0) {
      const int IFH = (handle->upd_pack_input == 1) ? handle->ifhp/handle->desc.u : IFHP;
      const int IFW = (handle->upd_pack_input == 1) ? handle->ifwp/handle->desc.v : IFWP;
      element_input_type *input_ptr_base = (handle->upd_pack_input == 1) ? (element_input_type*)((char*)handle->scratch + handle->upd_packing_padding_scratch_offset) : (element_input_type*)input_ptr_to_use;
      LIBXSMM_VLA_DECL(5, element_input_type, input_use, (element_input_type*)input_ptr_base, IFH, IFW, handle->blocksifm, handle->ifmblock);
      const float beta = ((handle->desc.N == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.f : 1.f;
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb * handle->upd_ofh_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

      /* If requested, pack input to avoid strided accesses */
      if (handle->upd_pack_input == 1) {
        LIBXSMM_VLA_DECL(5, element_input_type, input_src, (element_input_type*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
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
                    LIBXSMM_VLA_ACCESS(5, input_use, img, oj, oi, ifm1, ifm2, IFH, IFW, handle->blocksifm, handle->ifmblock) = LIBXSMM_VLA_ACCESS(5, input_src, img, ij, ii, ifm1, ifm2, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
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
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
                LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) = (element_filter_type)0;
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
                LIBXSMM_VLA_ACCESS(6, weight_global, kj, ki, ifm1, ifm2, ofm1, ofm2, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock) = (element_filter_type)0;
#endif
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
          ii = ki;
          for (oj = 0; oj < handle->ofh; oj += handle->upd_ofh_rb) {
            ij = oj * handle->desc.u + kj;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
            gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock),
                &LIBXSMM_VLA_ACCESS(5, input_use, img, ij, ii, ifm1, 0, IFH, IFW, handle->blocksifm, handle->ifmblock),
                &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
            gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock),
                &LIBXSMM_VLA_ACCESS(5, input_use, img, ij, ii, ifm1, 0, IFH, IFW, handle->blocksifm, handle->ifmblock),
                &LIBXSMM_VLA_ACCESS(6, weight_global, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock) );
#endif
          }
        }
      }
    } else {
      const float beta = ((handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.f : 1.f;
      gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
      gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb-1, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

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
              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, oj + j_br, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ij + j_br * handle->desc.u, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock);
              ind++;
            }
          }
          n_blocks = ind;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
          br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
          br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), &n_blocks);
#endif
        } else if (ki == 0) {
          ind = 0;
          for (img_br = 0; img_br < img_block_size; img_br++) {
            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, oj + j_br, oi + 1, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ij + j_br * handle->desc.u, ii + 1, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock);
              ind++;
            }
          }
          n_blocks = ind;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), &n_blocks);
#endif
        } else if (oi == handle->ofw-handle->fwd_ofw_rb  && ki == handle->desc.S-1) {
          ind = 0;
          for (img_br = 0; img_br < img_block_size; img_br++) {
            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, oj + j_br, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ij + j_br * handle->desc.u, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock);
              ind++;
            }
          }
          n_blocks = ind;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), &n_blocks);
#endif
        } else {
          if (kj == handle->desc.R-1) {
            ind = 0;
            for (img_br = 0; img_br < img_block_size; img_br++) {
              for (j_br = 0; j_br < handle->upd_ofh_rb-1; j_br++) {
                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, oj + j_br, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ij + j_br * handle->desc.u, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock);
                ind++;
              }
            }
            n_blocks = ind;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), &n_blocks);
#endif
          } else {
            ind = 0;
            for (img_br = 0; img_br < img_block_size; img_br++) {
              for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, oj + j_br, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ij + j_br * handle->desc.u, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock);
                ind++;
              }
            }
            n_blocks = ind;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), &n_blocks);
#endif
          }
        }
      }
    }
  } else {
    /* Here we are using batch-reduce kernel and hybrid minibatch/FM parallelization */
    /* FIXME: Hardcoed logic for N=27  */
    int group_size = (handle->desc.threads == 27 && handle->desc.N == 27 && handle->ofw == 14 && handle->desc.R == 1 && handle->desc.u == 1 && ltid >= 24) ? 3 : LIBXSMM_UPDIV(handle->desc.threads, handle->weight_copies);
    int tile_id = ltid / LIBXSMM_UPDIV(handle->desc.threads, handle->weight_copies);
    int tiles = handle->weight_copies;
    int img_per_tile = LIBXSMM_UPDIV(handle->desc.N, tiles);
    int my_in_tile_id = ltid % group_size;
    int ifms_per_thread = LIBXSMM_UPDIV(handle->blocksifm, group_size);
    int ofms_per_thread = LIBXSMM_UPDIV(handle->blocksofm, group_size);
    int my_R_start = 0;
    int my_R_end = handle->desc.R;
    const float beta = ((handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.f : 1.f;
    gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
    const float beta_flat = 0.0;
    gemm_br_function br_gemm_kernel_flat = libxsmm_smmdispatch_reducebatch_addr(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb, &LDA, &LDB, &LDC, NULL, &beta_flat, &l_flags, &prefetch_mode);
    element_filter_type *weight_ptr_group = (handle->weight_copies > 1) ? (element_filter_type*)((char*)handle->scratch + handle->upd_filter_scratch_offset) + tile_id * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S : (element_filter_type*)handle->grad_filter->data;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
    LIBXSMM_VLA_DECL(6, element_filter_type, weight_private_group, (element_filter_type*)weight_ptr_group, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
    LIBXSMM_VLA_DECL(6, element_filter_type, weight_private_group, (element_filter_type*)weight_ptr_group, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
#endif
    my_img_start = LIBXSMM_MIN(tile_id * img_per_tile, handle->desc.N);
    my_img_end = LIBXSMM_MIN((tile_id+1) * img_per_tile, handle->desc.N);
    my_ifm_start = LIBXSMM_MIN(my_in_tile_id * ifms_per_thread, handle->blocksifm  );
    my_ifm_end = LIBXSMM_MIN((my_in_tile_id+1) * ifms_per_thread, handle->blocksifm  );
    my_ofm_start = 0;
    my_ofm_end = handle->blocksofm;
    /* FIXME: Hardcoed logic for N=27  */
    if (handle->desc.threads == 27 && handle->desc.N == 27 && handle->desc.C == 256 && handle->desc.K == 1024 && handle->ofh == 14 && handle->desc.u == 1) {
      my_ofm_start = LIBXSMM_MIN(my_in_tile_id * ofms_per_thread, handle->blocksofm);
      my_ofm_end = LIBXSMM_MIN((my_in_tile_id+1) * ofms_per_thread, handle->blocksofm);
      my_ifm_start = 0;
      my_ifm_end = handle->blocksifm;
    }
    if (handle->desc.threads == 27 && handle->desc.N == 27 && handle->desc.R == 3 && handle->desc.S == 3 && handle->ofh == 14) {
      int r_per_tile = LIBXSMM_UPDIV(handle->desc.R, group_size);
      my_ifm_start = 0;
      my_ifm_end = handle->blocksifm;
      my_ofm_start = 0;
      my_ofm_end = handle->blocksofm;
      my_R_start = LIBXSMM_MIN(my_in_tile_id * r_per_tile, handle->desc.R);
      my_R_end = LIBXSMM_MIN((my_in_tile_id+1) * r_per_tile, handle->desc.R);
    }
    block_ofm = my_ofm_end-my_ofm_start+1;
    block_ifm = my_ifm_end-my_ifm_start+1;
    img_block_size = my_img_end - my_img_start;

    if (handle->desc.N != handle->desc.threads) {
      /* Use "flat" parallelism + reduction */
      const int work = handle->desc.R * handle->desc.S * handle->blocksofm * handle->blocksifm * handle->desc.N;
      const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
      const int work_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
      const int work_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
      int work_item;
      int Cb = handle->blocksifm;
      int Kb = handle->blocksofm;
      int R = handle->desc.R;
      int S = handle->desc.S;
      const int IFH = (handle->upd_pack_input == 1) ? handle->ifhp/handle->desc.u : IFHP;
      const int IFW = (handle->upd_pack_input == 1) ? handle->ifwp/handle->desc.v : IFWP;
      element_input_type *input_ptr_base = (handle->upd_pack_input == 1) ? (element_input_type*)((char*)handle->scratch + handle->upd_packing_padding_scratch_offset) : (element_input_type*)input_ptr_to_use;
      LIBXSMM_VLA_DECL(5, element_input_type, input_use, (element_input_type*)input_ptr_base, IFH, IFW, handle->blocksifm, handle->ifmblock);

      /* If requested, pack input to avoid strided accesses */
      if (handle->upd_pack_input == 1) {
        LIBXSMM_VLA_DECL(5, element_input_type, input_src, (element_input_type*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
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
                    LIBXSMM_VLA_ACCESS(5, input_use, img, oj, oi, ifm1, ifm2, IFH, IFW, handle->blocksifm, handle->ifmblock) = LIBXSMM_VLA_ACCESS(5, input_src, img, ij, ii, ifm1, ifm2, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
                  }
              }
            }
          }
        }
        libxsmm_barrier_wait(handle->barrier, ltid);
      }

      /* Initialize weights to zero */
      if (handle->upd_ofw_rb != handle->ofw) {
        for (work_item = work_begin; work_item < work_end; work_item++) {
          img = work_item/(Cb*Kb*R*S);
          ofm1 = (work_item%(Cb*Kb*R*S))/(Cb*R*S);
          ifm1 = ((work_item%(Cb*Kb*R*S))%(Cb*R*S))/(R*S);
          kj = (((work_item%(Cb*Kb*R*S))%(Cb*R*S))%(R*S))/S;
          ki = (((work_item%(Cb*Kb*R*S))%(Cb*R*S))%(R*S))%S;
          {
            element_filter_type *weight_ptr_current = (handle->weight_copies > 1) ? (element_filter_type*)((char*)handle->scratch + handle->upd_filter_scratch_offset)+ img * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S : (element_filter_type*)handle->grad_filter->data;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
            LIBXSMM_VLA_DECL(6, element_filter_type, weight_current, (element_filter_type*)weight_ptr_current, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
            LIBXSMM_VLA_DECL(6, element_filter_type, weight_current, (element_filter_type*)weight_ptr_current, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
#endif
            for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
              LIBXSMM_PRAGMA_SIMD
                for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
                  LIBXSMM_VLA_ACCESS(6, weight_current, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) = (element_filter_type)0;
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
                  LIBXSMM_VLA_ACCESS(6, weight_current, kj, ki, ifm1, ifm2, ofm1, ofm2, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock) = (element_filter_type)0;
#endif
                }
            }
          }
        }
      }

      for (work_item = work_begin; work_item < work_end; work_item++) {
        img = work_item/(Cb*Kb*R*S);
        ofm1 = (work_item%(Cb*Kb*R*S))/(Cb*R*S);
        ifm1 = ((work_item%(Cb*Kb*R*S))%(Cb*R*S))/(R*S);
        kj = (((work_item%(Cb*Kb*R*S))%(Cb*R*S))%(R*S))/S;
        ki = (((work_item%(Cb*Kb*R*S))%(Cb*R*S))%(R*S))%S;
        ii = 0 + ki;
        ij = 0 + kj;
        oj = 0;
        oi = 0;
        {
          element_filter_type *weight_ptr_current = (handle->weight_copies > 1) ? (element_filter_type*)((char*)handle->scratch + handle->upd_filter_scratch_offset) + img * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S : (element_filter_type*)handle->grad_filter->data;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
          LIBXSMM_VLA_DECL(6, element_filter_type, weight_current, (element_filter_type*)weight_ptr_current, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
          LIBXSMM_VLA_DECL(6, element_filter_type, weight_current, (element_filter_type*)weight_ptr_current, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
#endif
          ind = 0;
          for (j_br = 0; j_br < handle->ofh; j_br++) {
            A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img , oj + j_br, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
            B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input_use, img, ij + j_br * handle->desc.u, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock);
            ind++;
          }
          n_blocks = ind;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
          br_gemm_kernel_flat(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_current, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
          br_gemm_kernel_flat(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_current, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), &n_blocks);
#endif
        }
      }
    } else {
      /* May need to initialized private weights to zero  */
      if (!((handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw))) {
        for (ofm1 = my_ofm_start; ofm1 < my_ofm_end; ofm1++ ) {
          for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
            for (kj = my_R_start; kj < my_R_end; ++kj) {
              for (ki = 0; ki < handle->desc.S; ++ki) {
                for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++ ) {
                  for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
                    LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) = (element_filter_type)0;
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
                    LIBXSMM_VLA_ACCESS(6, weight_private_group, kj, ki, ifm1, ifm2, ofm1, ofm2, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock) = (element_filter_type)0;
#endif
                  }
                }
              }
            }
          }
        }
      }

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
                                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, oj + j_br, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
                                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ij + j_br * handle->desc.u, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock);
                                ind++;
                              }
                            }
                            n_blocks = ind;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
                            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
                            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_private_group, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), &n_blocks);
#endif
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
                                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, oj + j_br, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
                                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ij + j_br * handle->desc.u, ii, ifm1, 0, IFHP, IFWP, handle->blocksifm, handle->ifmblock);
                                ind++;
                              }
                            }
                            n_blocks = ind;
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_CUSTOM)
                            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
#endif
#if defined(LIBXSMM_DNN_TPL_UPD_DIRECT_GENERIC_NHWC_RSCK)
                            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_private_group, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), &n_blocks);
#endif
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
}

if (handle->weight_copies > 1) {
  /* reduce work-related variables  */
  const int fm_blocking = (handle->ofmblock % 16 == 0) ? 16 : handle->ofmblock;
  const int reduce_work = handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S * (handle->ofmblock/fm_blocking) * handle->ifmblock;
  const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
  const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
  const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;

  /* Perform reduction here  */
  libxsmm_barrier_wait(handle->barrier, ltid);

  for ( ij = reduce_thr_begin; ij < reduce_thr_end; ij++ ) {
    element_filter_type *weight_ptr_glb = (element_filter_type*) handle->grad_filter->data;
#if 1
    float weight_sum[64];
    int wtcnt = 0;
    assert( handle->ofmblock <= 64 );

    LIBXSMM_PRAGMA_SIMD
      for ( wtcnt = 0; wtcnt < fm_blocking; ++wtcnt ) {
        weight_sum[wtcnt] = 0.0f;
      }

    for ( ii = 0; ii < handle->weight_copies; ii++ ) {
      element_filter_type *weight_ptr_src = (element_filter_type*)((char*)handle->scratch + handle->upd_filter_scratch_offset)+ ii * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S + ij * fm_blocking;
      LIBXSMM_PRAGMA_SIMD
        for ( wtcnt = 0; wtcnt < fm_blocking; ++wtcnt ) {
          weight_sum[wtcnt] += weight_ptr_src[wtcnt];
        }
    }

    LIBXSMM_PRAGMA_SIMD
      for ( wtcnt = 0; wtcnt < fm_blocking; ++wtcnt ) {
        weight_ptr_glb[(ij*fm_blocking) + wtcnt] = weight_sum[wtcnt];
      }
#else
    __m512 weight_sum = _mm512_setzero_ps();
    for ( ii = 0; ii < handle->weight_copies; ii++ ) {
      element_filter_type *weight_ptr_src = (element_filter_type*)handle->scratch7 + ii * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S + ij * 16;
      weight_sum = _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(weight_ptr_src));
    }
    _mm512_storeu_ps(&weight_ptr_glb[ij*16], weight_sum);
#endif
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

