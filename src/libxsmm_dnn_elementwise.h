/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_ELEMENTWISE_H
#define LIBXSMM_DNN_ELEMENTWISE_H

#include <libxsmm.h>

#if !defined(LIBXSMM_DNN_ELTWISE_FTYPE)
# define LIBXSMM_DNN_ELTWISE_FTYPE float
#endif


LIBXSMM_API_INTERN void libxsmm_internal_matrix_zero(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_add(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *a, LIBXSMM_DNN_ELTWISE_FTYPE *b, LIBXSMM_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_mult(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *a, LIBXSMM_DNN_ELTWISE_FTYPE *b, LIBXSMM_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_transpose(libxsmm_blasint rows, libxsmm_blasint cols, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_copy(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_square(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_1D_2D(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint bm, libxsmm_blasint bn, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);

LIBXSMM_API_INTERN void libxsmm_internal_matrix_zero_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_add_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_sub_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_copy_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_fma_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_add_colvector_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, LIBXSMM_DNN_ELTWISE_FTYPE *colv);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_colvector_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, LIBXSMM_DNN_ELTWISE_FTYPE *colv);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_colvector_const_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, LIBXSMM_DNN_ELTWISE_FTYPE *colv, LIBXSMM_DNN_ELTWISE_FTYPE const_bias);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_cvt_bf16_fp32_colvector_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, libxsmm_bfloat16 *colv);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_cvt_bf16_fp32_colvector_const_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, libxsmm_bfloat16 *colv, LIBXSMM_DNN_ELTWISE_FTYPE const_bias);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);

LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_inverse_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_inverse_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_square_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_rne_mask_fp32_bfp16_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, float* src, float* dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_rne_cvt_fp32_bfp16_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, float* src, libxsmm_bfloat16* dst);
LIBXSMM_API_INTERN void libxsmm_internal_matrix_cvt_bf16_fp32_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_bfloat16 *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst);
#endif /*LIBXSMM_DNN_ELEMENTWISE_H*/

