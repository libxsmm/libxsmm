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
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_HELPER_RECURRENT_H
#define LIBXSMM_HELPER_RECURRENT_H

#if !defined(FTYPE)
# define FTYPE float /* TODO: undefine/remove generic symbol names as header-only interfers with user's code */
#endif

void libxsmm_internal_matinit(int seed, FTYPE *dst, 
  libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld, double scale);
void libxsmm_internal_matrix_add(libxsmm_blasint size, FTYPE *a, FTYPE *b, FTYPE *c);
void libxsmm_internal_matrix_eltwise_mult(libxsmm_blasint size, FTYPE *a, FTYPE *b, FTYPE *c);
void libxsmm_internal_matrix_sigmoid(libxsmm_blasint size, FTYPE *src, FTYPE *dst);
void libxsmm_internal_matrix_tanh(libxsmm_blasint size, FTYPE *src, FTYPE *dst);
void libxsmm_internal_matrix_relu(libxsmm_blasint size, FTYPE *src, FTYPE *dst);
void libxsmm_internal_matrix_sigmoid_inverse(libxsmm_blasint size, FTYPE *src, FTYPE *dst);
void libxsmm_internal_matrix_tanh_inverse(libxsmm_blasint size, FTYPE *src, FTYPE *dst);
void libxsmm_internal_matrix_relu_inverse(libxsmm_blasint size, FTYPE *src, FTYPE *dst, FTYPE *input);
void libxsmm_internal_matrix_transpose(libxsmm_blasint rows, libxsmm_blasint cols, FTYPE *src, FTYPE *dst);
void libxsmm_internal_matrix_copy(libxsmm_blasint size, FTYPE *src, FTYPE *dst);
void libxsmm_internal_matrix_complement(libxsmm_blasint size, FTYPE *src, FTYPE *dst);
void libxsmm_internal_matrix_complement_square(libxsmm_blasint size, FTYPE *src, FTYPE *dst);
void libxsmm_internal_recursive_step(libxsmm_bgemm_handle* handle, FTYPE* u, FTYPE* h, FTYPE* op1, FTYPE *op2,
  FTYPE *temp, FTYPE *dst, int act, libxsmm_blasint size, int tid, int nthreads);

#endif

