/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef LIBXSMM_GENERATOR_H
#define LIBXSMM_GENERATOR_H

void libxsmm_generator_dense_kernel(char**             io_generated_code,
                                    const unsigned int i_m,
                                    const unsigned int i_n,
                                    const unsigned int i_k,
                                    const unsigned int i_lda,
                                    const unsigned int i_ldb,
                                    const unsigned int i_ldc, 
                                    const int          i_alpha,
                                    const int          i_beta,
                                    const unsigned int i_aligned_a,
                                    const unsigned int i_aligned_c,
                                    const char*        i_arch,
                                    const char*        i_prefetch,
                                    const unsigned int i_single_precision);

void libxsmm_generator_dense(const char*        i_file_out,
                             const char*        i_routine_name,
                             const unsigned int i_m,
                             const unsigned int i_n,
                             const unsigned int i_k,
                             const unsigned int i_lda,
                             const unsigned int i_ldb,
                             const unsigned int i_ldc, 
                             const int          i_alpha,
                             const int          i_beta,
                             const unsigned int i_aligned_a,
                             const unsigned int i_aligned_c,
                             const char*        i_arch,
                             const char*        i_prefetch,
                             const unsigned int i_single_precision);

void libxsmm_generator_sparse(const char*        i_file_out,
                              const char*        i_routine_name,
                              const unsigned int i_m,
                              const unsigned int i_n,
                              const unsigned int i_k,
                              const unsigned int i_lda,
                              const unsigned int i_ldb,
                              const unsigned int i_ldc, 
                              const int          i_alpha,
                              const int          i_beta,
                              const unsigned int i_aligned_a,
                              const unsigned int i_aligned_c,
                              const char*        i_arch,
                              const char*        i_prefetch,
                              const unsigned int i_single_precision,
                              const char*        i_file_in);

#endif /* LIBXSMM_GENERATOR_H */

