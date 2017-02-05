/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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

#ifndef GENERATOR_GEMM_AVX512_MICROKERNEL_H
#define GENERATOR_GEMM_AVX512_MICROKERNEL_H

#include "generator_common.h"
#include "generator_gemm_common.h"

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_avx512_microkernel( libxsmm_generated_code*             io_generated_code,
                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                 const unsigned int                 i_n_blocking,
                                                 const unsigned int                 i_k_blocking );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_avx512_microkernel_qfma( libxsmm_generated_code*             io_generated_code,
                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                     const unsigned int                 i_n_blocking,
                                                     const unsigned int                 i_k_blocking );

#if 0
LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_avx512_microkernel_k_large( libxsmm_generated_code*             io_generated_code,
                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                         const unsigned int                 i_n_blocking,
                                                         const unsigned int                 i_k_blocking );
#endif

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_avx512_microkernel_k_large_n_nine( libxsmm_generated_code*             io_generated_code,
                                                                const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                const unsigned int                 i_k_blocking );

LIBXSMM_INTERNAL_API
unsigned int libxsmm_generator_gemm_avx512_kernel_kloop( libxsmm_generated_code*             io_generated_code,
                                                          libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                          const char*                        i_arch,
                                                          unsigned int                       i_n_blocking );

#endif /* GENERATOR_GEMM_AVX512_MICROKERNEL_H */

