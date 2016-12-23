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

#ifndef GENERATOR_GEMM_SSE3_AVX_AVX2_AVX512_H
#define GENERATOR_GEMM_SSE3_AVX_AVX2_AVX512_H

#include "generator_common.h"
#include "generator_gemm_common.h"

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_sse3_avx_avx2_avx512_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                         const char*                    i_arch );

LIBXSMM_INTERNAL_API
unsigned int libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_inital_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                                const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                                const char*                    i_arch );

LIBXSMM_INTERNAL_API
unsigned int libxsmm_generator_gemm_sse3_avx_avx2_avx512_update_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                            const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                            const char*                    i_arch,
                                                                            const unsigned int             i_current_m_blocking );

#endif /* GENERATOR_GEMM_SSE3_AVX_AVX2_AVX512_H */

