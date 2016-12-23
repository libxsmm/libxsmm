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

#ifndef GENERATOR_GEMM_COMMON_H
#define GENERATOR_GEMM_COMMON_H

#include "generator_common.h"

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_init_micro_kernel_config_fullvector( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                  const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                  const char*                    i_arch,
                                                                  const unsigned int             i_use_masking_a_c );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_init_micro_kernel_config_halfvector( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                  const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                  const char*                    i_arch,
                                                                  const unsigned int             i_use_masking_a_c );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_init_micro_kernel_config_scalar( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                              const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                              const char*                    i_arch,
                                                              const unsigned int             i_use_masking_a_c );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_add_flop_counter( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_header_kloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_k_blocking );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_footer_kloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_max_blocked_k,
                                           const unsigned int                 i_kloop_complete );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_header_nloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const unsigned int                 i_n_blocking );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_footer_nloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_n_blocking,
                                           const unsigned int                 i_n_done );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_header_mloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const unsigned int                 i_m_blocking );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_footer_mloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_m_done,
                                           const unsigned int                 i_k_unrolled );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_load_C( libxsmm_generated_code*             io_generated_code,
                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                     const unsigned int                 i_m_blocking,
                                     const unsigned int                 i_n_blocking );

LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_store_C( libxsmm_generated_code*             io_generated_code,
                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                      const unsigned int                 i_m_blocking,
                                      const unsigned int                 i_n_blocking );

#endif /* GENERATOR_GEMM_COMMON_H */
