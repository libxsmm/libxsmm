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

#ifndef GENERATOR_SPGEMM_CSR_ASPARSE_SOA_H
#define GENERATOR_SPGEMM_CSR_ASPARSE_SOA_H

#include "generator_common.h"
#include <libxsmm_generator.h>

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csr_asparse_soa( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                               const char*                     i_arch,
                                               const unsigned int*             i_row_idx,
                                               const unsigned int*             i_column_idx,
                                               const void*                     i_values );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csr_asparse_soa_avx512( libxsmm_generated_code*         io_generated_code,
                                                      const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                      const char*                     i_arch,
                                                      const unsigned int*             i_row_idx,
                                                      const unsigned int*             i_column_idx,
                                                      const void*                     i_values );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csr_asparse_soa_m_loop_avx512( libxsmm_generated_code*            io_generated_code,
                                                             const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                             libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                             const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                             const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                             const char*                        i_arch,
                                                             const unsigned int*                i_row_idx,
                                                             const unsigned int*                i_column_idx,
                                                             const void*                        i_values,
                                                             const unsigned int                 i_soa_width,
                                                             const unsigned int                 i_gen_m_trips,
                                                             const unsigned int                 i_a_is_dense,
                                                             const unsigned int                 i_num_c_cols );

#endif /* GENERATOR_SPGEMM_CSR_ASPARSE_SOA_H */

