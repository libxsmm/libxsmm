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
#ifndef GENERATOR_GEMM_AMX_MICROKERNEL_EMU_H
#define GENERATOR_GEMM_AMX_MICROKERNEL_EMU_H

#include "generator_common.h"
#include "generator_gemm_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_paired_tilestore_emu( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile0,
    int                                tile1,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_single_tilestore_emu( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_microkernel_emu( libxsmm_generated_code*            io_generated_code,
                                                     libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                     libxsmm_micro_kernel_config*       i_micro_kernel_config,
                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                     libxsmm_blocking_info_t*           n_blocking_info,
                                                     libxsmm_blocking_info_t*           m_blocking_info,
                                                     long long                          offset_A,
                                                     long long                          offset_B,
                                                     unsigned int                       is_last_k,
                                                     long long                          i_brgemm_loop,
                                                     unsigned int                       fully_unrolled_brloop  );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_kloop_emu( libxsmm_generated_code*            io_generated_code,
                                                      libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                      libxsmm_micro_kernel_config*       i_micro_kernel_config,
                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                      libxsmm_blocking_info_t*           n_blocking_info,
                                                      libxsmm_blocking_info_t*           m_blocking_info,
                                                      long long                          A_offs,
                                                      long long                          B_offs,
                                                      unsigned int                       fully_unrolled_brloop );

#endif /* GENERATOR_GEMM_AMX_MICROKERNEL_EMU_H */

