/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Stefan Remke (Univ. Jena)
******************************************************************************/
#ifndef GENERATOR_GEMM_SME_H
#define GENERATOR_GEMM_SME_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kloop_sme_het( libxsmm_generated_code*            io_generated_code,
                                                   libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                   const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                   const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                   const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                   const unsigned int                 i_m_blocking,
                                                   const unsigned int                 i_n_blocking,
                                                   const unsigned int                 i_blocking_scheme,
                                                   const unsigned int                 i_trans_size );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kernel_sme_het_blocking( libxsmm_generated_code*        io_generated_code,
                                                             const libxsmm_gemm_descriptor* i_xgemm_desc );

#endif /* GENERATOR_GEMM_SME_H */
