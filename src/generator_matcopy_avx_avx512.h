/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_MATCOPY_AVX_AVX512_H
#define GENERATOR_MATCOPY_AVX_AVX512_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_header_m_loop( libxsmm_generated_code*                   io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_matcopy_kernel_config*      i_kernel_config,
                                              const unsigned int                        i_gp_reg_m_loop );

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_footer_m_loop( libxsmm_generated_code*                       io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_matcopy_kernel_config*          i_kernel_config,
                                              const unsigned int                            i_gp_reg_m_loop,
                                              const unsigned int                            i_m );

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_header_n_loop( libxsmm_generated_code*                   io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_matcopy_kernel_config*      i_kernel_config,
                                              const unsigned int                        i_gp_reg_n_loop );

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_footer_n_loop( libxsmm_generated_code*                       io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_matcopy_kernel_config*          i_kernel_config,
                                              const unsigned int                            i_gp_reg_n_loop,
                                              const unsigned int                            i_n );

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_avx_avx512_kernel( libxsmm_generated_code*             io_generated_code,
                                                  const libxsmm_mcopy_descriptor*   i_trans_desc,
                                                  const char*                         i_arch );

#endif /* GENERATOR_MATCOPY_AVX_AVX512_H */

