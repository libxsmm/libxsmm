/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evanelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_MATELTWISE_TRANSFORM_COMMON_X86_H
#define GENERATOR_MATELTWISE_TRANSFORM_COMMON_X86_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_unpack_network_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                                 const char              i_vector_name,
                                                                 const unsigned char*    i_in_idx,
                                                                 const unsigned int      i_vec_reg_src_start,
                                                                 const unsigned int      i_vec_reg_dst_start,
                                                                 const unsigned int      i_out_offset,
                                                                 const unsigned int      i_even_instr,
                                                                 const unsigned int      i_odd_instr,
                                                                 const unsigned int      i_ways );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_full_load_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                            const char              i_vector_name,
                                                            const unsigned int      i_gp_reg_in,
                                                            const unsigned int      i_vec_reg_dst_start,
                                                            const unsigned int      i_ld,
                                                            const unsigned int      i_ld_instr,
                                                            const unsigned int      i_ways,
                                                            const unsigned int      i_valid_ways,
                                                            const unsigned int      i_use_masking,
                                                            const unsigned int      i_mask_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_full_store_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_gp_reg_out,
                                                             const unsigned int      i_vec_reg_src_start,
                                                             const unsigned int      i_ld,
                                                             const unsigned int      i_st_instr,
                                                             const unsigned int      i_use_masking,
                                                             const unsigned int      i_mask_reg,
                                                             const unsigned int      i_ways );

#endif /* GENERATOR_MATELTWISE_TRANSFORM_COMMON_X86_H */

