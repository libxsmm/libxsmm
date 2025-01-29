/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_COMMON_RV64_H
#define GENERATOR_COMMON_RV64_H

#include "generator_common.h"
#include "generator_rv64_instructions.h"

LIBXSMM_API_INTERN
void libxsmm_generator_loop_header_rv64( libxsmm_generated_code*     io_generated_code,
                                         libxsmm_loop_label_tracker* io_loop_label_tracker,
                                         const unsigned int          i_gp_reg_loop_cnt,
                                         const unsigned int          i_trips );

LIBXSMM_API_INTERN
void libxsmm_generator_loop_footer_rv64( libxsmm_generated_code*     io_generated_code,
                                         libxsmm_loop_label_tracker* io_loop_label_tracker,
                                         const unsigned int          i_gp_reg_loop_cnt,
                                         const unsigned int          i_loop_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_bcastload_masked_vreg_rv64( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_gp_reg_addr,
                                                   const unsigned int      i_gp_reg_scratch,
                                                   const unsigned int      i_vec_reg,
                                                   const unsigned int      i_datatype_size,
                                                   const unsigned int      i_masked_elems,
                                                   const unsigned int      i_vlen,
                                                   const unsigned int      i_avlen,
                                                   const unsigned int      i_adv_gpr );

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_rv64_rvv( libxsmm_generated_code* io_generated_code,
                                                 const libxsmm_datatype  i_datatype,
                                                 const unsigned int      i_gp_reg_addr,
                                                 const unsigned int      i_gp_reg_scratch,
                                                 const unsigned int      i_vec_length,
                                                 const unsigned int      i_vec_reg_count,
                                                 const unsigned int      i_m_blocking,
                                                 const unsigned int      i_n_blocking,
                                                 const unsigned int      i_ld,
                                                 const unsigned int      i_zero );

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_rv64_rvv( libxsmm_generated_code* io_generated_code,
                                                  const libxsmm_datatype  i_datatype,
                                                  const unsigned int      i_gp_reg_addr,
                                                  const unsigned int      i_gp_reg_scratch,
                                                  const unsigned int      i_vec_length,
                                                  const unsigned int      i_vec_reg_count,
                                                  const unsigned int      i_m_blocking,
                                                  const unsigned int      i_n_blocking,
                                                  const unsigned int      i_ld,
                                                  const libxsmm_datatype  i_inp_datatype,
                                                  const unsigned int      i_aux_gp_reg,
                                                  const unsigned int      i_reduce_on_output  );
#endif /* GENERATOR_COMMON_RV64_H */
