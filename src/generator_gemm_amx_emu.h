/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_GEMM_AMX_EMU_H
#define GENERATOR_GEMM_AMX_EMU_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_get_tileinfo( unsigned int tile_id, unsigned int *n_rows, unsigned int *n_cols, libxsmm_tile_config *tc);

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_tile_compute_emu( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_instruction_set,
                                           const unsigned int      i_tcompute_instr,
                                           const unsigned int      i_tile_src_reg_number_0,
                                           const unsigned int      i_tile_src_reg_number_1,
                                           const unsigned int      i_tile_dst_reg_number,
                                           libxsmm_micro_kernel_config*  i_micro_kernel_config);

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_tile_move_emu( libxsmm_generated_code*   io_generated_code,
                                        const unsigned int            i_instruction_set,
                                        const unsigned int            i_tmove_instr,
                                        const unsigned int            i_gp_reg_base,
                                        const unsigned int            i_gp_reg_idx,
                                        const unsigned int            i_scale,
                                        const int                     i_displacement,
                                        const unsigned int            i_tile_reg_number,
                                        libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                        unsigned int                  is_stride_0 );
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_C_amx_emu( libxsmm_generated_code*            io_generated_code,
    libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_C_amx_emu( libxsmm_generated_code*            io_generated_code,
    libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_setup_fusion_infra_emu( libxsmm_generated_code*            io_generated_code,
                                                    const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                    libxsmm_micro_kernel_config*  i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_setup_stack_frame_emu( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                  const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                  libxsmm_micro_kernel_config*        i_micro_kernel_config,
                                                  int                                 m_tiles,
                                                  int                                 n_tiles );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_destroy_stack_frame_emu( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                  const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                  libxsmm_micro_kernel_config*        i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_emu( libxsmm_generated_code*        io_generated_code,
                                        const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_mloop_emu( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_nloop_emu( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_convert_emu( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_instruction_set,
                                                   const unsigned int      i_vec_instr,
                                                   const char              i_vector_name,
                                                   const unsigned int      i_vec_reg_src_0,
                                                   const unsigned int      i_vec_reg_src_1,
                                                   const unsigned int      i_vec_reg_dst,
                                                   const unsigned int      i_shuffle_operand,
                                                   libxsmm_micro_kernel_config*  i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_emu( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_instruction_set,
                                              const unsigned int      i_vec_instr,
                                              const unsigned int      i_use_broadcast,
                                              const unsigned int      i_gp_reg_base,
                                              const unsigned int      i_gp_reg_idx,
                                              const unsigned int      i_scale,
                                              const int               i_displacement,
                                              const char              i_vector_name,
                                              const unsigned int      i_vec_reg_number_0,
                                              const unsigned int      i_vec_reg_number_1,
                                              libxsmm_micro_kernel_config*  i_micro_kernel_config );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_generic_loop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_generic_loop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg,
    unsigned int                       step,
    unsigned int                       bound);

#endif /* GENERATOR_GEMM_AMX_EMU_H */

