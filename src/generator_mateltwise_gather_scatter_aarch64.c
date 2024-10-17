/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_common.h"
#include "generator_mateltwise_gather_scatter_aarch64.h"

#if 0
#define USE_ENV_TUNING
#endif


LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_cols_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                                libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                                libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                                const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                                const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int l_m, l_n, l_im = 0, l_m_remainder_elements, l_m_trips, l_max_m_unrolling = 4, l_m_unroll_factor = 1, l_m_trips_loop = 0, l_peeled_m_trips = 0;
  unsigned int l_idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int l_mask_reg = 1;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch);
  unsigned int l_vlen = l_vector_length/i_micro_kernel_config->datatype_size_in;
  unsigned int l_is_gather = 1;
  unsigned int l_ld_reg_mat = 0;
  unsigned int l_dtype_size_reg_mat = 0;
  unsigned int l_gp_idx_mat_reg = 0, l_gp_idx_mat_reg_precision = 0, l_gp_reg_mat_reg = 0, l_gp_idx_mat_base_reg = 0;
  unsigned int l_gp_reg_ld = 0;
  unsigned int l_aux_vreg_start = 0;
#if defined(USE_ENV_TUNING)
  const char *const l_env_max_m_unroll = getenv("MAX_M_UNROLL_GATHER_SCATTER");
#endif
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

#if defined(USE_ENV_TUNING)
  if ( 0 == l_env_max_m_unroll ) {
  } else {
    l_max_m_unrolling = (unsigned int)LIBXSMM_MAX(1, atoi(l_env_max_m_unroll));
  }
#endif
  l_is_gather     = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? 1 : 0;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_AARCH64_GP_REG_X8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_AARCH64_GP_REG_X16;

  l_gp_idx_mat_base_reg   = LIBXSMM_AARCH64_GP_REG_X12;
  l_gp_reg_mat_reg        = LIBXSMM_AARCH64_GP_REG_X13;
  l_gp_idx_mat_reg        = LIBXSMM_AARCH64_GP_REG_X14;
  l_gp_idx_mat_reg_precision = (l_idx_tsize == 8) ? LIBXSMM_AARCH64_GP_REG_X14 : LIBXSMM_AARCH64_GP_REG_W14;
  l_gp_reg_ld             = LIBXSMM_AARCH64_GP_REG_X15;

  if (l_is_gather == 1) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_idx_mat_base_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_reg_mat_reg );
    l_ld_reg_mat = i_mateltwise_desc->ldo;
    l_dtype_size_reg_mat = i_micro_kernel_config->datatype_size_out;
  } else {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_reg_mat_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_idx_mat_base_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_ind_base );
    l_ld_reg_mat = i_mateltwise_desc->ldi;
    l_dtype_size_reg_mat = i_micro_kernel_config->datatype_size_in;
  }

  l_m                 = i_mateltwise_desc->m;
  l_n                 = i_mateltwise_desc->n;
  l_m_remainder_elements = l_m % l_vlen;
  l_m_trips           = (l_m + l_vlen - 1) / l_vlen;
  l_m_unroll_factor   = l_m_trips;
  l_m_trips_loop      = 1;
  l_peeled_m_trips    = 0;

  if (l_is_sve > 0) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 0, -1, i_gp_reg_mapping->gp_reg_scratch_0 );
  }
  if (l_m_remainder_elements > 0) {
    if (l_is_sve > 0) {
      /* Calculate mask reg 1 for reading/output-writing */
      if (i_micro_kernel_config->datatype_size_in == 4 || i_micro_kernel_config->datatype_size_in == 2) {
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_mask_reg, l_m_remainder_elements * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
      } else { /* should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    } else {
      l_mask_reg = l_m_remainder_elements;
    }
  }

  /* In this case we have to generate a loop for m */
  if (l_m_unroll_factor > l_max_m_unrolling) {
    l_m_unroll_factor = l_max_m_unrolling;
    l_m_trips_loop = l_m_trips/l_m_unroll_factor;
    l_peeled_m_trips = l_m_trips  - l_m_unroll_factor * l_m_trips_loop;
    if ((l_m_remainder_elements > 0) && (l_peeled_m_trips == 0)) {
      l_m_trips_loop--;
      l_peeled_m_trips = l_m_trips  - l_m_unroll_factor * l_m_trips_loop;
    }
  } else {
    if ((l_m_remainder_elements > 0) && (l_peeled_m_trips == 0)) {
      l_m_trips_loop--;
      l_peeled_m_trips = l_m_trips  - l_m_unroll_factor * l_m_trips_loop;
    }
  }

  /* Load idx mat ld bytes in a gp reg */
  if (l_is_gather == 1) {
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_ld, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
  } else {
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_ld, (long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_in );
  }

  /* Iterate over all indexed columns */
  libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, l_n);

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_EOR_SR, l_gp_idx_mat_reg, l_gp_idx_mat_reg, l_gp_idx_mat_reg, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST, i_gp_reg_mapping->gp_reg_ind_base, LIBXSMM_AARCH64_GP_REG_UNDEF, l_idx_tsize, l_gp_idx_mat_reg_precision );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, l_gp_idx_mat_reg, l_gp_reg_ld, l_gp_idx_mat_reg, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, l_gp_idx_mat_reg, l_gp_idx_mat_base_reg, l_gp_idx_mat_reg, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

  /* Iterate over full indexed column / reg. matrix column */
  if (l_m_trips_loop >= 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, l_m_trips_loop);
    for (l_im = 0; l_im < l_m_unroll_factor; l_im++) {
      unsigned int aux_vreg = l_aux_vreg_start + l_im;
      if (l_is_gather == 1) {
        /* Load gather vector */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, l_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg, l_dtype_size_reg_mat, 0, 1, 0, 0 );
        /* Store gathered vector */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg, l_dtype_size_reg_mat, 0, 1, 1, 0 );
      } else {
        /* Load vector to be scattered */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg, l_dtype_size_reg_mat, 0, 1, 0, 0 );
        /* Store vector to indexed column*/
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, l_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg, l_dtype_size_reg_mat, 0, 1, 1, 0 );
      }
    }
    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1);
  }

  if (l_peeled_m_trips > 0) {
    for (l_im = 0; l_im < l_peeled_m_trips; l_im++) {
      unsigned int l_masked_reg_elements = (l_im == l_peeled_m_trips - 1) ? l_m_remainder_elements : 0;
      unsigned int l_l_mask_reg = ((l_im == l_peeled_m_trips - 1) && (l_m_remainder_elements > 0)) ? l_mask_reg : 0;
      unsigned int aux_vreg = l_aux_vreg_start + l_im;
      if (l_is_gather == 1) {
        /* Load gather vector */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, l_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg, l_dtype_size_reg_mat, l_masked_reg_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_l_mask_reg) );
        /* Store gathered vector */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg, l_dtype_size_reg_mat, l_masked_reg_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_l_mask_reg) );
      } else {
        /* Load vector to be scattered */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg, l_dtype_size_reg_mat, l_masked_reg_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_l_mask_reg) );
        /* Store vector to indexed column*/
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, l_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg, l_dtype_size_reg_mat, l_masked_reg_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_l_mask_reg) );
      }
    }
  }

  if (((long long)l_ld_reg_mat * l_dtype_size_reg_mat - (long long)l_m * l_dtype_size_reg_mat) > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_reg_mat_reg, (long long)l_ld_reg_mat * l_dtype_size_reg_mat -(long long)l_m * l_dtype_size_reg_mat );
  }

  libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_aarch64_mn_loop_unrolled( libxsmm_generated_code*                              io_generated_code,
                                                                     libxsmm_mateltwise_gp_reg_mapping*                   i_gp_reg_mapping,
                                                                     const unsigned int                                   i_m_unroll_factor,
                                                                     const unsigned int                                   i_n_unroll_factor,
                                                                     const unsigned int                                   i_idx_vreg_start,
                                                                     const unsigned int                                   i_idx_tsize,
                                                                     const unsigned int                                   i_idx_mask_reg,
                                                                     const unsigned int                                   i_ld_idx_mat,
                                                                     const unsigned int                                   i_gather_instr,
                                                                     const unsigned int                                   i_scatter_instr,
                                                                     const unsigned int                                   i_vlen,
                                                                     const unsigned int                                   i_m_remainder_elements,
                                                                     const unsigned int                                   i_mask_reg,
                                                                     const unsigned int                                   i_mask_reg_full_frac_vlen,
                                                                     const unsigned int                                   i_is_gather,
                                                                     const unsigned int                                   i_gp_idx_mat_reg,
                                                                     const unsigned int                                   i_gp_reg_mat_reg,
                                                                     const unsigned int                                   i_dtype_size_reg_mat,
                                                                     const unsigned int                                   i_ld_reg_mat ) {
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int l_im = 0, l_in = 0;
  unsigned int l_ld_bytes_reg = i_ld_reg_mat * i_dtype_size_reg_mat;
  unsigned int l_m_adjust_reg = ( i_m_remainder_elements == 0 ) ? i_vlen * i_dtype_size_reg_mat * i_m_unroll_factor : i_dtype_size_reg_mat * ( (i_vlen * (i_m_unroll_factor-1)) + i_m_remainder_elements );
  unsigned int l_is_16bit_gs = (i_dtype_size_reg_mat == 2) ? 1 : 0;
  unsigned int l_is_64bit_idx = (i_idx_tsize == 8) ? 1 : 0;

  for (l_in = 0; l_in < i_n_unroll_factor; l_in++) {
    for (l_im = 0; l_im < i_m_unroll_factor; l_im++) {
      unsigned int l_idx_vreg_id = i_idx_vreg_start + l_im;
      unsigned int l_aux_vreg = i_idx_vreg_start + i_m_unroll_factor + l_im + l_in * i_m_unroll_factor;
      unsigned int l_idx_masked_elements = (l_im == i_m_unroll_factor - 1) ? i_m_remainder_elements : 0;
      unsigned int l_idx_mask_load = (l_im == i_m_unroll_factor - 1) ? ((i_m_remainder_elements > 0) ? i_idx_mask_reg : 0) : 0;

      unsigned int l_masked_reg_elements = ((l_is_16bit_gs == 0) && (l_is_64bit_idx == 0)) ? (l_im == i_m_unroll_factor - 1) ? i_m_remainder_elements : 0
                                                                                           : (l_im == i_m_unroll_factor - 1) ? (i_m_remainder_elements > 0) ? i_m_remainder_elements : i_vlen : i_vlen;
      unsigned int l_i_mask_reg = ((l_is_16bit_gs == 0) && (l_is_64bit_idx == 0)) ? (l_im == i_m_unroll_factor - 1) ? (i_m_remainder_elements > 0) ? i_mask_reg : 0 : 0
                                                                                : (l_im == i_m_unroll_factor - 1) ? (i_m_remainder_elements > 0) ? i_mask_reg : i_mask_reg_full_frac_vlen : i_mask_reg_full_frac_vlen;
      if (i_is_gather == 1) {
        /* Gather based on index vector im*/
        if (l_is_sve > 0) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, i_gather_instr, i_gp_idx_mat_reg, l_idx_vreg_id, 0, l_aux_vreg, l_idx_mask_load);
          if (l_is_16bit_gs || l_is_64bit_idx) {
            if ((l_is_16bit_gs > 0) && (l_is_64bit_idx> 0)) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, l_aux_vreg, l_aux_vreg, 0, l_aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
            }
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, l_aux_vreg, l_aux_vreg, 0, l_aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(i_dtype_size_reg_mat)) );
          }
        } else {
          libxsmm_generator_gather_scatter_vreg_asimd_aarch64( io_generated_code, i_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scratch_1,
              l_idx_vreg_id, i_idx_tsize, l_aux_vreg, i_dtype_size_reg_mat, l_idx_masked_elements, i_is_gather);
        }
        /* Store gathered vector */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_aux_vreg,
                                                          i_dtype_size_reg_mat, l_masked_reg_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_i_mask_reg) );
      } else {
        /* Load vector to be scattered */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_aux_vreg,
                                                          i_dtype_size_reg_mat, l_masked_reg_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_i_mask_reg) );
        /* Scatter based on index vector im*/
        if (l_is_sve > 0) {
          if (l_is_16bit_gs || l_is_64bit_idx) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, l_aux_vreg, l_aux_vreg, 0, l_aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(i_dtype_size_reg_mat)) );
            if ((l_is_16bit_gs > 0) && (l_is_64bit_idx> 0)) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, l_aux_vreg, l_aux_vreg, 0, l_aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
            }
          }
          libxsmm_aarch64_instruction_sve_move( io_generated_code, i_scatter_instr, i_gp_idx_mat_reg, l_idx_vreg_id, 0, l_aux_vreg, l_idx_mask_load);
        } else {
          libxsmm_generator_gather_scatter_vreg_asimd_aarch64( io_generated_code, i_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scratch_1,
              l_idx_vreg_id, i_idx_tsize, l_aux_vreg, i_dtype_size_reg_mat, l_idx_masked_elements, i_is_gather);
        }
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                  i_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_idx_mat_reg,
                                                  ((long long)i_ld_idx_mat) * i_dtype_size_reg_mat);
    if ( l_ld_bytes_reg != l_m_adjust_reg ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    i_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mat_reg,
                                                    ((long long)l_ld_bytes_reg - l_m_adjust_reg) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                               i_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_idx_mat_reg,
                                               (long long)i_ld_idx_mat*i_dtype_size_reg_mat*i_n_unroll_factor );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                               i_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mat_reg,
                                               (long long)l_ld_bytes_reg*i_n_unroll_factor );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                                libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                                libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                                const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                                const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int l_m, l_n, l_im = 0, l_m_remainder_elements, l_m_trips, l_max_m_unrolling = 4, l_m_unroll_factor = 1, l_n_unroll_factor = 4, l_m_trips_loop = 0, l_peeled_m_trips = 0;
  unsigned int l_idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int l_gather_instr = 0, l_scatter_instr = 0;
  unsigned int l_mask_reg = 1;
  unsigned int l_idx_maskreg = 2;
  unsigned int l_mask_reg_full_frac_vlen = 3;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch);
  unsigned int l_vlen = l_vector_length/l_idx_tsize; /* The granularity of work is determined by the idx size */
  unsigned int l_is_gather = 1;
  unsigned int l_ld_reg_mat = 0;
  unsigned int l_ld_idx_mat = 0;
  unsigned int l_dtype_size_reg_mat = 0;
  unsigned int l_gp_idx_mat_reg = 0, l_gp_reg_mat_reg = 0, l_gp_idx_mat_base_reg = 0;
  unsigned int l_idx_vreg_start = 0;
#if defined(USE_ENV_TUNING)
  const char *const l_env_max_m_unroll = getenv("MAX_M_UNROLL_GATHER_SCATTER");
  const char *const l_env_max_n_unroll = getenv("MAX_N_UNROLL_GATHER_SCATTER");
#endif
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

#if defined(USE_ENV_TUNING)
  if ( 0 == l_env_max_m_unroll ) {
  } else {
    l_max_m_unrolling = (unsigned int)LIBXSMM_MAX(1, atoi(l_env_max_m_unroll));
  }
  if ( 0 == l_env_max_n_unroll ) {
  } else {
    l_n_unroll_factor = (unsigned int)LIBXSMM_MAX(1, atoi(l_env_max_n_unroll));
  }
#endif

  /* Determine instructions based on arch and precisions */
  l_gather_instr  = (i_micro_kernel_config->datatype_size_in == 4) ? ((l_idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1W_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_LD1W_V_OFF64_SCALE)
                                                                 : ((l_idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1H_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_LD1H_V_OFF64_SCALE);
  l_scatter_instr = (i_micro_kernel_config->datatype_size_in == 4) ? ((l_idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_ST1W_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_ST1W_V_OFF64_SCALE)
                                                                 : ((l_idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_ST1H_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_ST1H_V_OFF64_SCALE);
  l_is_gather     = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? 1 : 0;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_AARCH64_GP_REG_X8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_AARCH64_GP_REG_X16;
  i_gp_reg_mapping->gp_reg_scratch_1 = LIBXSMM_AARCH64_GP_REG_X17;

  l_gp_idx_mat_base_reg   = LIBXSMM_AARCH64_GP_REG_X12;
  l_gp_reg_mat_reg        = LIBXSMM_AARCH64_GP_REG_X13;
  l_gp_idx_mat_reg        = LIBXSMM_AARCH64_GP_REG_X12;

  if (l_is_gather == 1) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_idx_mat_base_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_reg_mat_reg );
    l_ld_reg_mat = i_mateltwise_desc->ldo;
    l_ld_idx_mat = i_mateltwise_desc->ldi;
    l_dtype_size_reg_mat = i_micro_kernel_config->datatype_size_out;
  } else {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_reg_mat_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_idx_mat_base_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_ind_base );
    l_ld_reg_mat = i_mateltwise_desc->ldi;
    l_ld_idx_mat = i_mateltwise_desc->ldo;
    l_dtype_size_reg_mat = i_micro_kernel_config->datatype_size_in;
  }

  l_m                 = i_mateltwise_desc->m;
  l_n                 = i_mateltwise_desc->n;
  LIBXSMM_ASSERT(0 != l_vlen);
  l_m_remainder_elements = l_m % l_vlen;
  l_m_trips           = (l_m + l_vlen - 1) / l_vlen;
  l_m_unroll_factor   = l_m_trips;
  l_m_trips_loop      = 1;
  l_peeled_m_trips    = 0;

  while (l_n % l_n_unroll_factor > 0) {
    l_n_unroll_factor--;
  }

  if (l_is_sve > 0) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 0, -1, i_gp_reg_mapping->gp_reg_scratch_0 );
    if (i_micro_kernel_config->datatype_size_in == 2 || l_idx_tsize == 8) {
      /* Mask for full fractional vlen load/store */
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_mask_reg_full_frac_vlen, l_vlen * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
    }
  }
  if (l_m_remainder_elements > 0) {
    if (l_is_sve > 0) {
      /* Calculate mask reg 1 for reading/output-writing */
      /* define predicate registers 0 and 1 for loading */
      if (i_micro_kernel_config->datatype_size_in == 4 || i_micro_kernel_config->datatype_size_in == 2) {
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_mask_reg, l_m_remainder_elements * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_idx_maskreg, l_m_remainder_elements * l_idx_tsize, i_gp_reg_mapping->gp_reg_scratch_0 );
      } else { /* should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    } else {
      l_mask_reg = l_m_remainder_elements;
      l_idx_maskreg = l_m_remainder_elements;
    }
  }

  /* In this case we have to generate a loop for m */
  if (l_m_unroll_factor > l_max_m_unrolling) {
    l_m_unroll_factor = l_max_m_unrolling;
    l_m_trips_loop = l_m_trips/l_m_unroll_factor;
    l_peeled_m_trips = l_m_trips  - l_m_unroll_factor * l_m_trips_loop;
    if ((l_m_remainder_elements > 0) && (l_peeled_m_trips == 0)) {
      l_m_trips_loop--;
      l_peeled_m_trips = l_m_trips  - l_m_unroll_factor * l_m_trips_loop;
    }
  } else {
    if ((l_m_remainder_elements > 0) && (l_peeled_m_trips == 0)) {
      l_m_trips_loop--;
      l_peeled_m_trips = l_m_trips  - l_m_unroll_factor * l_m_trips_loop;
    }
  }

  if (l_m_trips_loop >= 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, l_m_trips_loop);

    /* Load Gather/Scatter indices */
    for (l_im = 0; l_im < l_m_unroll_factor; l_im++) {
      unsigned int idx_vreg_id = l_idx_vreg_start + l_im;
      unsigned int l_idx_masked_elements =  0;
      unsigned int l_idx_mask_load = 0;
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, idx_vreg_id,
                                                        l_idx_tsize, l_idx_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_idx_mask_load) );
    }

    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, l_n);


    libxsmm_generator_gather_scatter_rows_aarch64_mn_loop_unrolled( io_generated_code, i_gp_reg_mapping, l_m_unroll_factor, l_n_unroll_factor,
        l_idx_vreg_start, l_idx_tsize, l_idx_maskreg, l_ld_idx_mat, l_gather_instr, l_scatter_instr, l_vlen, 0,
        l_mask_reg, l_mask_reg_full_frac_vlen, l_is_gather, l_gp_idx_mat_reg, l_gp_reg_mat_reg, l_dtype_size_reg_mat, l_ld_reg_mat );


    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_reg_mat_reg, (long long)l_n_unroll_factor * l_ld_reg_mat * l_dtype_size_reg_mat );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_idx_mat_reg, (long long)l_n_unroll_factor * l_ld_idx_mat * l_dtype_size_reg_mat );


    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, l_n_unroll_factor);

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_reg_mat_reg, (long long)l_n * l_ld_reg_mat * l_dtype_size_reg_mat - (long long) l_m_unroll_factor * l_vlen * l_dtype_size_reg_mat);
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_idx_mat_reg, (long long)l_n * l_ld_idx_mat * l_dtype_size_reg_mat );

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1);
  }

  if (l_peeled_m_trips > 0) {
    /* Load Gather/Scatter indices */
    for (l_im = 0; l_im < l_peeled_m_trips; l_im++) {
      unsigned int idx_vreg_id = l_idx_vreg_start + l_im;
      unsigned int l_idx_masked_elements = (l_im == l_peeled_m_trips - 1) ? l_m_remainder_elements : 0;
      unsigned int l_idx_mask_load = (l_im == l_peeled_m_trips - 1) ? ((l_m_remainder_elements > 0) ? l_idx_maskreg : 0) : 0;
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, idx_vreg_id,
                                                        l_idx_tsize, l_idx_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_idx_mask_load) );
    }

    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, l_n);


    libxsmm_generator_gather_scatter_rows_aarch64_mn_loop_unrolled( io_generated_code, i_gp_reg_mapping, l_peeled_m_trips, l_n_unroll_factor,
        l_idx_vreg_start, l_idx_tsize, l_idx_maskreg, l_ld_idx_mat, l_gather_instr, l_scatter_instr, l_vlen, l_m_remainder_elements,
        l_mask_reg, l_mask_reg_full_frac_vlen, l_is_gather, l_gp_idx_mat_reg, l_gp_reg_mat_reg, l_dtype_size_reg_mat, l_ld_reg_mat );


    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_reg_mat_reg, (long long)l_n_unroll_factor * l_ld_reg_mat * l_dtype_size_reg_mat );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_idx_mat_reg, (long long)l_n_unroll_factor * l_ld_idx_mat * l_dtype_size_reg_mat );

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, l_n_unroll_factor);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_aarch64_mn_loop_unrolled( libxsmm_generated_code*                              io_generated_code,
                                                                     libxsmm_mateltwise_gp_reg_mapping*                   i_gp_reg_mapping,
                                                                     const unsigned int                                   i_m_unroll_factor,
                                                                     const unsigned int                                   i_n_unroll_factor,
                                                                     const unsigned int                                   i_gp_idx_base_reg,
                                                                     const unsigned int                                   i_idx_vlen,
                                                                     const unsigned int                                   i_idx_tsize,
                                                                     const unsigned int                                   i_idx_mask_reg,
                                                                     const unsigned int                                   i_ld_idx,
                                                                     const unsigned int                                   i_gather_instr,
                                                                     const unsigned int                                   i_scatter_instr,
                                                                     const unsigned int                                   i_vlen,
                                                                     const unsigned int                                   i_m_remainder_elements,
                                                                     const unsigned int                                   i_mask_reg,
                                                                     const unsigned int                                   i_mask_reg_full_frac_vlen,
                                                                     const unsigned int                                   i_is_gather,
                                                                     const unsigned int                                   i_gp_idx_mat_reg,
                                                                     const unsigned int                                   i_gp_reg_mat_reg,
                                                                     const unsigned int                                   i_dtype_size_reg_mat,
                                                                     const unsigned int                                   i_ld_reg_mat ) {
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int l_im= 0, l_in = 0;
  unsigned int l_aux_vreg_start = i_m_unroll_factor*i_n_unroll_factor;
  unsigned int l_idx_vreg_start = 0;
  unsigned int l_ld_bytes_idx = i_ld_idx * i_idx_tsize;
  unsigned int l_m_adjust_idx = ( i_m_remainder_elements == 0 ) ? i_idx_vlen * i_idx_tsize * i_m_unroll_factor : i_idx_tsize * ( (i_idx_vlen * (i_m_unroll_factor-1)) + i_m_remainder_elements );
  unsigned int l_ld_bytes_reg = i_ld_reg_mat * i_dtype_size_reg_mat;
  unsigned int l_m_adjust_reg = ( i_m_remainder_elements == 0 ) ? i_vlen * i_dtype_size_reg_mat * i_m_unroll_factor : i_dtype_size_reg_mat * ( (i_vlen * (i_m_unroll_factor-1)) + i_m_remainder_elements );
  unsigned int l_is_16bit_gs = (i_dtype_size_reg_mat == 2) ? 1 : 0;
  unsigned int l_is_64bit_idx = (i_idx_tsize == 8) ? 1 : 0;

  for (l_in = 0; l_in < i_n_unroll_factor; l_in++) {
    for (l_im = 0; l_im < i_m_unroll_factor; l_im++) {
      unsigned int l_idx_vreg_id = l_idx_vreg_start + l_im + l_in * i_m_unroll_factor;
      unsigned int l_aux_vreg = l_aux_vreg_start + l_im + l_in * i_m_unroll_factor;
      unsigned int l_idx_masked_elements = (l_im == i_m_unroll_factor - 1) ? i_m_remainder_elements : 0;
      unsigned int l_idx_mask_load = (l_im == i_m_unroll_factor - 1) ? ((i_m_remainder_elements > 0) ? i_idx_mask_reg : 0) : 0;

      unsigned int l_masked_reg_elements = ((l_is_16bit_gs == 0) && (l_is_64bit_idx == 0)) ? (l_im == i_m_unroll_factor - 1) ? i_m_remainder_elements : 0
                                                                                           : (l_im == i_m_unroll_factor - 1) ? (i_m_remainder_elements > 0) ? i_m_remainder_elements : i_vlen : i_vlen;
      unsigned int l_i_mask_reg = ((l_is_16bit_gs == 0) && (l_is_64bit_idx == 0)) ? (l_im == i_m_unroll_factor - 1) ? (i_m_remainder_elements > 0) ? i_mask_reg : 0 : 0
                                                                                : (l_im == i_m_unroll_factor - 1) ? (i_m_remainder_elements > 0) ? i_mask_reg : i_mask_reg_full_frac_vlen : i_mask_reg_full_frac_vlen;

      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_idx_base_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_idx_vreg_id,
                                                        i_idx_tsize, l_idx_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_idx_mask_load) );
      if (i_is_gather == 1) {
        /* Gather based on index vector im*/
        if (l_is_sve > 0) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, i_gather_instr, i_gp_idx_mat_reg, l_idx_vreg_id, 0, l_aux_vreg, l_idx_mask_load);
          if (l_is_16bit_gs || l_is_64bit_idx) {
            if ((l_is_16bit_gs > 0) && (l_is_64bit_idx> 0)) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, l_aux_vreg, l_aux_vreg, 0, l_aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
            }
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, l_aux_vreg, l_aux_vreg, 0, l_aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(i_dtype_size_reg_mat)) );
          }
        } else {
          libxsmm_generator_gather_scatter_vreg_asimd_aarch64( io_generated_code, i_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scratch_1,
              l_idx_vreg_id, i_idx_tsize, l_aux_vreg, i_dtype_size_reg_mat, l_idx_masked_elements, i_is_gather);
        }
        /* Store gathered vector */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_aux_vreg,
                                                          i_dtype_size_reg_mat, l_masked_reg_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_i_mask_reg) );
      } else {
        /* Load vector to be scattered */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_aux_vreg,
                                                          i_dtype_size_reg_mat, l_masked_reg_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_i_mask_reg) );
        /* Scatter based on index vector im*/
        if (l_is_sve > 0) {
          if (l_is_16bit_gs || l_is_64bit_idx) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, l_aux_vreg, l_aux_vreg, 0, l_aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(i_dtype_size_reg_mat)) );
            if ((l_is_16bit_gs > 0) && (l_is_64bit_idx> 0)) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, l_aux_vreg, l_aux_vreg, 0, l_aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
            }
          }
          libxsmm_aarch64_instruction_sve_move( io_generated_code, i_scatter_instr, i_gp_idx_mat_reg, l_idx_vreg_id, 0, l_aux_vreg, l_idx_mask_load);
        } else {
          libxsmm_generator_gather_scatter_vreg_asimd_aarch64( io_generated_code, i_gp_idx_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scratch_1,
              l_idx_vreg_id, i_idx_tsize, l_aux_vreg, i_dtype_size_reg_mat, l_idx_masked_elements, i_is_gather);
        }
      }
    }
    if ( l_ld_bytes_idx != l_m_adjust_idx ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    i_gp_idx_base_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_idx_base_reg,
                                                    ((long long)l_ld_bytes_idx - l_m_adjust_idx) );
    }
    if ( l_ld_bytes_reg != l_m_adjust_reg ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    i_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mat_reg,
                                                    ((long long)l_ld_bytes_reg - l_m_adjust_reg) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                               i_gp_idx_base_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_idx_base_reg,
                                               (long long)l_ld_bytes_idx*i_n_unroll_factor );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                               i_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mat_reg,
                                               (long long)l_ld_bytes_reg*i_n_unroll_factor );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                                libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                                libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                                const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                                const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int l_m, l_n, l_m_remainder_elements, l_m_trips, l_max_m_unrolling = 4, l_m_unroll_factor = 1, l_n_unroll_factor = 4, l_m_trips_loop = 0, l_peeled_m_trips = 0;
  unsigned int l_idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int l_gather_instr = 0, l_scatter_instr = 0;
  unsigned int l_mask_reg = 1;
  unsigned int l_idx_mask_reg = 2;
  unsigned int l_mask_reg_full_frac_vlen = 3;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch);
  unsigned int l_idx_vlen = l_vector_length/l_idx_tsize;
  unsigned int l_vlen = l_vector_length/l_idx_tsize; /* The granularity of work is determined by the idx size */
  unsigned int l_is_gather = 1;
  unsigned int l_ld_reg_mat = 0;
  unsigned int l_dtype_size_reg_mat = 0;
  unsigned int l_gp_idx_mat_reg = 0, l_gp_reg_mat_reg = 0, l_gp_idx_mat_base_reg = 0;
#if defined(USE_ENV_TUNING)
  const char *const l_env_max_m_unroll = getenv("MAX_M_UNROLL_GATHER_SCATTER");
  const char *const l_env_max_n_unroll = getenv("MAX_N_UNROLL_GATHER_SCATTER");
#endif
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

#if defined(USE_ENV_TUNING)
  if ( 0 == l_env_max_m_unroll ) {
  } else {
    l_max_m_unrolling = (unsigned int)LIBXSMM_MAX(1, atoi(l_env_max_m_unroll));
  }
  if ( 0 == l_env_max_n_unroll ) {
  } else {
    l_n_unroll_factor = (unsigned int)LIBXSMM_MAX(1, atoi(l_env_max_n_unroll));
  }
#endif

  /* Determine instructions based on arch and precisions */
  l_gather_instr  = (i_micro_kernel_config->datatype_size_in == 4) ? ((l_idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1W_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_LD1W_V_OFF64_SCALE)
                                                                 : ((l_idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1H_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_LD1H_V_OFF64_SCALE);
  l_scatter_instr = (i_micro_kernel_config->datatype_size_in == 4) ? ((l_idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_ST1W_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_ST1W_V_OFF64_SCALE)
                                                                 : ((l_idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_ST1H_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_ST1H_V_OFF64_SCALE);
  l_is_gather     = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? 1 : 0;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_AARCH64_GP_REG_X8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_AARCH64_GP_REG_X16;
  i_gp_reg_mapping->gp_reg_scratch_1 = LIBXSMM_AARCH64_GP_REG_X17;

  l_gp_idx_mat_base_reg   = LIBXSMM_AARCH64_GP_REG_X12;
  l_gp_reg_mat_reg        = LIBXSMM_AARCH64_GP_REG_X13;
  l_gp_idx_mat_reg        = LIBXSMM_AARCH64_GP_REG_X12;

  if (l_is_gather == 1) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_idx_mat_base_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_reg_mat_reg );
    l_ld_reg_mat = i_mateltwise_desc->ldo;
    l_dtype_size_reg_mat = i_micro_kernel_config->datatype_size_out;
  } else {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_reg_mat_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_idx_mat_base_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_ind_base );
    l_ld_reg_mat = i_mateltwise_desc->ldi;
    l_dtype_size_reg_mat = i_micro_kernel_config->datatype_size_in;
  }

  l_m                 = i_mateltwise_desc->m;
  l_n                 = i_mateltwise_desc->n;
  LIBXSMM_ASSERT(0 != l_vlen);
  l_m_remainder_elements = l_m % l_vlen;
  l_m_trips           = (l_m + l_vlen - 1) / l_vlen;
  l_m_unroll_factor   = l_m_trips;
  l_m_trips_loop      = 1;
  l_peeled_m_trips    = 0;

  while (l_n % l_n_unroll_factor > 0) {
    l_n_unroll_factor--;
  }

  if (l_is_sve > 0) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 0, -1, i_gp_reg_mapping->gp_reg_scratch_0 );
    if (i_micro_kernel_config->datatype_size_in == 2 || l_idx_tsize == 8) {
      /* Mask for full fractional vlen load/store */
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_mask_reg_full_frac_vlen, l_vlen * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
    }
  }
  if (l_m_remainder_elements > 0) {
    if (l_is_sve > 0) {
      /* Calculate mask reg 1 for reading/output-writing */
      /* define predicate registers 0 and 1 for loading */
      if (i_micro_kernel_config->datatype_size_in == 4 || i_micro_kernel_config->datatype_size_in == 2) {
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_mask_reg, l_m_remainder_elements * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_idx_mask_reg, l_m_remainder_elements * l_idx_tsize, i_gp_reg_mapping->gp_reg_scratch_0 );
      } else { /* should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    } else {
      l_mask_reg = l_m_remainder_elements;
      l_idx_mask_reg = l_m_remainder_elements;
    }
  }

  /* In this case we have to generate a loop for m */
  if (l_m_unroll_factor > l_max_m_unrolling) {
    l_m_unroll_factor = l_max_m_unrolling;
    l_m_trips_loop = l_m_trips/l_m_unroll_factor;
    l_peeled_m_trips = l_m_trips  - l_m_unroll_factor * l_m_trips_loop;
    if ((l_m_remainder_elements > 0) && (l_peeled_m_trips == 0)) {
      l_m_trips_loop--;
      l_peeled_m_trips = l_m_trips  - l_m_unroll_factor * l_m_trips_loop;
    }
  } else {
    if ((l_m_remainder_elements > 0) && (l_peeled_m_trips == 0)) {
      l_m_trips_loop--;
      l_peeled_m_trips = l_m_trips  - l_m_unroll_factor * l_m_trips_loop;
    }
  }

  if (l_m_trips_loop >= 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, l_n);
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, l_m_trips_loop);

    libxsmm_generator_gather_scatter_offs_aarch64_mn_loop_unrolled( io_generated_code, i_gp_reg_mapping, l_m_unroll_factor, l_n_unroll_factor,
        i_gp_reg_mapping->gp_reg_ind_base, l_idx_vlen, l_idx_tsize, l_idx_mask_reg, l_m, l_gather_instr, l_scatter_instr, l_vlen, 0,
        l_mask_reg, l_mask_reg_full_frac_vlen, l_is_gather, l_gp_idx_mat_reg, l_gp_reg_mat_reg, l_dtype_size_reg_mat, l_ld_reg_mat );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_reg_mat_reg, (long long)l_m_unroll_factor * l_vlen * l_dtype_size_reg_mat );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_ind_base, (long long)l_m_unroll_factor * l_idx_tsize * l_idx_vlen );

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1);

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_reg_mat_reg, ((long long)l_n_unroll_factor * l_ld_reg_mat - (long long)l_m_trips_loop * l_m_unroll_factor * l_vlen) * l_dtype_size_reg_mat);
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_ind_base, ((long long)l_n_unroll_factor * l_m - (long long)l_m_trips_loop * l_m_unroll_factor * l_idx_vlen) * l_idx_tsize);

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, l_n_unroll_factor);

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_reg_mat_reg, (long long)l_ld_reg_mat * l_dtype_size_reg_mat * l_n - (long long)l_m_trips_loop * l_m_unroll_factor * l_vlen * l_dtype_size_reg_mat );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_ind_base, (long long)l_m * l_idx_tsize * l_n - (long long)l_m_trips_loop * l_m_unroll_factor * l_idx_vlen * l_idx_tsize  );
  }

  if (l_peeled_m_trips > 0) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, l_n);

    libxsmm_generator_gather_scatter_offs_aarch64_mn_loop_unrolled( io_generated_code, i_gp_reg_mapping, l_peeled_m_trips, l_n_unroll_factor,
        i_gp_reg_mapping->gp_reg_ind_base, l_idx_vlen, l_idx_tsize, l_idx_mask_reg, l_m, l_gather_instr, l_scatter_instr, l_vlen, l_m_remainder_elements,
        l_mask_reg, l_mask_reg_full_frac_vlen, l_is_gather, l_gp_idx_mat_reg, l_gp_reg_mat_reg, l_dtype_size_reg_mat, l_ld_reg_mat );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, l_gp_reg_mat_reg, (long long)l_n_unroll_factor * l_ld_reg_mat * l_dtype_size_reg_mat);
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_ind_base, (long long)l_n_unroll_factor * l_m * l_idx_tsize);

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, l_n_unroll_factor);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                           libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                           libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                           const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                           const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS ) > 0 ) {
    libxsmm_generator_gather_scatter_cols_aarch64_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS ) > 0 ) {
    libxsmm_generator_gather_scatter_rows_aarch64_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_OFFS ) > 0 ) {
    libxsmm_generator_gather_scatter_offs_aarch64_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    /* SHOULD NOT HAPPEN */
  }
}
