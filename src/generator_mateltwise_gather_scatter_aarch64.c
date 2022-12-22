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
#include "generator_mateltwise_aarch64_sve.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_common.h"
#include "libxsmm_main.h"
#include "generator_mateltwise_gather_scatter_aarch64.h"

#if !defined(LIBXSMM_GENERATOR_MATELTWISE_GATHER_SCATTER_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
# define LIBXSMM_GENERATOR_MATELTWISE_GATHER_SCATTER_AARCH64_JUMP_LABEL_TRACKER_MALLOC
#endif

#if 0
#define USE_ENV_TUNING
#endif

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_aarch64_mn_loop_unrolled( libxsmm_generated_code*                        io_generated_code,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   n_unroll_factor,
    unsigned int                                   gp_idx_base_reg,
    unsigned int                                   idx_vlen,
    unsigned int                                   idx_tsize,
    unsigned int                                   idx_mask_reg,
    unsigned int                                   ld_idx,
    unsigned int                                   gather_instr,
    unsigned int                                   scatter_instr,
    unsigned int                                   vlen,
    unsigned int                                   m_remainder_elements,
    unsigned int                                   mask_reg,
    unsigned int                                   mask_reg_full_frac_vlen,
    unsigned int                                   is_gather,
    unsigned int                                   gp_idx_mat_reg,
    unsigned int                                   gp_reg_mat_reg,
    unsigned int                                   dtype_size_reg_mat,
    unsigned int                                   ld_reg_mat ) {
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int im = 0, in = 0;
  unsigned int aux_vreg_start = m_unroll_factor*n_unroll_factor;
  unsigned int idx_vreg_start = 0;
  unsigned int l_ld_bytes_idx = ld_idx * idx_tsize;
  unsigned int l_m_adjust_idx = ( m_remainder_elements == 0 ) ? idx_vlen * idx_tsize * m_unroll_factor : idx_tsize * ( (idx_vlen * (m_unroll_factor-1)) + m_remainder_elements );
  unsigned int l_ld_bytes_reg = ld_reg_mat * dtype_size_reg_mat;
  unsigned int l_m_adjust_reg = ( m_remainder_elements == 0 ) ? vlen * dtype_size_reg_mat * m_unroll_factor : dtype_size_reg_mat * ( (vlen * (m_unroll_factor-1)) + m_remainder_elements );
  unsigned int l_is_16bit_gs = (dtype_size_reg_mat == 2) ? 1 : 0;
  unsigned int l_is_64bit_idx = (idx_tsize == 8) ? 1 : 0;

  for (in = 0; in < n_unroll_factor; in++) {
    for (im = 0; im < m_unroll_factor; im++) {
      unsigned int idx_vreg_id = idx_vreg_start + im + in * m_unroll_factor;
      unsigned int aux_vreg = aux_vreg_start + im + in * m_unroll_factor;
      unsigned int l_idx_masked_elements = (im == m_unroll_factor - 1) ? m_remainder_elements : 0;
      unsigned int l_idx_mask_load = (im == m_unroll_factor - 1) ? ((m_remainder_elements > 0) ? idx_mask_reg : 0) : 0;

      unsigned int l_masked_reg_elements = ((l_is_16bit_gs == 0) && (l_is_64bit_idx == 0)) ? (im == m_unroll_factor - 1) ? m_remainder_elements : 0
                                                                                           : (im == m_unroll_factor - 1) ? (m_remainder_elements > 0) ? m_remainder_elements : vlen : vlen;
      unsigned int l_mask_reg = ((l_is_16bit_gs == 0) && (l_is_64bit_idx == 0)) ? (im == m_unroll_factor - 1) ? (m_remainder_elements > 0) ? mask_reg : 0 : 0
                                                                                : (im == m_unroll_factor - 1) ? (m_remainder_elements > 0) ? mask_reg : mask_reg_full_frac_vlen : mask_reg_full_frac_vlen;

      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, gp_idx_base_reg, i_gp_reg_mapping->gp_reg_scratch_0, idx_vreg_id,
                                                        idx_tsize, l_idx_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_idx_mask_load) );
      if (is_gather == 1) {
        /* Gather based on index vector im*/
        if (l_is_sve > 0) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, gather_instr, gp_idx_mat_reg, idx_vreg_id, 0, aux_vreg, l_idx_mask_load);
          if (l_is_16bit_gs || l_is_64bit_idx) {
            if ((l_is_16bit_gs > 0) && (l_is_64bit_idx> 0)) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, aux_vreg, aux_vreg, 0, aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
            }
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, aux_vreg, aux_vreg, 0, aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(dtype_size_reg_mat) );
          }
        } else {
        }
        /* Store gathered vector  */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg,
                                                          dtype_size_reg_mat, l_masked_reg_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_reg) );
      } else {
        /* Load vector to be scattered */
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg,
                                                          dtype_size_reg_mat, l_masked_reg_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_reg) );
        /* Scatter based on index vector im*/
        if (l_is_sve > 0) {
          if (l_is_16bit_gs || l_is_64bit_idx) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, aux_vreg, aux_vreg, 0, aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(dtype_size_reg_mat) );
            if ((l_is_16bit_gs > 0) && (l_is_64bit_idx> 0)) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, aux_vreg, aux_vreg, 0, aux_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
            }
          }
          libxsmm_aarch64_instruction_sve_move( io_generated_code, scatter_instr, gp_idx_mat_reg, idx_vreg_id, 0, aux_vreg, l_idx_mask_load);
        } else {
        }
      }
    }
    if ( l_ld_bytes_idx != l_m_adjust_idx ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    gp_idx_base_reg, i_gp_reg_mapping->gp_reg_scratch_0, gp_idx_base_reg,
                                                    ((long long)l_ld_bytes_idx - l_m_adjust_idx) );
    }
    if ( l_ld_bytes_reg != l_m_adjust_reg ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, gp_reg_mat_reg,
                                                    ((long long)l_ld_bytes_reg - l_m_adjust_reg) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                               gp_idx_base_reg, i_gp_reg_mapping->gp_reg_scratch_0, gp_idx_base_reg,
                                               (long long)l_ld_bytes_idx*n_unroll_factor );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                               gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, gp_reg_mat_reg,
                                               (long long)l_ld_bytes_reg*n_unroll_factor );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int m, n, m_remainder_elements, m_trips, max_m_unrolling = 4, m_unroll_factor = 1, n_unroll_factor = 4, m_trips_loop = 0, peeled_m_trips = 0;
  unsigned int idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int gather_instr = 0, scatter_instr = 0;
  unsigned int mask_reg = 1;
  unsigned int idx_mask_reg = 2;
  unsigned int mask_reg_full_frac_vlen = 3;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch);
  unsigned int idx_vlen = l_vector_length/idx_tsize;
  unsigned int vlen = l_vector_length/idx_tsize; /* The granularity of work is determined by the idx size */
  unsigned int is_gather = 1;
  unsigned int ld_reg_mat = 0;
  unsigned int dtype_size_reg_mat = 0;
  unsigned int gp_idx_mat_reg = 0, gp_reg_mat_reg = 0, gp_idx_mat_base_reg = 0;
#if defined(USE_ENV_TUNING)
  const char *const env_max_m_unroll = getenv("MAX_M_UNROLL_GATHER_SCATTER");
  const char *const env_max_n_unroll = getenv("MAX_N_UNROLL_GATHER_SCATTER");
#endif
#if defined(LIBXSMM_GENERATOR_MATELTWISE_GATHER_SCATTER_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
  libxsmm_jump_label_tracker* const p_jump_label_tracker = (libxsmm_jump_label_tracker*)malloc(sizeof(libxsmm_jump_label_tracker));
#else
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_jump_label_tracker* const p_jump_label_tracker = &l_jump_label_tracker;
#endif
  libxsmm_reset_jump_label_tracker(p_jump_label_tracker);

#if defined(USE_ENV_TUNING)
  if ( 0 == env_max_m_unroll ) {
  } else {
    max_m_unrolling = (unsigned int)LIBXSMM_MAX(1, atoi(env_max_m_unroll));
  }
  if ( 0 == env_max_n_unroll ) {
  } else {
    n_unroll_factor = (unsigned int)LIBXSMM_MAX(1, atoi(env_max_n_unroll));
  }
#endif

  /* Determine instructions based on arch and precisions */
  gather_instr  = (i_micro_kernel_config->datatype_size_in == 4) ? ((idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1W_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_LD1W_V_OFF64_SCALE)
                                                                 : ((idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1H_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_LD1H_V_OFF64_SCALE);
  scatter_instr = (i_micro_kernel_config->datatype_size_in == 4) ? ((idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_ST1W_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_ST1W_V_OFF64_SCALE)
                                                                 : ((idx_tsize == 4) ? LIBXSMM_AARCH64_INSTR_SVE_ST1H_V_OFF_SCALE : LIBXSMM_AARCH64_INSTR_SVE_ST1H_V_OFF64_SCALE);
  is_gather     = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? 1 : 0;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_AARCH64_GP_REG_X8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_AARCH64_GP_REG_X16;
  i_gp_reg_mapping->gp_reg_scratch_1 = LIBXSMM_AARCH64_GP_REG_X17;

  gp_idx_mat_base_reg   = LIBXSMM_AARCH64_GP_REG_X12;
  gp_reg_mat_reg        = LIBXSMM_AARCH64_GP_REG_X13;
  gp_idx_mat_reg        = LIBXSMM_AARCH64_GP_REG_X12;

  if (is_gather == 1) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, gp_idx_mat_base_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, gp_reg_mat_reg );
    ld_reg_mat = i_mateltwise_desc->ldo;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_out;
  } else {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, gp_reg_mat_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, gp_idx_mat_base_reg );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_ind_base );
    ld_reg_mat = i_mateltwise_desc->ldi;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_in;
  }

  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  m_remainder_elements = m % vlen;
  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  while (n % n_unroll_factor > 0) {
    n_unroll_factor--;
  }

  if (l_is_sve > 0) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 0, -1, i_gp_reg_mapping->gp_reg_scratch_0 );
    if (i_micro_kernel_config->datatype_size_in == 2 || idx_tsize == 8) {
      /* Mask for full fractional vlen load/store  */
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, mask_reg_full_frac_vlen, vlen * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
    }
  }
  if (m_remainder_elements > 0) {
    if (l_is_sve > 0) {
      /* Calculate mask reg 1 for reading/output-writing */
      /* define predicate registers 0 and 1 for loading */
      if (i_micro_kernel_config->datatype_size_in == 4 || i_micro_kernel_config->datatype_size_in == 2) {
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, mask_reg, m_remainder_elements * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, idx_mask_reg, m_remainder_elements * idx_tsize, i_gp_reg_mapping->gp_reg_scratch_0 );
      } else {
        /* should not happen */
#if defined(LIBXSMM_GENERATOR_MATELTWISE_GATHER_SCATTER_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
        free(p_jump_label_tracker);
#endif
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    } else {
      mask_reg = m_remainder_elements;
      idx_mask_reg = m_remainder_elements;
    }
  }

  /* In this case we have to generate a loop for m */
  if (m_unroll_factor > max_m_unrolling) {
    m_unroll_factor = max_m_unrolling;
    m_trips_loop = m_trips/m_unroll_factor;
    peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    if ((m_remainder_elements > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  } else {
    if ((m_remainder_elements > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  }

  if (m_trips_loop >= 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);

    libxsmm_generator_gather_scatter_offs_aarch64_mn_loop_unrolled( io_generated_code, i_gp_reg_mapping, m_unroll_factor, n_unroll_factor,
        i_gp_reg_mapping->gp_reg_ind_base, idx_vlen, idx_tsize, idx_mask_reg, m, gather_instr, scatter_instr, vlen, 0,
        mask_reg, mask_reg_full_frac_vlen, is_gather, gp_idx_mat_reg, gp_reg_mat_reg, dtype_size_reg_mat, ld_reg_mat );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, gp_reg_mat_reg, (long long)m_unroll_factor * vlen * dtype_size_reg_mat );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_ind_base, (long long)m_unroll_factor * idx_tsize * idx_vlen );

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1);

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, gp_reg_mat_reg, ((long long)n_unroll_factor * ld_reg_mat - (long long)m_trips_loop * m_unroll_factor * vlen) * dtype_size_reg_mat);
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_ind_base, ((long long)n_unroll_factor * m - (long long)m_trips_loop * m_unroll_factor * idx_vlen) * idx_tsize);

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n_unroll_factor);

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, gp_reg_mat_reg, (long long)ld_reg_mat * dtype_size_reg_mat * n - (long long)m_trips_loop * m_unroll_factor * vlen * dtype_size_reg_mat );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_ind_base, (long long)m * idx_tsize * n - (long long)m_trips_loop * m_unroll_factor * idx_vlen * idx_tsize  );
  }

  if (peeled_m_trips > 0) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);

    libxsmm_generator_gather_scatter_offs_aarch64_mn_loop_unrolled( io_generated_code, i_gp_reg_mapping, peeled_m_trips, n_unroll_factor,
        i_gp_reg_mapping->gp_reg_ind_base, idx_vlen, idx_tsize, idx_mask_reg, m, gather_instr, scatter_instr, vlen, m_remainder_elements,
        mask_reg, mask_reg_full_frac_vlen, is_gather, gp_idx_mat_reg, gp_reg_mat_reg, dtype_size_reg_mat, ld_reg_mat );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_scratch_0, gp_reg_mat_reg, (long long)n_unroll_factor * ld_reg_mat * dtype_size_reg_mat);
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_ind_base, (long long)n_unroll_factor * m * idx_tsize);

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n_unroll_factor);
  }
#if defined(LIBXSMM_GENERATOR_MATELTWISE_GATHER_SCATTER_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
  free(p_jump_label_tracker);
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS ) > 0 ) {
  } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS ) > 0 ) {
  } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_OFFS ) > 0 ) {
    libxsmm_generator_gather_scatter_offs_aarch64_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    /* SHOULD NOT HAPPEN  */
  }
}
