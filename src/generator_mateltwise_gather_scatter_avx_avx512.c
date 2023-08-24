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
#include "generator_common_x86.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_gather_scatter_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#if 0
#define USE_ENV_TUNING
#endif


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_unified_vec_move_ext( libxsmm_generated_code* io_generated_code,
                                                const unsigned int      i_vmove_instr,
                                                const unsigned int      i_gp_reg_base,
                                                const unsigned int      i_reg_idx,
                                                const unsigned int      i_scale,
                                                const int               i_displacement,
                                                const char              i_vector_name,
                                                const unsigned int      i_vec_reg_number_0,
                                                const unsigned int      i_use_masking,
                                                const unsigned int      i_mask_reg_number,
                                                const unsigned int      use_mask_move_instr,
                                                const unsigned int      use_m_scalar_loads_stores,
                                                const unsigned int      aux_gpr,
                                                const unsigned int      i_is_store ) {
  if ((i_use_masking == 0) || ((i_use_masking > 0) && (use_mask_move_instr > 0))) {
    libxsmm_x86_instruction_unified_vec_move( io_generated_code, i_vmove_instr, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_vector_name, i_vec_reg_number_0, i_use_masking, i_mask_reg_number, i_is_store );
  }
  if ((i_use_masking == 1) && (use_m_scalar_loads_stores > 0)) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVW, i_gp_reg_base, i_reg_idx, i_scale, i_displacement+(use_m_scalar_loads_stores-1)*2, aux_gpr, i_is_store );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_avx_avx512_mn_loop_unrolled( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   n_unroll_factor,
    unsigned int                                   idx_vload_instr,
    unsigned int                                   gp_idx_base_reg,
    unsigned int                                   idx_vlen,
    unsigned int                                   idx_tsize,
    char                                           idx_vname,
    unsigned int                                   idx_mask_reg,
    unsigned int                                   ld_idx,
    unsigned int                                   vload_instr,
    unsigned int                                   vstore_instr,
    unsigned int                                   vlen,
    char                                           vname_load,
    char                                           vname_store,
    unsigned int                                   use_m_masking,
    unsigned int                                   ones_mask_reg,
    unsigned int                                   mask_reg,
    unsigned int                                   help_mask_reg,
    unsigned int                                   is_gather,
    unsigned int                                   gp_idx_mat_reg,
    unsigned int                                   gp_reg_mat_reg,
    unsigned int                                   dtype_size_idx_mat,
    unsigned int                                   dtype_size_reg_mat,
    unsigned int                                   ld_reg_mat ) {

  unsigned int im = 0, in = 0;
  unsigned int aux_vreg_start = m_unroll_factor*n_unroll_factor;
  unsigned int idx_vreg_start = 0;

  for (in = 0; in < n_unroll_factor; in++) {
    for (im = 0; im < m_unroll_factor; im++) {
      unsigned int idx_vreg_id = idx_vreg_start + im + in * m_unroll_factor;
      unsigned int aux_vreg = aux_vreg_start + im + in * m_unroll_factor;

      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
          idx_vload_instr,
          gp_idx_base_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * idx_vlen * idx_tsize + in * ld_idx * idx_tsize,
          idx_vname,
          idx_vreg_id, ((use_m_masking > 0) && (im == m_unroll_factor-1)) ? 1 : 0, idx_mask_reg, 0 );

      if (is_gather == 1) {
        /* Gather based on index vector im*/
        if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
          if (use_m_masking > 0) {
            if (im == m_unroll_factor-1) {
              libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, mask_reg, mask_reg, help_mask_reg, 0);
            } else {
              libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, ones_mask_reg, ones_mask_reg, help_mask_reg, 0);
            }
          } else {
            libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, ones_mask_reg, ones_mask_reg, help_mask_reg, 0);
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              vload_instr,
              gp_idx_mat_reg,
              idx_vreg_id,
              dtype_size_idx_mat,
              0,
              vname_load, aux_vreg, help_mask_reg, 0, 0);
        } else {
          if (use_m_masking > 0) {
            if (im == m_unroll_factor-1) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', mask_reg, help_mask_reg );
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', ones_mask_reg, help_mask_reg );
            }
          } else {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', ones_mask_reg, help_mask_reg );
          }
          libxsmm_x86_instruction_vec_mask_move( io_generated_code,
              vload_instr,
              gp_idx_mat_reg,
              idx_vreg_id,
              dtype_size_idx_mat,
              0,
              vname_load,
              aux_vreg,
              help_mask_reg,
              0);
        }
        /* Store gathered vector */
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vstore_instr,
            gp_reg_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_reg_mat + in * ld_reg_mat * dtype_size_reg_mat,
            vname_store,
            aux_vreg, ((use_m_masking > 0) && (im == m_unroll_factor-1)) ? 1 : 0, mask_reg, 1 );
      } else {
        /* Load vector to be scattered */
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vload_instr,
            gp_reg_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_reg_mat + in*ld_reg_mat*dtype_size_reg_mat,
            vname_load,
            aux_vreg, ((use_m_masking > 0) && (im == m_unroll_factor-1)) ? 1 : 0, mask_reg, 0 );
        /* Scatter based on index vector im*/
        if (use_m_masking > 0) {
          if (im == m_unroll_factor-1) {
            libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, mask_reg, mask_reg, help_mask_reg, 0);
          } else {
            libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, ones_mask_reg, ones_mask_reg, help_mask_reg, 0);
          }
        } else {
          libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, ones_mask_reg, ones_mask_reg, help_mask_reg, 0);
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            io_generated_code->arch,
            vstore_instr,
            gp_idx_mat_reg,
            idx_vreg_id,
            dtype_size_idx_mat,
            0,
            vname_store, aux_vreg, help_mask_reg, 0, 0);
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_avx_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int m, n, use_m_masking, m_trips, max_m_unrolling = 4, m_unroll_factor = 1, n_unroll_factor = 4, m_trips_loop = 0, peeled_m_trips = 0, mask_inout_count = 0;
  unsigned int idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int gather_instr = 0, scatter_instr = 0;
  unsigned int vstore_instr = 0, vload_instr = 0, idx_vload_instr = 0;
  unsigned int mask_reg = 0;
  unsigned int idx_mask_reg = 0;
  unsigned int ones_mask_reg = 2;
  unsigned int help_mask_reg = 3;
  unsigned int vlen = 64/i_micro_kernel_config->datatype_size_in;
  unsigned int idx_vlen = (idx_tsize == 8) ? 8 : 16;
  char vname_load = 'z';
  char vname_store = 'z';
  char idx_vname = 'z';
  unsigned int is_gather = 1;
  unsigned int ld_reg_mat = 0;
  unsigned int dtype_size_idx_mat = 0;
  unsigned int dtype_size_reg_mat = 0;
  unsigned int gp_idx_mat_reg = 0, gp_reg_mat_reg = 0, gp_idx_mat_base_reg = 0;
#if defined(USE_ENV_TUNING)
  const char *const env_max_m_unroll = getenv("MAX_M_UNROLL_GATHER_SCATTER");
  const char *const env_max_n_unroll = getenv("MAX_N_UNROLL_GATHER_SCATTER");
#endif
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

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

  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
    gather_instr  = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_VGATHERQPS : LIBXSMM_X86_INSTR_VGATHERDPS;
  } else {
    gather_instr  = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_VGATHERQPS_VEX : LIBXSMM_X86_INSTR_VGATHERDPS_VEX;
  }
  scatter_instr = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_VSCATTERQPS : LIBXSMM_X86_INSTR_VSCATTERDPS;
  is_gather     = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? 1 : 0;
  vstore_instr  = (is_gather > 0) ? i_micro_kernel_config->vmove_instruction_out : scatter_instr;
  vload_instr   = (is_gather > 0) ? gather_instr :i_micro_kernel_config->vmove_instruction_in;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_X86_GP_REG_RDX;

  gp_idx_mat_base_reg   = LIBXSMM_X86_GP_REG_R10;
  gp_reg_mat_reg        = LIBXSMM_X86_GP_REG_R11;
  gp_idx_mat_reg        = LIBXSMM_X86_GP_REG_R10;

  if (is_gather == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_idx_mat_base_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        40,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_reg_mat_reg,
        0 );

    ld_reg_mat = i_mateltwise_desc->ldo;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_in;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_out;
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_reg_mat_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_idx_mat_base_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        72,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    ld_reg_mat = i_mateltwise_desc->ldi;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_out;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_in;
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
    n_unroll_factor = 1;
    vlen = 32/i_micro_kernel_config->datatype_size_in;
    vname_load = 'y';
    vname_store = 'y';
    idx_vname = 'y';
    idx_vlen = idx_vlen/2;
  }

  if (idx_tsize == 8) {
    idx_vload_instr = LIBXSMM_X86_INSTR_VMOVUPD;
    vlen = vlen/2;
    if (vname_load == 'z') {
      if (is_gather > 0) {
        vname_store = 'y';
      } else {
        vname_load = 'y';
      }
    } else /*if (vname_load == 'y')*/ {
      if (is_gather > 0) {
        vname_store = 'x';
      } else {
        vname_load = 'x';
      }
    }
#if 0
    else { /* should not happen */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return;
    }
#endif
  } else {
    idx_vload_instr = LIBXSMM_X86_INSTR_VMOVUPS;
  }

  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % vlen == 0 ) ? 0 : 1;
  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  while (n % n_unroll_factor > 0) {
    n_unroll_factor--;
  }

  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
    libxsmm_x86_instruction_alu_imm( io_generated_code,
      LIBXSMM_X86_INSTR_MOVQ,
      LIBXSMM_X86_GP_REG_RAX,
      0xffff );
    libxsmm_x86_instruction_mask_move( io_generated_code,
      LIBXSMM_X86_INSTR_KMOVW_GPR_LD,
      LIBXSMM_X86_GP_REG_RAX,
      ones_mask_reg );
  } else {
    ones_mask_reg = 13;
    help_mask_reg = 12;
    libxsmm_generator_initialize_avx_mask(io_generated_code, ones_mask_reg, 8, LIBXSMM_DATATYPE_F32);
  }

  if (use_m_masking == 1) {
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
      /* Calculate mask reg 1 for reading/output-writing */
      mask_inout_count = vlen - (m % vlen);
      mask_reg = 1;
      idx_mask_reg = mask_reg;
      if (i_micro_kernel_config->datatype_size_in == 4) {
        if (idx_tsize == 8) {
          libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_inout_count, LIBXSMM_DATATYPE_F64);
        } else {
          libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_inout_count, LIBXSMM_DATATYPE_F32);
        }
      } else if (i_micro_kernel_config->datatype_size_in == 2) {
        libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_inout_count, LIBXSMM_DATATYPE_BF16);
      } else { /* should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    } else {
      mask_reg = 15;
      idx_mask_reg = 14;
#if defined(USE_ENV_TUNING)
      if (max_m_unrolling > 7) {
        max_m_unrolling = 7;
      }
#endif
      libxsmm_generator_initialize_avx_mask(io_generated_code, mask_reg, m % vlen, LIBXSMM_DATATYPE_F32);
      if (idx_tsize == 8) {
        libxsmm_generator_initialize_avx_mask( io_generated_code, idx_mask_reg, m % vlen, LIBXSMM_DATATYPE_I64);
      } else {
        idx_mask_reg = mask_reg;
      }
    }
  }

  /* In this case we have to generate a loop for m */
  if (m_unroll_factor > max_m_unrolling) {
    m_unroll_factor = max_m_unrolling;
    m_trips_loop = m_trips/m_unroll_factor;
    peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  } else {
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  }

  if (m_trips_loop >= 1) {
    libxsmm_generator_generic_loop_header( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 0, n_unroll_factor );
    libxsmm_generator_generic_loop_header( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 0, 1 );

    libxsmm_generator_gather_scatter_offs_avx_avx512_mn_loop_unrolled( io_generated_code,
        m_unroll_factor, n_unroll_factor,
        idx_vload_instr, i_gp_reg_mapping->gp_reg_ind_base, idx_vlen, idx_tsize, idx_vname, idx_mask_reg, m,
        vload_instr, vstore_instr, vlen, vname_load, vname_store, 0, ones_mask_reg, mask_reg, help_mask_reg,
        is_gather, gp_idx_mat_reg, gp_reg_mat_reg, dtype_size_idx_mat, dtype_size_reg_mat, ld_reg_mat );

    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)m_unroll_factor * vlen * dtype_size_reg_mat);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_ind_base, (long long)m_unroll_factor * idx_tsize * idx_vlen);
    libxsmm_generator_generic_loop_footer( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, ((long long)n_unroll_factor * ld_reg_mat - (long long)m_trips_loop * m_unroll_factor * vlen) * dtype_size_reg_mat);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_ind_base, ((long long)n_unroll_factor * m - (long long)m_trips_loop * m_unroll_factor * idx_vlen) * idx_tsize);
    libxsmm_generator_generic_loop_footer( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_sub_instruction, gp_reg_mat_reg, (long long)ld_reg_mat * dtype_size_reg_mat * n - (long long)m_trips_loop * m_unroll_factor * vlen * dtype_size_reg_mat);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_ind_base, (long long)m * idx_tsize * n - (long long)m_trips_loop * m_unroll_factor * idx_vlen * idx_tsize );
  }

  if (peeled_m_trips > 0) {
    libxsmm_generator_generic_loop_header( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 0, n_unroll_factor );

    libxsmm_generator_gather_scatter_offs_avx_avx512_mn_loop_unrolled( io_generated_code,
        peeled_m_trips, n_unroll_factor,
        idx_vload_instr, i_gp_reg_mapping->gp_reg_ind_base, idx_vlen, idx_tsize, idx_vname, idx_mask_reg, m,
        vload_instr, vstore_instr, vlen, vname_load, vname_store, use_m_masking, ones_mask_reg, mask_reg, help_mask_reg,
        is_gather, gp_idx_mat_reg, gp_reg_mat_reg, dtype_size_idx_mat, dtype_size_reg_mat, ld_reg_mat );

    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)n_unroll_factor * ld_reg_mat * dtype_size_reg_mat);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_ind_base, (long long)n_unroll_factor * m * idx_tsize);
    libxsmm_generator_generic_loop_footer( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_avx_avx512_mn_loop_unrolled( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   n_unroll_factor,
    unsigned int                                   vload_instr,
    unsigned int                                   vstore_instr,
    unsigned int                                   vlen,
    char                                           vname_load,
    char                                           vname_store,
    unsigned int                                   use_m_masking,
    unsigned int                                   ones_mask_reg,
    unsigned int                                   mask_reg,
    unsigned int                                   help_mask_reg,
    unsigned int                                   is_gather,
    unsigned int                                   gp_idx_mat_reg,
    unsigned int                                   gp_reg_mat_reg,
    unsigned int                                   dtype_size_idx_mat,
    unsigned int                                   dtype_size_reg_mat,
    unsigned int                                   ld_idx_mat,
    unsigned int                                   ld_reg_mat ) {

  unsigned int im = 0, in = 0;
  unsigned int aux_vreg = m_unroll_factor;

  for (im = 0; im < m_unroll_factor; im++) {
    for (in = 0; in < n_unroll_factor; in++) {
      if (is_gather == 1) {
        /* Gather based on index vector im*/
        if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
          if (use_m_masking > 0) {
            if (im == m_unroll_factor-1) {
              libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, mask_reg, mask_reg, help_mask_reg, 0);
            } else {
              libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, ones_mask_reg, ones_mask_reg, help_mask_reg, 0);
            }
          } else {
            libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, ones_mask_reg, ones_mask_reg, help_mask_reg, 0);
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              vload_instr,
              gp_idx_mat_reg,
              im,
              dtype_size_idx_mat,
              in*ld_idx_mat*dtype_size_idx_mat,
              vname_load, aux_vreg, help_mask_reg, 0, 0);
        } else {
          if (use_m_masking > 0) {
            if (im == m_unroll_factor-1) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', mask_reg, help_mask_reg );
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', ones_mask_reg, help_mask_reg );
            }
          } else {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', ones_mask_reg, help_mask_reg );
          }
          libxsmm_x86_instruction_vec_mask_move( io_generated_code,
              vload_instr,
              gp_idx_mat_reg,
              im,
              dtype_size_idx_mat,
              in*ld_idx_mat*dtype_size_idx_mat,
              vname_load,
              aux_vreg,
              help_mask_reg,
              0);
        }
        /* Store gathered vector */
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vstore_instr,
            gp_reg_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_reg_mat + in*ld_reg_mat*dtype_size_reg_mat,
            vname_store,
            aux_vreg, ((use_m_masking > 0) && (im == m_unroll_factor-1)) ? 1 : 0, mask_reg, 1 );
      } else {
        /* Load vector to be scattered */
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vload_instr,
            gp_reg_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_reg_mat + in*ld_reg_mat*dtype_size_reg_mat,
            vname_load,
            aux_vreg, ((use_m_masking > 0) && (im == m_unroll_factor-1)) ? 1 : 0, mask_reg, 0 );
        /* Scatter based on index vector im*/
        if (use_m_masking > 0) {
          if (im == m_unroll_factor-1) {
            libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, mask_reg, mask_reg, help_mask_reg, 0);
          } else {
            libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, ones_mask_reg, ones_mask_reg, help_mask_reg, 0);
          }
        } else {
          libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORW, ones_mask_reg, ones_mask_reg, help_mask_reg, 0);
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            io_generated_code->arch,
            vstore_instr,
            gp_idx_mat_reg,
            im,
            dtype_size_idx_mat,
            in*ld_idx_mat*dtype_size_idx_mat,
            vname_store, aux_vreg, help_mask_reg, 0, 0);
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_cols_avx_avx512_m_loop( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    unsigned int                                   m_trips_loop,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   peeled_m_trips,
    unsigned int                                   vload_instr,
    unsigned int                                   vstore_instr,
    unsigned int                                   pf_instr,
    unsigned int                                   vlen,
    char                                           vname,
    unsigned int                                   use_m_masking,
    unsigned int                                   use_mask_move_instr,
    unsigned int                                   use_m_scalar_loads_stores,
    unsigned int                                   mask_reg,
    unsigned int                                   pf_dist,
    unsigned int                                   is_gather,
    unsigned int                                   gp_idx_mat_reg,
    unsigned int                                   gp_idx_mat_pf_reg,
    unsigned int                                   gp_reg_mat_reg,
    unsigned int                                   dtype_size_idx_mat,
    unsigned int                                   dtype_size_reg_mat ) {

  unsigned int im = 0;

  if (m_trips_loop >= 1) {
    if (m_trips_loop > 1) {
      libxsmm_generator_mateltwise_header_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
    }
    if (is_gather == 1) {
      for (im = 0; im < m_unroll_factor; im++) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vload_instr,
            gp_idx_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_idx_mat,
            vname,
            im, 0, 0, 0 );
        if (pf_dist > 0) {
          if ((im * vlen * dtype_size_idx_mat) % 64 == 0 ) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                pf_instr,
                gp_idx_mat_pf_reg,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * vlen * dtype_size_idx_mat);
          }
        }
      }
      for (im = 0; im < m_unroll_factor; im++) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vstore_instr,
            gp_reg_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_reg_mat,
            vname,
            im, 0, 0, 1 );
      }
    } else {
      for (im = 0; im < m_unroll_factor; im++) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vload_instr,
            gp_reg_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_reg_mat,
            vname,
            im, 0, 0, 0 );
      }
      for (im = 0; im < m_unroll_factor; im++) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vstore_instr,
            gp_idx_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_idx_mat,
            vname,
            im, 0, 0, 1 );
        if (pf_dist > 0) {
          if ((im * vlen * dtype_size_idx_mat) % 64 == 0 ) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                pf_instr,
                gp_idx_mat_pf_reg,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * vlen * dtype_size_idx_mat);
          }
        }
      }
    }
    if (m_trips_loop > 1) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)m_unroll_factor * vlen * dtype_size_reg_mat);
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx_mat_reg, (long long)m_unroll_factor * vlen * dtype_size_idx_mat);
      if (pf_dist > 0) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx_mat_pf_reg, (long long)m_unroll_factor * vlen * dtype_size_idx_mat);
      }
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
    }
  }
  if (peeled_m_trips > 0) {
    if (m_trips_loop == 1) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)m_unroll_factor * vlen * dtype_size_reg_mat);
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx_mat_reg, (long long)m_unroll_factor * vlen * dtype_size_idx_mat);
      if (pf_dist > 0) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx_mat_pf_reg, (long long)m_unroll_factor * vlen * dtype_size_idx_mat);
      }
    }
    if (is_gather == 1) {
      for (im = 0; im < peeled_m_trips; im++) {
        libxsmm_x86_instruction_unified_vec_move_ext( io_generated_code,
            vload_instr,
            gp_idx_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_idx_mat,
            vname,
            im, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_reg : 0, use_mask_move_instr, use_m_scalar_loads_stores, i_gp_reg_mapping->gp_reg_m_loop, 0 );
        if (pf_dist > 0) {
          if ((im * vlen * dtype_size_idx_mat) % 64 == 0 ) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                pf_instr,
                gp_idx_mat_pf_reg,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * vlen * dtype_size_idx_mat);
          }
        }
      }
      for (im = 0; im < peeled_m_trips; im++) {
        libxsmm_x86_instruction_unified_vec_move_ext( io_generated_code,
            vstore_instr,
            gp_reg_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_reg_mat,
            vname,
            im, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_reg : 0, use_mask_move_instr, use_m_scalar_loads_stores, i_gp_reg_mapping->gp_reg_m_loop, 1 );
      }
    } else {
      for (im = 0; im < peeled_m_trips; im++) {
        libxsmm_x86_instruction_unified_vec_move_ext( io_generated_code,
            vload_instr,
            gp_reg_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_reg_mat,
            vname,
            im, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_reg : 0, use_mask_move_instr, use_m_scalar_loads_stores, i_gp_reg_mapping->gp_reg_m_loop, 0 );
      }
      for (im = 0; im < peeled_m_trips; im++) {
        libxsmm_x86_instruction_unified_vec_move_ext( io_generated_code,
            vstore_instr,
            gp_idx_mat_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * dtype_size_idx_mat,
            vname,
            im, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_reg : 0, use_mask_move_instr, use_m_scalar_loads_stores, i_gp_reg_mapping->gp_reg_m_loop, 1);
        if (pf_dist > 0) {
          if ((im * vlen * dtype_size_idx_mat) % 64 == 0 ) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                pf_instr,
                gp_idx_mat_pf_reg,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * vlen * dtype_size_idx_mat);
          }
        }
      }
    }
  }

  /* Adjust m advancements */
  if (m_trips_loop >= 1) {
    if (peeled_m_trips > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, gp_reg_mat_reg, (long long)m_trips_loop * m_unroll_factor * vlen * dtype_size_reg_mat);
    } else {
      if (m_trips_loop > 1) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, gp_reg_mat_reg, (long long)m_trips_loop * m_unroll_factor * vlen * dtype_size_reg_mat);
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_cols_avx_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, use_m_masking, m_trips, max_m_unrolling = 4, m_unroll_factor = 1, m_trips_loop = 0, peeled_m_trips = 0, mask_inout_count = 0;
  unsigned int idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int ind_alu_mov_instr = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_MOVQ : LIBXSMM_X86_INSTR_MOVD;
  int pf_dist = 4, use_nts = 0, pf_instr = LIBXSMM_X86_INSTR_PREFETCHT1, pf_type = 1;
  unsigned int vstore_instr = 0, vload_instr = 0;
  unsigned int mask_reg = 0;
  unsigned int use_m_scalar_loads_stores = 0;
  unsigned int use_mask_move_instr = 0;
  unsigned int vlen = 64/i_micro_kernel_config->datatype_size_in;
  int n_pf_iters = 0;
  char vname = 'z';
  unsigned int is_gather = 1;
  unsigned int ld_idx_mat = 0, ld_reg_mat = 0;
  unsigned int dtype_size_idx_mat = 0;
  unsigned int dtype_size_reg_mat = 0;
  unsigned int gp_idx_mat_reg = 0, gp_idx_mat_pf_reg = 0, gp_reg_mat_reg = 0, gp_idx_mat_base_reg = 0;
  const char *const env_max_m_unroll = getenv("MAX_M_UNROLL_GATHER_SCATTER");
  const char *const env_pf_dist = getenv("PF_DIST_GATHER_SCATTER");
  const char *const env_pf_type = getenv("PF_TYPE_GATHER_SCATTER");
  const char *const env_nts     = getenv("NTS_GATHER_SCATTER");
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  if ( 0 == env_max_m_unroll ) {
  } else {
    max_m_unrolling = LIBXSMM_MAX(1, atoi(env_max_m_unroll));
  }
  if ( 0 == env_pf_dist ) {
  } else {
    pf_dist = atoi(env_pf_dist);
  }
  if ( 0 == env_pf_type ) {
  } else {
    pf_type = atoi(env_pf_type);
  }
  if ( 0 == env_nts ) {
  } else {
    use_nts = atoi(env_nts);
  }

  pf_instr      = (pf_type == 2) ? LIBXSMM_X86_INSTR_PREFETCHT1 : LIBXSMM_X86_INSTR_PREFETCHT0;
  vstore_instr  = (use_nts == 0) ? i_micro_kernel_config->vmove_instruction_out : LIBXSMM_X86_INSTR_VMOVNTPS;
  vload_instr   = i_micro_kernel_config->vmove_instruction_in;
  is_gather     = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? 1 : 0;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_X86_GP_REG_RDX;

  gp_idx_mat_base_reg   = LIBXSMM_X86_GP_REG_R10;
  gp_reg_mat_reg        = LIBXSMM_X86_GP_REG_R11;
  gp_idx_mat_reg        = LIBXSMM_X86_GP_REG_RSI;
  gp_idx_mat_pf_reg     = LIBXSMM_X86_GP_REG_RCX;

  if (is_gather == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_idx_mat_base_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        40,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_reg_mat_reg,
        0 );

    ld_idx_mat = i_mateltwise_desc->ldi;
    ld_reg_mat = i_mateltwise_desc->ldo;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_in;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_out;
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_reg_mat_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_idx_mat_base_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        72,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    ld_idx_mat = i_mateltwise_desc->ldo;
    ld_reg_mat = i_mateltwise_desc->ldi;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_out;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_in;
    pf_instr  = LIBXSMM_X86_INSTR_PREFETCHW;
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
    vlen = 32/i_micro_kernel_config->datatype_size_in;
    vname = 'y';
    vstore_instr  = LIBXSMM_X86_INSTR_VMOVUPS;
    vload_instr   = LIBXSMM_X86_INSTR_VMOVUPS;
  } else if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
    vlen = 32/i_micro_kernel_config->datatype_size_in;
    vname = 'y';
  }

  m                 = i_mateltwise_desc->m;
  use_m_masking     = ( m % vlen == 0 ) ? 0 : 1;
  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  if (use_m_masking == 1) {
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
      /* Calculate mask reg 1 for reading/output-writing */
      use_mask_move_instr = 1;
      mask_inout_count = vlen - (m % vlen);
      mask_reg = 1;
      if (i_micro_kernel_config->datatype_size_in == 4) {
        libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_inout_count, LIBXSMM_DATATYPE_F32);
      } else if (i_micro_kernel_config->datatype_size_in == 2) {
        libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_inout_count, LIBXSMM_DATATYPE_BF16);
      } else { /* should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    } else {
      mask_reg = 15;
      if (max_m_unrolling > 7) {
        max_m_unrolling = 7;
      }
      if (i_micro_kernel_config->datatype_size_in == 4) {
        use_mask_move_instr = 1;
        libxsmm_generator_initialize_avx_mask(io_generated_code, mask_reg, m % vlen, LIBXSMM_DATATYPE_F32);
      } else if (i_micro_kernel_config->datatype_size_in == 2) {
        unsigned int half_m = m/2;
        unsigned int half_vlen = vlen/2;
        if ( half_m % half_vlen > 0) {
          use_mask_move_instr = 1;
          libxsmm_generator_initialize_avx_mask(io_generated_code, mask_reg, half_m % half_vlen, LIBXSMM_DATATYPE_F32);
        }
        if ((m % vlen) % 2 == 1) {
          use_m_scalar_loads_stores = m % vlen;
        }
      } else { /* should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }
  }

  /* In this case we have to generate a loop for m */
  if (m_unroll_factor > max_m_unrolling) {
    m_unroll_factor = max_m_unrolling;
    m_trips_loop = m_trips/m_unroll_factor;
    peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  } else {
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  }

  n_pf_iters = LIBXSMM_MAX(0, i_mateltwise_desc->n - pf_dist);
  if (pf_dist <= 0) {
    n_pf_iters = 0;
  }

  if ((n_pf_iters > 0) && (pf_dist > 0)) {
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_n_loop, 0);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_n, n_pf_iters);

    libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, 1);
    libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, gp_idx_mat_reg, 0);
    libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, pf_dist * idx_tsize, gp_idx_mat_pf_reg, 0);
    libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, gp_idx_mat_reg, (long long)ld_idx_mat * dtype_size_idx_mat);
    libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, gp_idx_mat_pf_reg, (long long)ld_idx_mat * dtype_size_idx_mat);
    libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx_mat_base_reg, gp_idx_mat_reg);
    libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx_mat_base_reg, gp_idx_mat_pf_reg);

    libxsmm_generator_gather_scatter_cols_avx_avx512_m_loop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, m_trips_loop, m_unroll_factor, peeled_m_trips,
        vload_instr, vstore_instr, pf_instr, vlen, vname, use_m_masking, use_mask_move_instr, use_m_scalar_loads_stores, mask_reg, pf_dist,
        is_gather, gp_idx_mat_reg, gp_idx_mat_pf_reg, gp_reg_mat_reg, dtype_size_idx_mat, dtype_size_reg_mat );

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)ld_reg_mat * dtype_size_reg_mat);
    libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
  } else {
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_n_loop, 0);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_n, i_mateltwise_desc->n);
  }

  libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, 1);
  libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, gp_idx_mat_reg, 0);
  libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, gp_idx_mat_reg, (long long)ld_idx_mat * dtype_size_idx_mat);
  libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx_mat_base_reg, gp_idx_mat_reg);

  libxsmm_generator_gather_scatter_cols_avx_avx512_m_loop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, m_trips_loop, m_unroll_factor, peeled_m_trips,
      vload_instr, vstore_instr, pf_instr, vlen, vname, use_m_masking, use_mask_move_instr, use_m_scalar_loads_stores, mask_reg, 0,
      is_gather, gp_idx_mat_reg, gp_idx_mat_pf_reg, gp_reg_mat_reg, dtype_size_idx_mat, dtype_size_reg_mat );

  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)ld_reg_mat * dtype_size_reg_mat);
  libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_scalar_x86_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int m, n;
  unsigned int idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int load_instr = 0, store_instr = 0, idx_load_instr = 0, gp_idx = 0, gp_aux = 0;
  unsigned int is_gather = 1;
  unsigned int ld_idx_mat = 0, ld_reg_mat = 0;
  unsigned int dtype_size_idx_mat = 0;
  unsigned int dtype_size_reg_mat = 0;
  unsigned int gp_idx_mat_reg = 0, gp_reg_mat_reg = 0;
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_X86_GP_REG_RDX;

  gp_reg_mat_reg        = LIBXSMM_X86_GP_REG_R11;
  gp_idx_mat_reg        = LIBXSMM_X86_GP_REG_R10;
  gp_aux                = LIBXSMM_X86_GP_REG_RCX;
  gp_idx                = LIBXSMM_X86_GP_REG_R8;
  is_gather     = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? 1 : 0;

  if (is_gather == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_idx_mat_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        40,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_reg_mat_reg,
        0 );

    ld_idx_mat = i_mateltwise_desc->ldi;
    ld_reg_mat = i_mateltwise_desc->ldo;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_in;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_out;
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_reg_mat_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_idx_mat_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        72,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    ld_idx_mat = i_mateltwise_desc->ldo;
    ld_reg_mat = i_mateltwise_desc->ldi;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_out;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_in;
  }

  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  if ( idx_tsize == 8 ) {
    idx_load_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( idx_tsize == 4 ) {
    idx_load_instr = LIBXSMM_X86_INSTR_MOVD;
  }
  if (i_micro_kernel_config->datatype_size_in == 2) {
    load_instr = LIBXSMM_X86_INSTR_MOVW;
    store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if (i_micro_kernel_config->datatype_size_in == 4) {
    load_instr = LIBXSMM_X86_INSTR_MOVD;
    store_instr = LIBXSMM_X86_INSTR_MOVD;
  }

  /* M loop */
  libxsmm_generator_generic_loop_header_no_idx_inc( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 0);
  /* Load row index and add the corresponding offset to the idx_mat_reg*/
  libxsmm_x86_instruction_alu_mem( io_generated_code, idx_load_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_m_loop, idx_tsize, 0, gp_idx, 0 );
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_LEAQ, gp_idx_mat_reg, gp_idx, dtype_size_idx_mat, 0, gp_idx, 0);

  /* N loop */
  libxsmm_generator_generic_loop_header( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 0, 1);
  if (is_gather == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, load_instr,  gp_idx,         LIBXSMM_X86_GP_REG_UNDEF, 0, 0, gp_aux, 0 );
    libxsmm_x86_instruction_alu_mem( io_generated_code, store_instr, gp_reg_mat_reg, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, gp_aux, 1 );
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code, load_instr,  gp_reg_mat_reg, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, gp_aux, 0 );
    libxsmm_x86_instruction_alu_mem( io_generated_code, store_instr, gp_idx,         LIBXSMM_X86_GP_REG_UNDEF, 0, 0, gp_aux, 1 );
  }
  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)ld_reg_mat * dtype_size_reg_mat);
  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx,         (long long)ld_idx_mat * dtype_size_idx_mat);
  libxsmm_generator_generic_loop_footer( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);

  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_sub_instruction, gp_reg_mat_reg, (long long)ld_reg_mat * dtype_size_reg_mat * n - (long long)dtype_size_reg_mat);
  libxsmm_generator_generic_loop_footer_with_idx_inc( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1, m);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_scalar_x86_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int m, n;
  unsigned int idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int load_instr = 0, store_instr = 0, idx_load_instr = 0, gp_idx = 0, gp_aux = 0;
  unsigned int is_gather = 1;
  unsigned int ld_reg_mat = 0;
  unsigned int dtype_size_idx_mat = 0;
  unsigned int dtype_size_reg_mat = 0;
  unsigned int gp_idx_mat_reg = 0, gp_reg_mat_reg = 0;
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_X86_GP_REG_RDX;

  gp_reg_mat_reg        = LIBXSMM_X86_GP_REG_R11;
  gp_idx_mat_reg        = LIBXSMM_X86_GP_REG_R10;
  gp_aux                = LIBXSMM_X86_GP_REG_RCX;
  gp_idx                = LIBXSMM_X86_GP_REG_R8;
  is_gather     = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? 1 : 0;

  if (is_gather == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_idx_mat_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        40,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_reg_mat_reg,
        0 );

    ld_reg_mat = i_mateltwise_desc->ldo;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_in;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_out;
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_reg_mat_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_idx_mat_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        72,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    ld_reg_mat = i_mateltwise_desc->ldi;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_out;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_in;
  }

  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  if ( idx_tsize == 8 ) {
    idx_load_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( idx_tsize == 4 ) {
    idx_load_instr = LIBXSMM_X86_INSTR_MOVD;
  }
  if (i_micro_kernel_config->datatype_size_in == 2) {
    load_instr = LIBXSMM_X86_INSTR_MOVW;
    store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if (i_micro_kernel_config->datatype_size_in == 4) {
    load_instr = LIBXSMM_X86_INSTR_MOVD;
    store_instr = LIBXSMM_X86_INSTR_MOVD;
  }

  /* N loop */
  libxsmm_generator_generic_loop_header( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 0, 1 );
  /* M loop */
  libxsmm_generator_generic_loop_header_no_idx_inc( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 0);
  /* Load absolute offset */
  libxsmm_x86_instruction_alu_mem( io_generated_code, idx_load_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_m_loop, idx_tsize, 0, gp_idx, 0 );
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_LEAQ, gp_idx_mat_reg, gp_idx, dtype_size_idx_mat, 0, gp_idx, 0);
  if (is_gather == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, load_instr,  gp_idx,         LIBXSMM_X86_GP_REG_UNDEF       , 0                 , 0, gp_aux, 0 );
    libxsmm_x86_instruction_alu_mem( io_generated_code, store_instr, gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_m_loop, dtype_size_reg_mat, 0, gp_aux, 1 );
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code, load_instr,  gp_reg_mat_reg, i_gp_reg_mapping->gp_reg_m_loop, dtype_size_reg_mat, 0, gp_aux, 0 );
    libxsmm_x86_instruction_alu_mem( io_generated_code, store_instr, gp_idx,         LIBXSMM_X86_GP_REG_UNDEF       , 0                 , 0, gp_aux, 1 );
  }
  libxsmm_generator_generic_loop_footer_with_idx_inc( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1, m);
  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_ind_base, (long long)idx_tsize * m);
  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)ld_reg_mat * dtype_size_reg_mat);
  libxsmm_generator_generic_loop_footer( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_avx_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int m, n, im, use_m_masking, m_trips, max_m_unrolling = 4, m_unroll_factor = 1, n_unroll_factor = 4, m_trips_loop = 0, peeled_m_trips = 0, mask_inout_count = 0;
  unsigned int idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int gather_instr = 0, scatter_instr = 0;
  unsigned int vstore_instr = 0, vload_instr = 0, idx_vload_instr = 0;
  unsigned int mask_reg = 0;
  unsigned int idx_mask_reg = 0;
  unsigned int ones_mask_reg = 2;
  unsigned int help_mask_reg = 3;
  unsigned int vlen = 64/i_micro_kernel_config->datatype_size_in;
  unsigned int idx_vlen = (idx_tsize == 8) ? 8 : 16;
  char vname_load = 'z';
  char vname_store = 'z';
  char idx_vname = 'z';
  unsigned int is_gather = 1;
  unsigned int ld_idx_mat = 0, ld_reg_mat = 0;
  unsigned int dtype_size_idx_mat = 0;
  unsigned int dtype_size_reg_mat = 0;
  unsigned int gp_idx_mat_reg = 0, gp_reg_mat_reg = 0, gp_idx_mat_base_reg = 0;
#if defined(USE_ENV_TUNING)
  const char *const env_max_m_unroll = getenv("MAX_M_UNROLL_GATHER_SCATTER");
  const char *const env_max_n_unroll = getenv("MAX_N_UNROLL_GATHER_SCATTER");
#endif
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

#if defined(USE_ENV_TUNING)
  if ( 0 == env_max_m_unroll ) {
  } else {
    max_m_unrolling = LIBXSMM_MAX(1, atoi(env_max_m_unroll));
  }
  if ( 0 == env_max_n_unroll ) {
  } else {
    n_unroll_factor = LIBXSMM_MAX(1, atoi(env_max_n_unroll));
  }
#endif

  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
    gather_instr  = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_VGATHERQPS : LIBXSMM_X86_INSTR_VGATHERDPS;
  } else {
    gather_instr  = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_VGATHERQPS_VEX : LIBXSMM_X86_INSTR_VGATHERDPS_VEX;
  }
  scatter_instr = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_VSCATTERQPS : LIBXSMM_X86_INSTR_VSCATTERDPS;
  is_gather     = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? 1 : 0;
  vstore_instr  = (is_gather > 0) ? i_micro_kernel_config->vmove_instruction_out : scatter_instr;
  vload_instr   = (is_gather > 0) ? gather_instr :i_micro_kernel_config->vmove_instruction_in;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_X86_GP_REG_RDX;

  gp_idx_mat_base_reg   = LIBXSMM_X86_GP_REG_R10;
  gp_reg_mat_reg        = LIBXSMM_X86_GP_REG_R11;
  gp_idx_mat_reg        = LIBXSMM_X86_GP_REG_R10;

  if (is_gather == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_idx_mat_base_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        40,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_reg_mat_reg,
        0 );

    ld_idx_mat = i_mateltwise_desc->ldi;
    ld_reg_mat = i_mateltwise_desc->ldo;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_in;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_out;
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        gp_reg_mat_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        gp_idx_mat_base_reg,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        72,
        i_gp_reg_mapping->gp_reg_ind_base,
        0 );

    ld_idx_mat = i_mateltwise_desc->ldo;
    ld_reg_mat = i_mateltwise_desc->ldi;
    dtype_size_idx_mat = i_micro_kernel_config->datatype_size_out;
    dtype_size_reg_mat = i_micro_kernel_config->datatype_size_in;
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
    vlen = 32/i_micro_kernel_config->datatype_size_in;
    vname_load = 'y';
    vname_store = 'y';
    idx_vname = 'y';
    idx_vlen = idx_vlen/2;
  }

  if (idx_tsize == 8) {
    idx_vload_instr = LIBXSMM_X86_INSTR_VMOVUPD;
    vlen = vlen/2;
    if (vname_load == 'z') {
      if (is_gather > 0) {
        vname_store = 'y';
      } else {
        vname_load = 'y';
      }
    } else if (vname_load == 'y') {
      if (is_gather > 0) {
        vname_store = 'x';
      } else {
        vname_load = 'x';
      }
    }
#if 0
    else { /* should not happen */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return;
    }
#endif
  } else {
    idx_vload_instr = LIBXSMM_X86_INSTR_VMOVUPS;
  }

  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % vlen == 0 ) ? 0 : 1;
  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  while (n % n_unroll_factor > 0) {
    n_unroll_factor--;
  }

  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
    libxsmm_x86_instruction_alu_imm( io_generated_code,
      LIBXSMM_X86_INSTR_MOVQ,
      LIBXSMM_X86_GP_REG_RAX,
      0xffff );
    libxsmm_x86_instruction_mask_move( io_generated_code,
      LIBXSMM_X86_INSTR_KMOVW_GPR_LD,
      LIBXSMM_X86_GP_REG_RAX,
      ones_mask_reg );
  } else {
    ones_mask_reg = 13;
    help_mask_reg = 12;
    libxsmm_generator_initialize_avx_mask(io_generated_code, ones_mask_reg, 8, LIBXSMM_DATATYPE_F32);
  }

  if (use_m_masking == 1) {
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
      /* Calculate mask reg 1 for reading/output-writing */
      mask_inout_count = vlen - (m % vlen);
      mask_reg = 1;
      idx_mask_reg = mask_reg;
      if (i_micro_kernel_config->datatype_size_in == 4) {
        if (idx_tsize == 8) {
          libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_inout_count, LIBXSMM_DATATYPE_F64);
        } else {
          libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_inout_count, LIBXSMM_DATATYPE_F32);
        }
      } else if (i_micro_kernel_config->datatype_size_in == 2) {
        libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_inout_count, LIBXSMM_DATATYPE_BF16);
      } else { /* should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    } else {
      mask_reg = 15;
      idx_mask_reg = 14;
#if defined(USE_ENV_TUNING)
      if (max_m_unrolling > 7) {
        max_m_unrolling = 7;
      }
#endif
      libxsmm_generator_initialize_avx_mask(io_generated_code, mask_reg, m % vlen, LIBXSMM_DATATYPE_F32);
      if (idx_tsize == 8) {
        libxsmm_generator_initialize_avx_mask( io_generated_code, idx_mask_reg, m % vlen, LIBXSMM_DATATYPE_I64);
      } else {
        idx_mask_reg = mask_reg;
      }
    }
  }

  /* In this case we have to generate a loop for m */
  if (m_unroll_factor > max_m_unrolling) {
    m_unroll_factor = max_m_unrolling;
    m_trips_loop = m_trips/m_unroll_factor;
    peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  } else {
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  }


  if (m_trips_loop >= 1) {
    libxsmm_generator_generic_loop_header( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 0, 1 );

    /* Load Gather/Scatter indices */
    for (im = 0; im < m_unroll_factor; im++) {
      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
          idx_vload_instr,
          i_gp_reg_mapping->gp_reg_ind_base,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * idx_vlen * idx_tsize,
          idx_vname,
          im, 0, 0, 0 );
    }

    libxsmm_generator_generic_loop_header( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 0, n_unroll_factor );

    libxsmm_generator_gather_scatter_rows_avx_avx512_mn_loop_unrolled( io_generated_code,
        m_unroll_factor, n_unroll_factor,
        vload_instr, vstore_instr, vlen, vname_load, vname_store, 0, ones_mask_reg, mask_reg, help_mask_reg,
        is_gather, gp_idx_mat_reg, gp_reg_mat_reg, dtype_size_idx_mat, dtype_size_reg_mat, ld_idx_mat, ld_reg_mat );

    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)n_unroll_factor * ld_reg_mat * dtype_size_reg_mat);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx_mat_reg, (long long)n_unroll_factor * ld_idx_mat * dtype_size_idx_mat);
    libxsmm_generator_generic_loop_footer( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_sub_instruction, gp_reg_mat_reg, (long long)ld_reg_mat * dtype_size_reg_mat * n);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_sub_instruction, gp_idx_mat_reg, (long long)ld_idx_mat * dtype_size_idx_mat * n);

    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)m_unroll_factor * vlen * dtype_size_reg_mat);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_ind_base, (long long)m_unroll_factor * idx_tsize * idx_vlen);
    libxsmm_generator_generic_loop_footer( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
  }

  if (peeled_m_trips > 0) {
    /* Load Gather/Scatter indices */
    for (im = 0; im < peeled_m_trips; im++) {
      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
          idx_vload_instr,
          i_gp_reg_mapping->gp_reg_ind_base,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * idx_vlen * idx_tsize,
          idx_vname,
          im, ((use_m_masking > 0) && (im == peeled_m_trips-1)) ? 1 : 0, idx_mask_reg, 0 );
    }
    libxsmm_generator_generic_loop_header( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 0, n_unroll_factor );

    libxsmm_generator_gather_scatter_rows_avx_avx512_mn_loop_unrolled( io_generated_code,
        peeled_m_trips, n_unroll_factor,
        vload_instr, vstore_instr, vlen, vname_load, vname_store, use_m_masking, ones_mask_reg, mask_reg, help_mask_reg,
        is_gather, gp_idx_mat_reg, gp_reg_mat_reg, dtype_size_idx_mat, dtype_size_reg_mat, ld_idx_mat, ld_reg_mat );

    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_mat_reg, (long long)n_unroll_factor * ld_reg_mat * dtype_size_reg_mat);
    libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_idx_mat_reg, (long long)n_unroll_factor * ld_idx_mat * dtype_size_idx_mat);
    libxsmm_generator_generic_loop_footer( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_avx_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS ) > 0 ) {
    libxsmm_generator_gather_scatter_cols_avx_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS ) > 0 ) {
    if ( (i_micro_kernel_config->datatype_size_in == 4) && (i_micro_kernel_config->datatype_size_out == 4) &&
         ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) || ((io_generated_code->arch >= LIBXSMM_X86_AVX2) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER)))) {
      libxsmm_generator_gather_scatter_rows_avx_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      libxsmm_generator_gather_scatter_rows_scalar_x86_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
    }
  } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_OFFS ) > 0 ) {
    if ( (i_micro_kernel_config->datatype_size_in == 4) && (i_micro_kernel_config->datatype_size_out == 4) &&
         ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) || ((io_generated_code->arch >= LIBXSMM_X86_AVX2) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER)))) {
      libxsmm_generator_gather_scatter_offs_avx_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      libxsmm_generator_gather_scatter_offs_scalar_x86_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
    }
  } else {
    /* SHOULD NOT HAPPEN */
  }
}
