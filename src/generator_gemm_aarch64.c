/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_aarch64_instructions.h"
#include "generator_gemm_aarch64.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_loop_header_aarch64( libxsmm_generated_code*     io_generated_code,
                                                 libxsmm_loop_label_tracker* io_loop_label_tracker,
                                                 const unsigned int          i_gp_reg_loop_cnt,
                                                 const unsigned int          i_trips ) {
  libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_loop_cnt, (unsigned long long)i_trips );
  libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_loop_footer_aarch64( libxsmm_generated_code*     io_generated_code,
                                                 libxsmm_loop_label_tracker* io_loop_label_tracker,
                                                 const unsigned int          i_gp_reg_loop_cnt,
                                                 const unsigned int          i_loop_blocking ) {
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                 i_gp_reg_loop_cnt, i_gp_reg_loop_cnt, i_loop_blocking, 0 );
  libxsmm_aarch64_instruction_cond_jump_back_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBNZ,
                                                       i_gp_reg_loop_cnt, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_2dregblock_aarch64( libxsmm_generated_code* io_generated_code,
                                                     const unsigned int      i_gp_reg_addr,
                                                     const unsigned int      i_gp_reg_scratch_a,
                                                     const unsigned int      i_gp_reg_scratch_b,
                                                     const unsigned int      i_vec_length,
                                                     const unsigned int      i_vec_reg_count,
                                                     const unsigned int      i_m_blocking,
                                                     const unsigned int      i_n_blocking,
                                                     const unsigned int      i_ld,
                                                     const unsigned int      i_zero ) {
  unsigned int l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3];  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks;
  unsigned int l_m_bytes = 0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_vec_length)%(i_vec_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];
  l_m_bytes = l_m_blocks[0]*16 +  l_m_blocks[1]*8 + l_m_blocks[2]*4;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* load C accumulator */
  if (i_zero == 0) {
    if ( i_ld-l_m_bytes > 0 ) {
      libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_scratch_a, (unsigned long long)((unsigned long long)(i_ld-l_m_bytes)) );
    }
    libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_scratch_b, (unsigned long long)((unsigned long long)(i_ld*i_n_blocking)) );
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                                l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                                l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_D );
      }
      for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                                l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_S );
      }
      if ( i_ld-l_m_bytes > 0 ) {
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                             i_gp_reg_addr, i_gp_reg_scratch_a, i_gp_reg_addr,
                                                             0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
    }
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                         i_gp_reg_addr, i_gp_reg_scratch_b, i_gp_reg_addr,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  } else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n), 0,
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n), 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
      }
      for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n), 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_2dregblock_aarch64( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_addr,
                                                      const unsigned int      i_gp_reg_scratch_a,
                                                      const unsigned int      i_gp_reg_scratch_b,
                                                      const unsigned int      i_vec_length,
                                                      const unsigned int      i_vec_reg_count,
                                                      const unsigned int      i_m_blocking,
                                                      const unsigned int      i_n_blocking,
                                                      const unsigned int      i_ld ) {
  unsigned int l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3];  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks;
  unsigned int l_m_bytes = 0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_vec_length)%(i_vec_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];
  l_m_bytes = l_m_blocks[0]*16 +  l_m_blocks[1]*8 + l_m_blocks[2]*4;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* store C accumulator */
  if ( i_ld-l_m_bytes > 0 ) {
    libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_scratch_a, (unsigned long long)((unsigned long long)(i_ld-l_m_bytes)) );
  }
  libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_scratch_b, (unsigned long long)((unsigned long long)(i_ld*i_n_blocking)) );
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                              l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    }
    for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                              l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_D );
    }
    for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                              l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_S );
    }
    if ( i_ld-l_m_bytes > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                           i_gp_reg_addr, i_gp_reg_scratch_a, i_gp_reg_addr,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                       i_gp_reg_addr, i_gp_reg_scratch_b, i_gp_reg_addr,
                                                       0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
}

#if 0
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                       const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                       const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                       libxsmm_micro_kernel_config*       i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_destroy_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config );
#endif

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_aarch64( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                              const unsigned int             i_arch,
                                                              const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  if ( i_arch  == LIBXSMM_AARCH64_V81 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_AARCH64_V81;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'v';
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 2;
      io_micro_kernel_config->datatype_size = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else {
      /* should not happend */
    }
  } else {
    /* that should no happen */
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_get_max_n_blocking( const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                const unsigned int                  i_arch ) {
  LIBXSMM_UNUSED( i_micro_kernel_config );
  LIBXSMM_UNUSED( i_xgemm_desc );

  if ( i_arch == LIBXSMM_AARCH64_V81 ) {
    return 30;
  } else {
    return 0;
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_get_initial_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                    const unsigned int              i_arch ) {
  unsigned int l_m_blocking = 0;

  if ( ( i_arch == LIBXSMM_AARCH64_V81 ) && ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 16 ) {
      l_m_blocking = 16;
    } else {
      l_m_blocking = i_xgemm_desc->m;
      /* in case we don't have a full vector length, we use masking */
      if (l_m_blocking == 15) {  /* for 15 we would need 5 M registers :-( 4-4-4-2-1 */
        l_m_blocking = 12;
      }
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_V81 ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 8 ) {
      l_m_blocking = 8;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( io_micro_kernel_config, i_arch, i_xgemm_desc );

  return l_m_blocking;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_update_m_blocking( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                               const unsigned int             i_arch,
                                                               const unsigned int             i_current_m_blocking ) {
  unsigned int l_m_blocking = 0;

  if ( ( i_arch == LIBXSMM_AARCH64_V81 ) && ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32 out kernel with the same logic */
    if (i_current_m_blocking == 16) {
      l_m_blocking = i_xgemm_desc->m % 16;
      if (l_m_blocking == 15) { /* for 15 we would need 5 M registers 4-4-4-2-1 */
        l_m_blocking = 12;
      }
    } else if ( i_current_m_blocking == 12 && i_xgemm_desc->m != 12 ) {
      l_m_blocking = i_xgemm_desc->m % 4;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_V81 ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 8) {
      l_m_blocking = i_xgemm_desc->m % 8;
    } else {
      /* we are done with m_blocking */
    }
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( io_micro_kernel_config, i_arch, i_xgemm_desc );

  return l_m_blocking;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_n_blocking( libxsmm_generated_code*        io_generated_code,
                                                      libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                      const unsigned int             i_arch,
                                                      unsigned int*                  o_n_N,
                                                      unsigned int*                  o_n_n) {
  unsigned int max_n_blocking = libxsmm_generator_gemm_aarch64_get_max_n_blocking( io_micro_kernel_config, i_xgemm_desc, i_arch );
  const unsigned int init_m_blocking = libxsmm_generator_gemm_aarch64_get_initial_m_blocking( io_micro_kernel_config, i_xgemm_desc, i_arch );
  unsigned int init_m_blocks = 0;

  /* check for valid values */
  if ( max_n_blocking == 0 || init_m_blocking == 0 || io_micro_kernel_config->vector_length == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  init_m_blocks = LIBXSMM_UPDIV(init_m_blocking, io_micro_kernel_config->vector_length);

  /* increment m register blocking in case of 2 remainder registers */
  if ( init_m_blocking % io_micro_kernel_config->vector_length == 3 ) {
    init_m_blocks++;
  }
  while ((init_m_blocks * max_n_blocking + init_m_blocks + 1) > io_micro_kernel_config->vector_reg_count) {
    max_n_blocking--;
  }

  libxsmm_compute_equalized_blocking( i_xgemm_desc->n, max_n_blocking, &(o_n_N[0]), &(o_n_n[0]), &(o_n_N[1]), &(o_n_n[1]) );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_k_strides( libxsmm_generated_code*            io_generated_code,
                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                     const unsigned int                 i_m_blocking,
                                                     const unsigned int                 i_n_blocking ) {
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* register blocking counter in n */
  unsigned int l_n = 0;

  /* preload offset og B */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size;
  } else {
    l_b_offset = i_micro_kernel_config->datatype_size;
  }
  libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, (unsigned long long)l_b_offset );

  /* preload offset of A */
  libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                           (unsigned long long)((unsigned long long)(i_xgemm_desc->lda - i_m_blocking) * i_micro_kernel_config->datatype_size) );

  /* load b offsets */
  if ( i_n_blocking < 7 ) {
    for ( l_n = 1; l_n < i_n_blocking; l_n++ ) {
      /* handle trans B */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = l_n * i_micro_kernel_config->datatype_size;
      } else {
        l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size;
      }
      libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 + (l_n - 1), (unsigned long long)l_b_offset );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel( libxsmm_generated_code*            io_generated_code,
                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                 const unsigned int                 i_m_blocking,
                                                 const unsigned int                 i_n_blocking ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 0;
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3] = { 0 };  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks = 0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_micro_kernel_config->vector_length;                                            /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_micro_kernel_config->vector_length)/(i_micro_kernel_config->vector_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_micro_kernel_config->vector_length)%(i_micro_kernel_config->vector_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];

  /* start register of accumulator */
  l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_total_blocks);

  for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                            1 + l_m, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  }
  for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                            1 + l_m + l_m_blocks[0], LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }
  for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                            1 + l_m + l_m_blocks[0] + l_m_blocks[1], LIBXSMM_AARCH64_ASIMD_WIDTH_S );
  }

  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    if ( i_n_blocking > 6 ) {
      /* handle trans B */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = l_n * i_micro_kernel_config->datatype_size;
      } else {
        l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size;
      }

      libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2, (unsigned long long)l_b_offset );

      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                              i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, 0,
                                              0, (i_micro_kernel_config->datatype_size == 4) ?  LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D );
    } else {
      if ( l_n == 0 ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                                i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                                0, (i_micro_kernel_config->datatype_size == 4) ?  LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D );
      } else {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                                i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2 + (l_n - 1), 0,
                                                0, (i_micro_kernel_config->datatype_size == 4) ?  LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D );
      }
    }

    if (l_n == i_n_blocking - 1) {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                           i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                           i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    /* issude FMSa */
    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V,
                                                 1 + l_m, 0, 0, l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                 (i_micro_kernel_config->datatype_size == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
    }
    for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
      if ( i_micro_kernel_config->datatype_size == 4 ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V,
                                                   1 + l_m + l_m_blocks[0], 0, 0, l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2S );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S,
                                                   1 + l_m + l_m_blocks[0], 0, 0, l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      }
    }
    for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S,
                                                 1 + l_m + l_m_blocks[0] + l_m_blocks[1], 0, 0, l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                 LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kloop( libxsmm_generated_code*            io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_n_blocking ) {
  /* some hard coded parameters for k-blocking */
  unsigned int l_k_blocking = 4;
  unsigned int l_k_threshold = 23;

  /* apply multiple k_blocking strategies */
  /* 1. we are larger the k_threshold and a multiple of a predefined blocking parameter */
  if ((i_xgemm_desc->k % l_k_blocking) == 0 && (l_k_threshold < (unsigned int)i_xgemm_desc->k)) {
    unsigned int l_k;

    libxsmm_generator_gemm_aarch64_setup_k_strides(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                   i_xgemm_desc, i_m_blocking, i_n_blocking);

    libxsmm_generator_gemm_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, (unsigned int)i_xgemm_desc->k );

    for ( l_k = 0; l_k < l_k_blocking; l_k++) {
      libxsmm_generator_gemm_aarch64_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                 i_xgemm_desc, i_m_blocking, i_n_blocking);
    }

    libxsmm_generator_gemm_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_k_blocking );

    /* reset A pointer */
    libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                             (unsigned long long)((unsigned long long)i_xgemm_desc->k * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size) );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                         i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    /* reset B pointer */
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2,
                                               (unsigned long long)((unsigned long long)i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size) );
    } else {
      libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2,
                                               (unsigned long long)((unsigned long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size) );
    }
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                         i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_b,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  } else {
    /* 2. we want to fully unroll below the threshold */
    if ((unsigned int)i_xgemm_desc->k <= l_k_threshold) {
      unsigned int l_k;

      libxsmm_generator_gemm_aarch64_setup_k_strides(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                   i_xgemm_desc, i_m_blocking, i_n_blocking);

      for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
        libxsmm_generator_gemm_aarch64_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                   i_xgemm_desc, i_m_blocking, i_n_blocking);
      }

      /* reset A pointer */
      libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                               (unsigned long long)((unsigned long long)i_xgemm_desc->k * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size) );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                           i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* reset B pointer */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2,
                                                 (unsigned long long)((unsigned long long)i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size) );
      } else {
        libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2,
                                                 (unsigned long long)((unsigned long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size) );
      }
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                           i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_b,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
     /* 3. we are larger than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
    } else {
      unsigned int l_max_blocked_k = ((i_xgemm_desc->k)/l_k_blocking)*l_k_blocking;
      unsigned int l_k;

      libxsmm_generator_gemm_aarch64_setup_k_strides(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                   i_xgemm_desc, i_m_blocking, i_n_blocking);

      /* we can block as k is large enough */
      if ( l_max_blocked_k > 0 ) {
        libxsmm_generator_gemm_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_max_blocked_k );

        for ( l_k = 0; l_k < l_k_blocking; l_k++) {
          libxsmm_generator_gemm_aarch64_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                     i_xgemm_desc, i_m_blocking, i_n_blocking);
        }

        libxsmm_generator_gemm_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_k_blocking );
      }

      /* now we handle the remainder handling */
      for ( l_k = l_max_blocked_k; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
        libxsmm_generator_gemm_aarch64_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                   i_xgemm_desc, i_m_blocking, i_n_blocking);
      }

      /* reset A pointer */
      libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                               (unsigned long long)((unsigned long long)i_xgemm_desc->k * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size) );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                           i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      /* reset B pointer */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2,
                                                 (unsigned long long)((unsigned long long)i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size) );
      } else {
        libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2,
                                                 (unsigned long long)((unsigned long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size) );
      }
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                           i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_b,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kernel( libxsmm_generated_code*        io_generated_code,
                                            const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  unsigned int l_n_n[2] = {0,0};       /* blocking sizes for blocks */
  unsigned int l_n_N[2] = {0,0};       /* size of blocks */
  unsigned int l_n_count = 0;          /* array counter for blocking arrays */
  unsigned int l_n_done = 0;           /* progress tracker */
  unsigned int l_n_done_old = 0;

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_AARCH64_GP_REG_X4;
  /*l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_AARCH64_GP_REG_X5;*/
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_AARCH64_GP_REG_X7;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_X8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* compute n blocking, based on m blocking */
  libxsmm_generator_gemm_aarch64_setup_n_blocking( io_generated_code, &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch, l_n_N, l_n_n );

  /* check that l_n_N1 is non-zero */
  if ( l_n_N[0] == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code );

  /* apply n_blocking */
  while (l_n_done != (unsigned int)i_xgemm_desc->n) {
    unsigned int l_n_blocking = l_n_n[l_n_count];
    unsigned int l_m_done = 0;
    unsigned int l_m_done_old = 0;
    unsigned int l_m_blocking = 0;

    /* advance N */
    l_n_done_old = l_n_done;
    l_n_done += l_n_N[l_n_count];
    l_n_count++;

    /* open N loop */
    libxsmm_generator_gemm_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                                l_gp_reg_mapping.gp_reg_nloop, l_n_done - l_n_done_old );

    /* define the micro kernel code gen properties, especially m-blocking affects the vector instruction length */
    l_m_blocking = libxsmm_generator_gemm_aarch64_get_initial_m_blocking( &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch );

    /* apply m_blocking */
    while (l_m_done != (unsigned int)i_xgemm_desc->m) {
      if ( l_m_blocking == 0 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
      l_m_done_old = l_m_done;
      LIBXSMM_ASSERT(0 != l_m_blocking);
      /* coverity[divide_by_zero] */
      l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);

      if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
        /* open M loop */
        libxsmm_generator_gemm_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                                    l_gp_reg_mapping.gp_reg_mloop, l_m_done - l_m_done_old );
        /* load block of C */
        libxsmm_generator_gemm_load_2dregblock_aarch64( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1,
                                                        l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                        i_xgemm_desc->ldc * l_micro_kernel_config.datatype_size,
                                                        (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags) );

        /* compute outer product */
        libxsmm_generator_gemm_aarch64_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                              i_xgemm_desc, l_m_blocking, l_n_blocking );

        /* store block of C */
        libxsmm_generator_gemm_store_2dregblock_aarch64( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1,
                                                          l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                         i_xgemm_desc->ldc * l_micro_kernel_config.datatype_size );

        /* advance C pointer */
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_c,
                                                       l_m_blocking*l_micro_kernel_config.datatype_size, 0 );

        /* advance A pointer */
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_a,
                                                       l_m_blocking*l_micro_kernel_config.datatype_size, 0 );

        /* close M loop */
        libxsmm_generator_gemm_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                                    l_gp_reg_mapping.gp_reg_mloop, l_m_blocking );
      }

      /* switch to next smaller m_blocking */
      l_m_blocking = libxsmm_generator_gemm_aarch64_update_m_blocking( &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch, l_m_blocking );
    }
    /* reset C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_c,
                                                   i_xgemm_desc->m*l_micro_kernel_config.datatype_size, 0 );
    libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, l_gp_reg_mapping.gp_reg_help_1,
                                             (unsigned long long)((unsigned long long)l_n_blocking * i_xgemm_desc->ldc * l_micro_kernel_config.datatype_size) );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                         l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_c,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    /* reser A pointer */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                   l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_a,
                                                   i_xgemm_desc->m*l_micro_kernel_config.datatype_size, 0 );

    /* advance B pointer */
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, l_gp_reg_mapping.gp_reg_help_2,
                                               (unsigned long long)((unsigned long long)l_n_blocking * l_micro_kernel_config.datatype_size) );
    } else {
      libxsmm_aarch64_instruction_alu_set_i64( io_generated_code, l_gp_reg_mapping.gp_reg_help_2,
                                               (unsigned long long)((unsigned long long)l_n_blocking * i_xgemm_desc->ldb * l_micro_kernel_config.datatype_size) );
    }
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                         l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_b,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    /* close N loop */
    libxsmm_generator_gemm_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                                l_gp_reg_mapping.gp_reg_nloop, l_n_blocking );
  }

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code );
}

