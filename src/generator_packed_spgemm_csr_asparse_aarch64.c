/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_packed_spgemm_csr_asparse_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"


LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_aarch64( libxsmm_generated_code*         io_generated_code,
                                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                          const unsigned int*             i_row_idx,
                                                          const unsigned int*             i_column_idx,
                                                          const void*                     i_values,
                                                          const unsigned int              i_packed_width ) {
  unsigned int l_simd_packed_remainder = 0;
  unsigned int l_simd_packed_iters_full = 0;
  unsigned int l_simd_packed_width = 0;
  unsigned int l_n_max_block = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;
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
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_UNDEF;

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* select packed width */
  if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    if ( io_generated_code->arch >= LIBXSMM_AARCH64_SVE128 ) {
      if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE256 ) {
        l_simd_packed_width = 2;
      } else if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE512 ) {
        l_simd_packed_width = 4;
      } else {
        l_simd_packed_width = 8;
      }
    } else { /* asimd */
      l_simd_packed_width = 2;
    }
  } else {
    if ( io_generated_code->arch >= LIBXSMM_AARCH64_SVE128 ) {
      if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE256 ) {
        l_simd_packed_width = 4;
      } else if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE512 ) {
        l_simd_packed_width = 8;
      } else {
        l_simd_packed_width = 16;
      }
    } else { /* asimd */
      l_simd_packed_width = 4;
    }
  }

  /* calculate the packing count */
  l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
  l_simd_packed_iters_full = i_packed_width / l_simd_packed_width;

  if ( l_simd_packed_remainder != 0 ) {
    /* this is for now a general error */
    fprintf( stderr, "libxsmm_generator_packed_spgemm_csr_asparse_aarch64 right now only supports multiples of SIMD length!\n" );
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* select N blocking width as we have 32 ASIMD registers */
  l_n_max_block = 28;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xf );

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
    /* RDI holds the pointer to the struct, so lets first move this one into R15 */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR,
                                                         l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    /* A pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_reg_mapping.gp_reg_a );
    /* B pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_reg_mapping.gp_reg_b );
    /* C pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 96, l_gp_reg_mapping.gp_reg_c );
    if ( i_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 56, l_gp_reg_mapping.gp_reg_a_prefetch );
      /* B prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 88, l_gp_reg_mapping.gp_reg_b_prefetch );
    }
  } else {
#if 0
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
    return;
#endif
  }

  /* set P0 in case of SVE */
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P0,
                                                  -1,
                                                  l_gp_reg_mapping.gp_reg_help_0 );
  }

  /* loop over blocks of packing */
  if ( (l_simd_packed_iters_full > 1) /*|| (l_simd_packed_remainder > 0 && l_simd_packed_iters_full > 0 )*/ ) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_0, l_simd_packed_iters_full );

    /* save pointers for outer loop */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP,
                                                   32, 0 );
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 16,
                                               l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_b );
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0,
                                               l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_b_prefetch );
  }

  /* call N loop */
  if ( l_simd_packed_iters_full > 0 ) {
    libxsmm_generator_packed_spgemm_csr_asparse_aarch64_n_loop( io_generated_code,
                                                                        i_xgemm_desc,
                                                                        &l_loop_label_tracker,
                                                                        &l_micro_kernel_config,
                                                                        &l_gp_reg_mapping,
                                                                        i_row_idx,
                                                                        i_column_idx,
                                                                        i_values,
                                                                        l_n_max_block,
                                                                        i_packed_width,
                                                                        0 );
  }

  /* close packed loop */
  if ( (l_simd_packed_iters_full > 1) /*|| (l_simd_packed_remainder > 0 && l_simd_packed_iters_full > 0 )*/ ) {
    /* restore pointers from stack */
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 16,
                                               l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_b );
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0,
                                               l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_b_prefetch );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP,
                                                   32, 0 );

    /* advance B and C pointers */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                   (long long)l_simd_packed_width*l_micro_kernel_config.datatype_size_out );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                   (long long)l_simd_packed_width*l_micro_kernel_config.datatype_size_in );
#if 0
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   l_gp_reg_mapping.gp_reg_b_prefetch, l_gp_reg_mapping.gp_reg_b_prefetch, l_simd_packed_width*l_micro_kernel_config.datatype_size_in, 0 );
#endif
    libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_0, 1 );
  }

#if 0
  if ( l_simd_packed_remainder > 0 ) {
    /* TODO: */
  }
#endif

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xf );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_aarch64_n_loop( libxsmm_generated_code*            io_generated_code,
                                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                 libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                 const unsigned int*                i_row_idx,
                                                                 const unsigned int*                i_column_idx,
                                                                 const void*                        i_values,
                                                                 const unsigned int                 i_n_max_block,
                                                                 const unsigned int                 i_packed_width,
                                                                 const unsigned int                 i_packed_mask  ) {
  unsigned int l_gen_m_trips = 0;
  unsigned int l_a_is_dense = 0;
  unsigned int l_n_chunks = 0;
  unsigned int l_n_chunksize = 0;
  unsigned int l_n_remain = 0;
  unsigned int l_n_loop = 0;

  /* set function pointer based on supported SIMD arch */
  void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gemm_descriptor*, libxsmm_loop_label_tracker*,
                                  const libxsmm_micro_kernel_config*, const libxsmm_gp_reg_mapping*,
                                  const unsigned int*, const unsigned int*, const void*, const unsigned int,
                                  const unsigned int, const unsigned int, const unsigned int, const unsigned int);

  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    l_generator_microkernel = libxsmm_generator_packed_spgemm_csr_asparse_aarch64_m_loop_sve;
  } else {
    l_generator_microkernel = libxsmm_generator_packed_spgemm_csr_asparse_aarch64_m_loop_asimd;
  }

  /* test if we should generate a dense version */
  if ( i_row_idx[i_xgemm_desc->m] == (unsigned int)(i_xgemm_desc->m*i_xgemm_desc->k) ) {
    l_gen_m_trips = 1;
    l_a_is_dense = 1;
  } else {
    l_gen_m_trips = i_xgemm_desc->m;
    l_a_is_dense = 0;
  }

  /* calculate the chunk size of current columns to work on */
  l_n_chunks = ( (i_xgemm_desc->n % i_n_max_block) == 0 ) ? (i_xgemm_desc->n / i_n_max_block) : (i_xgemm_desc->n / i_n_max_block) + 1;
  l_n_chunksize = ( (i_xgemm_desc->n % l_n_chunks) == 0 ) ? (i_xgemm_desc->n / l_n_chunks) : (i_xgemm_desc->n / l_n_chunks) + 1;
  l_n_remain = ( ((i_xgemm_desc->n % l_n_chunksize) == 0) || ((unsigned int)i_xgemm_desc->n <= i_n_max_block) ) ? 0 : 1;
  l_n_loop = ( l_n_remain == 0 ) ? (l_n_chunks * l_n_chunksize) : ((l_n_chunks-1) * l_n_chunksize);

  /* loop over blocks of n */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, l_n_loop );

  /* do matix multiplicatoin for a block of N columns */
  l_generator_microkernel( io_generated_code, i_xgemm_desc, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping,
                           i_row_idx, i_column_idx, i_values,
                           l_gen_m_trips, l_a_is_dense, l_n_chunksize, i_packed_width, i_packed_mask );

  /* adjust B pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                 (long long)i_micro_kernel_config->datatype_size_in*i_packed_width*l_n_chunksize );

  /* advance B prefetch pointer */
#if 0
  if ( (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL2_VIA_C) > 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b_prefetch, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b_prefetch,
                                                   (((long long)i_micro_kernel_config->datatype_size_in*i_packed_width*i_xgemm_desc->ldb*i_xgemm_desc->m)
                                                     -(i_micro_kernel_config->datatype_size_in*i_packed_width*l_n_chunksize)) );
  }
#endif

  /* adjust C pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                 (long long)(((long long)i_micro_kernel_config->datatype_size_out*i_packed_width*i_xgemm_desc->ldc*i_xgemm_desc->m)
                                                   -((long long)i_micro_kernel_config->datatype_size_out*i_packed_width*l_n_chunksize)) );

  /* N loop jump back */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, l_n_chunksize );

  /* handle remainder of N loop */
  if ( l_n_remain != 0 ) {
    l_generator_microkernel( io_generated_code, i_xgemm_desc, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping,
                             i_row_idx, i_column_idx, i_values,
                             l_gen_m_trips, l_a_is_dense, i_xgemm_desc->n - (l_n_chunksize * (l_n_chunks - 1)), i_packed_width, i_packed_mask );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_aarch64_m_loop_asimd( libxsmm_generated_code*            io_generated_code,
                                                                       const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                       libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                       const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                       const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                       const unsigned int*                i_row_idx,
                                                                       const unsigned int*                i_column_idx,
                                                                       const void*                        i_values,
                                                                       const unsigned int                 i_gen_m_trips,
                                                                       const unsigned int                 i_a_is_dense,
                                                                       const unsigned int                 i_num_c_cols,
                                                                       const unsigned int                 i_packed_width,
                                                                       const unsigned int                 i_packed_mask ) {
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_z;
  unsigned int l_row_elements;
  unsigned int l_b_offset;

  LIBXSMM_UNUSED(i_values);

  /* do sparse times dense packed multiplication */
  for ( l_m = 0; l_m < i_gen_m_trips; l_m++ ) {
    /* handle b offset */
    l_b_offset = 0;

    /* generate M loop */
    if (i_a_is_dense != 0 ) {
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_mloop, i_xgemm_desc->m );
    }

    l_row_elements = i_row_idx[l_m+1] - i_row_idx[l_m];
    if (l_row_elements > 0) {
      /* load C accumulator */
      for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
        if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                     l_n, l_n, 0, l_n,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        } else {
          if ( i_packed_mask == 0 ) {
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                    i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                    i_packed_width*i_micro_kernel_config->datatype_size_out,
                                                    l_n, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
          } else {
            /* TODO: */
          }
        }
#if 0
        if ( (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL2_VIA_C) > 0 ) {
          /* TODO: */
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_micro_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_b_prefetch,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_n*i_packed_width*i_micro_kernel_config->datatype_size_in );
        }
#endif
      }
      /* reset C point for stores */
      if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                       (long long)i_num_c_cols*i_packed_width*i_micro_kernel_config->datatype_size_out );
      }

      /* loop over the non-zeros in A row m */
      for ( l_z = 0; l_z < l_row_elements; l_z++ ) {
        /* broadcast values of A */
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_in,
                                                i_num_c_cols, (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D );

        /* multiply with B */
        for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
          l_b_offset = ((i_column_idx[i_row_idx[l_m] + l_z]*i_micro_kernel_config->datatype_size_in*i_packed_width*i_xgemm_desc->ldb)
                                                     +(l_n*i_packed_width*i_micro_kernel_config->datatype_size_in));

          libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_1,
                                                     l_b_offset );

          if ( i_packed_mask == 0 ) {
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                                    i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, 0,
                                                    i_num_c_cols+1, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V,
                                                       i_num_c_cols+1, i_num_c_cols, 0, l_n,
                                                       (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
          } else {
            /* TODO: */
          }
        }
      }
      /* store C accumulator */
      for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
        if ( i_packed_mask == 0 ) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                                  i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  i_packed_width*i_micro_kernel_config->datatype_size_out,
                                                  l_n, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        } else {
          /* TODO: */
        }
      }
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                     (long long)i_num_c_cols*i_packed_width*i_micro_kernel_config->datatype_size_out );
    }
    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_micro_kernel_config->datatype_size_out*i_packed_width*i_xgemm_desc->ldc );

    /* advance B prefetch pointer */
#if 0
    if ( (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL2_VIA_C) > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_b_prefetch, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b_prefetch,
                                                     (long long)i_micro_kernel_config->datatype_size_in*i_packed_width*i_xgemm_desc->ldb );
    }
#endif

    /* generate M loop */
    if (i_a_is_dense != 0 ) {
      /* M loop jump back */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_mloop, 1 );
    }
  }

  /* reset A pointer */
  if (i_a_is_dense != 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)i_micro_kernel_config->datatype_size_in*i_xgemm_desc->k*i_xgemm_desc->m );
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)i_micro_kernel_config->datatype_size_in*i_row_idx[i_gen_m_trips] );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_aarch64_m_loop_sve( libxsmm_generated_code*            io_generated_code,
                                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                     libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                     const unsigned int*                i_row_idx,
                                                                     const unsigned int*                i_column_idx,
                                                                     const void*                        i_values,
                                                                     const unsigned int                 i_gen_m_trips,
                                                                     const unsigned int                 i_a_is_dense,
                                                                     const unsigned int                 i_num_c_cols,
                                                                     const unsigned int                 i_packed_width,
                                                                     const unsigned int                 i_packed_mask ) {
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_z;
  unsigned int l_row_elements;
  unsigned int l_b_offset;

  LIBXSMM_UNUSED(i_values);

  /* do sparse times dense packed multiplication */
  for ( l_m = 0; l_m < i_gen_m_trips; l_m++ ) {
    /* handle b offset */
    l_b_offset = 0;

    /* generate M loop */
    if (i_a_is_dense != 0 ) {
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_mloop, i_xgemm_desc->m );
    }

    l_row_elements = i_row_idx[l_m+1] - i_row_idx[l_m];
    if (l_row_elements > 0) {
      /* load C accumulator */
      for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
        if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   l_n, l_n, (unsigned char)-1, l_n,
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   LIBXSMM_AARCH64_SVE_TYPE_D );
        } else {
          if ( i_packed_mask == 0 ) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                                  i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                  l_n, LIBXSMM_AARCH64_SVE_REG_UNDEF );
            libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                           i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_c,
                                                           i_packed_width*i_micro_kernel_config->datatype_size_out, 0 );
          } else {
            /* TODO: */
          }
        }
#if 0
        if ( (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL2_VIA_C) > 0 ) {
          /* TODO: */
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_micro_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_b_prefetch,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_n*i_packed_width*i_micro_kernel_config->datatype_size_in );
        }
#endif
      }
      /* reset C point for stores */
      if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                       (long long)i_num_c_cols*i_packed_width*i_micro_kernel_config->datatype_size_out );
      }

      /* loop over the non-zeros in A row m */
      for ( l_z = 0; l_z < l_row_elements; l_z++ ) {
        /* broadcast values of A */
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                              i_gp_reg_mapping->gp_reg_a,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_num_c_cols, LIBXSMM_AARCH64_SVE_REG_P0 );
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_a,
                                                       i_micro_kernel_config->datatype_size_in, 0 );

        /* multiply with B */
        for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
          l_b_offset = ((i_column_idx[i_row_idx[l_m] + l_z]*i_packed_width*i_xgemm_desc->ldb)
                                                     +(l_n*i_packed_width));

          libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_1,
                                                     l_b_offset );

          if ( i_packed_mask == 0 ) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1W_SR : LIBXSMM_AARCH64_INSTR_SVE_LD1D_SR,
                                                  i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, 0,
                                                  i_num_c_cols+1, LIBXSMM_AARCH64_SVE_REG_P0 );

            libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                                     i_num_c_cols, i_num_c_cols+1,
                                                     (unsigned char)-1,
                                                     l_n, LIBXSMM_AARCH64_SVE_REG_P0,
                                                     (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_SVE_TYPE_S : LIBXSMM_AARCH64_SVE_TYPE_D );
          } else {
            /* TODO: */
          }
        }
      }
      /* store C accumulator */
      for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
        if ( i_packed_mask == 0 ) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                                i_gp_reg_mapping->gp_reg_c, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                l_n, LIBXSMM_AARCH64_SVE_REG_UNDEF );
          libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                         i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_c,
                                                         i_packed_width*i_micro_kernel_config->datatype_size_out, 0 );
        } else {
          /* TODO: */
        }
      }
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                     (long long)i_num_c_cols*i_packed_width*i_micro_kernel_config->datatype_size_out );
    }
    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_micro_kernel_config->datatype_size_out*i_packed_width*i_xgemm_desc->ldc );

    /* advance B prefetch pointer */
#if 0
    if ( (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL2_VIA_C) > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_b_prefetch, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b_prefetch,
                                                     (long long)i_micro_kernel_config->datatype_size_in*i_packed_width*i_xgemm_desc->ldb );
    }
#endif

    /* generate M loop */
    if (i_a_is_dense != 0 ) {
      /* M loop jump back */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_mloop, 1 );
    }
  }

  /* reset A pointer */
  if (i_a_is_dense != 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)i_micro_kernel_config->datatype_size_in*i_xgemm_desc->k*i_xgemm_desc->m );
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)i_micro_kernel_config->datatype_size_in*i_row_idx[i_gen_m_trips] );
  }
}
