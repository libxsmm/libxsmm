/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena)
******************************************************************************/

#include "generator_gemm_power.h"
#include "generator_power_instructions.h"

LIBXSMM_API_INTERN
unsigned char libxsmm_generator_gemm_power_load_store_vsx( libxsmm_generated_code * io_generated_code,
                                                           unsigned int             i_m_blocking_full,
                                                           unsigned int             i_n_blocking,
                                                           unsigned int             i_remainder_size,
                                                           unsigned int             i_stride,
                                                           unsigned char            i_load_store,
                                                           unsigned char            i_precision,
                                                           unsigned char            i_endianness,
                                                           unsigned char            i_gpr_ptr,
                                                           unsigned char            i_gpr_scratch[3],
                                                           unsigned char            i_vsr_first ) {
  unsigned int l_vsr = i_vsr_first;

  unsigned int l_ops_vsx[3] = {0};
  if( i_load_store == 0 ) {
    l_ops_vsx[0] = LIBXSMM_POWER_INSTR_VSX_LXVW4X;
    l_ops_vsx[1] = LIBXSMM_POWER_INSTR_VSX_LXVLL;
    l_ops_vsx[2] = LIBXSMM_POWER_INSTR_VSX_XXBRW;
  }
  else {
    l_ops_vsx[0] = LIBXSMM_POWER_INSTR_VSX_STXVW4X;
    l_ops_vsx[1] = LIBXSMM_POWER_INSTR_VSX_STXVLL;
    l_ops_vsx[2] = LIBXSMM_POWER_INSTR_VSX_XXBRW;
  }

  libxsmm_power_instruction_3( io_generated_code,
                               LIBXSMM_POWER_INSTR_FIP_ORI,
                               i_gpr_scratch[0],
                               i_gpr_ptr,
                               0 );

  if( i_remainder_size > 0 ) {
    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 i_gpr_scratch[2],
                                 0,
                                 i_remainder_size );
    libxsmm_power_instruction_4( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_RLDICR,
                                 i_gpr_scratch[2],
                                 i_gpr_scratch[2],
                                 64-8,
                                 63 );
  }

  for( unsigned int l_bn = 0; l_bn < i_n_blocking; l_bn++ ) {
    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ORI,
                                 i_gpr_scratch[1],
                                 i_gpr_scratch[0],
                                 0 );

    /* 128bit loads/stores */
    for( unsigned int l_bm = 0; l_bm < i_m_blocking_full; l_bm++ ) {
      libxsmm_power_instruction_3( io_generated_code,
                                   l_ops_vsx[0],
                                   l_vsr,
                                   0,
                                   i_gpr_scratch[1] );

      libxsmm_power_instruction_3( io_generated_code,
                                   LIBXSMM_POWER_INSTR_FIP_ADDI,
                                   i_gpr_scratch[1],
                                   i_gpr_scratch[1],
                                   16 );

      l_vsr++;
    }

    /* remainder load/store (if required) */
    if( i_remainder_size > 0 ) {
      /* reverse byte-order before storing for LE */
      if( i_endianness == 0 &&
          i_load_store == 1 ) {
        libxsmm_power_instruction_2( io_generated_code,
                                     l_ops_vsx[2],
                                     l_vsr,
                                     l_vsr );
      }

      libxsmm_power_instruction_3( io_generated_code,
                                   l_ops_vsx[1],
                                   l_vsr,
                                   i_gpr_scratch[1],
                                   i_gpr_scratch[2] );

      /* reverse byte-order after loading for LE */
      if( i_endianness == 0 &&
          i_load_store == 0 ) {
        libxsmm_power_instruction_2( io_generated_code,
                                     l_ops_vsx[2],
                                     l_vsr,
                                     l_vsr );
      }
      l_vsr++;
    }

    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 i_gpr_scratch[0],
                                 i_gpr_scratch[0],
                                 i_stride );
  }

  return l_vsr - i_vsr_first;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_power_microkernel_vsx( libxsmm_generated_code        * io_generated_code,
                                                   libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                   unsigned int                    i_m_blocking,
                                                   unsigned int                    i_n_blocking,
                                                   unsigned int                    i_k_blocking ) {
  unsigned int l_vector_length = 4;

  unsigned char l_gpr_a = LIBXSMM_POWER_GPR_R3;
  unsigned char l_gpr_b = LIBXSMM_POWER_GPR_R4;
  unsigned char l_gpr_c = LIBXSMM_POWER_GPR_R5;

  unsigned char l_gpr_scratch[3] = { 6, 7, 8 };

  unsigned int l_bytes_per_val = 4;

  unsigned int l_stride_a = i_xgemm_desc->lda * l_bytes_per_val;
  unsigned int l_stride_b = i_xgemm_desc->ldb * l_bytes_per_val;
  unsigned int l_stride_c = i_xgemm_desc->ldc * l_bytes_per_val;


  /* derive blocking in terms of 128bit and remainder ops */
  unsigned int l_m_blocking_full = i_m_blocking / l_vector_length;
  unsigned int l_remainder_size = i_m_blocking % l_vector_length;
               l_remainder_size *= l_bytes_per_val;

  /* load accumulator block */
  unsigned char l_vsr_c_first = 0;
  unsigned char l_n_vsr_c = 0;
  l_n_vsr_c = libxsmm_generator_gemm_power_load_store_vsx( io_generated_code,
                                                           l_m_blocking_full,
                                                           i_n_blocking,
                                                           l_remainder_size,
                                                           l_stride_c,
                                                           0,
                                                           l_bytes_per_val == 4 ? 0 : 1,
                                                           0,
                                                           l_gpr_c,
                                                           l_gpr_scratch,
                                                           l_vsr_c_first );

  unsigned char l_vsr_a_first = l_vsr_c_first + l_n_vsr_c;

  /* iterate over K */
  for( unsigned short l_k = 0; l_k < i_k_blocking; l_k++ ) {
    /* load (partial) column of A */
    unsigned int l_n_vsr_a = 0;
    l_n_vsr_a = libxsmm_generator_gemm_power_load_store_vsx( io_generated_code,
                                                             l_m_blocking_full,
                                                             1,
                                                             l_remainder_size,
                                                             l_stride_a,
                                                             0,
                                                             l_bytes_per_val == 4 ? 0 : 1,
                                                             0,
                                                             l_gpr_a,
                                                             l_gpr_scratch,
                                                             l_vsr_a_first );

    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 l_gpr_a,
                                 l_gpr_a,
                                 l_stride_a );

    unsigned char l_vsr_b = l_vsr_a_first + l_n_vsr_a;

    /* iterate over entries of B */
    for( unsigned short l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* bcast entry of B */
      libxsmm_power_instruction_3( io_generated_code,
                                   LIBXSMM_POWER_INSTR_VSX_LXVWSX,
                                   l_vsr_b,
                                   0,
                                   l_gpr_b );
      if( l_n != i_n_blocking-1 ) {
        libxsmm_power_instruction_3( io_generated_code,
                                     LIBXSMM_POWER_INSTR_FIP_ADDI,
                                     LIBXSMM_POWER_GPR_R4,
                                     LIBXSMM_POWER_GPR_R4,
                                     l_stride_b );
      }
      else {
        libxsmm_power_instruction_3( io_generated_code,
                                     LIBXSMM_POWER_INSTR_FIP_ADDI,
                                     LIBXSMM_POWER_GPR_R4,
                                     LIBXSMM_POWER_GPR_R4,
                                     -(i_n_blocking-1)*l_stride_b + l_bytes_per_val );
      }

      /* perform FMAs */
      unsigned int l_n_ops = l_m_blocking_full;
      if( l_remainder_size > 1 ) {
        l_n_ops++;
      }
      for( unsigned short l_op = 0; l_op < l_n_ops; l_op++ ) {
        unsigned int l_vsr_c_off = l_n * l_n_ops;
                     l_vsr_c_off += l_op;

        libxsmm_power_instruction_3( io_generated_code,
                                     LIBXSMM_POWER_INSTR_VSX_XVMADDASP,
                                     l_vsr_c_first + l_vsr_c_off,
                                     l_vsr_a_first + l_op,
                                     l_vsr_b );
      }
    }
  }

  /* store accumulator block */
  l_n_vsr_c = libxsmm_generator_gemm_power_load_store_vsx( io_generated_code,
                                                           l_m_blocking_full,
                                                           i_n_blocking,
                                                           l_remainder_size,
                                                           l_stride_c,
                                                           1,
                                                           l_bytes_per_val == 4 ? 0 : 1,
                                                           0,
                                                           l_gpr_c,
                                                           l_gpr_scratch,
                                                           l_vsr_c_first );
}

LIBXSMM_API_INTERN
int libxsmm_generator_gemm_power_kernel( libxsmm_generated_code        * io_generated_code,
                                         libxsmm_gemm_descriptor const * i_xgemm_desc ) {
  libxsmm_power_instruction_open_stream( io_generated_code,
                                         31,
                                         31,
                                         31 );

  libxsmm_generator_gemm_power_microkernel_vsx( io_generated_code,
                                                i_xgemm_desc,
                                                i_xgemm_desc->m,
                                                i_xgemm_desc->n,
                                                i_xgemm_desc->k );

  libxsmm_power_instruction_close_stream( io_generated_code,
                                          31,
                                          31,
                                          31 );
  return 0;
}