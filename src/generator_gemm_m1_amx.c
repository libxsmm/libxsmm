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
#include "generator_gemm_m1_amx.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_amx( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_instr,
                                      const unsigned char     i_operand ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int  l_code_head = io_generated_code->code_size / 4;
    unsigned int* l_code      = (unsigned int *) io_generated_code->generated_code;

    /* set LIBXSMM-internal bits to 0 */
    l_code[l_code_head] = (unsigned int)(0xffffffe0 & i_instr);

    /* derive mask selecting only valid bits of operand */
    unsigned int l_n_bits = i_instr & 0x7;
    unsigned int l_mask = ~0u;
    l_mask = l_mask >> (32 - l_n_bits);

    /* apply mask to operand and set resulting bits */
    l_code[l_code_head] |= (unsigned int) (l_mask & i_operand);

    /* increase code-size by 32 bits / 4 bytes */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_amx: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_m1_amx_lsblock( uint8_t                  i_bytes_per_val,
                                            uint8_t                  i_gp_c,
                                            uint8_t                  i_block_m,
                                            uint8_t                  i_block_n,
                                            uint32_t                 i_ldc,
                                            uint8_t                  i_gp_stride_vec,
                                            uint8_t                  i_gp_stride_c,
                                            uint8_t                  i_gp_scratch[2],
                                            bool                     i_load,
                                            libxsmm_generated_code * io_generated_code ) {
  uint8_t l_off_xview = 32;
  uint32_t l_vec_length = 64 / i_bytes_per_val;

  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  // set up column jump
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                             i_gp_scratch[0] + l_off_xview,
                                             64/l_vec_length - i_block_m );

  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                             i_gp_scratch[4] + l_off_xview,
                                             i_ldc*i_bytes_per_val - i_block_m*64 );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       i_gp_scratch[4] + l_off_xview,
                                                       i_gp_scratch[0] + l_off_xview,
                                                       i_gp_scratch[4] + l_off_xview,
                                                       56,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                       LIBXSMM_AARCH64_GP_REG_XZR,
                                                       i_gp_c + l_off_xview,
                                                       i_gp_scratch[2] + l_off_xview,
                                                       0,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  // loop-counters over n-blocks and vector-size
  libxsmm_generator_loop_header_aarch64( io_generated_code,
                                         &l_loop_label_tracker,
                                         i_gp_scratch[0] + l_off_xview,
                                         i_block_n );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                       LIBXSMM_AARCH64_GP_REG_XZR,
                                                       i_gp_scratch[2] + l_off_xview,
                                                       i_gp_scratch[3] + l_off_xview,
                                                       0,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  libxsmm_generator_loop_header_aarch64( io_generated_code,
                                         &l_loop_label_tracker,
                                         i_gp_scratch[1] + l_off_xview,
                                         l_vec_length );

  // unroll m
  for( uint8_t l_m = 0; l_m < i_block_m; l_m++ ) {
    if( i_load ) {
      libxsmm_aarch64_instruction_amx( io_generated_code,
                                       LIBXSMM_AARCH64_INSTR_AMX_LDZ,
                                       i_gp_scratch[3] );
    }
    else {
      libxsmm_aarch64_instruction_amx( io_generated_code,
                                       LIBXSMM_AARCH64_INSTR_AMX_STZ,
                                       i_gp_scratch[3] );
    }

    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                         i_gp_scratch[3] + l_off_xview,
                                                         i_gp_stride_vec + l_off_xview,
                                                         i_gp_scratch[3] + l_off_xview,
                                                         0,
                                                         LIBXSMM_AARCH64_SHIFTMODE_LSL );
  }

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       i_gp_scratch[3] + l_off_xview,
                                                       i_gp_scratch[4] + l_off_xview,
                                                       i_gp_scratch[3] + l_off_xview,
                                                       0,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  // end of vec-loop
  libxsmm_generator_loop_footer_aarch64( io_generated_code,
                                         &l_loop_label_tracker,
                                         i_gp_scratch[1] + l_off_xview,
                                         1 );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       i_gp_scratch[2] + l_off_xview,
                                                       i_gp_stride_c   + l_off_xview,
                                                       i_gp_scratch[2] + l_off_xview,
                                                       0,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  // end of n-loop
  libxsmm_generator_loop_footer_aarch64( io_generated_code,
                                         &l_loop_label_tracker,
                                         i_gp_scratch[0] + l_off_xview,
                                         1 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_m1_amx_kloop( uint8_t                  i_bytes_per_val,
                                          uint32_t                 i_block_m,
                                          uint32_t                 i_block_n,
                                          uint8_t                  i_gp_a,
                                          uint8_t                  i_gp_b,
                                          uint8_t                  i_gp_c,
                                          uint32_t                 i_k,
                                          uint32_t                 i_lda,
                                          uint32_t                 i_ldb,
                                          uint32_t                 i_ldc,
                                          libxsmm_generated_code * io_generated_code ) {
  uint8_t l_off_xview = 32;
  uint8_t l_vec_length = 64 / i_bytes_per_val;

  // register mapping
  uint8_t l_gp_scratch[5] = { 3, 4, 5, 6, 7 };

  uint8_t l_gp_stride_vec = 8;
  uint8_t l_gp_stride_a = 9;
  uint8_t l_gp_stride_c = 10;

  uint8_t l_gp_fma[8] = { 11, 12, 13, 14, 15, 16, 17, 18 };

  uint8_t l_gp_tmp_a[2] = { 19, 20 };
  uint8_t l_gp_tmp_b[2] = { 21, 22 };

  uint8_t l_gp_kloop = 23;

  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  // set stride in vector dim
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                             l_gp_scratch[0] + l_off_xview,
                                             1 );

  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                             l_gp_stride_vec + l_off_xview,
                                             64 );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       l_gp_stride_vec + l_off_xview,
                                                       l_gp_scratch[0] + l_off_xview,
                                                       l_gp_stride_vec + l_off_xview,
                                                       56,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  // set stride in a
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                             l_gp_scratch[0] + l_off_xview,
                                             i_block_m );

  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                             l_gp_stride_a + l_off_xview,
                                             i_lda*i_bytes_per_val );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       l_gp_stride_a   + l_off_xview,
                                                       l_gp_scratch[0] + l_off_xview,
                                                       l_gp_stride_a   + l_off_xview,
                                                       56,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  // set stride in c
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                             l_gp_scratch[0] + l_off_xview,
                                             i_block_m );

  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                             l_gp_stride_c + l_off_xview,
                                             l_vec_length*i_ldc*i_bytes_per_val );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       l_gp_stride_c   + l_off_xview,
                                                       l_gp_scratch[0] + l_off_xview,
                                                       l_gp_stride_c   + l_off_xview,
                                                       56,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  // load C-block
  libxsmm_generator_gemm_m1_amx_lsblock( i_bytes_per_val,
                                         i_gp_c,
                                         i_block_m,
                                         i_block_n,
                                         i_ldc,
                                         l_gp_stride_vec,
                                         l_gp_stride_c,
                                         l_gp_scratch,
                                         true,
                                         io_generated_code );

  // set AMX offsets
  for( uint8_t l_n = 0; l_n < i_block_n; l_n++ ) {
    for( uint8_t l_m = 0; l_m < i_block_m; l_m++ ) {
      uint8_t l_id = l_n*i_block_m + l_m;

      if( l_n > 0 ) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                                   l_gp_fma[l_id] + l_off_xview,
                                                   64*l_n );
      }
      else {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                                   l_gp_fma[l_id] + l_off_xview,
                                                   0 );
      }

      if( l_m > 0 ) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                                   l_gp_scratch[0] + l_off_xview,
                                                   64*l_m );

        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                             LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                             l_gp_fma[l_id]  + l_off_xview,
                                                             l_gp_scratch[0] + l_off_xview,
                                                             l_gp_fma[l_id]  + l_off_xview,
                                                             10,
                                                             LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }

      if( l_id > 0 ) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                                   l_gp_scratch[0] + l_off_xview,
                                                   l_id );

        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                             LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                             l_gp_fma[l_id]  + l_off_xview,
                                                             l_gp_scratch[0] + l_off_xview,
                                                             l_gp_fma[l_id]  + l_off_xview,
                                                             20,
                                                             LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
    }
  }

  // tmp pointer to A
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                       LIBXSMM_AARCH64_GP_REG_XZR,
                                                       i_gp_a + l_off_xview,
                                                       l_gp_tmp_a[0] + l_off_xview,
                                                       0,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  // tmp pointer to B
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                       LIBXSMM_AARCH64_GP_REG_XZR,
                                                       i_gp_b + l_off_xview,
                                                       l_gp_tmp_b[0] + l_off_xview,
                                                       0,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  // k-loop
  libxsmm_generator_loop_header_aarch64( io_generated_code,
                                         &l_loop_label_tracker,
                                         l_gp_kloop + l_off_xview,
                                         i_k );

  // load A columns
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                       LIBXSMM_AARCH64_GP_REG_XZR,
                                                       l_gp_tmp_a[0] + l_off_xview,
                                                       l_gp_tmp_a[1] + l_off_xview,
                                                       0,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  for( unsigned short l_m = 0; l_m < i_block_m; l_m++ ) {
    libxsmm_aarch64_instruction_amx( io_generated_code,
                                     LIBXSMM_AARCH64_INSTR_AMX_LDX,
                                     l_gp_tmp_a[1] );

    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                         l_gp_tmp_a[1]   + l_off_xview,
                                                         l_gp_stride_vec + l_off_xview,
                                                         l_gp_tmp_a[1]   + l_off_xview,
                                                         0,
                                                         LIBXSMM_AARCH64_SHIFTMODE_LSL );
  }

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 l_gp_tmp_a[0] + l_off_xview,
                                                 l_gp_scratch[0],
                                                 l_gp_tmp_a[0] + l_off_xview,
                                                 i_lda * i_bytes_per_val );

  // load B-row
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                       LIBXSMM_AARCH64_GP_REG_XZR,
                                                       l_gp_tmp_b[0] + l_off_xview,
                                                       l_gp_tmp_b[1] + l_off_xview,
                                                       0,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );

  for( unsigned short l_n = 0; l_n < i_block_n; l_n++ ) {
    libxsmm_aarch64_instruction_amx( io_generated_code,
                                     LIBXSMM_AARCH64_INSTR_AMX_LDY,
                                     l_gp_tmp_b[1] );

    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                         l_gp_tmp_b[1]   + l_off_xview,
                                                         l_gp_stride_vec + l_off_xview,
                                                         l_gp_tmp_b[1]   + l_off_xview,
                                                         0,
                                                         LIBXSMM_AARCH64_SHIFTMODE_LSL );
  }

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 l_gp_tmp_b[0] + l_off_xview,
                                                 l_gp_scratch[0],
                                                 l_gp_tmp_b[0] + l_off_xview,
                                                 i_ldb * i_bytes_per_val );

  // do the FMAs
  for( unsigned short l_in = 0; l_in < i_block_m*i_block_n; l_in++ ) {
    if( i_bytes_per_val == 2 ) {
      libxsmm_aarch64_instruction_amx( io_generated_code,
                                       LIBXSMM_AARCH64_INSTR_AMX_FMA16,
                                       l_gp_fma[l_in] );
    }
    else if( i_bytes_per_val == 4 ) {
      libxsmm_aarch64_instruction_amx( io_generated_code,
                                       LIBXSMM_AARCH64_INSTR_AMX_FMA32,
                                       l_gp_fma[l_in] );
    }
    else if( i_bytes_per_val == 8 ) {
      libxsmm_aarch64_instruction_amx( io_generated_code,
                                       LIBXSMM_AARCH64_INSTR_AMX_FMA64,
                                       l_gp_fma[l_in] );
    }
  }

  // end of k-loop
  libxsmm_generator_loop_footer_aarch64( io_generated_code,
                                         &l_loop_label_tracker,
                                         l_gp_kloop + l_off_xview,
                                         1 );

  // store C-block
  libxsmm_generator_gemm_m1_amx_lsblock( i_bytes_per_val,
                                         i_gp_c,
                                         i_block_m,
                                         i_block_n,
                                         i_ldc,
                                         l_gp_stride_vec,
                                         l_gp_stride_c,
                                         l_gp_scratch,
                                         false,
                                         io_generated_code );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_m1_amx_generic( uint8_t                  i_bytes_per_val,
                                            uint32_t                 i_m,
                                            uint32_t                 i_n,
                                            uint32_t                 i_k,
                                            uint32_t                 i_lda,
                                            uint32_t                 i_ldb,
                                            uint32_t                 i_ldc,
                                            libxsmm_generated_code * io_generated_code ) {
  uint8_t l_off_xview = 32;
  uint32_t l_vec_length = 64 / i_bytes_per_val;

  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  // store according to procedure call standard
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xff0 );

  // enable AMX instructions
  libxsmm_aarch64_instruction_amx( io_generated_code,
                                  LIBXSMM_AARCH64_INSTR_AMX_ENABLE,
                                  0 );

  // register mapping
  uint8_t l_gp_a = 0;
  uint8_t l_gp_b = 1;
  uint8_t l_gp_c = 2;

  uint8_t l_gp_scratch[1] = { 25 };

  uint8_t l_gp_tmp_a = 26;
  uint8_t l_gp_tmp_b = 27;
  uint8_t l_gp_tmp_c = 28;

  uint32_t l_gp_loop_m = 29;
  uint32_t l_gp_loop_n = 30;

  // max blocks
  uint32_t l_block_max_m = 0;
  uint32_t l_block_max_n = 0;

  l_block_max_m = i_m;
  l_block_max_m /= 64;
  l_block_max_n = i_n;
  l_block_max_n /= 64;

  // raw micro-kernel for small enough matrices
  if( (i_m/l_vec_length) * (i_n/l_vec_length) <= i_bytes_per_val ) {
    uint32_t l_block_m = i_m / l_vec_length;
    uint32_t l_block_n = i_n / l_vec_length;

    libxsmm_generator_gemm_m1_amx_kloop( i_bytes_per_val,
                                         l_block_m,
                                         l_block_n,
                                         0,
                                         1,
                                         2,
                                         i_k,
                                         i_lda,
                                         i_ldb,
                                         i_ldc,
                                         io_generated_code );
  }
  else if( l_block_max_m > 0 ) {
    uint32_t l_block_m = 64 / l_vec_length;
    uint32_t l_block_n = 1;

    // loop over n
    libxsmm_generator_loop_header_aarch64( io_generated_code,
                                           &l_loop_label_tracker,
                                           l_gp_loop_n + l_off_xview,
                                           i_n / l_vec_length );


    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                         LIBXSMM_AARCH64_GP_REG_XZR,
                                                         l_gp_a     + l_off_xview,
                                                         l_gp_tmp_a + l_off_xview,
                                                         0,
                                                         LIBXSMM_AARCH64_SHIFTMODE_LSL );

    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                         LIBXSMM_AARCH64_GP_REG_XZR,
                                                         l_gp_b     + l_off_xview,
                                                         l_gp_tmp_b + l_off_xview,
                                                         0,
                                                         LIBXSMM_AARCH64_SHIFTMODE_LSL );

    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                         LIBXSMM_AARCH64_GP_REG_XZR,
                                                         l_gp_c     + l_off_xview,
                                                         l_gp_tmp_c + l_off_xview,
                                                         0,
                                                         LIBXSMM_AARCH64_SHIFTMODE_LSL );

    // loop over m
    libxsmm_generator_loop_header_aarch64( io_generated_code,
                                           &l_loop_label_tracker,
                                           l_gp_loop_m + l_off_xview,
                                           l_block_max_m );

    libxsmm_generator_gemm_m1_amx_kloop( i_bytes_per_val,
                                         l_block_m,
                                         l_block_n,
                                         l_gp_tmp_a,
                                         l_gp_tmp_b,
                                         l_gp_tmp_c,
                                         i_k,
                                         i_lda,
                                         i_ldb,
                                         i_ldc,
                                         io_generated_code );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_tmp_a + l_off_xview,
                                                   l_gp_scratch[0] + l_off_xview,
                                                   l_gp_tmp_a + l_off_xview,
                                                   l_block_m*l_vec_length*i_bytes_per_val );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_tmp_c + l_off_xview,
                                                   l_gp_scratch[0] + l_off_xview,
                                                   l_gp_tmp_c + l_off_xview,
                                                   l_block_m*l_vec_length*i_bytes_per_val );

    // end of loop over m
    libxsmm_generator_loop_footer_aarch64( io_generated_code,
                                           &l_loop_label_tracker,
                                           l_gp_loop_m + l_off_xview,
                                           1 );

    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   l_gp_b + l_off_xview,
                                                   l_gp_b + l_off_xview,
                                                   64,
                                                   0 );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_c          + l_off_xview,
                                                   l_gp_scratch[0] + l_off_xview,
                                                   l_gp_c          + l_off_xview,
                                                   i_ldc*l_vec_length*i_bytes_per_val );

     // end of loop over n
     libxsmm_generator_loop_footer_aarch64( io_generated_code,
                                            &l_loop_label_tracker,
                                            l_gp_loop_n + l_off_xview,
                                            1 );
  }
  else {
    fprintf( stderr, "libxsmm_generator_gemm_m1_amx_generic: kernel configuration not supported!\n");
    exit(-1);
  }

  // disable AMX-instructions
  libxsmm_aarch64_instruction_amx( io_generated_code,
                                   LIBXSMM_AARCH64_INSTR_AMX_ENABLE,
                                   1 );

  // restore according to procedure call standard and return
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xff0 );
}

LIBXSMM_API_INTERN
int libxsmm_generator_gemm_m1_amx_kernel( libxsmm_generated_code        * io_generated_code,
                                          libxsmm_gemm_descriptor const * i_xgemm_desc ) {
  uint8_t l_bytes_per_val = 0;
  if( LIBXSMM_GEMM_PRECISION_F16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_bytes_per_val = 2;
  }
  else if( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_bytes_per_val = 4;
  }
  else if( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_bytes_per_val = 8;
  }
  else {
    return 1;
  }
  uint32_t l_vec_length = 64 / l_bytes_per_val;

  // check for support
  if( i_xgemm_desc->m % l_vec_length != 0 ) return 1;
  if( i_xgemm_desc->n % l_vec_length != 0 ) return 1;
  if(  (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) ) return 1;
  if( !(i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) ) return 1;
  if( (i_xgemm_desc->m/l_vec_length) * (i_xgemm_desc->n/l_vec_length) > l_bytes_per_val &&
       i_xgemm_desc->m % 64 != 0 ) {
        return 1;
  }


  libxsmm_generator_gemm_m1_amx_generic( l_bytes_per_val,
                                         i_xgemm_desc->m,
                                         i_xgemm_desc->n,
                                         i_xgemm_desc->k,
                                         i_xgemm_desc->lda,
                                         i_xgemm_desc->ldb,
                                         i_xgemm_desc->ldc,
                                         io_generated_code );
  return 0;
}