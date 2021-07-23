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
#ifndef GENERATOR_GEMM_M1_AMX_H
#define GENERATOR_GEMM_M1_AMX_H

#include "stdint.h"
#include "generator_common.h"
#include "../include/libxsmm_typedefs.h"

/* Descriptor
 * bits 5-31: https://gist.github.com/dougallj/7a75a3be1ec69ca550e7c36dc75e0d6f
 * bits 3-4: unused
 * bits 0-2: number of valid bits in the operand.
 */
/* define AMX load / store instructions */
#define LIBXSMM_AARCH64_INSTR_AMX_LDX    0x00201005
#define LIBXSMM_AARCH64_INSTR_AMX_LDY    0x00201025
#define LIBXSMM_AARCH64_INSTR_AMX_STX    0x00201045
#define LIBXSMM_AARCH64_INSTR_AMX_STY    0x00201065
#define LIBXSMM_AARCH64_INSTR_AMX_LDZ    0x00201085
#define LIBXSMM_AARCH64_INSTR_AMX_STZ    0x002010A5

/* define AMX compute instructions */
#define LIBXSMM_AARCH64_INSTR_AMX_FMA16  0x002011e5
#define LIBXSMM_AARCH64_INSTR_AMX_FMA32  0x00201185
#define LIBXSMM_AARCH64_INSTR_AMX_FMA64  0x00201145

/* define AMX on/off-switch */
#define LIBXSMM_AARCH64_INSTR_AMX_ENABLE 0x00201221

/**
 * Generates an AMX-instruction.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_instr AMX instruction.
 * @param i_operand 64-bit GP-register for most, 0/1 for enable/disable.
 **/
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_amx( libxsmm_generated_code * io_generated_code,
                                      const unsigned int       i_instr,
                                      const unsigned char      i_operand );

/**
 * Loads or store a block of C according to the kernels internal mapping to z.
 *
 * @param i_bytes_per_val number of bytes per value (2: FP16, 4: FP32, 8: FP64).
 * @param i_gp_c general purpose register holding the address to C.
 * @param i_block_m number of times the vector-length fits in the M dimension of the block.
 * @param i_block_n number of times the vector-length fits in the N-dimension of the block.
 * @param i_ldc leading dimension of C.
 * @param i_gp_stride_vec stride in z-register and C when jumping in the vector dimension.
 * @param i_gp_stride_c stride in z-register and C when jumping in the N-dimension w.r.t. C.
 * @param i_gp_scratch general purpose scratch registers.
 * @param i_load if true: data is loaded from memory to z-register, otherwise stored from z to memory.
 * @param io_kernel kernel which is modified.
 **/
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
                                            libxsmm_generated_code * io_generated_code );

/**
 * Generates a matrix kernel which performs the operation C += A*B^T for a block of C.
 * with possible C block configs:
 * (2x1, 1x2, 1x1) for HP (given as multiples of 32 HP-values).
 * (4x1, 3x1, 2x1, 1x1, 1x2, 1x3, 1x4, 2x2) for SP (given as multiples of 16 SP-values).
 * (8x1, 7x1, [...], 1x1, [...], 1x7, 1x8, 2x4, 4x2) for DP (given as multiples of 8 SP-values).
 *
 * K is arbitrary.
 *
 * The generated kernel assumes the addressed of A, B and C w.r.t. the block in x0, x1, x2.
 * x0-x2 remain unmodified.
 * x3 - x23 are used as scratch registers.
 *
 * @param i_bytes_per_val number of bytes per value (2: FP16, 4: FP32, 8: FP64).
 * @param i_block_m number of times the vector-length fits in the M dimension of the block.
 * @param i_block_n number of times the vector-length fits in the N-dimension of the block.
 * @param i_gp_a register containing respective address of A.
 * @param i_gp_b register containing respective address of B.
 * @param i_gp_c register containing respective address of C.
 * @param i_k BLAS-parameter K.
 * @param i_lda leading dimension of A.
 * @param i_ldb leading dimension of B.
 * @param i_ldc leading dimension of C.
 * @param io_generated_code code stream to which the generated instructions are added.
 **/
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
                                          libxsmm_generated_code * io_generated_code );

/**
 * Generates a generic matrix kernel which performs the operation C += A*B^T.
 * Currently, two modes are supported:
 *   1) A raw-forward to the micro-kernel.
 *   2) A wrapped execution of micro-kernel 2x1 (HP), 4x1 (SP) or 8x1 (DP).
 *      Here, M has to be a multiple of 64 and N a multiple of 32 (HP), 16 (SP) or 8 (DP).
 *      K is arbitrary.
 *
 * @param i_bytes_per_val number of bytes per value (2: FP16, 4: FP32, 8: FP64).
 * @param i_m BLAS-parameter m.
 * @param i_n BLAS-parameter n.
 * @param i_k BLAS-parameter k.
 * @param i_lda leading dimension of A.
 * @param i_ldb leading dimension of B.
 * @param i_ldc leading dimension of C.
 * @param io_generated_code code stream to which the generated instructions are added.
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_m1_amx_generic( uint8_t                  i_bytes_per_val,
                                            uint32_t                 i_m,
                                            uint32_t                 i_n,
                                            uint32_t                 i_k,
                                            uint32_t                 i_lda,
                                            uint32_t                 i_ldb,
                                            uint32_t                 i_ldc,
                                            libxsmm_generated_code * io_generated_code );

/**
 * Wrapper of generic which accepts an LIBXSMM-description of the matrix kernel.
 *
 * @param io_generated_code code stream to which the generated instructions are added (if supported).
 * @param i_xgemm_desc description of the matrix kernel.
 * @return 0 if the kernel config is supported, 1 otherwise.
 **/
LIBXSMM_API_INTERN
int libxsmm_generator_gemm_m1_amx_kernel( libxsmm_generated_code        * io_generated_code,
                                          libxsmm_gemm_descriptor const * i_xgemm_desc );

#endif