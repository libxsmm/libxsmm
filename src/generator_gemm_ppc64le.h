/******************************************************************************
* Copyright (c) 2021, Friedrich Schiller University Jena                      *
* Copyright (c) 2024, IBM Corporation                                         *
* - All rights reserved.                                                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Will Trojak (IBM Corp.)
******************************************************************************/

#ifndef GENERATOR_GEMM_PPC64LE_H
#define GENERATOR_GEMM_PPC64LE_H

#include "generator_common.h"
#include "generator_ppc64le_instructions.h"
#include "../include/libxsmm_typedefs.h"

/**
 * Loads a block of a matrix to vector status and control registers.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_m_blocking_full number of full vectors (each 128-bit) in M-dimension.
 * @param i_n_blocking number of entries in N-dimension.
 * @param i_remainder_size size in bytes of the remainder in M-dimension.
 * @param i_stride stride in N-dimension.
 * @param i_precision 0: FP32, !=0: FP64
 * @param i_gpr_ptr GPR which has the address from which we load or to which we store data.
 * @param i_gpr_scratch GPRs which are used as scratch registers.
 *                      2+#chunks are required (see chunk desc. below).
 * @param i_vsr_first first VSR to which data is loaded or to which data is written.
 * @return number of VSR which were written.
 **/
LIBXSMM_API_INTERN
unsigned char libxsmm_generator_gemm_ppc64le_load_vsx( libxsmm_generated_code * io_generated_code,
                                                       unsigned int             i_m_blocking_full,
                                                       unsigned int             i_n_blocking,
                                                       unsigned int             i_remainder_size,
                                                       unsigned int             i_stride,
                                                       unsigned char            i_precision,
                                                       unsigned char            i_gpr_ptr,
                                                       unsigned char          * i_gpr_scratch,
                                                       unsigned char            i_vsr_first );

/**
 * Stores a block of a matrix to vector status and control registers.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_m_blocking_full number of full vectors (each 128-bit) in M-dimension.
 * @param i_n_blocking number of entries in N-dimension.
 * @param i_remainder_size size in bytes of the remainder in M-dimension.
 * @param i_stride stride in N-dimension.
 * @param i_precision 0: FP32, !=0: FP64
 * @param i_gpr_ptr GPR which has the address from which we load or to which we store data.
 * @param i_gpr_scratch GPRs which are used as scratch registers.
 *                      2+#chunks are required (see chunk desc. below).
 * @param i_vsr_first first VSR to which data is loaded or to which data is written.
 * @return number of VSR which were written.
 **/
LIBXSMM_API_INTERN
unsigned char libxsmm_generator_gemm_ppc64le_store_vsx( libxsmm_generated_code * io_generated_code,
                                                        unsigned int             i_m_blocking_full,
                                                        unsigned int             i_n_blocking,
                                                        unsigned int             i_remainder_size,
                                                        unsigned int             i_stride,
                                                        unsigned char            i_precision,
                                                        unsigned char            i_gpr_ptr,
                                                        unsigned char          * i_gpr_scratch,
                                                        unsigned char            i_vsr_first );

/**
 * Generates a microkernel using VSX.
 *
 * The generator unrolls M, N.
 * K is possibly unrolled (partially) and a loop is used if not unrolled completely.
 * M can be arbitrary, i.e., does not have to be a multiple of the vector length.
 * The kernel splits M into 128-bit chunks and (if required) one additional part (< 128 bits).
 *
 * E.g., for M=19 and single precision arithmetic,
 * four 128-bit chunks are used and one for the remaining 3 values.
 *
 * In general, the generator will use:
 *  1) N * #chunks VSX-registers for the accumulator block (C).
 *  2) #chunks VSX-register for the (partial) column of A.
 *  3) One VSX-register for an entry B.
 *
 * The blocking has to be chosen accordingly, i.e., it has to fit in the 64 available VSX registers.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 * @param i_gpr_a register holding address to A (contents are preserved).
 * @param i_gpr_b register holding address to B (contents are preserved).
 * @param i_gpr_c register holding address to C (contents are preserved).
 * @param i_m_blocking used blocking for M.
 * @param i_n_blocking used blocking for N.
 * @param i_k_blocking used blokcing for K.
 */
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_microkernel_vsx( libxsmm_generated_code        * io_generated_code,
                                                     libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                     unsigned char                   i_gpr_a,
                                                     unsigned char                   i_gpr_b,
                                                     unsigned char                   i_gpr_c,
                                                     unsigned int                    i_m_blocking,
                                                     unsigned int                    i_n_blocking,
                                                     unsigned int                    i_k_blocking );

/**
 * Generates a kernel which loops over M-blocks.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 * @param i_bytes_per_val number of bytes per value.
 * @param i_gpr_a register holding address to A (contents are not preserved).
 * @param i_gpr_b register holding address to B (contents are not preserved).
 * @param i_gpr_c register holding address to C (contents are not preserved).
 * @param i_gpr_scratch scratch register.
 * @param i_max_block_m maximum size of a block w.r.t. M.
 * @param i_n used size of the blocks w.r.t. N.
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_m_loop_vsx( libxsmm_generated_code        * io_generated_code,
                                                libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                unsigned int                    i_bytes_per_val,
                                                unsigned char                   i_gpr_a,
                                                unsigned char                   i_gpr_b,
                                                unsigned char                   i_gpr_c,
                                                unsigned char                   i_gpr_scratch,
                                                unsigned int                    i_max_block_m,
                                                unsigned int                    i_n );


LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_ppc64le_vsx_bytes( libxsmm_generated_code        * io_generated_code,
                                                       libxsmm_gemm_descriptor const * i_xgemm_desc );

/**
 * Generates a matrix kernel for PPC64LE-VSX.
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_vsx_wt( libxsmm_generated_code        * io_generated_code,
                                                   libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                   libxsmm_ppc64le_reg           * reg_tracker );

/**
 * Generates a matrix kernel for PPC64LE-VSX.
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_vsx( libxsmm_generated_code        * io_generated_code,
                                                libxsmm_gemm_descriptor const * i_xgemm_desc );

/**
 * Generates a matrix kernel for PPC64LE.
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel( libxsmm_generated_code        * io_generated_code,
                                            libxsmm_gemm_descriptor const * i_xgemm_desc );

#endif
