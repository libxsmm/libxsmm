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

#ifndef GENERATOR_GEMM_POWER_H
#define GENERATOR_GEMM_POWER_H

#include "generator_common.h"
#include "../include/libxsmm_typedefs.h"

/**
 * Loads or stores a block of a matrix to vector status and control registers.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_m_blocking_full number of full vectors (each 128-bit) in M-dimension.
 * @param i_n_blocking number of entries in N-dimension.
 * @param i_remainder_size size in bytes of the remainder in M-dimension.
 * @param i_stride stride in N-dimension.
 * @param i_load_store 0: load, !=0: store.
 * @param i_precision 0: FP32, !=0: FP64
 * @param i_endianness 0: little endian, !=0 big endian.
 * @param i_gpr_ptr GPR which has the address from which we load or to which we store data.
 * @param i_gpr_scratch GPRs which are used as scratch registers.
 * @param i_vsr_first first VSR to which data is loaded or to which data is written.
 * @return number of VSR which were written.
 **/
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
                                                           unsigned char            i_vsr_first );

/**
 * Generates a microkernel using VSX.
 *
 * The generator unrolls M, N, and K.
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
 * @param i_m_blocking used blocking for M.
 * @param i_n_blocking used blocking for N.
 */
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_power_microkernel_vsx( libxsmm_generated_code        * io_generated_code,
                                                   libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                   unsigned int                    i_m_blocking,
                                                   unsigned int                    i_n_blocking );

/**
 * Genrates a matrix kernel for POWER.
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 **/
LIBXSMM_API_INTERN
int libxsmm_generator_gemm_power_kernel( libxsmm_generated_code        * io_generated_code,
                                         libxsmm_gemm_descriptor const * i_xgemm_desc );

#endif