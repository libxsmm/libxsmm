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
#include "generator_gemm_mma_microkernel.h"
#include "generator_gemm_vsx_microkernel.h"
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
unsigned char libxsmm_generator_gemm_ppc64le_load_vsx( libxsmm_generated_code *io_generated_code,
                                                       unsigned int            i_m_blocking_full,
                                                       unsigned int            i_n_blocking,
                                                       unsigned int            i_remainder_size,
                                                       unsigned int            i_stride,
                                                       unsigned char           i_precision,
                                                       unsigned char           i_gpr_ptr,
                                                       unsigned char          *i_gpr_scratch,
                                                       unsigned char           i_vsr_first );

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
unsigned char libxsmm_generator_gemm_ppc64le_store_vsx( libxsmm_generated_code *io_generated_code,
                                                        unsigned int            i_m_blocking_full,
                                                        unsigned int            i_n_blocking,
                                                        unsigned int            i_remainder_size,
                                                        unsigned int            i_stride,
                                                        unsigned char           i_precision,
                                                        unsigned char           i_gpr_ptr,
                                                        unsigned char          *i_gpr_scratch,
                                                        unsigned char           i_vsr_first );


/**
 * Generators the inner m-loop for PPC64LE VSX GEMM kernel
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 * @param io_reg_tracker register tracking structure.
 * @param io_loop_labels pointer to libxsmm loop label tracker
 * @param i_blocking pointer to array containing the bocking ordered [n,m,k]
 * @param i_a_prt_gpr number of register containing current pointer to a block
 * @param i_a_prt_gpr number of register containing current pointer to b block
 * @param i_a_prt_gpr number of register containing current pointer to c block
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_vsx_m_loop( libxsmm_generated_code        *io_generated_code,
                                                libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking      *io_blocking,
                                                libxsmm_ppc64le_reg           *io_reg_tracker,
                                                libxsmm_loop_label_tracker    *io_loop_labels,
                                                unsigned char const            i_a,
                                                unsigned char const            i_b,
                                                unsigned char const            i_c );


/**
 * Generators the inner m-loop for PPC64LE MMA GEMM kernel
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 * @param io_reg_tracker register tracking structure.
 * @param io_loop_labels pointer to libxsmm loop label tracker
 * @param i_blocking pointer to array containing the bocking ordered [n,m,k]
 * @param i_acc array of accumulation registers
 * @param i_a_prt_gpr number of register containing current pointer to a block
 * @param i_a_prt_gpr number of register containing current pointer to b block
 * @param i_a_prt_gpr number of register containing current pointer to c block
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_mma_m_loop( libxsmm_generated_code        *io_generated_code,
                                                libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking      *i_blocking,
                                                libxsmm_ppc64le_reg           *io_reg_tracker,
                                                libxsmm_loop_label_tracker    *io_loop_labels,
                                                unsigned int                  *i_acc,
                                                unsigned char const            i_a,
                                                unsigned char const            i_b,
                                                unsigned char const            i_c );

/**
 * Generates a matrix kernel for PPC64LE VSX.
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 * @param io_reg_tracker register tracking structure.
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_vsx( libxsmm_generated_code        *io_generated_code,
                                                libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking      *io_blocking,
                                                libxsmm_ppc64le_reg           *io_reg_tracker );

/**
 * Generates a matrix kernel for PPC64LE MMA.
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 * @param io_reg_tracker register tracking structure.
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_mma( libxsmm_generated_code        *io_generated_code,
                                                libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking      *i_blocking,
                                                libxsmm_ppc64le_reg           *io_reg_tracker );


/**
 * Generates a matrix kernel for PPC64LE.
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_xgemm_desc description of the matrix-operation.
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel( libxsmm_generated_code        *io_generated_code,
                                            libxsmm_gemm_descriptor const *i_xgemm_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_setup_blocking( libxsmm_generated_code        *io_generated_code,
                                                    const libxsmm_gemm_descriptor *i_xgemm_desc,
                                                    libxsmm_ppc64le_blocking      *io_blocking );

#endif
