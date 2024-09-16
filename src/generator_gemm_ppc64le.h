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


typedef void (*libxsmm_ppc64le_reg_func)( unsigned int const i_vec_len,
                                          unsigned int      *i_blocking,
                                          unsigned int      *o_reg );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_reg_vsx( unsigned int const i_vec_len,
                                             unsigned int      *i_blocking,
                                             unsigned int      *o_reg);


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_reg_mma( unsigned int const i_vec_len,
                                             unsigned int      *i_blocking,
                                             unsigned int      *o_reg );


LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_ppc64le_n_reg( unsigned int const       i_vec_len,
                                                   unsigned int            *i_blocking,
                                                   libxsmm_ppc64le_reg_func i_reg_func );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_blocking_iter( unsigned int const       i_reg_max,
                                                   unsigned int const       i_vec_len,
                                                   unsigned int const       i_comp_bytes,
                                                   unsigned int            *i_dims,
                                                   unsigned int            *i_increment,
                                                   unsigned int            *i_weights,
                                                   unsigned int const       i_nweight,
                                                   unsigned int            *o_blocking,
                                                   libxsmm_ppc64le_reg_func i_reg_func );


void libxsmm_generator_gemm_ppc64le_create_blocking( libxsmm_generated_code        *io_generated_code,
                                                     libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                     libxsmm_ppc64le_blocking      *io_blocking );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_setup_blocking( libxsmm_generated_code        *io_generated_code,
                                                    libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                    libxsmm_ppc64le_blocking      *io_blocking );



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


#endif
