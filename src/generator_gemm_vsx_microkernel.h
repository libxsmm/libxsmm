/******************************************************************************
* Copyright (c) 2024, IBM Corporation                                         *
* - All rights reserved.                                                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#ifndef GENERATOR_PPC64LE_VSX_MICROKERNEL_H
#define GENERATOR_PPC64LE_VSX_MICROKERNEL_H

#include "generator_ppc64le_instructions.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load_trans( libxsmm_generated_code * io_generated_code,
                                               libxsmm_datatype const   datatype,
                                               libxsmm_datatype const   comptype, /* currently unsuded */
                                               libxsmm_ppc64le_reg    * reg_tracker,
                                               unsigned int           * loaded_regs,
                                               unsigned int             i_ptr_gpr,
                                               unsigned int             n_rows,
                                               unsigned int             n_cols,
                                               unsigned int             stride );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load( libxsmm_generated_code * io_generated_code,
                                         libxsmm_datatype const   datatype,
                                         libxsmm_datatype const   comptype, /* currently unsuded */
                                         libxsmm_ppc64le_reg    * reg_tracker,
                                         unsigned int           * loaded_regs,
                                         unsigned int             i_ptr_gpr,
                                         unsigned int             n_rows,
                                         unsigned int             n_cols,
                                         unsigned int             stride );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load_bcast( libxsmm_generated_code * io_generated_code,
                                                   libxsmm_datatype const   datatype,
                                                   libxsmm_datatype const   comptype, /* currently unsuded */
                                                   libxsmm_ppc64le_reg    * reg_tracker,
                                                   unsigned int           * loaded_regs,
                                                   unsigned int             i_ptr_gpr,
                                                   unsigned int             n_rows,
                                                   unsigned int             n_cols,
                                                   unsigned int             stride );


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_block_fma_b_bcast( libxsmm_generated_code * io_generated_code,
                                              libxsmm_datatype         datatype,
                                              unsigned int             m,
                                              unsigned int             n,
                                              unsigned int             k,
                                              unsigned int           * a,
                                              unsigned int             lda,
                                              unsigned int           * b,
                                              unsigned int             ldb,
                                              unsigned int             beta,
                                              unsigned int           * c,
                                              unsigned int             ldc );


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_microkernel( libxsmm_generated_code        * io_generated_code,
                                        libxsmm_gemm_descriptor const * i_xgemm_desc,
                                        libxsmm_ppc64le_reg           * reg_tracker,
                                        libxsmm_loop_label_tracker    * loop_labels,
                                        unsigned int                  * blocking,
                                        unsigned char const             i_a_ptr_gpr,
                                        unsigned char const             i_b_ptr_gpr,
                                        unsigned char const             i_c_ptr_gpr );

#endif
