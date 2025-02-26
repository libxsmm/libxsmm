/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#include "generator_gemm_s390x.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_reg( unsigned int  i_vec_ele,
                                            unsigned int *i_blocking,
                                            unsigned int *o_reg ) {
  /* a reg := [0], b reg := [1], c reg := [2] */
  o_reg[0] = ( ( i_blocking[0] + i_vec_ele - 1 ) / i_vec_ele )*i_blocking[2];
  o_reg[1] = i_blocking[1]*i_blocking[2];
  o_reg[2] = ( ( i_blocking[0] + i_vec_ele - 1 ) / i_vec_ele )*i_blocking[1];
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_s390x_reg_sum( unsigned int const     i_vec_ele,
                                                   unsigned int          *i_blocking,
                                                   libxsmm_s390x_reg_func i_reg_func ) {
  unsigned int l_reg[3];
  i_reg_func( i_vec_ele, i_blocking, l_reg );
  return l_reg[0] + l_reg[1] + l_reg[2];
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_blocking_maximise( libxsmm_generated_code        *io_generated_code,
                                                          const libxsmm_gemm_descriptor *i_xgemm_desc,
                                                          libxsmm_s390x_blocking        *io_blocking,
                                                          libxsmm_s390x_reg_func         i_reg_func ) {
  unsigned int l_vec_ele = io_blocking->vector_len_comp;
  unsigned int l_inc[3], l_blocking[3], l_dims[3], l_reg[3], l_full[3];
  unsigned int l_weights[] = {0, 1, 0, 1, 2};
  unsigned int l_nweight = (unsigned int)(sizeof(l_weights) / sizeof(l_weights[0]));
  unsigned int l_nreg, l_reg_max, l_step, i;

  l_inc[0] = l_vec_ele;
  l_inc[1] = 1;
  l_inc[2] = 1;
  l_blocking[0] = l_vec_ele;
  l_blocking[1] = 1;
  l_blocking[2] = 1;

  l_dims[0] = i_xgemm_desc->m;
  l_dims[1] = i_xgemm_desc->n;
  l_dims[2] = i_xgemm_desc->k;
  for ( i = 0; i < 3 ; ++i ) {
    l_reg[i] = 0;
    l_full[i] = (l_blocking[i] >= l_dims[i]) ? 1 : 0;
  }

  l_nreg = libxsmm_generator_gemm_s390x_reg_sum( l_vec_ele, l_blocking, i_reg_func );
  l_reg_max = libxsmm_s390x_vec_nreg( io_generated_code ) - libxsmm_s390x_vec_nscratch( io_generated_code );

  l_step = 0;
  while ( l_nreg < l_reg_max && !( l_full[0] && l_full[1] && l_full[2] ) ) {
    unsigned int l_b = l_weights[l_step % l_nweight];

    if( !l_full[l_b] ) {
      unsigned int l_temp_nreg;
      l_blocking[l_b] += l_inc[l_b];
      l_temp_nreg = libxsmm_generator_gemm_s390x_reg_sum( l_vec_ele, l_blocking, i_reg_func );

      if ( l_temp_nreg > l_reg_max ) {
        l_blocking[l_b] -= l_inc[l_b];
        l_full[l_b] = 1;
      } else {
        l_nreg = l_temp_nreg;
      }
      l_full[l_b] |= ( l_blocking[l_b] >= l_dims[l_b] );
    }
    ++l_step;
  }

  l_nreg = libxsmm_generator_gemm_s390x_reg_sum( l_vec_ele, l_blocking, i_reg_func );
  if ( libxsmm_generator_gemm_s390x_reg_sum( l_vec_ele, l_blocking, i_reg_func ) > l_reg_max ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Set blocking and reg layout based on result */
  io_blocking->block_m = l_blocking[0];
  io_blocking->block_n = l_blocking[1];
  io_blocking->block_k = l_blocking[2];

  i_reg_func( io_blocking->vector_len_comp, l_blocking, l_reg );
  io_blocking->n_reg_a = l_reg[0];
  io_blocking->n_reg_b = l_reg[1];
  io_blocking->n_reg_c = l_reg[2];

  io_blocking->reg_lda = (io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;
  io_blocking->reg_ldb = io_blocking->block_k;
  io_blocking->reg_ldc = ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_blocking_init( libxsmm_generated_code        *io_generated_code,
                                                      const libxsmm_gemm_descriptor *i_xgemm_desc,
                                                      libxsmm_s390x_blocking        *io_blocking ) {
  unsigned int l_v_bytes = LIBXSMM_S390X_VR_WIDTH / 8;
  libxsmm_s390x_reg_func l_reg_func = NULL;

  io_blocking->vector_len_a = l_v_bytes / libxsmm_s390x_bytes( io_generated_code, LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) );
  io_blocking->vector_len_b = l_v_bytes / libxsmm_s390x_bytes( io_generated_code, LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ) );
  io_blocking->vector_len_c = l_v_bytes / libxsmm_s390x_bytes( io_generated_code, LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) );
  io_blocking->comp_bytes = libxsmm_s390x_bytes( io_generated_code, LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ) );
  io_blocking->vector_len_comp = l_v_bytes / io_blocking->comp_bytes;

  if ( LIBXSMM_S390X_ARCH11 <= io_generated_code->arch ) {
    l_reg_func = &libxsmm_generator_gemm_s390x_vxrs_reg;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  /* Perform a rough maximisation of blocking */
  libxsmm_generator_gemm_s390x_vxrs_blocking_maximise( io_generated_code, i_xgemm_desc, io_blocking, l_reg_func );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_m_loop( libxsmm_generated_code        *io_generated_code,
                                               libxsmm_s390x_defer           *io_deferred_code,
                                               const libxsmm_gemm_descriptor *i_xgemm_desc,
                                               libxsmm_s390x_reg             *io_reg_tracker,
                                               libxsmm_s390x_blocking        *i_blocking,
                                               libxsmm_loop_label_tracker    *io_loop_labels,
                                               unsigned int                   i_a,
                                               unsigned int                   i_b,
                                               unsigned int                   i_c ) {
  unsigned int l_a, l_c, l_m_iters, l_m_loop, l_part;

  /* m loop values */
  l_m_iters = i_xgemm_desc->m / i_blocking->block_m ;

  /* Check if there is a partial m part */
  l_part = ( 0 != ( i_xgemm_desc->m % i_blocking->block_m ) ) ? 1 : 0 ;

  if ( 1 < l_m_iters || 1 == l_part ) {
    l_a = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_a, l_a, 0 );

    l_c = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_c, l_c, 0 );
  } else {
    l_a = i_a;
    l_c = i_c;
  }

  /* Set jump point if we nned to loop */
  if ( 1 < l_m_iters ) {
    l_m_loop = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_set_value( io_generated_code, l_m_loop, l_m_iters );
    libxsmm_s390x_instr_register_jump_label( io_generated_code, io_loop_labels );
  }

  /* Microkernel for full blocking */
  if ( 0 < l_m_iters ) {
    libxsmm_generator_vxrs_microkernel( io_generated_code,
                                        i_xgemm_desc,
                                        io_reg_tracker,
                                        io_loop_labels,
                                        i_blocking,
                                        l_a,
                                        i_b,
                                        l_c );
  }

  /* Increment a and c pointers if required */
  if ( 1 < l_m_iters || ( 0 < l_m_iters && 1 == l_part ) ) {
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, l_a, l_a, i_blocking->comp_bytes*i_blocking->block_m );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, l_c, l_c, i_blocking->comp_bytes*i_blocking->block_m );
  }

  /* Jump back if we need to m-loop */
  if ( 1 < l_m_iters ) {
    libxsmm_s390x_instr_branch_count_jump_label( io_generated_code, l_m_loop, io_loop_labels );
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_m_loop );
  }

  /* Mickrokernel for remaining m part */
  if ( 1 == l_part ) {
    unsigned int l_block_m = i_blocking->block_m;
    i_blocking->block_m = i_xgemm_desc->m % l_block_m;
    libxsmm_generator_vxrs_microkernel( io_generated_code,
                                        i_xgemm_desc,
                                        io_reg_tracker,
                                        io_loop_labels,
                                        i_blocking,
                                        l_a,
                                        i_b,
                                        l_c );
    i_blocking->block_m = l_block_m;
  }

  if ( 1 < l_m_iters || 1 == l_part ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_a );
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_c );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_kernel( libxsmm_generated_code        *io_generated_code,
                                               libxsmm_s390x_defer           *io_deferred_code,
                                               const libxsmm_gemm_descriptor *i_xgemm_desc,
                                               libxsmm_s390x_reg             *io_reg_tracker,
                                               libxsmm_s390x_blocking        *i_blocking ) {
  unsigned int i_a, i_b, i_c, l_n_iters, l_n_loop, l_part;
  libxsmm_loop_label_tracker l_loop_labels;

  /* Reset loop labels as this is a new kernel */
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* Start stream based on ABI */
  libxsmm_s390x_instr_open_stack( io_generated_code );

  /* Unpack the args from the LIBXSMM matrix arg struct, place them in arg GPR */
  libxsmm_s390x_instr_unpack_args( io_generated_code, io_reg_tracker );
  i_a = LIBXSMM_S390X_GPR_ARG0;
  i_b = LIBXSMM_S390X_GPR_ARG1;
  i_c = LIBXSMM_S390X_GPR_ARG2;

  /* n loop values */
  l_n_iters = i_xgemm_desc->n / i_blocking->block_n ;

  /* Check if there is partial n part */
  l_part = ( 0 != ( i_xgemm_desc->n % i_blocking->block_n ) ) ? 1 : 0 ;

  /* Set jump point if we need an n loop */
  if ( 1 < l_n_iters ) {
    l_n_loop = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_set_value( io_generated_code, l_n_loop, l_n_iters );
    libxsmm_s390x_instr_register_jump_label( io_generated_code, &l_loop_labels );
  }

  /* Microkernel for packed n */
  if ( 0 < l_n_iters ) {
    libxsmm_generator_gemm_s390x_vxrs_m_loop( io_generated_code,
                                              io_deferred_code,
                                              i_xgemm_desc,
                                              io_reg_tracker,
                                              i_blocking,
                                              &l_loop_labels,
                                              i_a,
                                              i_b,
                                              i_c );
  }

  /* Increment b and c pointers if required */
  if ( ( 1 < l_n_iters ) || ( 0 < l_n_iters && 1 == l_part ) ) {
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_b, i_b, i_xgemm_desc->ldb*i_blocking->comp_bytes*i_blocking->block_n );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_c, i_c, i_xgemm_desc->ldc*i_blocking->comp_bytes*i_blocking->block_n );
  }

  /* Jump if looping */
  if ( 1 < l_n_iters ) {
    libxsmm_s390x_instr_branch_count_jump_label( io_generated_code, l_n_loop, &l_loop_labels );
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_n_loop );
  }

  /* Mickrokernel for remaining n part */
  if ( 1 == l_part ) {
    unsigned int l_block_n = i_blocking->block_n;
    i_blocking->block_n = i_xgemm_desc->n % l_block_n;
    libxsmm_generator_gemm_s390x_vxrs_m_loop( io_generated_code,
                                              io_deferred_code,
                                              i_xgemm_desc,
                                              io_reg_tracker,
                                              i_blocking,
                                              &l_loop_labels,
                                              i_a,
                                              i_b,
                                              i_c );
    i_blocking->block_n = l_block_n;
  }

  /* Collapse and return */
  libxsmm_s390x_instr_collapse_stack( io_generated_code );

  /* Resovle deferred instructions */
  libxsmm_s390x_instr_deferred_resolve( io_generated_code, io_deferred_code );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_kernel( libxsmm_generated_code        *io_generated_code,
                                          const libxsmm_gemm_descriptor *i_xgemm_desc ) {
  libxsmm_s390x_reg l_reg_tracker;
  libxsmm_s390x_blocking l_blocking;
  libxsmm_s390x_defer *l_deferred_code = libxsmm_s390x_defer_init( io_generated_code );

  /* Initial register tracking structure */
  l_reg_tracker = libxsmm_s390x_reg_init( io_generated_code );

  if ( LIBXSMM_S390X_ARCH11 <= io_generated_code->arch  ) {
    /* Create a blocking of the matrix, later switches  */
    libxsmm_generator_gemm_s390x_vxrs_blocking_init( io_generated_code, i_xgemm_desc, &l_blocking );

    /* jit the kernel */
    libxsmm_generator_gemm_s390x_vxrs_kernel( io_generated_code, l_deferred_code, i_xgemm_desc, &l_reg_tracker, &l_blocking );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  libxsmm_s390x_defer_destroy( io_generated_code, l_deferred_code );
  libxsmm_s390x_reg_destroy( io_generated_code, &l_reg_tracker );
  return;
}
