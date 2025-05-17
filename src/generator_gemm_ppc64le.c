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

#include <string.h>
#include "generator_gemm_ppc64le.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_reg_vsx( unsigned int const i_vec_len,
                                             unsigned int      *i_blocking,
                                             unsigned int      *o_reg ) {
  o_reg[0] = ( ( i_blocking[0] + i_vec_len - 1 ) / i_vec_len )*i_blocking[2];
  o_reg[1] = i_blocking[1]*i_blocking[2];
  o_reg[2] = ( ( i_blocking[0] + i_vec_len - 1 ) / i_vec_len )*i_blocking[1];
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_reg_mma( unsigned int const i_vec_len,
                                             unsigned int      *i_blocking,
                                             unsigned int      *o_reg ) {
  unsigned int l_acc_vec_len = LIBXSMM_PPC64LE_ACC_WIDTH / LIBXSMM_PPC64LE_VSR_WIDTH;
  unsigned int l_n_acc_c = ( ( ( i_blocking[1] + i_vec_len - 1 ) / i_vec_len ) *
                             ( ( i_blocking[0] + i_vec_len - 1 ) / i_vec_len ) );
  o_reg[0] = ( ( i_blocking[0] + i_vec_len - 1 ) / i_vec_len )*i_blocking[2];
  o_reg[1] = ( ( i_blocking[1] + i_vec_len - 1 ) / i_vec_len )*i_blocking[2];
  o_reg[2] = l_n_acc_c*l_acc_vec_len;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_ppc64le_n_reg( unsigned int const       i_vec_len,
                                                   unsigned int            *i_blocking,
                                                   libxsmm_ppc64le_reg_func i_reg_func ) {
  unsigned int l_reg[3];
  i_reg_func( i_vec_len, i_blocking, l_reg );
  return l_reg[0] + l_reg[1] + l_reg[2];
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_blocking_iter( unsigned int const       i_reg_max,
                                                   unsigned int const       i_vec_len,
                                                   unsigned int const       i_comp_bytes,
                                                   unsigned int            *i_dims,
                                                   unsigned int            *i_increment,
                                                   unsigned int            *i_weights,
                                                   unsigned int const       i_nweight,
                                                   unsigned int            *o_blocking,
                                                   libxsmm_ppc64le_reg_func i_reg_func ) {
  unsigned int l_nreg;
  unsigned int l_maxed, l_full[] = {0, 0, 0};
  unsigned int i, l_step;

  /* Initial blocking */
  o_blocking[0] = i_increment[0];
  o_blocking[1] = i_increment[1];
  o_blocking[2] = i_increment[2];
  l_nreg = libxsmm_generator_gemm_ppc64le_n_reg( i_vec_len, o_blocking, i_reg_func );

  for ( i = 0; i < 3; ++i ) {
    l_full[i] |= (o_blocking[i] >= i_dims[i]);
  }

  l_maxed = ( ( l_full[0] != 0 ) & ( l_full[1] != 0 ) & ( l_full[2] != 0 ) );

  l_step = 0;
  while ( l_nreg < i_reg_max && l_maxed == 0 ) {
    unsigned int l_b = i_weights[l_step % i_nweight];

    if( l_full[l_b] == 0 ) {
      unsigned int l_temp_nreg;

      o_blocking[l_b] += i_increment[l_b];
      l_temp_nreg = libxsmm_generator_gemm_ppc64le_n_reg( i_vec_len, o_blocking, i_reg_func );

      if ( l_temp_nreg > i_reg_max ) {
        o_blocking[l_b] -= i_increment[l_b];
        l_full[l_b] = 1;
      } else {
        l_nreg = l_temp_nreg;
      }

      l_full[l_b] |= (o_blocking[l_b] >= i_dims[l_b]);
    }

    l_maxed |= ((l_full[0] != 0) & (l_full[1] != 0) & (l_full[2] !=0));

    ++l_step;
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_create_blocking( libxsmm_generated_code        *io_generated_code,
                                                     const libxsmm_gemm_descriptor *i_xgemm_desc,
                                                     libxsmm_ppc64le_blocking      *io_blocking ) {
  unsigned int l_reg_max = LIBXSMM_PPC64LE_VSR_NMAX - LIBXSMM_PPC64LE_VSR_SCRATCH;
  unsigned int l_vector_len = io_blocking->vector_len_comp;
  unsigned int l_comp_bytes = io_blocking->comp_bytes;
  libxsmm_ppc64le_reg_func l_reg_func = NULL;

  unsigned int l_inc[3], l_dims[3];
  unsigned int l_weight[] = {0, 1, 0, 1, 2};
  unsigned int l_nweight = (unsigned int)(sizeof(l_weight)/sizeof(l_weight[0]));
  unsigned int l_blocking[] = {0, 0, 0};

  l_dims[0] = i_xgemm_desc->m;
  l_dims[1] = i_xgemm_desc->n;
  l_dims[2] = i_xgemm_desc->k;
  l_inc[0] = (unsigned int)io_blocking->m_ele;
  l_inc[1] = (unsigned int)io_blocking->n_ele;
  l_inc[2] = (unsigned int)io_blocking->k_ele;

  if ( io_generated_code->arch == LIBXSMM_PPC64LE_VSX ) {
    l_reg_func = &libxsmm_generator_gemm_ppc64le_reg_vsx;
  } else if ( io_generated_code->arch == LIBXSMM_PPC64LE_MMA ) {
    l_reg_func = &libxsmm_generator_gemm_ppc64le_reg_mma;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  libxsmm_generator_gemm_ppc64le_blocking_iter( l_reg_max, l_vector_len, l_comp_bytes, l_dims, l_inc, l_weight, l_nweight, l_blocking, l_reg_func );

  io_blocking->block_m = l_blocking[0];
  io_blocking->block_n = l_blocking[1];
  io_blocking->block_k = l_blocking[2];
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_setup_blocking( libxsmm_generated_code        *io_generated_code,
                                                    const libxsmm_gemm_descriptor *i_xgemm_desc,
                                                    libxsmm_ppc64le_blocking      *io_blocking ) {
  unsigned int l_v_bytes = LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  libxsmm_ppc64le_reg_func l_reg_func = NULL;
  unsigned int l_vector_len;
  unsigned int l_blocking[3], l_reg[3], l_n_reg;

  memset( io_blocking, 0, sizeof( libxsmm_ppc64le_blocking ) );
  io_blocking->vector_len_a = l_v_bytes / libxsmm_ppc64le_instr_bytes( io_generated_code, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) );
  io_blocking->vector_len_b = l_v_bytes / libxsmm_ppc64le_instr_bytes( io_generated_code, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ) );
  io_blocking->vector_len_c = l_v_bytes / libxsmm_ppc64le_instr_bytes( io_generated_code, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) );
  io_blocking->comp_bytes = libxsmm_ppc64le_instr_bytes( io_generated_code, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ) );
  io_blocking->vector_len_comp = l_v_bytes / io_blocking->comp_bytes;
  l_vector_len = io_blocking->vector_len_comp;

  /* Vector len */
  switch ( io_generated_code->arch ) {
    case LIBXSMM_PPC64LE_VSX: {
      io_blocking->m_ele = l_vector_len;
      io_blocking->n_ele = l_vector_len;
      io_blocking->k_ele = l_vector_len;
    } break;
    case LIBXSMM_PPC64LE_MMA: {
      io_blocking->m_ele = l_vector_len;
      io_blocking->n_ele = (LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F64) ? 2*l_vector_len : l_vector_len;
      io_blocking->k_ele = l_vector_len;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }

  libxsmm_generator_gemm_ppc64le_create_blocking( io_generated_code, i_xgemm_desc, io_blocking );
  io_blocking->n_block_m_full = i_xgemm_desc->m / io_blocking->block_m;
  io_blocking->n_block_n_full = i_xgemm_desc->n / io_blocking->block_n;
  io_blocking->n_block_k_full = i_xgemm_desc->k / io_blocking->block_k;

  switch ( io_generated_code->arch ) {
    case LIBXSMM_PPC64LE_VSX: {
      l_reg_func = &libxsmm_generator_gemm_ppc64le_reg_vsx;

      io_blocking->reg_lda = (io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;
      io_blocking->reg_ldb = io_blocking->block_k;
      io_blocking->reg_ldc = ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;
    } break;
    case LIBXSMM_PPC64LE_MMA: {
      l_reg_func = &libxsmm_generator_gemm_ppc64le_reg_mma;

      io_blocking->reg_lda = ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;
      io_blocking->reg_ldb = io_blocking->block_k;
      io_blocking->reg_ldc = ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }

  l_blocking[0] = io_blocking->block_m;
  l_blocking[1] = io_blocking->block_n;
  l_blocking[2] = io_blocking->block_k;
  l_reg_func( io_blocking->vector_len_comp, l_blocking, l_reg );
  io_blocking->n_reg_a = l_reg[0];
  io_blocking->n_reg_b = l_reg[1];
  io_blocking->n_reg_c = l_reg[2];

  l_n_reg = libxsmm_generator_gemm_ppc64le_n_reg( io_blocking->vector_len_comp, l_blocking, l_reg_func );
  if ( l_n_reg > LIBXSMM_PPC64LE_VSR_NMAX ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_br_vsx_m_loop( libxsmm_generated_code        *io_generated_code,
                                                   libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                   libxsmm_ppc64le_blocking      *io_blocking,
                                                   libxsmm_ppc64le_reg           *io_reg_tracker,
                                                   libxsmm_loop_label_tracker    *io_loop_labels,
                                                   unsigned int                   i_a,
                                                   unsigned int                   i_b,
                                                   unsigned int                   i_b_n_offset,
                                                   unsigned int                   i_c,
                                                   unsigned int                   i_br,
                                                   unsigned int                   i_a_offset,
                                                   unsigned int                   i_b_offset ) {
  libxsmm_datatype l_c_datatype = (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_comptype = (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int l_a, l_b, l_c;
  unsigned int i;
  unsigned int l_a_m_offset = 0, l_a_ptr = 0, l_b_ptr = 0;
  unsigned int l_m_iters = i_xgemm_desc->m / io_blocking->block_m;
  unsigned int l_m_loop = 0, l_br_loop = 0;
  unsigned int l_packed = ( 0 == i_xgemm_desc->m % io_blocking->block_m ) ? 1 : 0;
  unsigned int l_c_reg[LIBXSMM_PPC64LE_VSR_NMAX];

  /* Create registers to store A and B pointers */
  l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  l_b = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  /* Create local pointer for C, if required */
  if ( 1 < l_m_iters || !l_packed ) {
    l_c = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_c, l_c );
  } else {
    l_c = i_c;
  }

  /* If required, allocate an A m-loop offset register and zero */
  if ( 1 < l_m_iters || ( 0 < l_m_iters && !l_packed ) ) {
    l_a_m_offset = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_a_m_offset, 0 );
  }

  /* Get a register for the pointer offset counter, if required */
  if ( ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) ||
       ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) {
    l_a_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_b_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  }

  /* Allocate VSR accumulators for C, due to persistence required for batch-reduce */
  libxsmm_ppc64le_alloc_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, io_blocking->n_reg_c, 1, l_c_reg );

  /* Get a register for the batch-reduce loop counter */
  l_br_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  /* Set jump point if required */
  if ( l_m_iters > 1 ) {
    l_m_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_m_loop, l_m_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );
  }

  /* Main packed m-loop */
  if ( l_m_iters > 0 ) {
    /* For BR-address copy the initial pointer pointer */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) {
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a, l_a_ptr );
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_b, l_b_ptr );
    /* For BR-offset, reset the offsets */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a_offset, l_a_ptr );
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_b_offset, l_b_ptr );
    /* For BR-stride set the initial A and B pointers, including the loop offsets */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, i_a, l_a_m_offset, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, i_b, i_b_n_offset, 0, 0 );
    }

    /* Zero the accumulators */
    for ( i = 0; i < io_blocking->n_reg_c; ++i ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXSPLTIB, l_c_reg[i], 0, ( 0x20 & l_c_reg[i] ) >> 5 );
    }

    /* Set batch reduce jump point */
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_br, l_br_loop );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );

    /* Load the A and B pointers for BR-address */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_a, l_a_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, l_a_m_offset, 0, 0 );

      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_b, l_b_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b_n_offset, 0, 0 );
    /* Load the offset and set the A and B pointers for BR-offset */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_a, l_a_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, i_a, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, l_a_m_offset, 0, 0 );

      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_b, l_b_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b_n_offset, 0, 0 );
    }

    /* Call batch-reduce k-loop microkernel */
    libxsmm_generator_br_vsx_microkernel( io_generated_code,
                                          i_xgemm_desc,
                                          io_blocking,
                                          io_reg_tracker,
                                          io_loop_labels,
                                          l_a,
                                          l_b,
                                          l_c,
                                          l_c_reg );

    /* For BR-stride increment the A and B pointers by the constant strinde */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, l_a, l_a, i_xgemm_desc->c1 );
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, l_b, l_b, i_xgemm_desc->c2 );
    }

    /* Decrement, compare, and jump for batch-loop */
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_br_loop, l_br_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_br_loop, io_loop_labels );

    /* Store accumulated C block */
    libxsmm_generator_gemm_vsx_block_store_vsr( io_generated_code,
                                                io_reg_tracker,
                                                l_c_datatype,
                                                l_comptype,
                                                l_c,
                                                io_blocking->block_m,
                                                io_blocking->block_n,
                                                i_xgemm_desc->ldc,
                                                l_c_reg,
                                                io_blocking->reg_ldc );
  }

  /* Increment A m-loop offset and C pointer if required */
  if ( 1 < l_m_iters || ( 0 < l_m_iters && !l_packed ) ) {
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_a_m_offset,
                                     l_a_m_offset,
                                     io_blocking->comp_bytes*io_blocking->block_m );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_c,
                                     l_c,
                                     io_blocking->comp_bytes*io_blocking->block_m );
  }

  /* Decrement, compare, and jump if required for m-loop */
  if ( 1 < l_m_iters ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_m_loop, l_m_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_m_loop, io_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_loop );
  }

  /* Partial M block */
  if ( !l_packed ) {
    unsigned int l_block_m = io_blocking->block_m;
    io_blocking->block_m = ( i_xgemm_desc->m % l_block_m );

    /* For BR-address copy the initial pointer pointer */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) {
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a, l_a_ptr );
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_b, l_b_ptr );
    /* For BR-offset, reset the offsets */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a_offset, l_a_ptr );
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_b_offset, l_b_ptr );
    /* For BR-stride set the initial A and B pointers, including the loop offsets */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, i_a, l_a_m_offset, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, i_b, i_b_n_offset, 0, 0 );
    }

    /* Zero the accumulators */
    for ( i = 0; i < io_blocking->n_reg_c; ++i ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXSPLTIB, l_c_reg[i], 0, ( 0x20 & l_c_reg[i] ) >> 5 );
    }

    /* Set batch reduce jump point */
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_br, l_br_loop );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );

    /* Load the A and B pointers for BR-address */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_a, l_a_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, l_a_m_offset, 0, 0 );

      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_b, l_b_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b_n_offset, 0, 0 );
    /* Load the offset and set the A and B pointers for BR-offset */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_a, l_a_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, i_a, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, l_a_m_offset, 0, 0 );

      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_b, l_b_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b_n_offset, 0, 0 );
    }

    /* Call batch-reduce k-loop microkernel */
    libxsmm_generator_br_vsx_microkernel( io_generated_code,
                                          i_xgemm_desc,
                                          io_blocking,
                                          io_reg_tracker,
                                          io_loop_labels,
                                          l_a,
                                          l_b,
                                          l_c,
                                          l_c_reg );

    /* For BR-stride increment the A and B pointers by the constant strinde */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, l_a, l_a, i_xgemm_desc->c1 );
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, l_b, l_b, i_xgemm_desc->c2 );
    }

    /* Decrement, compare, and jump for batch-loop */
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_br_loop, l_br_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_br_loop, io_loop_labels );

    /* Store accumulated C block */
    libxsmm_generator_gemm_vsx_block_store_vsr( io_generated_code,
                                                io_reg_tracker,
                                                l_c_datatype,
                                                l_comptype,
                                                l_c,
                                                io_blocking->block_m,
                                                io_blocking->block_n,
                                                i_xgemm_desc->ldc,
                                                l_c_reg,
                                                io_blocking->reg_ldc );

    /* Reset m-blocking */
    io_blocking->block_m = l_block_m;
  }

  /* Free C acc registers */
  for ( i = 0; i < io_blocking->n_reg_c ; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_ACC, l_c_reg[i] );
  }

  /* Free batch reduce counter */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_br_loop );

  /* Free the pointer offset counter, if required */
  if ( ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) ||
       ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_ptr );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_ptr );
  }

  /* Free A m-loop offset register, if required */
  if ( 1 < l_m_iters || ( 0 < l_m_iters && !l_packed ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_m_offset );
  }

  /* Free C pointers if required */
  if ( 1 < l_m_iters || !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  }

  /* Free A and B pointers */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b );

}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_vsx_m_loop( libxsmm_generated_code        *io_generated_code,
                                                libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking      *io_blocking,
                                                libxsmm_ppc64le_reg           *io_reg_tracker,
                                                libxsmm_loop_label_tracker    *io_loop_labels,
                                                unsigned int                   i_a,
                                                unsigned int                   i_b,
                                                unsigned int                   i_c ) {
  unsigned int l_a, l_c;
  unsigned int l_m_iters = i_xgemm_desc->m / io_blocking->block_m;
  unsigned int l_m_loop = 0;
  unsigned int l_packed = ( 0 == i_xgemm_desc->m % io_blocking->block_m ) ? 1 : 0;

  /* Create local pointer for a */
  if ( l_m_iters > 1 || !l_packed ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a, l_a );
    l_c = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_c, l_c );
  } else {
    l_a = i_a;
    l_c = i_c;
  }

  /* Set jump point if required */
  if ( l_m_iters > 1 ) {
    l_m_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_m_loop, l_m_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );
  }

  /* Call k-loop */
  if ( l_m_iters > 0 ) {
    libxsmm_generator_vsx_microkernel( io_generated_code,
                                       i_xgemm_desc,
                                       io_blocking,
                                       io_reg_tracker,
                                       io_loop_labels,
                                       l_a,
                                       i_b,
                                       l_c );
  }

  /* Increment a and c pointers if required */
  if ( l_m_iters > 1 || ( l_m_iters > 0 && !l_packed ) ) {
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_a,
                                     l_a,
                                     io_blocking->comp_bytes*io_blocking->block_m );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_c,
                                     l_c,
                                     io_blocking->comp_bytes*io_blocking->block_m );
  }

  /* Compare and jump if required */
  if ( l_m_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_m_loop, l_m_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_m_loop, io_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_loop );
  }

  /* Partial M block */
  if ( !l_packed ) {
    unsigned int l_block_m = io_blocking->block_m;
    io_blocking->block_m = ( i_xgemm_desc->m % l_block_m );

    libxsmm_generator_vsx_microkernel( io_generated_code,
                                       i_xgemm_desc,
                                       io_blocking,
                                       io_reg_tracker,
                                       io_loop_labels,
                                       l_a,
                                       i_b,
                                       l_c );

    io_blocking->block_m = l_block_m;
  }

  /* Free a and c pointers if required */
  if ( l_m_iters > 1 || !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_br_kernel_vsx( libxsmm_generated_code         *io_generated_code,
                                                   libxsmm_gemm_descriptor const  *i_xgemm_desc,
                                                   libxsmm_ppc64le_blocking       *io_blocking,
                                                   libxsmm_ppc64le_reg            *io_reg_tracker ) {
  libxsmm_loop_label_tracker l_loop_labels;
  unsigned int l_a, l_b, l_c;
  unsigned int l_br, l_a_offset = 0, l_b_offset = 0;
  unsigned int l_n_iters, l_n_loop, l_packed;
  unsigned int l_b_n_offset = 0;

  /* loop labels reset */
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, io_reg_tracker );

  /* Unpack the matrix pointers */
  libxsmm_ppc64le_instr_unpack_brargs( io_generated_code, i_xgemm_desc, io_reg_tracker );

  /* GPRs holding pointers to A, B, and C */
  l_a = LIBXSMM_PPC64LE_GPR_ARG0;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  l_b = LIBXSMM_PPC64LE_GPR_ARG1;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b );
  l_c = LIBXSMM_PPC64LE_GPR_ARG2;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );

  /* GPR holding br count */
  l_br = LIBXSMM_PPC64LE_GPR_ARG3;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_br );

  /* If using BR-offset then allocate the A and B offset registers */
  if ( 0 < ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) {
    l_a_offset = LIBXSMM_PPC64LE_GPR_ARG4;
    libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_offset );
    l_b_offset = LIBXSMM_PPC64LE_GPR_ARG5;
    libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_offset );
  }

  /* n-loop values */
  l_n_iters = i_xgemm_desc->n / io_blocking->block_n;
  l_packed = ( 0 == i_xgemm_desc->n % io_blocking->block_n ) ? 1 : 0;

  /* Set jump point for n-loop if required */
  if ( l_n_iters > 1 ) {
    l_n_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_n_loop, l_n_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, &l_loop_labels );
  }

  /* Call packed m-loop */
  if ( l_n_iters > 0 ) {
    libxsmm_generator_gemm_ppc64le_br_vsx_m_loop( io_generated_code,
                                                  i_xgemm_desc,
                                                  io_blocking,
                                                  io_reg_tracker,
                                                  &l_loop_labels,
                                                  l_a,
                                                  l_b,
                                                  l_b_n_offset,
                                                  l_c,
                                                  l_br,
                                                  l_a_offset,
                                                  l_b_offset );
  }

  /* Increment b and c pointers if required */
  if ( ( l_n_iters > 1 ) || ( l_n_iters > 0 && !l_packed ) ) {
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_b_n_offset,
                                     l_b_n_offset,
                                     i_xgemm_desc->ldb*io_blocking->comp_bytes*io_blocking->block_n );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_c,
                                     l_c,
                                     i_xgemm_desc->ldc*io_blocking->comp_bytes*io_blocking->block_n );
  }

  /* Compare and jump if required */
  if ( l_n_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_n_loop, l_n_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_n_loop, &l_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_n_loop );
  }

  /* Partial N block */
  if ( !l_packed ) {
    unsigned int l_block_n = io_blocking->block_n;
    io_blocking->block_n = i_xgemm_desc->n % io_blocking->block_n;
    libxsmm_generator_gemm_ppc64le_br_vsx_m_loop( io_generated_code,
                                                  i_xgemm_desc,
                                                  io_blocking,
                                                  io_reg_tracker,
                                                  &l_loop_labels,
                                                  l_a,
                                                  l_b,
                                                  l_b_n_offset,
                                                  l_c,
                                                  l_br,
                                                  l_a_offset,
                                                  l_b_offset );
    io_blocking->block_n = l_block_n;
  }

  /* Free B offset register if required */
  if ( ( 1 < l_n_iters ) || ( 0 < l_n_iters && !l_packed ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_n_offset );
  }

  /* Free the input registers */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_br );
  if ( 0 < ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_offset );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_offset );
  }

  /* Colapse stack frame */
  libxsmm_ppc64le_instr_colapse_stack( io_generated_code, io_reg_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_vsx( libxsmm_generated_code         *io_generated_code,
                                                libxsmm_gemm_descriptor const  *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking       *io_blocking,
                                                libxsmm_ppc64le_reg            *io_reg_tracker ) {
  libxsmm_loop_label_tracker l_loop_labels;
  unsigned int i_a, i_b, i_c;
  unsigned int l_n_iters, l_n_loop, l_packed;

  /* loop labels reset */
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, io_reg_tracker );

  /* Unpack the matrix pointers */
  libxsmm_ppc64le_instr_unpack_args( io_generated_code, io_reg_tracker );

  /* GPRs holding pointers to A, B, and C */
  i_a = LIBXSMM_PPC64LE_GPR_ARG0;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, i_a );
  i_b = LIBXSMM_PPC64LE_GPR_ARG1;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, i_b );
  i_c = LIBXSMM_PPC64LE_GPR_ARG2;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, i_c );

  /* n loop values */
  l_n_iters = i_xgemm_desc->n / io_blocking->block_n;
  l_packed = ( 0 == i_xgemm_desc->n % io_blocking->block_n ) ? 1 : 0;

  /* Set jump point if required */
  if ( l_n_iters > 1 ) {
    l_n_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_n_loop, l_n_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, &l_loop_labels );
  }

  /* Call packed m-loop */
  if ( l_n_iters > 0 ) {
    libxsmm_generator_gemm_ppc64le_vsx_m_loop( io_generated_code,
                                               i_xgemm_desc,
                                               io_blocking,
                                               io_reg_tracker,
                                               &l_loop_labels,
                                               i_a,
                                               i_b,
                                               i_c );
  }

  /* Increment b and c pointers if required */
  if ( ( l_n_iters > 1 ) || ( l_n_iters > 0 && !l_packed ) ) {
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     i_b,
                                     i_b,
                                     i_xgemm_desc->ldb*io_blocking->comp_bytes*io_blocking->block_n );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     i_c,
                                     i_c,
                                     i_xgemm_desc->ldc*io_blocking->comp_bytes*io_blocking->block_n );
  }

  /* Compare and jump if required */
  if ( l_n_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_n_loop, l_n_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_n_loop, &l_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_n_loop );
  }

  /* Partial N block */
  if ( !l_packed ) {
    unsigned int l_block_n = io_blocking->block_n;
    io_blocking->block_n = i_xgemm_desc->n % io_blocking->block_n;
    libxsmm_generator_gemm_ppc64le_vsx_m_loop( io_generated_code,
                                               i_xgemm_desc,
                                               io_blocking,
                                               io_reg_tracker,
                                               &l_loop_labels,
                                               i_a,
                                               i_b,
                                               i_c );
    io_blocking->block_n = l_block_n;
  }

  /* Colapse stack frame */
  libxsmm_ppc64le_instr_colapse_stack( io_generated_code, io_reg_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_br_mma_m_loop( libxsmm_generated_code         *io_generated_code,
                                                   libxsmm_gemm_descriptor const  *i_xgemm_desc,
                                                   libxsmm_ppc64le_blocking       *io_blocking,
                                                   libxsmm_ppc64le_reg            *io_reg_tracker,
                                                   libxsmm_loop_label_tracker     *io_loop_labels,
                                                   unsigned int                    i_a,
                                                   unsigned int                    i_b,
                                                   unsigned int                    i_b_n_offset,
                                                   unsigned int                    i_c,
                                                   unsigned int                    i_br,
                                                   unsigned int                    i_a_offset,
                                                   unsigned int                    i_b_offset ) {
  libxsmm_datatype l_c_datatype = (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_comptype = (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int l_a, l_b, l_c;
  unsigned int i;
  unsigned int l_a_m_offset = 0, l_a_ptr = 0, l_b_ptr = 0;
  unsigned int l_m_iters = i_xgemm_desc->m / io_blocking->block_m;
  unsigned int l_m_loop = 0, l_br_loop = 0;
  unsigned int l_packed = ( 0 == i_xgemm_desc->m % io_blocking->block_m ) ? 1 : 0;
  unsigned int l_c_acc[LIBXSMM_PPC64LE_ACC_NMAX];

  /* Create registers to store A and B pointers */
  l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  l_b = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  /* Create local pointer for C, if required */
  if ( 1 < l_m_iters || !l_packed ) {
    l_c = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_c, l_c );
  } else {
    l_c = i_c;
  }

  /* If required, allocate an A m-loop offset register and zero */
  if ( 1 < l_m_iters || ( 0 < l_m_iters && !l_packed ) ) {
    l_a_m_offset = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_a_m_offset, 0 );
  }

  /* Get a register for the pointer offset counter, if required */
  if ( ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) ||
       ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) {
    l_a_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_b_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  }

  /* Allocate accumulators for C,
   * This done first due to the limitastion on C-acc registers and for
   * the persistence required for batch-reduce
   */
  for ( i = 0; i < io_blocking->n_acc_c; ++i ) {
    l_c_acc[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_ACC );
  }

  /* Get a register for the batch-reduce loop counter */
  l_br_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  /* Set m-loop jump point if required */
  if ( 1 < l_m_iters ) {
    l_m_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_m_loop, l_m_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );
  }

  /* Main packed m-loop */
  if ( 0 < l_m_iters ) {
    /* For BR-address copy the initial pointer pointer */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) {
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a, l_a_ptr );
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_b, l_b_ptr );
    /* For BR-offset, reset the offsets */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a_offset, l_a_ptr );
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_b_offset, l_b_ptr );
    /* For BR-stride set the initial A and B pointers, including the loop offsets */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, i_a, l_a_m_offset, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, i_b, i_b_n_offset, 0, 0 );
    }

    /* Zero the accumulators */
    for ( i = 0; i < io_blocking->n_acc_c; ++i ) {
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXSETACCZ, l_c_acc[i] );
    }

    /* Set batch reduce jump point */
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_br, l_br_loop );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );

    /* Load the A and B pointers for BR-address */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_a, l_a_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, l_a_m_offset, 0, 0 );

      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_b, l_b_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b_n_offset, 0, 0 );
    /* Load the offset and set the A and B pointers for BR-offset */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_a, l_a_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, i_a, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, l_a_m_offset, 0, 0 );

      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_b, l_b_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b_n_offset, 0, 0 );
    }

    /* Call batch-reduce k-loop microkernel */
    libxsmm_generator_br_mma_microkernel( io_generated_code,
                                          i_xgemm_desc,
                                          io_blocking,
                                          io_reg_tracker,
                                          io_loop_labels,
                                          l_a,
                                          l_b,
                                          l_c,
                                          l_c_acc );

    /* For BR-stride increment the A and B pointers by the constant strinde */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, l_a, l_a, i_xgemm_desc->c1 );
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, l_b, l_b, i_xgemm_desc->c2 );
    }

    /* Decrement, compare, and jump for batch-loop */
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_br_loop, l_br_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_br_loop, io_loop_labels );

    /* Store accumulated C block */
    libxsmm_generator_gemm_mma_block_store_acc( io_generated_code,
                                                io_reg_tracker,
                                                l_c_datatype,
                                                l_comptype,
                                                i_c,
                                                io_blocking->block_m,
                                                io_blocking->block_n,
                                                i_xgemm_desc->ldc,
                                                l_c_acc,
                                                io_blocking->reg_ldc );
  }

  /* Increment A m-loop offset and C pointer if required */
  if ( 1 < l_m_iters || ( 0 < l_m_iters && !l_packed ) ) {
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_a_m_offset,
                                     l_a_m_offset,
                                     io_blocking->comp_bytes*io_blocking->block_m );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_c,
                                     l_c,
                                     io_blocking->comp_bytes*io_blocking->block_m );
  }

  /* Decrement, compare, and jump if required for m-loop */
  if ( 1 < l_m_iters ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_m_loop, l_m_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_m_loop, io_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_loop );
  }

  /* Partial M block */
  if ( !l_packed ) {
    unsigned int l_block_m = io_blocking->block_m;
    io_blocking->block_m = ( i_xgemm_desc->m % l_block_m );

    /* For BR-address copy the initial pointer pointer */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) {
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a, l_a_ptr );
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_b, l_b_ptr );
    /* For BR-offset, reset the offsets */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a_offset, l_a_ptr );
      libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_b_offset, l_b_ptr );
    /* For BR-stride set the initial A and B pointers, including the loop offsets */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, i_a, l_a_m_offset, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, i_b, i_b_n_offset, 0, 0 );
    }

    /* Zero the C accumulators */
    for ( i = 0; i < io_blocking->n_acc_c; ++i ) {
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXSETACCZ, l_c_acc[i] );
    }

    /* Set batch reduce jump point */
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_br, l_br_loop );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );

    /* Load the A and B pointers for BR-address */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_a, l_a_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, l_a_m_offset, 0, 0 );

      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_b, l_b_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b_n_offset, 0, 0 );
    /* Load the offset and set the A and B pointers for BR-offset */
    } else if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_a, l_a_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, i_a, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_a, l_a, l_a_m_offset, 0, 0 );

      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_LDU, l_b, l_b_ptr, 8 >> 2 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b, 0, 0 );
      libxsmm_ppc64le_instr_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_b, l_b, i_b_n_offset, 0, 0 );
    }

    /* Call batch-reduce k-loop microkernel */
    libxsmm_generator_br_mma_microkernel( io_generated_code,
                                          i_xgemm_desc,
                                          io_blocking,
                                          io_reg_tracker,
                                          io_loop_labels,
                                          l_a,
                                          l_b,
                                          l_c,
                                          l_c_acc );

    /* For BR-stride increment the A and B pointers by the constant strinde */
    if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, l_a, l_a, i_xgemm_desc->c1 );
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, l_b, l_b, i_xgemm_desc->c2 );
    }

    /* Decrement, compare, and jump for batch-loop */
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_br_loop, l_br_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_br_loop, io_loop_labels );

    /* Store accumulated C block */
    libxsmm_generator_gemm_mma_block_store_acc( io_generated_code,
                                                io_reg_tracker,
                                                l_c_datatype,
                                                l_comptype,
                                                i_c,
                                                io_blocking->block_m,
                                                io_blocking->block_n,
                                                i_xgemm_desc->ldc,
                                                l_c_acc,
                                                io_blocking->reg_ldc );

    /* Reset the m-blocking */
    io_blocking->block_m = l_block_m;
  }

  /* Free C VSR registers */
  for ( i = 0; i < io_blocking->n_acc_c ; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_c_acc[i] );
  }

  /* Free batch reduce counter */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_br_loop );

  /* Free the pointer offset counter, if required */
  if ( ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) ||
       ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_ptr );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_ptr );
  }

  /* Free A m-loop offset register, if required */
  if ( 1 < l_m_iters || ( 0 < l_m_iters && !l_packed ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_m_offset );
  }

  /* Free C pointers if required */
  if ( 1 < l_m_iters || !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  }

  /* Free A and B pointers */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_mma_m_loop( libxsmm_generated_code         *io_generated_code,
                                                libxsmm_gemm_descriptor const  *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking       *io_blocking,
                                                libxsmm_ppc64le_reg            *io_reg_tracker,
                                                libxsmm_loop_label_tracker     *io_loop_labels,
                                                unsigned int                    i_a,
                                                unsigned int                    i_b,
                                                unsigned int                    i_c ) {
  unsigned int l_a, l_c;
  unsigned int l_m_iters = i_xgemm_desc->m / io_blocking->block_m;
  unsigned int l_m_loop;
  unsigned int l_packed = ( 0 == i_xgemm_desc->m % io_blocking->block_m ) ? 1 : 0;

  /* Create local pointer for a */
  if ( 1 < l_m_iters || !l_packed ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_a, l_a );
    l_c = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_c, l_c );
  } else {
    l_a = i_a;
    l_c = i_c;
  }

  /* Set jump point if required */
  if ( 1 < l_m_iters ) {
    l_m_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_m_loop, l_m_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );
  }

  /* Call k-loop */
  if ( 0 < l_m_iters ) {
    libxsmm_generator_mma_microkernel( io_generated_code,
                                       i_xgemm_desc,
                                       io_blocking,
                                       io_reg_tracker,
                                       io_loop_labels,
                                       l_a,
                                       i_b,
                                       l_c );
  }

  /* Increment a and c pointers if required */
  if ( 1 < l_m_iters || ( 0 < l_m_iters && !l_packed ) ) {
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_a,
                                     l_a,
                                     io_blocking->comp_bytes*io_blocking->block_m );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_c,
                                     l_c,
                                     io_blocking->comp_bytes*io_blocking->block_m );
  }

  /* Compare and jump if required */
  if ( 1 < l_m_iters ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_m_loop, l_m_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_m_loop, io_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_loop );
  }

  /* Partial M block */
  if ( !l_packed ) {
    unsigned int l_block_m = io_blocking->block_m;
    io_blocking->block_m = ( i_xgemm_desc->m % l_block_m );
    libxsmm_generator_mma_microkernel( io_generated_code,
                                       i_xgemm_desc,
                                       io_blocking,
                                       io_reg_tracker,
                                       io_loop_labels,
                                       l_a,
                                       i_b,
                                       l_c );
    io_blocking->block_m = l_block_m;
  }

  /* Free a and c pointers if required */
  if ( 1 < l_m_iters || !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_br_kernel_mma( libxsmm_generated_code        *io_generated_code,
                                                   libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                   libxsmm_ppc64le_blocking      *io_blocking,
                                                   libxsmm_ppc64le_reg           *io_reg_tracker ) {
  libxsmm_loop_label_tracker l_loop_labels;
  unsigned int l_n_iters, l_n_loop, l_packed;
  unsigned int l_a, l_b, l_c;
  unsigned int l_br, l_a_offset = 0, l_b_offset = 0;
  unsigned int l_acc_vec_len = LIBXSMM_PPC64LE_ACC_WIDTH / LIBXSMM_PPC64LE_VSR_WIDTH;
  unsigned int l_b_n_offset = 0;

  /* loop labels reset */
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, io_reg_tracker );

  /* Unpack the matrix pointers */
  libxsmm_ppc64le_instr_unpack_brargs( io_generated_code, i_xgemm_desc, io_reg_tracker );

  /* GPRs holding pointers to A, B, and C */
  l_a = LIBXSMM_PPC64LE_GPR_ARG0;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  l_b = LIBXSMM_PPC64LE_GPR_ARG1;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b );
  l_c = LIBXSMM_PPC64LE_GPR_ARG2;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );

  /* GPR holding br count */
  l_br = LIBXSMM_PPC64LE_GPR_ARG3;
  libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_br );

  /* If using BR-offset then allocate the A and B offset registers */
  if ( 0 < ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) {
    l_a_offset = LIBXSMM_PPC64LE_GPR_ARG4;
    libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_offset );
    l_b_offset = LIBXSMM_PPC64LE_GPR_ARG5;
    libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_offset );
  }

  /* n loop values */
  l_n_iters = i_xgemm_desc->n / io_blocking->block_n;
  l_packed = ( 0 == i_xgemm_desc->n % io_blocking->block_n ) ? 1 : 0;

  /* If offset registers are needed for B and C, then allocate and zero */
  if ( ( 1 < l_n_iters ) || ( 0 < l_n_iters && !l_packed ) ) {
    l_b_n_offset = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_b_n_offset, 0 );
  }

  /* Set the number of acc registers */
  io_blocking->n_acc_c = io_blocking->n_reg_c / l_acc_vec_len;

  /* Set jump point for N-loop if required */
  if ( 1 < l_n_iters ) {
    l_n_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_n_loop, l_n_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, &l_loop_labels );
  }

  /* Call m-loop */
  if ( 0 < l_n_iters ) {
    libxsmm_generator_gemm_ppc64le_br_mma_m_loop( io_generated_code,
                                                  i_xgemm_desc,
                                                  io_blocking,
                                                  io_reg_tracker,
                                                  &l_loop_labels,
                                                  l_a,
                                                  l_b,
                                                  l_b_n_offset,
                                                  l_c,
                                                  l_br,
                                                  l_a_offset,
                                                  l_b_offset );
  }

  /* Increment b and c pointers if required */
  if ( ( 1 < l_n_iters ) || ( 0 < l_n_iters && !l_packed ) ) {
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_b_n_offset,
                                     l_b_n_offset,
                                     i_xgemm_desc->ldb*io_blocking->comp_bytes*io_blocking->block_n );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     l_c,
                                     l_c,
                                     i_xgemm_desc->ldc*io_blocking->comp_bytes*io_blocking->block_n );
  }

  /* Compare and jump if required */
  if ( 1 < l_n_iters ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_n_loop, l_n_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_n_loop, &l_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_n_loop );
  }

  /* Partial N block */
  if ( !l_packed ) {
    unsigned int l_block_n = io_blocking->block_n;
    io_blocking->block_n = i_xgemm_desc->n % io_blocking->block_n;
    libxsmm_generator_gemm_ppc64le_br_mma_m_loop( io_generated_code,
                                                  i_xgemm_desc,
                                                  io_blocking,
                                                  io_reg_tracker,
                                                  &l_loop_labels,
                                                  l_a,
                                                  l_b,
                                                  l_b_n_offset,
                                                  l_c,
                                                  l_br,
                                                  l_a_offset,
                                                  l_b_offset );
    io_blocking->block_n = l_block_n;
  }

  /* Free B offset register if required */
  if ( ( 1 < l_n_iters ) || ( 0 < l_n_iters && !l_packed ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_n_offset );
  }

  /* Free the input registers */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_br );
  if ( 0 < ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_offset );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_offset );
  }

  /* Colapse stack frame */
  libxsmm_ppc64le_instr_colapse_stack( io_generated_code, io_reg_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_mma( libxsmm_generated_code         *io_generated_code,
                                                libxsmm_gemm_descriptor const  *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking       *io_blocking,
                                                libxsmm_ppc64le_reg            *io_reg_tracker ) {
  libxsmm_loop_label_tracker l_loop_labels;
  unsigned int l_n_iters, l_n_loop, l_packed;
  unsigned int i_a, i_b, i_c;
  unsigned int l_acc_vec_len = LIBXSMM_PPC64LE_ACC_WIDTH / LIBXSMM_PPC64LE_VSR_WIDTH;

  /* loop labels reset */
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, io_reg_tracker );

  /* Unpack the matrix pointers */
  libxsmm_ppc64le_instr_unpack_args( io_generated_code, io_reg_tracker );

  /* GPRs holding pointers to A, B, and C */
  i_a = LIBXSMM_PPC64LE_GPR_ARG0;
  i_b = LIBXSMM_PPC64LE_GPR_ARG1;
  i_c = LIBXSMM_PPC64LE_GPR_ARG2;

  /* n loop values */
  l_n_iters = i_xgemm_desc->n / io_blocking->block_n;
  l_packed = ( 0 == i_xgemm_desc->n % io_blocking->block_n ) ? 1 : 0;

  /* Set the number of acc registers */
  io_blocking->n_acc_c = io_blocking->n_reg_c / l_acc_vec_len;

  /* Set jump point for N-loop if required */
  if ( 1 < l_n_iters ) {
    l_n_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_n_loop, l_n_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, &l_loop_labels );
  }

  /* Call m-loop */
  if ( 0 < l_n_iters ) {
    libxsmm_generator_gemm_ppc64le_mma_m_loop( io_generated_code,
                                               i_xgemm_desc,
                                               io_blocking,
                                               io_reg_tracker,
                                               &l_loop_labels,
                                               i_a,
                                               i_b,
                                               i_c );
  }

  /* Increment b and c pointers if required */
  if ( ( 1 < l_n_iters ) || ( 0 < l_n_iters && !l_packed ) ) {
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     i_b,
                                     i_b,
                                     i_xgemm_desc->ldb*io_blocking->comp_bytes*io_blocking->block_n );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     i_c,
                                     i_c,
                                     i_xgemm_desc->ldc*io_blocking->comp_bytes*io_blocking->block_n );
  }

  /* Compare and jump if required */
  if ( 1 < l_n_iters ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_n_loop, l_n_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_n_loop, &l_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_n_loop );
  }

  /* Partial N block */
  if ( !l_packed ) {
    unsigned int l_block_n = io_blocking->block_n;
    io_blocking->block_n = i_xgemm_desc->n % io_blocking->block_n;
    libxsmm_generator_gemm_ppc64le_mma_m_loop( io_generated_code,
                                               i_xgemm_desc,
                                               io_blocking,
                                               io_reg_tracker,
                                               &l_loop_labels,
                                               i_a,
                                               i_b,
                                               i_c );
    io_blocking->block_n = l_block_n;
  }

  /* Colapse stack frame */
  libxsmm_ppc64le_instr_colapse_stack( io_generated_code, io_reg_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel( libxsmm_generated_code        *io_generated_code,
                                            const libxsmm_gemm_descriptor *i_xgemm_desc ) {
  void (*l_generator_kernel)( libxsmm_generated_code *,
                              libxsmm_gemm_descriptor const *,
                              libxsmm_ppc64le_blocking *,
                              libxsmm_ppc64le_reg * );
  char l_br = ( ( 0 < ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) ) ||
                ( 0 < ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) ||
                ( 0 < ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) ) ) ? 1 : 0;
  libxsmm_ppc64le_reg l_reg_tracker;
  libxsmm_ppc64le_blocking l_blocking;

  if ( l_br ) {
    if ( LIBXSMM_PPC64LE_VSX == io_generated_code->arch ) {
      l_generator_kernel = &libxsmm_generator_gemm_ppc64le_br_kernel_vsx;
    } else if ( LIBXSMM_PPC64LE_MMA == io_generated_code->arch ) {
      l_generator_kernel = &libxsmm_generator_gemm_ppc64le_br_kernel_mma;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return;
    }
  } else {
    if ( LIBXSMM_PPC64LE_VSX == io_generated_code->arch ) {
      l_generator_kernel = &libxsmm_generator_gemm_ppc64le_kernel_vsx;
    } else if ( LIBXSMM_PPC64LE_MMA == io_generated_code->arch ) {
      l_generator_kernel = &libxsmm_generator_gemm_ppc64le_kernel_mma;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return;
    }
  }

  /* Transpose not currently supported */
  if ( ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A ) > 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
    return;
  }
  if ( ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B ) > 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
    return;
  }

  /* Initialise reg tracker */
  l_reg_tracker = libxsmm_ppc64le_reg_init();

  /* Initialise blocking */
  libxsmm_generator_gemm_ppc64le_setup_blocking( io_generated_code, i_xgemm_desc, &l_blocking );

  /* Generate kernel */
  l_generator_kernel( io_generated_code, i_xgemm_desc, &l_blocking, &l_reg_tracker );

  return;
}
