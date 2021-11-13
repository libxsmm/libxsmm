/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_spgemm_csr_asparse_reg.h"
#include "generator_x86_instructions.h"
#include "generator_common_x86.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common.h"
#include "generator_gemm_common_aarch64.h"
#include "libxsmm_main.h"

#define LIBXSMM_ASPARSE_REG_MAX_M_BLOCK 4

/* 65k should be enough for anybody */
#define LIBXSMM_ASPARSE_REG_MAX_OPS 65536

typedef struct {
  unsigned long long b_disp;
  unsigned long long c_disps[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  unsigned short n;
  unsigned short src_vals[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  signed char src_sgns[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  unsigned char acc_idxs[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  unsigned char flags[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
} libxsmm_asparse_reg_op;

#define LIBXSMM_ASPARSE_REG_FLAG_FIRST  0x1
#define LIBXSMM_ASPARSE_REG_FLAG_LAST   0x2


LIBXSMM_API_INTERN
void libxsmm_analyse_sparse_nnz( unsigned int   i_n_row_idx,
                                 const double*  i_values,
                                 unsigned int*  o_unique,
                                 double*        o_unique_values,
                                 unsigned int*  o_unique_pos,
                                 int*           o_unique_sgn ) {
  unsigned int l_unique = 1;
  unsigned int l_hit, l_m, l_z;

  o_unique_values[0] = fabs(i_values[0]);
  o_unique_pos[0] = 0;
  o_unique_sgn[0] = (i_values[0] > 0) ? 1 : -1;
  for ( l_m = 1; l_m < i_n_row_idx; l_m++ ) {
    l_hit = 0;
    /* search for the value */
    for ( l_z = 0; l_z < l_unique; l_z++ ) {
      if ( !(o_unique_values[l_z] < fabs(i_values[l_m])) && !(o_unique_values[l_z] > fabs(i_values[l_m])) ) {
        o_unique_pos[l_m] = l_z;
        l_hit = 1;
      }
    }
    /* value was not found */
    if ( !l_hit ) {
      o_unique_values[l_unique] = fabs(i_values[l_m]);
      o_unique_pos[l_m] = l_unique++;
    }
    o_unique_sgn[l_m] = (i_values[l_m] > 0) ? 1 : -1;
  }
  *o_unique = l_unique;
}

LIBXSMM_API_INTERN
void libxsmm_asparse_reg_sequence( unsigned int i_m,
                                   unsigned int i_k,
                                   unsigned int i_m_blocking,
                                   const unsigned int* i_row_idx,
                                   const unsigned int* i_column_idx,
                                   const unsigned int* i_unique_pos,
                                   const int*          i_unique_sgn,
                                   unsigned int i_max_ops,
                                   libxsmm_asparse_reg_op* o_ops,
                                   unsigned int* o_n_ops) {
  unsigned int l_done = 0, l_op_idx = 0;
  unsigned char* l_done_rows = (unsigned char*) calloc(i_m, sizeof(unsigned char));
  unsigned int* l_nnz_mask = (unsigned int*) malloc(i_k*(i_m_blocking + 1)*sizeof(unsigned int));
  unsigned int* l_grp_nnz_count = l_nnz_mask + i_k*i_m_blocking;
  unsigned int l_grp_rows[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];

  /* Zero the operation count in order to signify an error state */
  *o_n_ops = 0;

  /* Check the allocations were successful */
  if ( NULL == l_done_rows || NULL == l_nnz_mask ) {
    goto cleanup;
  }

  /* Process the rows */
  for ( l_done = 0; l_done < i_m; ) {
    unsigned int l_m, l_y, l_z, l_nnz, l_u, l_ngrp = 0;
    unsigned int l_max_nnz = 0, l_arg_max_nnz = ~0;

    /* Reset the arrays */
    memset( l_nnz_mask, ~0, i_k*i_m_blocking*sizeof(unsigned int) );
    memset( l_grp_nnz_count, 0, i_k*sizeof(unsigned int) );
    memset( l_grp_rows, ~0, LIBXSMM_ASPARSE_REG_MAX_M_BLOCK*sizeof(unsigned int) );

    /* Find the pending row with the most non-zeros */
    for ( l_m = 0; l_m < i_m; l_m++ ) {
      l_nnz = i_row_idx[l_m + 1] - i_row_idx[l_m];
      if ( 0 == l_done_rows[l_m] && (~0 == l_arg_max_nnz || l_nnz > l_max_nnz) ) {
        l_max_nnz = l_nnz;
        l_arg_max_nnz = l_m;
      }
    }

    /* Compute the NNZ pattern for the row */
    for ( l_z = i_row_idx[l_arg_max_nnz]; l_z < i_row_idx[l_arg_max_nnz] + l_max_nnz; l_z++) {
      l_nnz_mask[i_column_idx[l_z]] = l_z;
      l_grp_nnz_count[i_column_idx[l_z]]++;
    }

    /* Add the row to the group and mark it as done */
    l_grp_rows[l_ngrp++] = l_arg_max_nnz;
    l_done_rows[l_arg_max_nnz] = 1;
    l_done++;

    /* Construct a group around this row */
    while ( l_done < i_m && l_ngrp < i_m_blocking ) {
      unsigned int l_overlap, l_max_overlap = ~0, l_arg_max_overlap = ~0;

      /* Find the best row to add to the group */
      for ( l_m = 0; l_m < i_m; l_m++ ) {
        if ( 0 == l_done_rows[l_m] ) {
          l_overlap = 0;
          for ( l_z = i_row_idx[l_m]; l_z < i_row_idx[l_m + 1]; l_z++ ) {
            l_overlap += !!l_grp_nnz_count[i_column_idx[l_z]];
          }

          if ( ~0 == l_arg_max_overlap || l_overlap > l_max_overlap ) {
            l_max_overlap = l_overlap;
            l_arg_max_overlap = l_m;
          }
        }
      }

      /* Compute the NNZ pattern for the row */
      for ( l_z = i_row_idx[l_arg_max_overlap]; l_z < i_row_idx[l_arg_max_overlap + 1]; l_z++) {
        l_u = i_column_idx[l_z];
        l_nnz_mask[l_ngrp*i_k + l_u] = l_z;
        l_grp_nnz_count[l_u]++;
      }

      /* Add the row to the group and mark it as done */
      l_grp_rows[l_ngrp++] = l_arg_max_overlap;
      l_done_rows[l_arg_max_overlap] = 1;
      l_done++;
    }

    /* Sequence the dot products for the group */
    for ( l_y = 0; l_y < i_k; l_y++ ) {
      /* See if any rows in the group have a non-zero in column l_y */
      if ( l_grp_nnz_count[l_y] != 0 ) {
        o_ops[l_op_idx].n = l_grp_nnz_count[l_y];
        o_ops[l_op_idx].b_disp = l_y;

        for ( l_z = 0, l_u = 0; l_z < l_ngrp; l_z++ ) {
          if ( l_nnz_mask[l_z*i_k + l_y] != ~0 ) {
            o_ops[l_op_idx].flags[l_u] = 0;

            /* Note if this is the first non-zero for the row */
            if ( l_y == i_column_idx[i_row_idx[l_grp_rows[l_z]]] ) {
              o_ops[l_op_idx].flags[l_u] |= LIBXSMM_ASPARSE_REG_FLAG_FIRST;
            }

            /* Note if this is the first non-zero for the row */
            if ( l_y == i_column_idx[i_row_idx[l_grp_rows[l_z] + 1] - 1] ) {
              o_ops[l_op_idx].flags[l_u] |= LIBXSMM_ASPARSE_REG_FLAG_LAST;
            }

            o_ops[l_op_idx].c_disps[l_u] = l_grp_rows[l_z];
            o_ops[l_op_idx].src_vals[l_u] = i_unique_pos[l_nnz_mask[l_z*i_k + l_y]];
            o_ops[l_op_idx].src_sgns[l_u] = i_unique_sgn[l_nnz_mask[l_z*i_k + l_y]];
            o_ops[l_op_idx].acc_idxs[l_u] = l_z;
            l_u++;
          }
        }

        if ( ++l_op_idx == i_max_ops ) {
          goto cleanup;
        }
      }
    }
  }

  /* Save the number of sequenced ops */
  *o_n_ops = l_op_idx;

cleanup:
  free( l_nnz_mask );
  free( l_done_rows );
}

LIBXSMM_API_INTERN
int libxsmm_asparse_reg_pick_bcast_reg( const unsigned int* i_vals,
                                        unsigned int i_nvals,
                                        const libxsmm_asparse_reg_op* i_ops,
                                        unsigned int i_nops ) {
  int l_nuse = 0, l_arg_nuse = 0, l_x, l_y, l_z, l_hit;

  /* See when the value in reg l_x is next used */
  for ( l_x = 0; l_x < i_nvals; l_x++ ) {
    for ( l_y = 0, l_hit = 0; l_y < i_nops && !l_hit; l_y++ ) {
      for ( l_z = 0; l_z < i_ops[l_y].n && !l_hit; l_z++ ) {
        if ( i_ops[l_y].src_vals[l_z] == i_vals[l_x] ) {
          if ( l_y > l_nuse ) {
            l_nuse = l_y;
            l_arg_nuse = l_x;
          }

          l_hit = 1;
        }
      }
    }

    /* Value is never used again; we're done */
    if ( !l_hit ) {
      return l_x;
    }
  }

  return l_arg_nuse;
}

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csr_asparse_reg_x86( libxsmm_generated_code*         io_generated_code,
                                                   const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                   const unsigned int*             i_row_idx,
                                                   const unsigned int*             i_column_idx,
                                                   const double*                   i_values ) {
  unsigned int l_n, l_z;
  unsigned int l_unique, l_uoff, l_poff;
  unsigned int l_m_blocking, l_n_blocking;
  unsigned int l_n_row_idx = i_row_idx[i_xgemm_desc->m];

  double *const l_unique_values = (double*)(0 != l_n_row_idx ? malloc(sizeof(double) * l_n_row_idx) : NULL);
  unsigned int *const l_unique_pos = (unsigned int*)(0 != l_n_row_idx ? malloc(sizeof(unsigned int) * l_n_row_idx) : NULL);
  int *const l_unique_sgn = (int*)(0 != l_n_row_idx ? malloc(sizeof(int) * l_n_row_idx) : NULL);

  const unsigned int l_perm_consts[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

  unsigned int l_need_bcast_reg = 0;
  unsigned int l_bcast_reg_vals[31], l_base_bcast_reg = ~0, l_nbcast_regs, l_cur_bcast_reg;

  const unsigned int l_fp64 = LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype );
  const unsigned int l_fbytes = (l_fp64) ? 8 : 4;

  const unsigned int l_broadcast_insn = (l_fp64) ? LIBXSMM_X86_INSTR_VBROADCASTSD : LIBXSMM_X86_INSTR_VBROADCASTSS;

  const unsigned int l_c_is_nt =  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT & i_xgemm_desc->flags;
  const unsigned int l_beta0 = LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags;

  unsigned int l_num_reg, l_used_reg = 0, l_values_per_reg;
  unsigned int l_breg_unique, l_preg_unique;
  unsigned int l_base_c_reg, l_ld_reg = 0, l_base_perm_reg, l_vbytes;

  libxsmm_asparse_reg_op *l_ops = (libxsmm_asparse_reg_op*) malloc(sizeof(libxsmm_asparse_reg_op)*LIBXSMM_ASPARSE_REG_MAX_OPS);
  unsigned int l_n_ops, l_op_idx;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_const_data_tracker l_const_data_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  const unsigned int l_fma_tbl[2][2][2] = {
    /* [single][+a*b][+c, -c] */
    { { LIBXSMM_X86_INSTR_VFMADD231PS, LIBXSMM_X86_INSTR_VFMSUB231PS },
    /* [single][-a*b][+c, -c] */
      { LIBXSMM_X86_INSTR_VFNMADD231PS, LIBXSMM_X86_INSTR_VFNMSUB231PS } },
    /* [double][+a*b][+c, -c] */
    { { LIBXSMM_X86_INSTR_VFMADD231PD, LIBXSMM_X86_INSTR_VFMSUB231PD },
    /* [double][-a*b][+c, -c] */
      { LIBXSMM_X86_INSTR_VFNMADD231PD, LIBXSMM_X86_INSTR_VFNMSUB231PD } },
  };

  /* Tracks which accumulators are negated due to use of VMULP[SD] */
  unsigned int l_acc_neg_tbl[4][LIBXSMM_ASPARSE_REG_MAX_M_BLOCK] = {};

  /* Check if mallocs were successful */
  if ( 0 == l_unique_values || 0 == l_unique_pos || 0 == l_unique_sgn || 0 == l_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_ALLOC_DATA );
    goto cleanup;
  }

  /* Check that the arch is supported */
  if ( io_generated_code->arch < LIBXSMM_X86_AVX2 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    goto cleanup;
  }

  memset( l_bcast_reg_vals, ~0, sizeof(l_bcast_reg_vals) );
  libxsmm_reset_const_data_tracker(&l_const_data_tracker);

  /* Define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc, 0 );

  /* Inner chunk size */
  if ( i_xgemm_desc->n == l_micro_kernel_config.vector_length ) {
    l_n_blocking = 1;
  } else if ( i_xgemm_desc->n == 2*l_micro_kernel_config.vector_length ) {
    l_n_blocking = 2;
  } else if ( i_xgemm_desc->n == 4*l_micro_kernel_config.vector_length ) {
    l_n_blocking = 4;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    goto cleanup;
  }

  /* Init config */
  if ( io_generated_code->arch == LIBXSMM_X86_AVX2 ) {
    l_num_reg = 16;

    l_preg_unique = 0;
    l_base_perm_reg = l_nbcast_regs = (unsigned int)-1;
  } else {
    l_num_reg = 32;

    if ( l_fp64 ) {
      l_preg_unique = (32 - l_n_blocking - 1 - 8)*8;
    } else {
      l_preg_unique = (32 - l_n_blocking - 1 - 16)*16;
    }
  }

  l_breg_unique = l_num_reg - l_n_blocking;
  l_values_per_reg = l_micro_kernel_config.vector_length;
  l_vbytes = l_values_per_reg*l_fbytes;

  /* prerequisite */
  assert(0 != i_values);

  /* Let's figure out how many unique values we have */
  libxsmm_analyse_sparse_nnz( l_n_row_idx, i_values, &l_unique,
                              l_unique_values, l_unique_pos, l_unique_sgn );

  /* Check that there are not too many unique values */
  if ( l_unique > 1280 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNIQUE_VAL );
    goto cleanup;
  }

  /* If needed cast from double to float */
  if ( !l_fp64 ) {
    for ( l_z = 0; l_z < l_unique; l_z++ ) {
      float l_fval = (float) l_unique_values[l_z];
      memcpy( ((float*) l_unique_values) + l_z, &l_fval, sizeof(l_fval) );
    }
  }

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
#endif
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;

  /* Define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* Open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );

  /* Copy the unique values into the data segment with 64-byte alignment */
  l_uoff = libxsmm_x86_instruction_add_data( io_generated_code,
                                             (unsigned char*) l_unique_values,
                                             l_unique*l_fbytes, 64, 1,
                                             &l_const_data_tracker );

  /* Load the base address of the data + 0x300 to reduce imm sizes */
  libxsmm_x86_instruction_lea_data( io_generated_code, LIBXSMM_X86_GP_REG_R9,
                                    l_uoff + 0x300, &l_const_data_tracker );

  /* Try to store A entirely in broadcasted registers */
  if ( l_unique <= l_breg_unique ) {
    /* Broadcast the unique values into registers */
    for ( l_z = 0; l_z < l_unique; l_z++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_micro_kernel_config.instruction_set,
                                        l_broadcast_insn,
                                        LIBXSMM_X86_GP_REG_R9,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_z*l_fbytes - 0x300,
                                        l_micro_kernel_config.vector_name,
                                        l_z, 0, 0, 0 );
    }

    /* Update the register count */
    l_used_reg += l_unique;
  /* Else, see if we can store A entirely in packed registers */
  } else if ( l_unique <= l_preg_unique ) {
    /* Load the packed unique values into registers */
    for ( l_z = 0; l_z < l_unique; l_z += l_values_per_reg ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_micro_kernel_config.instruction_set,
                                        l_micro_kernel_config.c_vmove_instruction,
                                        LIBXSMM_X86_GP_REG_R9,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_z*l_fbytes - 0x300,
                                        l_micro_kernel_config.vector_name,
                                        l_z / l_values_per_reg, 0, 0, 0 );
    }

    /* Update the register count */
    l_used_reg += LIBXSMM_UPDIV( l_unique, l_values_per_reg );
    l_base_perm_reg = l_used_reg;

    /* Copy the permutation constants into the data segment */
    l_poff = libxsmm_x86_instruction_add_data( io_generated_code,
                                               (unsigned char*) l_perm_consts,
                                               sizeof(l_perm_consts), 8, 1,
                                               &l_const_data_tracker );

    /* Broadcast permute constants into registers */
    for ( l_z = 0; l_z < l_values_per_reg; l_z++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_micro_kernel_config.instruction_set,
                                        l_broadcast_insn,
                                        LIBXSMM_X86_GP_REG_R9,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_poff - l_uoff + l_z*l_fbytes - 0x300,
                                        l_micro_kernel_config.vector_name,
                                        l_base_perm_reg + l_z, 0, 0, 0 );
    }

    /* Update the register count */
    l_used_reg += l_values_per_reg;

    /* Mark that we also need at least one broadcast register */
    l_need_bcast_reg = 1;
  /* Otherwise, load the entries of A from memory at runtime */
  } else {
    /* Mark that we also need at least one broadcast register */
    l_need_bcast_reg = 1;
  }

  /* See if we have registers spare for m blocking */
  if ( l_used_reg + 1 + l_need_bcast_reg + 4*l_n_blocking <= l_num_reg ) {
    l_m_blocking = 4;
    l_ld_reg = l_used_reg++;
  } else if ( l_used_reg + 1 + l_need_bcast_reg + 2*l_n_blocking <= l_num_reg ) {
    l_m_blocking = 2;
    l_ld_reg = l_used_reg++;
  } else {
    LIBXSMM_ASSERT( l_used_reg + l_need_bcast_reg + l_n_blocking <= l_num_reg );
    l_m_blocking = 1;
  }

  /* Update the register count */
  l_base_c_reg = l_used_reg;
  l_used_reg += l_m_blocking*l_n_blocking;

  /* If we're broadcasting then allocate some broadcast registers */
  if ( l_need_bcast_reg ) {
    l_base_bcast_reg = l_used_reg;
    l_nbcast_regs = l_num_reg - l_base_bcast_reg;
  }

  /* Sequence the operations */
  libxsmm_asparse_reg_sequence( i_xgemm_desc->m, i_xgemm_desc->k, l_m_blocking,
                                i_row_idx, i_column_idx, l_unique_pos, l_unique_sgn,
                                LIBXSMM_ASPARSE_REG_MAX_OPS, l_ops, &l_n_ops );

  /* Ensure it worked */
  if ( 0 == l_n_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    goto cleanup;
  }

  /* Start the n loop */
  libxsmm_generator_generic_loop_header( io_generated_code, &l_loop_label_tracker,
                                         l_gp_reg_mapping.gp_reg_nloop, 0,
                                         l_values_per_reg*l_n_blocking );

  for ( l_op_idx = 0; l_op_idx < l_n_ops; l_op_idx++ ) {
    libxsmm_asparse_reg_op op = l_ops[l_op_idx];
    unsigned int l_rvb = l_ld_reg;
    unsigned long long l_b_disp = op.b_disp*i_xgemm_desc->ldb*l_fbytes;

    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      for ( l_z = 0; l_z < op.n; l_z++ ) {
        unsigned int l_fma_insn, l_mov_insn;
        unsigned int l_acc_idx = op.acc_idxs[l_z];
        unsigned int l_u = op.src_vals[l_z], l_v;
        unsigned int l_uneg = op.src_sgns[l_z] == -1;
        unsigned int l_rva = (l_unique > l_breg_unique) ? ~0 : l_u;
        unsigned int l_rvc = l_base_c_reg + l_n_blocking*l_acc_idx;
        unsigned long long l_c_disp = op.c_disps[l_z]*i_xgemm_desc->ldc*l_fbytes;

        /* Look up our FMA instruction */
        l_fma_insn = l_fma_tbl[l_fp64][l_uneg][l_acc_neg_tbl[l_n][l_acc_idx]];
        l_acc_neg_tbl[l_n][l_acc_idx] = 0;

        /* See if we need to load/zero the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_FIRST & op.flags[l_z] ) {
          /* Zero */
          if ( l_beta0 ) {
            /* Transform (c = 0; c += a*b) => c = a*b */
            if ( 0 == l_uneg ) {
              l_fma_insn = (l_fp64) ? LIBXSMM_X86_INSTR_VMULPD : LIBXSMM_X86_INSTR_VMULPS;
            /* Transform (c = 0; c -= a*b) => c = a*b with deferred negation */
            } else if ( 1 == l_uneg && 0 == (LIBXSMM_ASPARSE_REG_FLAG_LAST & op.flags[l_z]) ) {
              l_fma_insn = (l_fp64) ? LIBXSMM_X86_INSTR_VMULPD : LIBXSMM_X86_INSTR_VMULPS;

              /* Ensure the next operation on this accumulator negates it */
              l_acc_neg_tbl[l_n][l_acc_idx] = 1;
            /* Issue c = 0 */
            } else {
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                       l_micro_kernel_config.instruction_set,
                                                       l_micro_kernel_config.vxor_instruction,
                                                       l_micro_kernel_config.vector_name,
                                                       l_rvc + l_n, l_rvc + l_n, l_rvc + l_n );
            }

            /* As we'll be writing to C later, consider pre-fetching into cache */
            if ( 0 == l_c_is_nt ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                                                LIBXSMM_X86_INSTR_PREFETCHW,
                                                l_gp_reg_mapping.gp_reg_c,
                                                LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                l_c_disp + l_n*l_vbytes );
            }
          /* Load */
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              l_micro_kernel_config.instruction_set,
                                              l_micro_kernel_config.c_vmove_instruction,
                                              l_gp_reg_mapping.gp_reg_c,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              l_c_disp + l_n*l_vbytes,
                                              l_micro_kernel_config.vector_name,
                                              l_rvc + l_n, 0, 1, 0 );
          }
        }

        /* If necessary, broadcast a unique value */
        if ( l_unique > l_breg_unique ) {
          /* See if we already have it broadcasted */
          for ( l_v = 0; l_v < l_nbcast_regs; l_v++ ) {
            if ( l_bcast_reg_vals[l_v] == l_u ) {
              l_rva = l_base_bcast_reg + l_v;
              break;
            }
          }

          /* Otherwise pick a register to broadcast into */
          if ( ~0 == l_rva ) {
            l_cur_bcast_reg = libxsmm_asparse_reg_pick_bcast_reg( l_bcast_reg_vals, l_nbcast_regs,
                                                                  l_ops + l_op_idx + 1,
                                                                  l_n_ops - l_op_idx - 1 );
            l_rva = l_base_bcast_reg + l_cur_bcast_reg;

            /* Broadcast from memory */
            if ( l_unique > l_preg_unique ) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                                                l_micro_kernel_config.instruction_set,
                                                l_broadcast_insn,
                                                LIBXSMM_X86_GP_REG_R9,
                                                LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                l_u*l_fbytes - 0x300,
                                                l_micro_kernel_config.vector_name,
                                                l_rva, 0, 0, 0 );
            /* Broadcast from a packed register */
            } else {
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                       l_micro_kernel_config.instruction_set,
                                                       LIBXSMM_X86_INSTR_VPERMD,
                                                       l_micro_kernel_config.vector_name,
                                                       l_u / l_values_per_reg,
                                                       l_base_perm_reg + l_u % l_values_per_reg,
                                                       l_rva );
            }

            /* Update our records */
            l_bcast_reg_vals[l_cur_bcast_reg] = l_u;
          }
        }

        /* If B will be reused, load to a register */
        if ( op.n > 1 ) {
          /* Load on first use */
          if ( 0 == l_z ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              l_micro_kernel_config.instruction_set,
                                              l_micro_kernel_config.c_vmove_instruction,
                                              l_gp_reg_mapping.gp_reg_b,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              l_b_disp + l_n*l_vbytes,
                                              l_micro_kernel_config.vector_name,
                                              l_rvb, 0, 1, 0 );
          }

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   l_micro_kernel_config.instruction_set,
                                                   l_fma_insn,
                                                   l_micro_kernel_config.vector_name,
                                                   l_rva, l_rvb, l_rvc + l_n );
        /* Otherwise load as part of the FMA */
        } else {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                   l_micro_kernel_config.instruction_set,
                                                   l_fma_insn, 0,
                                                   l_gp_reg_mapping.gp_reg_b,
                                                   LIBXSMM_X86_GP_REG_UNDEF,
                                                   0, l_b_disp + l_n*l_vbytes,
                                                   l_micro_kernel_config.vector_name,
                                                   l_rva, l_rvc + l_n );
        }

        /* See if we need to save the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_LAST & op.flags[l_z] ) {
          /* Handle non-temporal stores */
          if ( 0 != l_c_is_nt ) {
            l_mov_insn = (l_fp64) ? LIBXSMM_X86_INSTR_VMOVNTPD : LIBXSMM_X86_INSTR_VMOVNTPS;
          } else {
            l_mov_insn = l_micro_kernel_config.c_vmove_instruction;
          }

          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            l_micro_kernel_config.instruction_set,
                                            l_mov_insn,
                                            l_gp_reg_mapping.gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_c_disp + l_n*l_vbytes,
                                            l_micro_kernel_config.vector_name,
                                            l_rvc + l_n, 0, 0, 1 );
        }
      }
    }
  }

  /* Advance B and C */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                   l_gp_reg_mapping.gp_reg_b, l_vbytes*l_n_blocking );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                   l_gp_reg_mapping.gp_reg_c, l_vbytes*l_n_blocking );

  /* Test the loop condition */
  libxsmm_generator_generic_loop_footer( io_generated_code, &l_loop_label_tracker,
                                         l_gp_reg_mapping.gp_reg_nloop, i_xgemm_desc->c1 );

  /* Close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );
  libxsmm_x86_instruction_close_data( io_generated_code, &l_const_data_tracker );

cleanup:
  free( l_unique_values );
  free( l_unique_pos );
  free( l_unique_sgn );
  free( l_ops );
}

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csr_asparse_reg_aarch64_neon( libxsmm_generated_code*         io_generated_code,
                                                            const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                            const unsigned int*             i_row_idx,
                                                            const unsigned int*             i_column_idx,
                                                            const double*                   i_values ) {
  unsigned int l_n, l_z;
  unsigned int l_unique, l_uoff;
  unsigned int l_m_blocking, l_n_blocking;
  unsigned int l_n_row_idx = i_row_idx[i_xgemm_desc->m];
  double *const l_unique_values = (double*)(0 != l_n_row_idx ? malloc(sizeof(double) * l_n_row_idx) : NULL);
  unsigned int *const l_unique_pos = (unsigned int*)(0 != l_n_row_idx ? malloc(sizeof(unsigned int) * l_n_row_idx) : NULL);
  int *const l_unique_sgn = (int*)(0 != l_n_row_idx ? malloc(sizeof(int) * l_n_row_idx) : NULL);
  const unsigned int l_fp64 = LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype );
  unsigned int l_reg_unique, l_base_c_reg, l_base_c_gp_reg, l_base_ld_reg;

  unsigned int l_bcast_reg_vals[120], l_nbcast_vals = 0;

  const libxsmm_aarch64_asimd_tupletype l_tuplet = (l_fp64) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;
  const libxsmm_aarch64_asimd_width l_width = (l_fp64) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S;
  const unsigned int l_values_per_reg = (l_fp64) ? 2 : 4;
  const unsigned int l_fbytes = (l_fp64) ? 8 : 4;

  const unsigned int l_c_is_nt =  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT & i_xgemm_desc->flags;
  const unsigned int l_beta0 = LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags;

  libxsmm_asparse_reg_op *l_ops = (libxsmm_asparse_reg_op*) malloc(sizeof(libxsmm_asparse_reg_op)*LIBXSMM_ASPARSE_REG_MAX_OPS);
  unsigned int l_n_ops, l_op_idx;

  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_const_data_tracker l_const_data_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* Check if mallocs were successful */
  if ( 0 == l_unique_values || 0 == l_unique_pos || 0 == l_unique_sgn || 0 == l_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_ALLOC_DATA );
    goto cleanup;
  }

  /* Inner chunk size */
  if ( i_xgemm_desc->n == l_values_per_reg ) {
    l_n_blocking = 1;
  } else if ( i_xgemm_desc->n == 2*l_values_per_reg ) {
    l_n_blocking = 2;
  } else if ( i_xgemm_desc->n == 4*l_values_per_reg ) {
    l_n_blocking = 4;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    goto cleanup;
  }

  /* Init config */
  l_reg_unique = l_values_per_reg*(32 - l_n_blocking - ((l_n_blocking > 1) ? 2 : 1));

  /* prerequisite */
  LIBXSMM_ASSERT(0 != i_values);

  /* Let's figure out how many unique values we have */
  libxsmm_analyse_sparse_nnz( l_n_row_idx, i_values, &l_unique,
                              l_unique_values, l_unique_pos, l_unique_sgn );

  /* Check that there are not too many unique values */
  if ( l_unique > 1280 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNIQUE_VAL );
    goto cleanup;
  }

  /* If needed cast from double to float */
  if ( !l_fp64 ) {
    for ( l_z = 0; l_z < l_unique; l_z++ ) {
      float l_fval = (float) l_unique_values[l_z];
      memcpy( ((float*) l_unique_values) + l_z, &l_fval, sizeof(l_fval) );
    }
  }

  /* A stored entirely in registers */
  if ( l_unique <= l_reg_unique ) {
    l_base_ld_reg = LIBXSMM_UPDIV( l_unique, l_values_per_reg );
    l_base_c_reg = l_base_ld_reg + ((l_n_blocking > 1) ? 2 : 1);
    l_base_c_gp_reg = LIBXSMM_AARCH64_GP_REG_X12;

    /* See if we have registers spare for m blocking */
    if ( l_base_c_reg + 4*l_n_blocking <= 32 ) {
      l_m_blocking = 4;
    } else if ( l_base_c_reg + 2*l_n_blocking <= 32 ) {
      l_m_blocking = 2;
    } else {
      l_m_blocking = 1;
    }
  /* A loaded in from memory */
  } else {
    l_m_blocking = 4;
    l_base_c_gp_reg = LIBXSMM_AARCH64_GP_REG_X12;
    l_base_c_reg = 32 - l_m_blocking*l_n_blocking;
    l_base_ld_reg = l_base_c_reg - ((l_n_blocking > 1) ? 2 : 1);
    l_nbcast_vals = l_base_ld_reg*l_values_per_reg;
  }

  /* Sequence the operations */
  libxsmm_asparse_reg_sequence( i_xgemm_desc->m, i_xgemm_desc->k, l_m_blocking,
                                i_row_idx, i_column_idx, l_unique_pos, l_unique_sgn,
                                LIBXSMM_ASPARSE_REG_MAX_OPS, l_ops, &l_n_ops );

  /* Ensure it worked */
  if ( 0 == l_n_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    goto cleanup;
  }

  /* Define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X24;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X25;

  memset( l_bcast_reg_vals, ~0, sizeof(l_bcast_reg_vals) );
  libxsmm_reset_const_data_tracker( &l_const_data_tracker );

  /* Define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* Open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xfff );

  /* Pre-load A into registers */
  if ( l_unique <= l_reg_unique ) {
    /* Copy the unique values into the data segment with 16-byte alignment */
    l_uoff = libxsmm_aarch64_instruction_add_data( io_generated_code,
                                                   (unsigned char*) l_unique_values,
                                                   l_unique*l_fbytes, 16, 1,
                                                   &l_const_data_tracker );

    /* Pad the segment to be a multiple of 16 */
    if ( (l_unique*l_fbytes) % 16 != 0 ) {
      unsigned char l_pad[15] = {};
      libxsmm_aarch64_instruction_add_data( io_generated_code, l_pad,
                                            (l_unique*l_fbytes) % 16, 1, 1,
                                            &l_const_data_tracker );
    }

    libxsmm_aarch64_instruction_adr_data( io_generated_code,
                                          l_gp_reg_mapping.gp_reg_help_0,
                                          l_uoff, &l_const_data_tracker );

    /* Perform the loads */
    if ( (l_base_ld_reg % 2) != 0 ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                              l_gp_reg_mapping.gp_reg_help_0, 0, 0,
                                              0, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    }

    for ( l_z = l_base_ld_reg % 2; l_z < l_base_ld_reg; l_z += 2 ) {
      libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF,
                                                   l_gp_reg_mapping.gp_reg_help_0, 16*l_z,
                                                   l_z, l_z + 1, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    }
  }

  /* Start the n loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                         LIBXSMM_AARCH64_GP_REG_X23, i_xgemm_desc->c1 );

  /* Reset the data segment pointer */
  if ( l_unique > l_reg_unique ) {
    libxsmm_aarch64_instruction_adr_data( io_generated_code,
                                          LIBXSMM_AARCH64_GP_REG_X26,
                                          0, &l_const_data_tracker );
  }

  for ( l_op_idx = 0; l_op_idx < l_n_ops; l_op_idx++ ) {
    libxsmm_asparse_reg_op op = l_ops[l_op_idx];
    unsigned int l_rvb = l_base_ld_reg;
    unsigned long long l_b_disp = op.b_disp*i_xgemm_desc->ldb*l_fbytes;

    /* Load the address of B */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_0,
                                                   l_gp_reg_mapping.gp_reg_help_1, l_b_disp );

    if ( 1 == l_n_blocking ) {
      /* Load B itself */
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                              l_gp_reg_mapping.gp_reg_help_1, 0, 0,
                                              l_rvb, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

      for ( l_z = 0; l_z < op.n; l_z++ ) {
        unsigned int l_u = op.src_vals[l_z], l_v;
        unsigned int l_rg = l_base_c_gp_reg + op.acc_idxs[l_z];
        unsigned int l_rva = (l_unique > l_reg_unique) ? ~0 : l_u;
        unsigned int l_rvc = l_base_c_reg + op.acc_idxs[l_z];
        unsigned long long l_c_disp = op.c_disps[l_z]*i_xgemm_desc->ldc*l_fbytes;
        unsigned int l_fma_insn = (op.src_sgns[l_z] == 1) ? LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V : LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_E_V;

        /* See if we need to load/zero the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_FIRST & op.flags[l_z] ) {
          /* Save offset into a GPR for later reuse */
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                         l_rg, l_c_disp );

          /* Zero (elide if constant is +ve) */
          if ( l_beta0 ) {
            if ( 1 == op.src_sgns[l_z] ) {
              l_fma_insn = LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_E_V;
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                         l_rvc, l_rvc, 0, l_rvc,
                                                         LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            }
          /* Load */
          } else {
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                                    l_rg, 0, 0, l_rvc, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
          }
        }

        /* If necessary, load a unique value from memory */
        if ( l_unique > l_reg_unique ) {
          /* See if we already have it in a register */
          for ( l_v = 0; l_v < l_nbcast_vals; l_v++ ) {
            if ( l_bcast_reg_vals[l_v] == l_u ) {
              l_rva = l_v;
              break;
            }
          }

          /* Otherwise pick a register and lane to load it into */
          if ( ~0 == l_rva ) {
            l_rva = libxsmm_asparse_reg_pick_bcast_reg( l_bcast_reg_vals, l_nbcast_vals,
                                                        l_ops + l_op_idx + 1, l_n_ops - l_op_idx - 1 );

            /* Add the unique value to the data segment */
            libxsmm_aarch64_instruction_add_data( io_generated_code,
                                                  ((unsigned char*) l_unique_values) + l_fbytes*l_u,
                                                  l_fbytes, l_fbytes, 1,
                                                  &l_const_data_tracker );

            /* Load */
            libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_ASIMD_LD1_I_POST,
                                                           LIBXSMM_AARCH64_GP_REG_X26, 0, l_fbytes,
                                                           l_rva / l_values_per_reg,
                                                           l_rva % l_values_per_reg,
                                                           l_width );

            /* Update our records */
            l_bcast_reg_vals[l_rva] = l_u;
          }
        }

        /* Perform the computation */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_fma_insn,
                                                   l_rvb,
                                                   l_rva / l_values_per_reg,
                                                   l_rva % l_values_per_reg,
                                                   l_rvc, l_tuplet );

        /* See if we need to save the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_LAST & op.flags[l_z] ) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                                  l_rg, 0, 0, l_rvc, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        }
      }
    } else {
      for ( l_n = 0; l_n < l_n_blocking; l_n += 2 ) {
        /* Load B itself */
        libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF,
                                                     l_gp_reg_mapping.gp_reg_help_1, 16*l_n,
                                                     l_rvb, l_rvb + 1, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

        for ( l_z = 0; l_z < op.n; l_z++ ) {
          unsigned int l_u = op.src_vals[l_z], l_v;
          unsigned int l_rg = l_base_c_gp_reg + op.acc_idxs[l_z];
          unsigned int l_rva = (l_unique > l_reg_unique) ? ~0 : l_u;
          unsigned int l_rvc = l_base_c_reg + l_n_blocking*op.acc_idxs[l_z];
          unsigned long long l_c_disp = op.c_disps[l_z]*i_xgemm_desc->ldc*l_fbytes;
          unsigned int l_fma_insn = (op.src_sgns[l_z] == 1) ? LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V : LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_E_V;
          unsigned int l_stp_insn = (l_c_is_nt) ? LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF : LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;

          /* See if we need to load/zero the accumulator */
          if ( LIBXSMM_ASPARSE_REG_FLAG_FIRST & op.flags[l_z] ) {
            /* Save offset into a GPR for later reuse */
            if ( 0 == l_n ) {
              libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                             l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                             l_rg, l_c_disp );
            }

            /* Zero (elide if constant is +ve) */
            if ( l_beta0 ) {
              if ( 1 == op.src_sgns[l_z] ) {
                l_fma_insn = LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_E_V;
              } else {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                            l_rvc + l_n, l_rvc + l_n, 0, l_rvc + l_n,
                                                            LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                            l_rvc + l_n + 1, l_rvc + l_n + 1, 0, l_rvc + l_n + 1,
                                                            LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
            /* Load paired */
            } else {
              libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF,
                                                           l_rg, 16*l_n, l_rvc + l_n, l_rvc + l_n + 1,
                                                           LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
            }
          }

          /* If necessary, load a unique value from memory */
          if ( l_unique > l_reg_unique ) {
            /* See if we already have it in a register */
            for ( l_v = 0; l_v < l_nbcast_vals; l_v++ ) {
              if ( l_bcast_reg_vals[l_v] == l_u ) {
                l_rva = l_v;
                break;
              }
            }

            /* Otherwise pick a register and lane to load it into */
            if ( ~0 == l_rva ) {
              l_rva = libxsmm_asparse_reg_pick_bcast_reg( l_bcast_reg_vals, l_nbcast_vals,
                                                          l_ops + l_op_idx + 1, l_n_ops - l_op_idx - 1 );

              /* Add the unique value to the data segment */
              libxsmm_aarch64_instruction_add_data( io_generated_code,
                                                    ((unsigned char*) l_unique_values) + l_fbytes*l_u,
                                                    l_fbytes, l_fbytes, 1,
                                                    &l_const_data_tracker );

              /* Load */
              libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code,
                                                             LIBXSMM_AARCH64_INSTR_ASIMD_LD1_I_POST,
                                                             LIBXSMM_AARCH64_GP_REG_X26, 0, l_fbytes,
                                                             l_rva / l_values_per_reg,
                                                             l_rva % l_values_per_reg,
                                                             l_width );

              /* Update our records */
              l_bcast_reg_vals[l_rva] = l_u;
            }
          }

          /* Perform the computation */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_fma_insn,
                                                     l_rvb,
                                                     l_rva / l_values_per_reg,
                                                     l_rva % l_values_per_reg,
                                                     l_rvc + l_n, l_tuplet );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_fma_insn,
                                                     l_rvb + 1,
                                                     l_rva / l_values_per_reg,
                                                     l_rva % l_values_per_reg,
                                                     l_rvc + l_n + 1, l_tuplet );

          /* See if we need to save the accumulator */
          if ( LIBXSMM_ASPARSE_REG_FLAG_LAST & op.flags[l_z] ) {
            libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, l_stp_insn,
                                                         l_rg, 16*l_n, l_rvc + l_n, l_rvc + l_n + 1,
                                                         LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
          }
        }
      }
    }
  }

  /* Advance B and C */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_b,
                                                 16*l_n_blocking, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_c,
                                                 16*l_n_blocking, 0 );

  /* Test the loop condition */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                         LIBXSMM_AARCH64_GP_REG_X23,
                                         l_values_per_reg*l_n_blocking );

  /* Close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xfff );
  libxsmm_aarch64_instruction_close_data( io_generated_code, &l_const_data_tracker );

cleanup:
  free( l_unique_values );
  free( l_unique_pos );
  free( l_unique_sgn );
  free( l_ops );
}

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csr_asparse_reg_aarch64_sve( libxsmm_generated_code*         io_generated_code,
                                                           const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                           const unsigned int*             i_row_idx,
                                                           const unsigned int*             i_column_idx,
                                                           const double*                   i_values ) {
  unsigned int l_n, l_z;
  unsigned int l_unique, l_uoff;
  unsigned int l_m_blocking, l_n_blocking;
  unsigned int l_n_row_idx = i_row_idx[i_xgemm_desc->m];
  double *const l_unique_values = (double*)(0 != l_n_row_idx ? malloc(sizeof(double) * l_n_row_idx) : NULL);
  unsigned int *const l_unique_pos = (unsigned int*)(0 != l_n_row_idx ? malloc(sizeof(unsigned int) * l_n_row_idx) : NULL);
  int *const l_unique_sgn = (int*)(0 != l_n_row_idx ? malloc(sizeof(int) * l_n_row_idx) : NULL);
  unsigned int l_reg_unique, l_base_c_reg, l_base_c_gp_reg, l_ld_reg, l_used_reg = 0;

  unsigned int l_bcast_reg_vals[30], l_nbcast_vals = 0;

  const unsigned int l_fp64 = LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype );
  const unsigned int l_fbytes = (l_fp64) ? 8 : 4;
  unsigned int l_vlen, l_vbytes;
  unsigned int l_npacked_reg, l_npacked_values_per_reg;

  const unsigned int l_beta0 = LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags;

  libxsmm_aarch64_sve_type l_svet;

  libxsmm_asparse_reg_op *l_ops = (libxsmm_asparse_reg_op*) malloc(sizeof(libxsmm_asparse_reg_op)*LIBXSMM_ASPARSE_REG_MAX_OPS);
  unsigned int l_n_ops, l_op_idx;

  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_const_data_tracker l_const_data_tracker;
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* check if mallocs were successful */
  if ( 0 == l_unique_values || 0 == l_unique_pos || 0 == l_unique_sgn || 0 == l_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_ALLOC_DATA );
    goto cleanup;
  }

  /* Define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );
  l_vlen = l_micro_kernel_config.vector_length;
  l_vbytes = l_fbytes*l_vlen;

  /* Inner chunk size */
  if ( i_xgemm_desc->n == l_vlen ) {
    l_n_blocking = 1;
  } else if ( i_xgemm_desc->n == 2*l_vlen ) {
    l_n_blocking = 2;
  } else if ( i_xgemm_desc->n == 4*l_vlen ) {
    l_n_blocking = 4;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    goto cleanup;
  }

  /* Init config */
  if ( l_fp64 ) {
    l_npacked_reg = 16;
    l_npacked_values_per_reg = 2;
    l_svet = LIBXSMM_AARCH64_SVE_TYPE_D;
  } else {
    l_npacked_reg = 8;
    l_npacked_values_per_reg = 4;
    l_svet = LIBXSMM_AARCH64_SVE_TYPE_S;
  }

  l_reg_unique = l_npacked_reg*l_npacked_values_per_reg + (32 - l_n_blocking - 1 - l_npacked_reg);

  /* prerequisite */
  LIBXSMM_ASSERT(0 != i_values);

  /* Let's figure out how many unique values we have */
  libxsmm_analyse_sparse_nnz( l_n_row_idx, i_values, &l_unique,
                              l_unique_values, l_unique_pos, l_unique_sgn );

  /* Check that there are not too many unique values */
  if ( l_unique > 1280 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNIQUE_VAL );
    goto cleanup;
  }

  /* If needed cast from double to float */
  if ( !l_fp64 ) {
    for ( l_z = 0; l_z < l_unique; l_z++ ) {
      float l_fval = (float) l_unique_values[l_z];
      memcpy( ((float*) l_unique_values) + l_z, &l_fval, sizeof(l_fval) );
    }
  }

  /* Define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X24;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X25;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X26;

  memset( l_bcast_reg_vals, ~0, sizeof(l_bcast_reg_vals) );
  libxsmm_reset_const_data_tracker( &l_const_data_tracker );

  /* Define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* Open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xfff );

  libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, LIBXSMM_AARCH64_SVE_REG_P0,
                                                -1, l_gp_reg_mapping.gp_reg_help_0 );

  /* Copy the unique values into the data segment with 16-byte alignment */
  l_uoff = libxsmm_aarch64_instruction_add_data( io_generated_code,
                                                 (unsigned char*) l_unique_values,
                                                 l_unique*l_fbytes, 16, 1,
                                                 &l_const_data_tracker );

  libxsmm_aarch64_instruction_adr_data( io_generated_code,
                                        l_gp_reg_mapping.gp_reg_help_2,
                                        l_uoff, &l_const_data_tracker );

  /* A stored entirely in registers */
  if ( l_unique <= l_reg_unique ) {
    /* Pad the segment to be a multiple of 16 */
    if ( (l_unique*l_fbytes) % 16 != 0 ) {
      unsigned char l_pad[15] = {};
      libxsmm_aarch64_instruction_add_data( io_generated_code, l_pad,
                                            (l_unique*l_fbytes) % 16, 1, 1,
                                            &l_const_data_tracker );
    }

    /* Pre-load A into registers */
    l_n = 0;
    while ( 1 ) {
      unsigned int l_inc, l_ld_insn;

      /* Packed register; load with 128-bit replication */
      if ( l_used_reg < l_npacked_reg ) {
        l_inc = l_npacked_values_per_reg;
        l_ld_insn = LIBXSMM_AARCH64_INSTR_SVE_LD1RQD_I_OFF;
      /* Broadcast register; load with 32-/64-bit replication */
      } else {
        l_inc = 1;
        l_ld_insn = (l_fp64) ? LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF;
      }

      libxsmm_aarch64_instruction_sve_move( io_generated_code, l_ld_insn,
                                            l_gp_reg_mapping.gp_reg_help_2, 0, 0,
                                            l_used_reg++, LIBXSMM_AARCH64_SVE_REG_P0 );
      l_n += l_inc;

      if ( l_n < l_unique ) {
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       l_gp_reg_mapping.gp_reg_help_2,
                                                       l_gp_reg_mapping.gp_reg_help_2,
                                                       l_inc*l_fbytes, 0 );
      } else {
        break;
      }
    }

    l_ld_reg = l_used_reg++;
    l_base_c_reg = l_used_reg;
    l_base_c_gp_reg = LIBXSMM_AARCH64_GP_REG_X12;

    /* See if we have registers spare for m blocking */
    if ( l_base_c_reg + 4*l_n_blocking <= 32 ) {
      l_m_blocking = 4;
    } else if ( l_base_c_reg + 2*l_n_blocking <= 32 ) {
      l_m_blocking = 2;
    } else {
      l_m_blocking = 1;
    }
  /* A loaded in from memory */
  } else {
    l_m_blocking = 4;
    l_base_c_gp_reg = LIBXSMM_AARCH64_GP_REG_X12;
    l_base_c_reg = 32 - l_m_blocking*l_n_blocking;
    l_ld_reg = l_base_c_reg - 1;
    l_nbcast_vals = l_ld_reg;
    l_npacked_reg = 0;
  }

  /* Sequence the operations */
  libxsmm_asparse_reg_sequence( i_xgemm_desc->m, i_xgemm_desc->k, l_m_blocking,
                                i_row_idx, i_column_idx, l_unique_pos, l_unique_sgn,
                                LIBXSMM_ASPARSE_REG_MAX_OPS, l_ops, &l_n_ops );

  /* Ensure it worked */
  if ( 0 == l_n_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    goto cleanup;
  }

  /* Start the n loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                         LIBXSMM_AARCH64_GP_REG_X23, i_xgemm_desc->c1 );

  for ( l_op_idx = 0; l_op_idx < l_n_ops; l_op_idx++ ) {
    libxsmm_asparse_reg_op op = l_ops[l_op_idx];
    unsigned int l_rvb = l_ld_reg;
    unsigned long long l_b_disp = op.b_disp*i_xgemm_desc->ldb*l_fbytes;

    /* Load B */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_0,
                                                   l_gp_reg_mapping.gp_reg_help_1, l_b_disp );

    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      /* Load B itself */
      libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                            l_gp_reg_mapping.gp_reg_help_1, 0, l_n,
                                            l_rvb, LIBXSMM_AARCH64_SVE_REG_P0 );

      for ( l_z = 0; l_z < op.n; l_z++ ) {
        unsigned int l_u = op.src_vals[l_z], l_v;
        unsigned int l_rva, l_idx;
        unsigned int l_rg = l_base_c_gp_reg + op.acc_idxs[l_z];
        unsigned int l_rvc = l_base_c_reg + l_n_blocking*op.acc_idxs[l_z];
        unsigned long long l_c_disp = op.c_disps[l_z]*i_xgemm_desc->ldc*l_fbytes;
        unsigned int l_fma_insn, l_ld_insn;

        /* Constant is packed in its register */
        if ( l_u < l_npacked_reg*l_npacked_values_per_reg ) {
          l_rva = l_u / l_npacked_values_per_reg;
          l_idx = l_u % l_npacked_values_per_reg;
          l_fma_insn = (op.src_sgns[l_z] == 1) ? LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_I : LIBXSMM_AARCH64_INSTR_SVE_FMLS_V_I;
        /* Constant is broadcasted in its register or will be loaded */
        } else {
          l_rva = l_u - l_npacked_reg*(l_npacked_values_per_reg - 1);
          l_idx = 0;
          l_fma_insn = (op.src_sgns[l_z] == 1) ? LIBXSMM_AARCH64_INSTR_SVE_FMLA_V : LIBXSMM_AARCH64_INSTR_SVE_FMLS_V;
        }

        /* If necessary, broadcast a unique value from memory */
        if ( l_unique > l_reg_unique ) {
          /* See if we already have it in a register */
          for ( l_v = 0, l_rva = ~0; l_v < l_nbcast_vals; l_v++ ) {
            if ( l_bcast_reg_vals[l_v] == l_u ) {
              l_rva = l_v;
              break;
            }
          }

          /* Otherwise pick a register to broadcast into */
          if ( ~0 == l_rva ) {
            l_rva = libxsmm_asparse_reg_pick_bcast_reg( l_bcast_reg_vals, l_nbcast_vals,
                                                        l_ops + l_op_idx + 1, l_n_ops - l_op_idx - 1 );

            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                           l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_0,
                                                           l_gp_reg_mapping.gp_reg_help_1, l_u*l_fbytes );

            /* Load */
            l_ld_insn = (l_fp64) ? LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF;
            libxsmm_aarch64_instruction_sve_move( io_generated_code, l_ld_insn,
                                                  l_gp_reg_mapping.gp_reg_help_1, 0, 0,
                                                  l_rva, LIBXSMM_AARCH64_SVE_REG_P0 );

            /* Update our records */
            l_bcast_reg_vals[l_rva] = l_u;
          }
        }

        /* See if we need to load/zero the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_FIRST & op.flags[l_z] ) {
          /* Save offset into a GPR for later reuse */
          if ( 0 == l_n ) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                          l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                          l_rg, l_c_disp );
          }

          /* Zero (elide if constant is +ve) */
          if ( l_beta0 ) {
            if ( 1 == op.src_sgns[l_z] && l_rva < l_npacked_reg ) {
              l_fma_insn = LIBXSMM_AARCH64_INSTR_SVE_FMUL_V_I;
            } else if ( 1 == op.src_sgns[l_z] ) {
              l_fma_insn = LIBXSMM_AARCH64_INSTR_SVE_FMUL_V;
            } else {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                       l_rvc + l_n, l_rvc + l_n, 0, l_rvc + l_n,
                                                       LIBXSMM_AARCH64_GP_REG_UNDEF, l_svet );
            }
          /* Load */
          } else {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                                  l_rg, 0, l_n,
                                                  l_rvc + l_n, LIBXSMM_AARCH64_SVE_REG_P0 );
          }
        }

        /* Perform the computation */
        libxsmm_aarch64_instruction_sve_compute ( io_generated_code, l_fma_insn,
                                                  l_rvb, l_rva, l_idx, l_rvc + l_n,
                                                  LIBXSMM_AARCH64_SVE_REG_P0, l_svet );

        /* See if we need to save the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_LAST & op.flags[l_z] ) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                                l_rg, 0, l_n, l_rvc + l_n,
                                                LIBXSMM_AARCH64_SVE_REG_P0 );
        }
      }
    }
  }

  /* Advance B and C */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_b,
                                                 l_vbytes*l_n_blocking, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_c,
                                                 l_vbytes*l_n_blocking, 0 );

  /* Test the loop condition */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                         LIBXSMM_AARCH64_GP_REG_X23,
                                         l_vlen*l_n_blocking );

  /* Close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xfff );
  libxsmm_aarch64_instruction_close_data( io_generated_code, &l_const_data_tracker );

cleanup:
  free( l_unique_values );
  free( l_unique_pos );
  free( l_unique_sgn );
  free( l_ops );
}
