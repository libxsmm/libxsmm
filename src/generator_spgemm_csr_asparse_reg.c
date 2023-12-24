/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.), Freddie Witherden
******************************************************************************/
#include "generator_spgemm_csr_asparse_reg.h"
#include "generator_x86_instructions.h"
#include "generator_common_x86.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common.h"
#include "generator_gemm_common_aarch64.h"

#define LIBXSMM_ASPARSE_REG_MAX_M_BLOCK 12

/* 65k should be enough for anybody */
#define LIBXSMM_ASPARSE_REG_MAX_OPS 65536

#define LIBXSMM_ASPARSE_REG_FLAG_FIRST  0x1
#define LIBXSMM_ASPARSE_REG_FLAG_LAST   0x2


typedef struct {
  unsigned int b_disp;
  unsigned int c_disps[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  unsigned short n;
  unsigned short src_vals[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  signed char src_sgns[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  unsigned char acc_idxs[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  unsigned char flags[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
} libxsmm_asparse_reg_op;


LIBXSMM_API_INTERN
void libxsmm_analyse_sparse_nnz( unsigned int   i_n_row_idx,
                                 const double*  i_values,
                                 unsigned int*  o_unique,
                                 double*        o_unique_values,
                                 unsigned int*  o_unique_pos,
                                 int*           o_unique_sgn );

LIBXSMM_API_INTERN
void libxsmm_asparse_reg_sequence( unsigned int i_m,
                                   unsigned int i_m_blocking,
                                   const unsigned int* i_row_idx,
                                   const unsigned int* i_column_idx,
                                   const unsigned int* i_unique_pos,
                                   const int*          i_unique_sgn,
                                   unsigned int i_max_ops,
                                   libxsmm_asparse_reg_op* o_ops,
                                   unsigned int* o_n_ops);

LIBXSMM_API_INTERN
unsigned int libxsmm_asparse_reg_pick_bcast_reg( const unsigned int* i_vals,
                                                 unsigned int i_nvals,
                                                 const libxsmm_asparse_reg_op* i_ops,
                                                 unsigned int i_nops );

LIBXSMM_API_INTERN
void libxsmm_analyse_sparse_nnz( unsigned int   i_n_row_idx,
                                 const double*  i_values,
                                 unsigned int*  o_unique,
                                 double*        o_unique_values,
                                 unsigned int*  o_unique_pos,
                                 int*           o_unique_sgn ) {
  unsigned int l_unique = 1;
  unsigned int l_hit, l_m, l_z;

  o_unique_values[0] = LIBXSMM_FABS(i_values[0]);
  o_unique_pos[0] = 0;
  o_unique_sgn[0] = (i_values[0] > 0) ? 1 : -1;
  for ( l_m = 1; l_m < i_n_row_idx; l_m++ ) {
    l_hit = 0;
    /* search for the value */
    for ( l_z = 0; l_z < l_unique; l_z++ ) {
      if ( !(o_unique_values[l_z] < LIBXSMM_FABS(i_values[l_m])) && !(o_unique_values[l_z] > LIBXSMM_FABS(i_values[l_m])) ) {
        o_unique_pos[l_m] = l_z;
        l_hit = 1;
      }
    }
    /* value was not found */
    if ( !l_hit ) {
      o_unique_values[l_unique] = LIBXSMM_FABS(i_values[l_m]);
      o_unique_pos[l_m] = l_unique++;
    }
    o_unique_sgn[l_m] = (i_values[l_m] > 0) ? 1 : -1;
  }
  *o_unique = l_unique;
}

LIBXSMM_API_INTERN
void libxsmm_asparse_reg_sequence( unsigned int i_m,
                                   unsigned int i_m_blocking,
                                   const unsigned int* i_row_idx,
                                   const unsigned int* i_column_idx,
                                   const unsigned int* i_unique_pos,
                                   const int*          i_unique_sgn,
                                   unsigned int i_max_ops,
                                   libxsmm_asparse_reg_op* o_ops,
                                   unsigned int* o_n_ops) {
  unsigned int l_done = 0, l_op_idx = 0, l_r;
  unsigned int l_row_offs[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  unsigned int l_acc_idxs[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  unsigned int l_grp_rows[LIBXSMM_ASPARSE_REG_MAX_M_BLOCK][LIBXSMM_ASPARSE_REG_MAX_M_BLOCK];
  unsigned char* l_done_rows = (unsigned char*) calloc(i_m, sizeof(unsigned char));

  /* Zero the operation count in order to signify an error state */
  *o_n_ops = 0;

  /* Check the allocations were successful */
  if ( NULL == l_done_rows ) {
    goto cleanup;
  }

  /* Mark all-zero rows as done */
  for ( l_r = 0; l_r < i_m; l_r++) {
    if ( i_row_idx[l_r + 1] == i_row_idx[l_r] ) {
      l_done_rows[l_r] = 1;
      l_done++;
    }
  }

  /* Process the rows */
  while ( l_done < i_m ) {
    int l_r_nnz = 0, l_avg_nnz = 0;
    unsigned int l_msz, l_mtot;

    /* Reset the arrays */
    memset( l_row_offs, 0, sizeof(l_row_offs) );
    memset( l_acc_idxs, ~0, sizeof(l_acc_idxs) );
    memset( l_grp_rows, ~0, sizeof(l_grp_rows) );

    /* Construct a bundle of row groups */
    for ( l_msz = 0, l_mtot = 0; l_done < i_m && l_mtot < i_m_blocking; l_msz++ ) {
      unsigned int l_m, l_z, l_ngrp = 0;

      /* Pick the best pending row */
      for ( l_m = 0, l_r = ~0U; l_m < i_m; l_m++ ) {
        if ( 0 == l_done_rows[l_m] ) {
          int l_m_nnz = i_row_idx[l_m + 1] - i_row_idx[l_m];

          /* No rows in the bundle, so go with this one */
          if ( 0 == l_msz ) {
            l_r = l_m;
            l_r_nnz = l_m_nnz;
            break;
          /* Some rows in the bundle, so try and match the nnz count */
          } else if ( ~0U == l_r || LIBXSMM_DELTA( l_avg_nnz, l_m_nnz ) < LIBXSMM_DELTA( l_avg_nnz, l_r_nnz ) ) {
            l_r = l_m;
            l_r_nnz = l_m_nnz;
          }
        }
      }

      /* Start a new row group in our bundle and mark the row as done */
      l_grp_rows[l_msz][0] = l_r;
      l_done_rows[l_r] = 1;
      l_acc_idxs[l_msz] = l_mtot;
      l_ngrp++; l_mtot++; l_done++;

      /* Update the average NNZ count of our bundle */
      l_avg_nnz = ((l_mtot - 1)*l_avg_nnz + l_r_nnz) / l_mtot;

      /* Add in any rows which share this rows non-zero pattern */
      for ( l_m = 0; l_m < i_m && l_done < i_m && l_mtot < i_m_blocking; l_m++ ) {
        if ( 0 == l_done_rows[l_m] ) {
          unsigned int l_m_nnz = i_row_idx[l_m + 1] - i_row_idx[l_m];

          /* First check NNZ */
          if ( l_m_nnz != i_row_idx[l_r + 1] - i_row_idx[l_r] ) {
            continue;
          }

          /* If this matches see if the values themselves check out */
          for ( l_z = 0; l_z < l_m_nnz; l_z++ ) {
            if ( i_column_idx[i_row_idx[l_r] + l_z] != i_column_idx[i_row_idx[l_m] + l_z] ) {
              break;
            }
          }

          /* Add the row to the group and mark it as done */
          if ( l_z == l_m_nnz ) {
            l_grp_rows[l_msz][l_ngrp] = l_m;
            l_done_rows[l_m] = 1;
            l_ngrp++; l_mtot++; l_done++;

            /* Update the average NNZ count of our bundle */
            l_avg_nnz = ((l_mtot - 1)*l_avg_nnz + l_r_nnz) / l_mtot;
          }
        }
      }
    }

    /* Sequence the dot products for all row groups in the bundle */
    while ( 1 ) {
      unsigned int l_y, l_z, l_issued = 0;

      /* Iterate over each row group in the bundle */
      for ( l_y = 0; l_y < l_msz; l_y++ ) {
        unsigned int l_g_row = l_grp_rows[l_y][0];
        unsigned int l_g_off = i_row_idx[l_g_row] + l_row_offs[l_y];

        /* See if the row group still has operations to be issued */
        if ( l_g_off < i_row_idx[l_g_row + 1] ) {

          /* Iterate through each row in the group */
          for ( l_z = 0; l_z < i_m_blocking && ~0U != l_grp_rows[l_y][l_z]; l_z++ ) {
            unsigned int l_row = l_grp_rows[l_y][l_z];
            unsigned int l_off = i_row_idx[l_row] + l_row_offs[l_y];

            /* Zero the flags */
            o_ops[l_op_idx].flags[l_z] = 0;

            /* Note if this is the first non-zero */
            if ( l_off == i_row_idx[l_row] ) {
              o_ops[l_op_idx].flags[l_z] |= LIBXSMM_ASPARSE_REG_FLAG_FIRST;
            }

            /* Note if this is the last non-zero */
            if ( l_off == i_row_idx[l_row + 1] - 1 ) {
              o_ops[l_op_idx].flags[l_z] |= LIBXSMM_ASPARSE_REG_FLAG_LAST;
            }

            o_ops[l_op_idx].c_disps[l_z] = l_row;
            o_ops[l_op_idx].acc_idxs[l_z] = (unsigned char)(l_acc_idxs[l_y] + l_z);
            o_ops[l_op_idx].src_vals[l_z] = (unsigned short)i_unique_pos[l_off];
            o_ops[l_op_idx].src_sgns[l_z] = (char)i_unique_sgn[l_off];
          }

          o_ops[l_op_idx].n = (unsigned short)l_z;
          o_ops[l_op_idx].b_disp = i_column_idx[l_g_off];

          if ( ++l_op_idx == i_max_ops ) {
            goto cleanup;
          }

          /* March the row pointer forwards for this group */
          l_row_offs[l_y]++;

          /* Note that we issued an operation */
          l_issued = 1;
        }
      }

      /* If no row groups in the bundle issued we're done */
      if ( 0 == l_issued ) {
        break;
      }
    }
  }

  /* Save the number of sequenced ops */
  *o_n_ops = l_op_idx;

cleanup:
  free( l_done_rows );
}

LIBXSMM_API_INTERN
unsigned int libxsmm_asparse_reg_pick_bcast_reg( const unsigned int* i_vals,
                                                 unsigned int i_nvals,
                                                 const libxsmm_asparse_reg_op* i_ops,
                                                 unsigned int i_nops ) {
  unsigned int l_nuse = 0, l_arg_nuse = 0, l_x, l_y, l_z, l_hit;

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

  unsigned int l_perm_consts[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

  unsigned int l_need_bcast_reg = 0;
  unsigned int l_bcast_reg_vals[31], l_base_bcast_reg = ~0U, l_nbcast_regs = 0, l_cur_bcast_reg = 0;

  const unsigned int l_fp64 = LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype );
  const unsigned int l_fbytes = (l_fp64) ? 8 : 4;

  const unsigned int l_movu_insn = (l_fp64) ? LIBXSMM_X86_INSTR_VMOVUPD : LIBXSMM_X86_INSTR_VMOVUPS;
  const unsigned int l_broadcast_insn = (l_fp64) ? LIBXSMM_X86_INSTR_VBROADCASTSD : LIBXSMM_X86_INSTR_VBROADCASTSS;

  const unsigned int l_beta0 = LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags;

  unsigned int l_num_reg, l_used_reg = 0, l_values_per_reg, l_packed_values_per_reg = 1;
  unsigned int l_breg_unique, l_preg_unique;
  unsigned int l_base_c_reg = 0, l_ld_reg = 0, l_base_perm_reg = 0, l_vbytes;

  libxsmm_asparse_reg_op *l_ops = (libxsmm_asparse_reg_op*) malloc(sizeof(libxsmm_asparse_reg_op)*LIBXSMM_ASPARSE_REG_MAX_OPS);
  unsigned int l_n_ops, l_op_idx, l_mov_insn;

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
  unsigned int l_acc_neg_tbl[4][LIBXSMM_ASPARSE_REG_MAX_M_BLOCK] = { { 0 }, { 0 } };

  /* Check if mallocs were successful */
  if ( 0 == l_unique_values || 0 == l_unique_pos || 0 == l_unique_sgn || 0 == l_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_ALLOC_DATA );
    goto cleanup;
  }

  /* Check that the arch is supported */
  if ( io_generated_code->arch < LIBXSMM_X86_AVX2 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    goto cleanup;
  }

  memset( l_bcast_reg_vals, ~0, sizeof(l_bcast_reg_vals) );
  libxsmm_reset_const_data_tracker(&l_const_data_tracker);

  /* Define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc, 0 );

  /* Decide about NTS-hint (leading dimension is already considered in micro-kernel config) */
  l_mov_insn = (LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT == (LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT & i_xgemm_desc->flags)
    ? l_micro_kernel_config.c_vmove_nts_instruction : l_micro_kernel_config.c_vmove_instruction);

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
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX2) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX) ) {
    l_num_reg = 16;

    l_preg_unique = 0;
    l_base_perm_reg = l_nbcast_regs = (unsigned int)-1;
  } else {
    l_num_reg = 32;

    if ( l_fp64 ) {
      l_preg_unique = (32 - l_n_blocking - 1 - 7)*8;
    } else {
      l_preg_unique = (32 - l_n_blocking - 1 - 15)*16;
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
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
#endif
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;

  /* Define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* Open asm */
  libxsmm_x86_instruction_open_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
    /* RDI holds the pointer to the struct, so lets first move this one into R15 */
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1 );
    /* A pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 32, l_gp_reg_mapping.gp_reg_a, 0 );
    /* B pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, l_gp_reg_mapping.gp_reg_b, 0 );
    /* C pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 96, l_gp_reg_mapping.gp_reg_c, 0 );
#if 0
    if ( i_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 56, l_gp_reg_mapping.gp_reg_a_prefetch, 0 );
      /* B prefetch pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 88, l_gp_reg_mapping.gp_reg_b_prefetch, 0 );
    }
#endif
  } else {
#if 0
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
    return;
#endif
  }

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
    /* Determine the optimal number of values to pack into each register */
    for ( l_z = 2, l_n = l_unique; l_z <= l_values_per_reg; l_z++ ) {
      if ( LIBXSMM_UPDIV( l_unique, l_z ) + l_z - 1 < l_n ) {
        l_n = LIBXSMM_UPDIV( l_unique, l_z ) + l_z - 1;
        l_packed_values_per_reg = l_z;
      }
    }

    /* Load the packed unique values into registers */
    for ( l_z = 0; l_z < l_unique; l_z += l_packed_values_per_reg ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_micro_kernel_config.instruction_set,
                                        l_movu_insn,
                                        LIBXSMM_X86_GP_REG_R9,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_z*l_fbytes - 0x300,
                                        l_micro_kernel_config.vector_name,
                                        l_z / l_packed_values_per_reg, 0, 0, 0 );
    }

    /* Update the register count */
    l_used_reg += LIBXSMM_UPDIV( l_unique, l_packed_values_per_reg );
    l_base_perm_reg = l_used_reg;

    /* Copy the permutation constants into the data segment */
    l_poff = libxsmm_x86_instruction_add_data( io_generated_code,
                                               (unsigned char*) (l_perm_consts + l_fp64),
                                               sizeof(l_perm_consts), 8, 1,
                                               &l_const_data_tracker );

    /* Broadcast permute constants into registers */
    for ( l_z = 0; l_z < l_packed_values_per_reg - 1; l_z++ ) {
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
    l_used_reg += l_packed_values_per_reg - 1;

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
  libxsmm_asparse_reg_sequence( i_xgemm_desc->m, l_m_blocking, i_row_idx,
                                i_column_idx, l_unique_pos, l_unique_sgn,
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
    unsigned int l_b_disp = op.b_disp*i_xgemm_desc->ldb*l_fbytes;

    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      for ( l_z = 0; l_z < op.n; l_z++ ) {
        unsigned int l_acc_idx = op.acc_idxs[l_z];
        unsigned int l_u = op.src_vals[l_z], l_v;
        unsigned int l_uneg = op.src_sgns[l_z] == -1;
        unsigned int l_rva = (l_unique > l_breg_unique) ? ~0U : l_u;
        unsigned int l_rvc = l_base_c_reg + l_n_blocking*l_acc_idx;
        unsigned int l_c_disp = op.c_disps[l_z]*i_xgemm_desc->ldc*l_fbytes;

        /* Look up our FMA instruction */
        unsigned int l_fma_insn = l_fma_tbl[l_fp64][l_uneg][l_acc_neg_tbl[l_n][l_acc_idx]];
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
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                        l_micro_kernel_config.vxor_instruction,
                                                        l_micro_kernel_config.vector_name,
                                                        l_rvc + l_n, l_rvc + l_n, l_rvc + l_n );
            }

            /* As we will be writing to C later, consider pre-fetching into cache */
            if ( l_mov_insn == l_micro_kernel_config.c_vmove_instruction ) {
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
          if ( ~0U == l_rva ) {
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
            /* Broadcast from lane zero of a packed register */
            } else if ( 0 == l_u % l_packed_values_per_reg ) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                        l_broadcast_insn,
                                                        l_micro_kernel_config.vector_name,
                                                        l_u / l_packed_values_per_reg,
                                                        l_rva );
            /* Broadcast from a packed register */
            } else {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                        LIBXSMM_X86_INSTR_VPERMD,
                                                        l_micro_kernel_config.vector_name,
                                                        l_u / l_packed_values_per_reg,
                                                        l_base_perm_reg + l_u % l_packed_values_per_reg - 1,
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

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    l_fma_insn,
                                                    l_micro_kernel_config.vector_name,
                                                    l_rva, l_rvb, l_rvc + l_n );
        /* Otherwise load as part of the FMA */
        } else {
          libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                        l_fma_insn,
                                                        l_micro_kernel_config.vector_name,
                                                        l_gp_reg_mapping.gp_reg_b,
                                                        LIBXSMM_X86_GP_REG_UNDEF,
                                                        0, l_b_disp + l_n*l_vbytes, 0,
                                                        l_rva, l_rvc + l_n );
        }

        /* See if we need to save the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_LAST & op.flags[l_z] ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            l_micro_kernel_config.instruction_set,
                                            l_mov_insn, /* Handle non-temporal stores */
                                            l_gp_reg_mapping.gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_c_disp + l_n*l_vbytes,
                                            l_micro_kernel_config.vector_name,
                                            l_rvc + l_n, 0, 0, 1 );
        }
      }
    }
  }

  /* In the case of beta = 0 handle all-zero rows */
  if ( l_beta0 ) {
    unsigned int l_zeroed = 0;

    for ( l_z = 0; l_z < i_xgemm_desc->m; l_z++ ) {
      if ( i_row_idx[l_z + 1] == i_row_idx[l_z] ) {
        unsigned int l_c_disp = l_z*i_xgemm_desc->ldc*l_fbytes;

        if ( !l_zeroed ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    l_micro_kernel_config.vxor_instruction,
                                                    l_micro_kernel_config.vector_name,
                                                    l_base_c_reg, l_base_c_reg, l_base_c_reg );
          l_zeroed = 1;
        }

        for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            l_micro_kernel_config.instruction_set,
                                            l_mov_insn, /* Handle non-temporal stores */
                                            l_gp_reg_mapping.gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_c_disp + l_n*l_vbytes,
                                            l_micro_kernel_config.vector_name,
                                            l_base_c_reg, 0, 0, 1 );
        }
      }
    }
  }

  /* Advance B and C */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                   l_gp_reg_mapping.gp_reg_b, (long long)l_vbytes*l_n_blocking );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                   l_gp_reg_mapping.gp_reg_c, (long long)l_vbytes*l_n_blocking );

  /* Test the loop condition */
  libxsmm_generator_generic_loop_footer( io_generated_code, &l_loop_label_tracker,
                                         l_gp_reg_mapping.gp_reg_nloop, (unsigned int)i_xgemm_desc->c1 );

  /* Close asm */
  libxsmm_x86_instruction_close_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );
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
  const unsigned int l_fp64 = LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype );
  unsigned int l_reg_unique, l_base_c_reg, l_base_c_gp_reg, l_base_ld_reg;
  int l_curr_b_disp = 0, l_curr_rvb_disp = -1;

  unsigned int l_bcast_reg_vals[120], l_nbcast_vals = 0;

  const libxsmm_aarch64_asimd_tupletype l_tuplet = (l_fp64) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;
  const libxsmm_aarch64_asimd_width l_width = (l_fp64) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S;
  const unsigned int l_values_per_reg = (l_fp64) ? 2 : 4;
  const unsigned int l_fbytes = (l_fp64) ? 8 : 4;

  /* Decide about NTS based on hint/flag and leading dimension */
  const unsigned int l_c_is_nt = ((LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT == (LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT & i_xgemm_desc->flags) &&
    0 == (i_xgemm_desc->ldc % l_values_per_reg)) ? 1/*true*/ : 0/*false*/);

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
  libxsmm_asparse_reg_sequence( i_xgemm_desc->m, l_m_blocking, i_row_idx,
                                i_column_idx, l_unique_pos, l_unique_sgn,
                                LIBXSMM_ASPARSE_REG_MAX_OPS, l_ops, &l_n_ops );

  /* Ensure it worked */
  if ( 0 == l_n_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    goto cleanup;
  }

  /* Define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;
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

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR,
                                                         l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    /* B pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_reg_mapping.gp_reg_b );
    /* C pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 96, l_gp_reg_mapping.gp_reg_c );
#if 0
    if ( i_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 56, l_gp_reg_mapping.gp_reg_a_prefetch );
      /* B prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 88, l_gp_reg_mapping.gp_reg_b_prefetch );
    }
#endif
  } else {
#if 0
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
    return;
#endif
  }

  /* Pre-load A into registers */
  if ( l_unique <= l_reg_unique ) {
    /* Copy the unique values into the data segment with 16-byte alignment */
    l_uoff = libxsmm_aarch64_instruction_add_data( io_generated_code,
                                                   (unsigned char*) l_unique_values,
                                                   l_unique*l_fbytes, 16, 1,
                                                   &l_const_data_tracker );

    /* Pad the segment to be a multiple of 16 */
    if ( (l_unique*l_fbytes) % 16 != 0 ) {
      unsigned char l_pad[15] = { 0 };
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
                                         LIBXSMM_AARCH64_GP_REG_X23, (unsigned int)i_xgemm_desc->c1 );

  /* Reset the data segment pointer */
  if ( l_unique > l_reg_unique ) {
    libxsmm_aarch64_instruction_adr_data( io_generated_code,
                                          LIBXSMM_AARCH64_GP_REG_X26,
                                          0, &l_const_data_tracker );
  }

  /* Copy our B pointer to a GPR */
  libxsmm_generator_mov_aarch64( io_generated_code, l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1 );

  for ( l_op_idx = 0; l_op_idx < l_n_ops; l_op_idx++ ) {
    libxsmm_asparse_reg_op op = l_ops[l_op_idx];
    unsigned int l_rvb = l_base_ld_reg;
    int l_b_disp = op.b_disp*i_xgemm_desc->ldb*l_fbytes;

    /* Offset our B pointer */
    if ( l_b_disp != l_curr_b_disp ) {
      unsigned int l_disp_insn = (l_b_disp > l_curr_b_disp) ? LIBXSMM_AARCH64_INSTR_GP_META_ADD : LIBXSMM_AARCH64_INSTR_GP_META_SUB;
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, l_disp_insn,
                                                     l_gp_reg_mapping.gp_reg_help_1,
                                                     l_gp_reg_mapping.gp_reg_help_0,
                                                     l_gp_reg_mapping.gp_reg_help_1,
                                                     LIBXSMM_DELTA( l_b_disp, l_curr_b_disp ) );
      l_curr_b_disp = l_b_disp;
    }

    if ( 1 == l_n_blocking ) {
      /* Load B itself (elide if already loaded) */
      if ( l_curr_rvb_disp != l_b_disp ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                                l_gp_reg_mapping.gp_reg_help_1, 0, 0,
                                                l_rvb, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        l_curr_rvb_disp = l_b_disp;
      }


      for ( l_z = 0; l_z < op.n; l_z++ ) {
        unsigned int l_u = op.src_vals[l_z], l_v;
        unsigned int l_rg = l_base_c_gp_reg + op.acc_idxs[l_z];
        unsigned int l_rva = (l_unique > l_reg_unique) ? ~0U : l_u;
        unsigned int l_rvc = l_base_c_reg + op.acc_idxs[l_z];
        unsigned int l_c_disp = op.c_disps[l_z]*i_xgemm_desc->ldc*l_fbytes;
        unsigned int l_neg = 0;
        unsigned int l_fma_insn = (op.src_sgns[l_z] == 1) ? LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V : LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_E_V;

        /* See if we need to load/zero the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_FIRST & op.flags[l_z] ) {
          /* Save offset into a GPR for later reuse */
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                         l_rg, l_c_disp );

          /* Zero (elide through FMUL + optional FNEG) */
          if ( l_beta0 ) {
            l_fma_insn = LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_E_V;
            l_neg = op.src_sgns[l_z] == -1;
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
          if ( ~0U == l_rva ) {
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
                                                           (short)(l_rva % l_values_per_reg),
                                                           l_width );

            /* Update our records */
            l_bcast_reg_vals[l_rva] = l_u;
          }
        }

        /* Perform the computation */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_fma_insn,
                                                   l_rvb,
                                                   l_rva / l_values_per_reg,
                                                   (unsigned char)(l_rva % l_values_per_reg),
                                                   l_rvc, l_tuplet );

        if ( l_neg ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V,
                                                     l_rvc, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, l_rvc, l_tuplet );
        }

        /* See if we need to save the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_LAST & op.flags[l_z] ) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                                  l_rg, 0, 0, l_rvc, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        }
      }
    } else {
      for ( l_n = 0; l_n < l_n_blocking; l_n += 2 ) {
        /* Load B itself (elide if already loaded) */
        if ( l_curr_rvb_disp != (l_b_disp + (int)l_n*16) ) {
          libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF,
                                                       l_gp_reg_mapping.gp_reg_help_1, 16*l_n,
                                                       l_rvb, l_rvb + 1, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
          l_curr_rvb_disp = l_b_disp + 16*l_n;
        }

        for ( l_z = 0; l_z < op.n; l_z++ ) {
          unsigned int l_u = op.src_vals[l_z], l_v;
          unsigned int l_rg = l_base_c_gp_reg + op.acc_idxs[l_z];
          unsigned int l_rva = (l_unique > l_reg_unique) ? ~0U : l_u;
          unsigned int l_rvc = l_base_c_reg + l_n_blocking*op.acc_idxs[l_z];
          unsigned int l_c_disp = op.c_disps[l_z]*i_xgemm_desc->ldc*l_fbytes;
          unsigned int l_neg = 0;
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

            /* Zero (elide through FMUL + optional FNEG) */
            if ( l_beta0 ) {
              l_fma_insn = LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_E_V;
              l_neg = op.src_sgns[l_z] == -1;
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
            if ( ~0U == l_rva ) {
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
                                                             (short)(l_rva % l_values_per_reg),
                                                             l_width );

              /* Update our records */
              l_bcast_reg_vals[l_rva] = l_u;
            }
          }

          /* Perform the computation */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_fma_insn,
                                                     l_rvb,
                                                     l_rva / l_values_per_reg,
                                                     (unsigned char)(l_rva % l_values_per_reg),
                                                     l_rvc + l_n, l_tuplet );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_fma_insn,
                                                     l_rvb + 1,
                                                     l_rva / l_values_per_reg,
                                                     (unsigned char)(l_rva % l_values_per_reg),
                                                     l_rvc + l_n + 1, l_tuplet );

          if ( l_neg ) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V,
                                                       l_rvc + l_n , LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, l_rvc + l_n, l_tuplet );
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V,
                                                       l_rvc + l_n + 1, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, l_rvc + l_n + 1, l_tuplet );
          }

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

  /* Handle rows which are all zero */
  if ( l_beta0 ) {
    unsigned int l_zeroed = 0;
    unsigned int l_stp_insn;

    if ( 1 == l_n_blocking ) {
      l_stp_insn = LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF;
    } else {
      l_stp_insn = (l_c_is_nt) ? LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF : LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
    }

    for ( l_z = 0; l_z < i_xgemm_desc->m; l_z++ ) {
      if ( i_row_idx[l_z + 1] == i_row_idx[l_z] ) {
        if ( !l_zeroed ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                     l_base_c_reg, l_base_c_reg, 0, l_base_c_reg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          l_zeroed = 1;
        }

        /* Compute the displacement */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                       l_base_c_gp_reg, (unsigned long long)l_z*i_xgemm_desc->ldc*l_fbytes );

        /* Issue the moves */
        if ( 1 == l_n_blocking ) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, l_stp_insn,
                                                  l_base_c_gp_reg, 0, 0, l_base_c_reg,
                                                  LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        } else {
          for ( l_n = 0; l_n < l_n_blocking; l_n += 2 ) {
            libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, l_stp_insn,
                                                         l_base_c_gp_reg, 16*l_n, l_base_c_reg, l_base_c_reg,
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
  int l_curr_b_disp = 0, l_curr_rvb_disp = -1;

  unsigned int l_bcast_reg_vals[30], l_nbcast_vals = 0;

  const unsigned int l_fp64 = LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype );
  const unsigned int l_fbytes = (l_fp64) ? 8 : 4;
  unsigned int l_vlen, l_vbytes, l_c_is_nt;
  unsigned int l_npacked_reg, l_npacked_values_per_reg;
  unsigned int l_na_off_reg = 0, l_base_a_off_reg = 0;
  const unsigned int l_beta0 = !!(LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags);

  libxsmm_asparse_reg_op *l_ops = (libxsmm_asparse_reg_op*) malloc(sizeof(libxsmm_asparse_reg_op)*LIBXSMM_ASPARSE_REG_MAX_OPS);
  unsigned int l_n_ops, l_op_idx;

  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_const_data_tracker l_const_data_tracker;
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_aarch64_sve_type l_svet;

  /* Load instruction table [single, double] */
  const unsigned int l_ld_tbl[2] = {
    LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
    LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF
  };

  /* Load w/replication instruction table [single, double] */
  const unsigned int l_ldr_tbl[2] = {
    LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
    LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF
  };

  /* Store instruction table [single, double][regular, non-temporal] */
  const unsigned int l_st_tbl[2][2] = {
    { LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF, LIBXSMM_AARCH64_INSTR_SVE_STNT1W_I_OFF },
    { LIBXSMM_AARCH64_INSTR_SVE_ST1D_I_OFF, LIBXSMM_AARCH64_INSTR_SVE_STNT1D_I_OFF }
  };

  /* check if mallocs were successful */
  if ( 0 == l_unique_values || 0 == l_unique_pos || 0 == l_unique_sgn || 0 == l_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_ALLOC_DATA );
    goto cleanup;
  }

  /* Define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );
  l_vlen = l_micro_kernel_config.vector_length;
  l_vbytes = l_fbytes*l_vlen;

  /* Decide about NTS based on hint/flag and leading dimension */
  l_c_is_nt = ((LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT == (LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT & i_xgemm_desc->flags) &&
    0 == (i_xgemm_desc->ldc % l_vlen)) ? 1/*true*/ : 0/*false*/);

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

  l_reg_unique = (32 - l_n_blocking - 2)*l_npacked_values_per_reg;

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

  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X28;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X29;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X30;

  memset( l_bcast_reg_vals, ~0, sizeof(l_bcast_reg_vals) );
  libxsmm_reset_const_data_tracker( &l_const_data_tracker );

  /* Define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* Open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xfff );

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR,
                                                         l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    /* B pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_reg_mapping.gp_reg_b );
    /* C pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 96, l_gp_reg_mapping.gp_reg_c );
#if 0
    if ( i_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 56, l_gp_reg_mapping.gp_reg_a_prefetch );
      /* B prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 88, l_gp_reg_mapping.gp_reg_b_prefetch );
    }
#endif
  } else {
#if 0
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
    return;
#endif
  }

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
      unsigned char l_pad[15] = { 0 };
      libxsmm_aarch64_instruction_add_data( io_generated_code, l_pad,
                                            (l_unique*l_fbytes) % 16, 1, 1,
                                            &l_const_data_tracker );
    }

    /* In the overflow case Z[0] is needed as a scratch register */
    if ( l_unique > l_npacked_values_per_reg*l_npacked_reg ) {
      l_used_reg++;
    }

    /* Pre-load A into registers */
    l_n = 0;
    while ( 1 ) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RQD_I_OFF,
                                            l_gp_reg_mapping.gp_reg_help_2, 0, 0,
                                            l_used_reg++, LIBXSMM_AARCH64_SVE_REG_P0 );
      l_n += l_npacked_values_per_reg;

      if ( l_n < l_unique ) {
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       l_gp_reg_mapping.gp_reg_help_2,
                                                       l_gp_reg_mapping.gp_reg_help_2,
                                                       l_npacked_values_per_reg*l_fbytes, 0 );
      } else {
        break;
      }
    }

    l_ld_reg = l_used_reg++;
    l_base_c_reg = l_used_reg;
    l_base_c_gp_reg = LIBXSMM_AARCH64_GP_REG_X3;

    /* Use any remaining registers for m blocking */
    l_m_blocking = LIBXSMM_MIN( (32 - l_base_c_reg) / l_n_blocking, LIBXSMM_ASPARSE_REG_MAX_M_BLOCK );
  /* A loaded in from memory */
  } else {
    l_m_blocking = 4;
    l_base_c_gp_reg = LIBXSMM_AARCH64_GP_REG_X3;
    l_base_c_reg = 32 - l_m_blocking*l_n_blocking;
    l_ld_reg = l_base_c_reg - 1;
    l_nbcast_vals = l_ld_reg;
    l_npacked_reg = 0;

    /* Load A + offsets into GPRs to enable loads with immediate offsets */
    l_base_a_off_reg = l_base_c_gp_reg + l_m_blocking;
    l_na_off_reg = l_gp_reg_mapping.gp_reg_help_0 - l_base_a_off_reg;

    libxsmm_generator_mov_aarch64( io_generated_code, l_gp_reg_mapping.gp_reg_help_2, l_base_a_off_reg );

    for ( l_n = 1; l_n < l_na_off_reg && 64*l_n < l_unique; l_n++ ) {
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     l_base_a_off_reg + l_n - 1,
                                                     l_base_a_off_reg + l_n,
                                                     64*l_fbytes, 0 );
    }
  }

  /* Sequence the operations */
  libxsmm_asparse_reg_sequence( i_xgemm_desc->m, l_m_blocking, i_row_idx,
                                i_column_idx, l_unique_pos, l_unique_sgn,
                                LIBXSMM_ASPARSE_REG_MAX_OPS, l_ops, &l_n_ops );

  /* Ensure it worked */
  if ( 0 == l_n_ops ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    goto cleanup;
  }

  /* Start the n loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                         LIBXSMM_AARCH64_GP_REG_X0, (unsigned int)i_xgemm_desc->c1 );

  /* Copy our B pointer to a GPR */
  libxsmm_generator_mov_aarch64( io_generated_code, l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1 );

  for ( l_op_idx = 0; l_op_idx < l_n_ops; l_op_idx++ ) {
    libxsmm_asparse_reg_op op = l_ops[l_op_idx];
    unsigned int l_rvb = l_ld_reg;
    int l_b_disp = op.b_disp*i_xgemm_desc->ldb*l_fbytes;
    unsigned int l_disp_insn = (l_b_disp > l_curr_b_disp) ? LIBXSMM_AARCH64_INSTR_GP_META_ADD : LIBXSMM_AARCH64_INSTR_GP_META_SUB;

    /* Offset our B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, l_disp_insn,
                                                   l_gp_reg_mapping.gp_reg_help_1,
                                                   l_gp_reg_mapping.gp_reg_help_0,
                                                   l_gp_reg_mapping.gp_reg_help_1,
                                                   LIBXSMM_DELTA( l_b_disp, l_curr_b_disp ) );
    l_curr_b_disp = l_b_disp;

    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      /* Load B itself (elide if already loaded) */
      if ( l_curr_rvb_disp != l_b_disp + (int)(l_n*l_vbytes) ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, l_ld_tbl[l_fp64],
                                              l_gp_reg_mapping.gp_reg_help_1, 0, l_n,
                                              l_rvb, LIBXSMM_AARCH64_SVE_REG_P0 );
        l_curr_rvb_disp = l_b_disp + l_n*l_vbytes;
      }

      for ( l_z = 0; l_z < op.n; l_z++ ) {
        unsigned int l_u = op.src_vals[l_z], l_v;
        unsigned int l_rva;
        unsigned char l_idx;
        unsigned int l_rg = l_base_c_gp_reg + op.acc_idxs[l_z];
        unsigned int l_rvc = l_base_c_reg + l_n_blocking*op.acc_idxs[l_z];
        unsigned int l_c_disp = op.c_disps[l_z]*i_xgemm_desc->ldc*l_fbytes;
        unsigned int l_neg = 0;
        unsigned int l_fma_insn;

        /* Constants are packed in registers */
        if ( l_unique <= l_reg_unique ) {
          l_fma_insn = (op.src_sgns[l_z] == 1) ? LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_I : LIBXSMM_AARCH64_INSTR_SVE_FMLS_V_I;

          l_rva = l_u / l_npacked_values_per_reg;
          l_idx = (unsigned char)(l_u % l_npacked_values_per_reg);

          /* Handle the overflow case */
          if ( l_unique > l_npacked_values_per_reg*l_npacked_reg ) {
            l_rva++;

            /* See if we need to copy from Z[l_rva] to Z[0] */
            if ( l_rva >= l_npacked_reg ) {
              if ( l_bcast_reg_vals[0] != l_rva ) {
                /* Move */
                libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                         l_rva, l_rva, 0, 0,
                                                         LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                         LIBXSMM_AARCH64_SVE_TYPE_D );

                /* Update our records */
                l_bcast_reg_vals[0] = l_rva;
              }

              l_rva = 0;
            }
          }
        /* Else, constants are read in from memory */
        } else {
          l_fma_insn = (op.src_sgns[l_z] == 1) ? LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P : LIBXSMM_AARCH64_INSTR_SVE_FMLS_V_P;
          l_idx = 0;

          /* See if we already have it in a register */
          for ( l_v = 0, l_rva = ~0U; l_v < l_nbcast_vals; l_v++ ) {
            if ( l_bcast_reg_vals[l_v] == l_u ) {
              l_rva = l_v;
              break;
            }
          }

          /* Otherwise pick a register to broadcast into */
          if ( ~0U == l_rva ) {
            int l_rsrc, l_disp;

            l_rva = libxsmm_asparse_reg_pick_bcast_reg( l_bcast_reg_vals, l_nbcast_vals,
                                                        l_ops + l_op_idx + 1, l_n_ops - l_op_idx - 1 );

            /* See if we can load directly using a GPR + immediate */
            if ( l_u < 64*l_na_off_reg ) {
              l_rsrc = l_base_a_off_reg + l_u / 64;
              l_disp = l_fbytes*(l_u % 64);
            /* Otherwise, use a helper register */
            } else {
              l_rsrc = l_gp_reg_mapping.gp_reg_help_2;
              l_disp = 0;

              libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                             l_base_a_off_reg, l_gp_reg_mapping.gp_reg_help_0,
                                                             l_gp_reg_mapping.gp_reg_help_2, (unsigned long long)l_u*l_fbytes );
            }

            /* Load */
            libxsmm_aarch64_instruction_sve_move( io_generated_code, l_ldr_tbl[l_fp64],
                                                  l_rsrc, 0, l_disp,
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

          /* Zero (elide through FMUL + optional FNEG) */
          if ( l_beta0 ) {
            l_fma_insn = (l_rva < l_npacked_reg) ? LIBXSMM_AARCH64_INSTR_SVE_FMUL_V_I : LIBXSMM_AARCH64_INSTR_SVE_FMUL_V;
            l_neg = op.src_sgns[l_z] == -1;
          /* Load */
          } else {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, l_ld_tbl[l_fp64],
                                                  l_rg, 0, l_n,
                                                  l_rvc + l_n, LIBXSMM_AARCH64_SVE_REG_P0 );
          }
        }

        /* Perform the computation */
        libxsmm_aarch64_instruction_sve_compute ( io_generated_code, l_fma_insn,
                                                  l_rvb, l_rva, l_idx, l_rvc + l_n,
                                                  LIBXSMM_AARCH64_SVE_REG_P0, l_svet );

        if ( l_neg ) {
          libxsmm_aarch64_instruction_sve_compute ( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FNEG_V_P,
                                                    l_rvc + l_n, 0, 0, l_rvc + l_n,
                                                    LIBXSMM_AARCH64_SVE_REG_P0, l_svet );
        }

        /* See if we need to save the accumulator */
        if ( LIBXSMM_ASPARSE_REG_FLAG_LAST & op.flags[l_z] ) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, l_st_tbl[l_fp64][l_c_is_nt],
                                                l_rg, 0, l_n, l_rvc + l_n,
                                                LIBXSMM_AARCH64_SVE_REG_P0 );
        }
      }
    }
  }

  /* Handle rows which are all zero */
  if ( l_beta0 ) {
    unsigned int l_zeroed = 0;

    for ( l_z = 0; l_z < i_xgemm_desc->m; l_z++ ) {
      if ( i_row_idx[l_z + 1] == i_row_idx[l_z] ) {
        if ( !l_zeroed ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   l_base_c_reg, l_base_c_reg, 0, l_base_c_reg,
                                                   LIBXSMM_AARCH64_SVE_REG_P0, LIBXSMM_AARCH64_SVE_TYPE_B );
          l_zeroed = 1;
        }

        /* Compute the displacement */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                       l_base_c_gp_reg, (unsigned long long)l_z*i_xgemm_desc->ldc*l_fbytes );

        /* Issue the moves */
        for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, l_st_tbl[l_fp64][l_c_is_nt],
                                                l_base_c_gp_reg, 0, l_n, l_base_c_reg,
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
                                         LIBXSMM_AARCH64_GP_REG_X0,
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
