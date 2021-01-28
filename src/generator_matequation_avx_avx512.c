/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_matequation_avx_avx512.h"
#include "generator_mateltwise_avx_avx512.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"
#include "generator_common_x86.h"

#define M_ADJUSTMENT 0
#define N_ADJUSTMENT 1

LIBXSMM_API_INTERN
unsigned int get_start_of_register_block(libxsmm_matequation_kernel_config *i_micro_kernel_config, unsigned int i_reg_block_id) {
  unsigned int result;
  result = i_micro_kernel_config->reserved_zmms + i_reg_block_id * i_micro_kernel_config->register_block_size;
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_init_micro_kernel_config( libxsmm_generated_code*         io_generated_code,
    libxsmm_matequation_kernel_config*    io_micro_kernel_config) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE ) {
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
    io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
    io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
    io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
    io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
    io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    io_micro_kernel_config->vmove_instruction_out= LIBXSMM_X86_INSTR_VMOVUPS;
    io_micro_kernel_config->vector_name = 'z';
  } else {
    /* That should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
  }
}

LIBXSMM_API_INTERN
int libxsmm_generator_mateqn_get_rbp_relative_offset( libxsmm_meqn_stack_var stack_var ) {
  /* The stack at exit of setup looks like this:
   *
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      Param_struct_ptr2                         <-- RBP-8
   *      Param_struct_ptr1                         <-- RBP-16
   *      Param_struct_ptr0                         <-- RBP-24
   *      Scratch ptr in stack (to be filled)       <-- RBP-32
   *      Placeholder for stack var                 <-- RBP-40
   *      Placeholder for stack var                 <-- RBP-48
   *      Placeholder for stack var                 <-- RBP-56
   *      Placeholder for stack var                 <-- RBP-64
   *      Placeholder for stack var                 <-- RBP-72
   *      Placeholder for stack var                 <-- RBP-80
   *
   * * */

  switch ( stack_var ) {
    case LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR0:
      return -24;
    case LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR1:
      return -16;
    case LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR2:
      return -8;
    case LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR:
      return -32;
    default:
      return 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_var( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_mateqn_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, i_gp_reg);
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg, offset);
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getval_stack_var( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_mateqn_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_tmp_i( libxsmm_generated_code*            io_generated_code,
                                                unsigned int                        i_tmp_offset_i,
                                                unsigned int                        i_gp_reg ) {
  libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, i_gp_reg );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg, i_tmp_offset_i);
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_setval_stack_var( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_mateqn_get_rbp_relative_offset(stack_var);
  /* make sure we requested to set  a legal stack var */
  if (offset >= 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 1 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_setup_stack_frame( libxsmm_generated_code*   io_generated_code,
                                              const libxsmm_meqn_descriptor*                      i_mateqn_desc,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matrix_eqn*                                 i_eqn,
                                              unsigned int                                        i_strategy ) {

  unsigned int temp_reg                     = LIBXSMM_X86_GP_REG_R8;
  unsigned int skip_pushpops_callee_gp_reg  = 0;
  unsigned int allocate_scratch = 1;

  i_micro_kernel_config->skip_pushpops_callee_gp_reg = skip_pushpops_callee_gp_reg;
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 80 );

  /* The stack now looks like this:
   *
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      Param_struct_ptr2                         <-- RBP-8
   *      Param_struct_ptr1                         <-- RBP-16
   *      Param_struct_ptr0                         <-- RBP-24
   *      Scratch ptr in stack (to be filled)       <-- RBP-32
   *      Placeholder for stack var                 <-- RBP-40
   *      Placeholder for stack var                 <-- RBP-48
   *      Placeholder for stack var                 <-- RBP-56
   *      Placeholder for stack var                 <-- RBP-64
   *      Placeholder for stack var                 <-- RBP-72
   *      Placeholder for stack var                 <-- RBP-80
   *
   * * */

  if (allocate_scratch > 0) {
    unsigned int scratch_size = 0;
    /* Strategy 0 is via tmp scratch in stack, Strategy 1 is via tmp register blocks */
    if (i_strategy == 0) {
      /*TODO: Now we allocate tmps with dsize float */
      libxsmm_blasint n_tmp = i_eqn->eqn_root->reg_score;
      unsigned int tmp_size = i_eqn->eqn_root->max_tmp_size * 4;
      scratch_size = tmp_size * n_tmp;
      i_micro_kernel_config->tmp_size = tmp_size;
    } else {
      unsigned int n_args = i_eqn->eqn_root->n_args;
      i_micro_kernel_config->n_args = n_args;
      scratch_size = n_args * 8;
    }
    /* make scratch size multiple of 64b */
    scratch_size = (scratch_size % 64 == 0) ? scratch_size : ((scratch_size + 63)/64) * 64;

    /* Now align RSP to 64 byte boundary  */
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, i_micro_kernel_config->alu_mov_instruction, temp_reg, 0xFFFFFFFFFFFFFFC0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ANDQ, temp_reg, LIBXSMM_X86_GP_REG_RSP);

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, scratch_size );
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
  }

  /* Now push to RSP the callee-save registers  */
  if (skip_pushpops_callee_gp_reg == 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBX );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
  }

  /* The stack at exit of setup looks like this:
   *
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      Param_struct_ptr2                         <-- RBP-8
   *      Param_struct_ptr1                         <-- RBP-16
   *      Param_struct_ptr0                         <-- RBP-24
   *      Scratch ptr in stack (to be filled)       <-- RBP-32
   *      Placeholder for stack var                 <-- RBP-40
   *      Placeholder for stack var                 <-- RBP-48
   *      Placeholder for stack var                 <-- RBP-56
   *      Placeholder for stack var                 <-- RBP-64
   *      Placeholder for stack var                 <-- RBP-72
   *      Placeholder for stack var                 <-- RBP-80
   *      [ Potentianl  pad for 64b align ]
   *      Scratch, 64b aligned                      <-- (RBP-32) contains this address
   *      Callee-saved registers                    <-- RSP
   *
   * * */
}


LIBXSMM_API_INTERN
void libxsmm_generator_matequation_destroy_stack_frame( libxsmm_generated_code*   io_generated_code,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config ) {
  if (i_micro_kernel_config->skip_pushpops_callee_gp_reg == 0) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBX );
  }

  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RSP);
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
}

LIBXSMM_API_INTERN
libxsmm_matrix_eqn_elem* find_op_at_timestamp(libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint timestamp) {
  libxsmm_matrix_eqn_elem *result = NULL;
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    result = NULL;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    if (cur_node->visit_timestamp == timestamp) {
      result = cur_node;
    } else {
      result = find_op_at_timestamp(cur_node->le, timestamp);
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    if (cur_node->visit_timestamp == timestamp) {
      result = cur_node;
    } else {
      result = find_op_at_timestamp(cur_node->le, timestamp);
      if (result == NULL) {
        result = find_op_at_timestamp(cur_node->ri, timestamp);
      }
    }
  }
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_create_unary_descriptor( libxsmm_matrix_eqn_elem *cur_op, libxsmm_meltw_descriptor **desc) {
  libxsmm_descriptor_blob blob;
  *desc = libxsmm_meltw_descriptor_init2(&blob, cur_op->tmp.dtype, cur_op->info.u_op.dtype, cur_op->tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->tmp.m, cur_op->tmp.n, cur_op->tmp.ld, cur_op->tmp.ld, 0, 0, (unsigned short)cur_op->info.u_op.flags, cur_op->info.u_op.type, LIBXSMM_MELTW_OPERATION_UNARY);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_create_binary_descriptor( libxsmm_matrix_eqn_elem *cur_op, libxsmm_meltw_descriptor **desc) {
  libxsmm_descriptor_blob blob;
  *desc = libxsmm_meltw_descriptor_init2(&blob, cur_op->tmp.dtype, cur_op->info.b_op.dtype, cur_op->tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->tmp.m, cur_op->tmp.n, cur_op->tmp.ld, cur_op->tmp.ld, 0, 0, (unsigned short)cur_op->info.b_op.flags, cur_op->info.b_op.type, LIBXSMM_MELTW_OPERATION_BINARY);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_input_in_stack_param_struct( libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_matrix_eqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        ptr_id ) {

  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0,
        temp_reg,
        0 );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        temp_reg,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        cur_node->info.arg.in_pos*8,
        temp_reg,
        0 );
  } else {
    libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, cur_node->tmp.id * i_micro_kernel_config->tmp_size, temp_reg);
  }
  if (ptr_id == 0) {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR0, temp_reg );
  } else {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR1, temp_reg );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_output_in_stack_param_struct(libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_matrix_eqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        is_last_op ) {
  if (is_last_op > 0) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        8,
        temp_reg,
        0 );
  } else {
    libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, cur_node->tmp.id * i_micro_kernel_config->tmp_size, temp_reg);
  }
  libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR2, temp_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_stack_scratch_avx_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc) {

  libxsmm_matequation_gp_reg_mapping  l_gp_reg_mapping;
  libxsmm_matequation_kernel_config   l_kernel_config;
  libxsmm_meltw_descriptor            *meltw_desc;
  libxsmm_loop_label_tracker          l_loop_label_tracker;

  unsigned int eqn_idx = i_mateqn_desc->eqn_idx;
  libxsmm_matrix_eqn *eqn = libxsmm_matrix_eqn_get_equation( eqn_idx );
  unsigned int timestamp = 0;
  unsigned int last_timestamp;
  unsigned int temp_reg = LIBXSMM_X86_GP_REG_R8;

  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation doesn't exist... nothing to JIT,,,\n" );
    return;
  } else {
    last_timestamp = eqn->eqn_root->visit_timestamp;
  }

  /* Some basic initialization of the config kernel */
  libxsmm_generator_matequation_init_micro_kernel_config(io_generated_code, &l_kernel_config);

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif

  /* Open asm */
  libxsmm_x86_instruction_open_stream_matequation( io_generated_code, l_gp_reg_mapping.gp_reg_param_struct);

  /* Setup the stack */
  libxsmm_generator_matequation_setup_stack_frame( io_generated_code, i_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, eqn, 0);

  l_gp_reg_mapping.gp_reg_mapping_eltwise.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RSI;
  libxsmm_generator_meqn_getaddr_stack_var(  io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR0, LIBXSMM_X86_GP_REG_RSI );

  /* Iterate over the equation tree based on the optimal traversal order and call the proper JITer */
  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_matrix_eqn_elem *cur_op = find_op_at_timestamp(eqn->eqn_root, timestamp);
    if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
      /* Prepare struct param */
      libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, &l_kernel_config, &l_gp_reg_mapping, cur_op->le,
          temp_reg, 0);
      libxsmm_generator_matequation_set_output_in_stack_param_struct( io_generated_code, &l_kernel_config, &l_gp_reg_mapping, cur_op,
          temp_reg, (timestamp == last_timestamp) );
      /* Prepare descriptor  */
      libxsmm_generator_matequation_create_unary_descriptor( cur_op, &meltw_desc);
    } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
      libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, &l_kernel_config, &l_gp_reg_mapping, cur_op->le,
          temp_reg, 0);
      libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, &l_kernel_config, &l_gp_reg_mapping, cur_op->ri,
          temp_reg, 1);
      libxsmm_generator_matequation_set_output_in_stack_param_struct( io_generated_code, &l_kernel_config, &l_gp_reg_mapping, cur_op,
          temp_reg, (timestamp == last_timestamp) );
      libxsmm_generator_matequation_create_binary_descriptor( cur_op, &meltw_desc);
    } else {
      /* This should not happen */
    }
    /* Configure the unary-binary microkernel */
    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_kernel_config.meltw_kernel_config, io_generated_code->arch, meltw_desc);
    /* Call unary-binary JITer */
    libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping.gp_reg_mapping_eltwise, &l_kernel_config.meltw_kernel_config, meltw_desc );
  }

  /* Destroy stack frame */
  libxsmm_generator_matequation_destroy_stack_frame(  io_generated_code,  &l_kernel_config );

  /* Close asm */
  libxsmm_x86_instruction_close_stream_matequation( io_generated_code );
}

LIBXSMM_API_INTERN
void libxsmm_generator_copy_input_args(libxsmm_generated_code*        io_generated_code,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    libxsmm_matrix_eqn_elem             *cur_node,
    unsigned int                        *arg_id,
    libxsmm_matrix_eqn_arg              *arg_info,
    unsigned int                        input_reg) {

  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;
  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    unsigned int cur_pos = *arg_id;
    if (cur_pos < i_micro_kernel_config->n_avail_gpr) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          input_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          cur_node->info.arg.in_pos*8,
          i_micro_kernel_config->gpr_pool[cur_pos],
          0 );
    } else {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          input_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          cur_node->info.arg.in_pos*8,
          temp_reg,
          0 );
      libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, cur_pos * 8, temp_reg2);
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          temp_reg2,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          temp_reg,
          1 );
    }
    arg_info[cur_pos] = cur_node->info.arg;
    *arg_id = cur_pos + 1;
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    if (cur_node->le->reg_score >= cur_node->ri->reg_score) {
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
    } else {
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
    }
  } else {
    /* This should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_adjust_args_addr(libxsmm_generated_code*        io_generated_code,
    const libxsmm_meqn_descriptor       *i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    unsigned int                        i_adjust_instr,
    unsigned int                        i_adjust_amount,
    unsigned int                        i_adjust_type,
    libxsmm_matrix_eqn_arg              *arg_info) {

  unsigned int n_args = i_micro_kernel_config->n_args;
  unsigned int i;
  unsigned int adjust_val = 0;
  unsigned int output_tsize = libxsmm_typesize(LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype));
  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;

  /* Adjust input args */
  for (i = 0; i < n_args; i++) {
    unsigned int tsize = libxsmm_typesize(arg_info[i].dtype);
    if (i_adjust_type == M_ADJUSTMENT) {
      adjust_val = i_adjust_amount * tsize;
    } else if (i_adjust_type == N_ADJUSTMENT) {
      adjust_val = i_adjust_amount * arg_info[i].ld * tsize;
    }
    if ( i < i_micro_kernel_config->n_avail_gpr ) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_micro_kernel_config->gpr_pool[i], adjust_val);
    } else {
      libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, i * 8, temp_reg);
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          temp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          temp_reg2,
          0 );
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, temp_reg2, adjust_val);
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          temp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          temp_reg2,
          1 );
    }
  }
  /* Adjust output  */
  if (i_adjust_type == M_ADJUSTMENT) {
    adjust_val = i_adjust_amount * output_tsize;
  } else if (i_adjust_type == N_ADJUSTMENT) {
    adjust_val = i_adjust_amount * i_mateqn_desc->ldo * output_tsize;
  }
  libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_out, adjust_val);
}

LIBXSMM_API_INTERN
void libxsmm_configure_mateqn_microkernel_loops( libxsmm_matequation_kernel_config       *i_micro_kernel_config,
                                                 libxsmm_matrix_eqn                      *i_eqn,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n,
                                                 unsigned int                            i_use_m_input_masking,
                                                 unsigned int*                           i_m_trips,
                                                 unsigned int*                           i_n_trips,
                                                 unsigned int*                           i_m_unroll_factor,
                                                 unsigned int*                           i_n_unroll_factor,
                                                 unsigned int*                           i_m_assm_trips,
                                                 unsigned int*                           i_n_assm_trips) {

  unsigned int m_trips, n_trips, m_unroll_factor, n_unroll_factor, m_assm_trips, n_assm_trips;
  unsigned int max_nm_unrolling = 32;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int n_tmp_reg_blocks = i_eqn->eqn_root->reg_score + 1;
  i_micro_kernel_config->n_tmp_reg_blocks = n_tmp_reg_blocks;

  m_trips               = (i_m + i_vlen_in - 1) / i_vlen_in;
  n_trips               = i_n;

  max_nm_unrolling = max_nm_unrolling - reserved_zmms;
  max_nm_unrolling = max_nm_unrolling / n_tmp_reg_blocks;

  if (max_nm_unrolling < 1) {
    printf("Can't generate run this code variant, ran out of zmm registers...\n");
  }
  if ((max_nm_unrolling < m_trips) && (i_use_m_input_masking == 1)) {
    printf("Can't generate run this code variant, ran out of zmm registers and we want to mask M...\n");
  }

  if (i_use_m_input_masking == 1) {
    m_unroll_factor = m_trips;
  } else {
    m_unroll_factor = LIBXSMM_MIN(m_trips,16);
  }

  if (m_unroll_factor > max_nm_unrolling) {
    m_unroll_factor = max_nm_unrolling;
  }

  while (m_trips % m_unroll_factor != 0) {
    m_unroll_factor--;
  }

  n_unroll_factor = n_trips;
  while (m_unroll_factor * n_unroll_factor > max_nm_unrolling) {
    n_unroll_factor--;
  }

  while (n_trips % n_unroll_factor != 0) {
    n_unroll_factor--;
  }

  m_assm_trips = m_trips/m_unroll_factor;
  n_assm_trips = n_trips/n_unroll_factor;

  *i_m_trips = m_trips;
  *i_n_trips = n_trips;
  *i_m_unroll_factor = m_unroll_factor;
  *i_n_unroll_factor = n_unroll_factor;
  *i_m_assm_trips = m_assm_trips;
  *i_n_assm_trips = n_assm_trips;
}

LIBXSMM_API_INTERN
void libxsmm_meqn_setup_input_output_masks( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_tmp_reg,
                                                 unsigned int                            i_m,
                                                 unsigned int*                           i_use_m_input_masking,
                                                 unsigned int*                           i_mask_reg_in,
                                                 unsigned int*                           i_use_m_output_masking,
                                                 unsigned int*                           i_mask_reg_out) {

  unsigned int mask_in_count, mask_out_count, mask_reg_in = 0, mask_reg_out = 0, use_m_input_masking, use_m_output_masking;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned int reserved_mask_regs = 1;

  use_m_input_masking   = (i_m % i_vlen_in == 0 ) ? 0 : 1;
  use_m_output_masking  = (i_m % i_vlen_out == 0 ) ? 0 : 1;

  if (use_m_input_masking == 1) {
    mask_in_count = i_vlen_in - i_m % i_vlen_in;
    mask_reg_in   = reserved_mask_regs;
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_in, mask_in_count, i_micro_kernel_config->arg_info[0].dtype);
    reserved_mask_regs++;
  }

  if (use_m_output_masking == 1) {
    if (i_vlen_in == i_vlen_out) {
      mask_reg_out = mask_reg_in;
    } else {
      mask_out_count = i_vlen_out - i_m % i_vlen_out;
      mask_reg_out   = reserved_mask_regs;
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_out, mask_out_count, LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype));
      reserved_mask_regs++;
    }
  }

  i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
  *i_mask_reg_in = mask_reg_in;
  *i_use_m_input_masking = use_m_input_masking;
  *i_mask_reg_out = mask_reg_out;
  *i_use_m_output_masking = use_m_output_masking;
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_load_arg_to_2d_reg_block( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_arg_id,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg ) {

  unsigned int in, im;
  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;
  unsigned int cur_vreg;
  libxsmm_matrix_eqn_arg  *arg_info = i_micro_kernel_config->arg_info;
  unsigned int i_start_vreg = get_start_of_register_block(i_micro_kernel_config, i_reg_block_id);
  unsigned int input_reg = 0;

  if (i_arg_id < i_micro_kernel_config->n_avail_gpr) {
    input_reg = i_micro_kernel_config->gpr_pool[i_arg_id];
  } else {
    libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, i_arg_id * 8, temp_reg2);
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        temp_reg2,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0,
        temp_reg,
        0 );
    input_reg = temp_reg;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          input_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * i_vlen + in * arg_info[i_arg_id].ld) * libxsmm_typesize(arg_info[i_arg_id].dtype),
          'z',
          cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 1, 0 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_store_2d_reg_block( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg ) {
  unsigned int in, im;
  unsigned int cur_vreg;
  unsigned int i_start_vreg = get_start_of_register_block(i_micro_kernel_config, i_reg_block_id);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * i_vlen + in * i_meqn_desc->ldo) * libxsmm_typesize(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)),
          'z',
          cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 0, 1 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_unary_op_2d_reg_block( libxsmm_generated_code*     io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_unary_type                i_op_type,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking ) {

  unsigned int i_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_reg_block_id);
  unsigned int im, in, cur_vreg;
  libxsmm_mateltwise_kernel_config *i_micro_kernel_config = &(i_meqn_micro_kernel_config->meltw_kernel_config);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_X2) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'z', cur_vreg, cur_vreg, cur_vreg );
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VXORPS, 'z', cur_vreg, i_micro_kernel_config->neg_signs_vreg, cur_vreg );
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_INC) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, 'z', cur_vreg, i_micro_kernel_config->vec_ones, cur_vreg );
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRCP14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRSQRT14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,LIBXSMM_X86_INSTR_VRSQRT14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,LIBXSMM_X86_INSTR_VRCP14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
        libxsmm_generator_exp_ps_3dts_avx512( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_y,
            i_micro_kernel_config->vec_z,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_log2e);
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_TANH || i_op_type == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV ) {
        libxsmm_generator_tanh_ps_rational_78_avx512( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_x2,
            i_micro_kernel_config->vec_nom,
            i_micro_kernel_config->vec_denom,
            i_micro_kernel_config->mask_hi,
            i_micro_kernel_config->mask_lo,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c1_d,
            i_micro_kernel_config->vec_c2_d,
            i_micro_kernel_config->vec_c3_d,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            i_micro_kernel_config->vec_ones,
            i_micro_kernel_config->vec_neg_ones);

        if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VFNMSUB213PS, 'z', i_micro_kernel_config->vec_neg_ones, cur_vreg, cur_vreg );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
        libxsmm_generator_sigmoid_ps_rational_78_avx512( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_x2,
            i_micro_kernel_config->vec_nom,
            i_micro_kernel_config->vec_denom,
            i_micro_kernel_config->mask_hi,
            i_micro_kernel_config->mask_lo,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c1_d,
            i_micro_kernel_config->vec_c2_d,
            i_micro_kernel_config->vec_c3_d,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            i_micro_kernel_config->vec_ones,
            i_micro_kernel_config->vec_neg_ones,
            i_micro_kernel_config->vec_halves );

        if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VSUBPS, 'z', cur_vreg, i_micro_kernel_config->vec_ones, i_micro_kernel_config->vec_x2 );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VMULPS, 'z', i_micro_kernel_config->vec_x2, cur_vreg, cur_vreg );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_GELU) {
        libxsmm_generator_gelu_ps_minimax3_avx512( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_xr,
            i_micro_kernel_config->vec_xa,
            i_micro_kernel_config->vec_index,
            i_micro_kernel_config->vec_C0,
            i_micro_kernel_config->vec_C1,
            i_micro_kernel_config->vec_C2,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2 );
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
        libxsmm_generator_gelu_inv_ps_minimax3_avx512( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_xr,
            i_micro_kernel_config->vec_xa,
            i_micro_kernel_config->vec_index,
            i_micro_kernel_config->vec_C0,
            i_micro_kernel_config->vec_C1,
            i_micro_kernel_config->vec_C2,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2 );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_binary_op_2d_reg_block( libxsmm_generated_code*    io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_binary_type               i_op_type,
                                                 unsigned int                            i_left_reg_block_id,
                                                 unsigned int                            i_right_reg_block_id,
                                                 unsigned int                            i_dst_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking ) {
  unsigned int i_left_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_left_reg_block_id);
  unsigned int i_right_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_right_reg_block_id);
  unsigned int i_dst_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_dst_reg_block_id);
  unsigned int im, in, left_vreg, right_vreg, dst_vreg;
  unsigned int binary_op_instr = 0;

  switch (i_op_type) {
    case LIBXSMM_MELTW_TYPE_BINARY_ADD: {
      binary_op_instr = LIBXSMM_X86_INSTR_VADDPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MUL: {
      binary_op_instr = LIBXSMM_X86_INSTR_VMULPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_SUB: {
      binary_op_instr = LIBXSMM_X86_INSTR_VSUBPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_DIV: {
      binary_op_instr = LIBXSMM_X86_INSTR_VDIVPS;
    } break;
    default:;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      left_vreg = i_left_start_vreg + in * i_m_blocking + im;
      right_vreg = i_right_start_vreg + in * i_m_blocking + im;
      dst_vreg = i_dst_start_vreg + in * i_m_blocking + im;
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, binary_op_instr, 'z', right_vreg, left_vreg, dst_vreg );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_2d_microkernel( libxsmm_generated_code*                    io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 libxsmm_matrix_eqn                      *i_eqn,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n,
                                                 unsigned int                            skip_n_loop_reg_cleanup ) {

  unsigned int use_m_input_masking, use_m_output_masking, m_trips, m_unroll_factor, m_assm_trips, n_trips, n_unroll_factor, n_assm_trips;
  unsigned int mask_reg_in, mask_reg_out;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int last_timestamp = i_eqn->eqn_root->visit_timestamp;
  unsigned int arg_id = 0;
  unsigned int aux_reg_block = 0;
  unsigned int right_reg_block = 0;
  unsigned int left_reg_block = 0;
  unsigned int timestamp = 0;

  /* Configure microkernel masks */
  libxsmm_meqn_setup_input_output_masks( io_generated_code, i_micro_kernel_config, i_meqn_desc,
      temp_reg, i_m, &use_m_input_masking, &mask_reg_in, &use_m_output_masking, &mask_reg_out);

  /* Configure microkernel loops */
  libxsmm_configure_mateqn_microkernel_loops( i_micro_kernel_config, i_eqn, i_m, i_n, use_m_input_masking,
      &m_trips, &n_trips, &m_unroll_factor, &n_unroll_factor, &m_assm_trips, &n_assm_trips);

  i_micro_kernel_config->register_block_size = m_unroll_factor * n_unroll_factor;

  aux_reg_block = i_micro_kernel_config->n_tmp_reg_blocks - 1;

  /* Headers of microkernel loops */
  if (n_assm_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 0, n_unroll_factor);
  }

  if (m_assm_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 0, m_unroll_factor);
  }

  /* Traverse equation tree based on optimal execution plan and emit code  */
  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_matrix_eqn_elem *cur_op = find_op_at_timestamp(i_eqn->eqn_root, timestamp);
    if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
      if (cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        /* We have to load the input from the argument tensor using the node's assigned tmp reg block */
        left_reg_block = cur_op->tmp.id;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, left_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
        arg_id++;
      }
      libxsmm_generator_mateqn_compute_unary_op_2d_reg_block( io_generated_code, i_micro_kernel_config,
          cur_op->info.u_op.type, cur_op->tmp.id, m_unroll_factor, n_unroll_factor);
    } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
      if ((cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG)) {
        /* We have to load the input from the argument tensor using the node's assigned tmp reg block */
        left_reg_block = cur_op->tmp.id;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, left_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
        arg_id++;
        right_reg_block = aux_reg_block;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, right_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
        arg_id++;
      } else if ((cur_op->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_op->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG)) {
        left_reg_block = cur_op->le->tmp.id;
        right_reg_block = cur_op->ri->tmp.id;
      } else {
        if (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
          /* We have to load the input from the argument tensor using the auxiliary tmp reg block */
          left_reg_block = cur_op->le->tmp.id;
          right_reg_block = aux_reg_block;
          libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
              arg_id, i_vlen_in, aux_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
          arg_id++;
        } else {
          left_reg_block = aux_reg_block;
          right_reg_block = cur_op->ri->tmp.id;
          libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
              arg_id, i_vlen_in, aux_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
          arg_id++;
        }
      }
      libxsmm_generator_mateqn_compute_binary_op_2d_reg_block( io_generated_code, i_micro_kernel_config,
          cur_op->info.b_op.type, left_reg_block, right_reg_block, cur_op->tmp.id, m_unroll_factor, n_unroll_factor);
    } else {
      /* This should not happen */
    }
  }

  /* Store the computed register block to output  */
  libxsmm_generator_mateqn_store_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
      i_vlen_out, i_eqn->eqn_root->tmp.id, m_unroll_factor, n_unroll_factor, use_m_output_masking, mask_reg_out );

  /* Footers of microkernel loops */
  if (m_assm_trips > 1) {
    libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_add_instruction, m_unroll_factor * i_vlen_in , M_ADJUSTMENT, i_micro_kernel_config->arg_info);
    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips);
    libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_sub_instruction, m_unroll_factor * i_vlen_in * m_assm_trips, M_ADJUSTMENT, i_micro_kernel_config->arg_info);
  }

  if (n_assm_trips > 1) {
    libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_add_instruction, n_unroll_factor, N_ADJUSTMENT, i_micro_kernel_config->arg_info);
    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
    if (skip_n_loop_reg_cleanup == 0) {
      libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_sub_instruction, n_unroll_factor * n_assm_trips, N_ADJUSTMENT, i_micro_kernel_config->arg_info);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_configure_M_N_blocking( libxsmm_matrix_eqn *i_eqn, unsigned int m, unsigned int n, unsigned int vlen, unsigned int *m_blocking, unsigned int *n_blocking) {
  /* The m blocking is done in chunks of vlen */
  unsigned int m_chunks = (m+vlen-1)/vlen;
  unsigned int m_chunk_remainder = 8;
  unsigned int m_range, m_block_size, foo1, foo2;
  unsigned int reserved_zmms = 0;
  unsigned int n_tmp_reg_blocks = i_eqn->eqn_root->reg_score + 1;
  unsigned int max_nm_unrolling = 32 - reserved_zmms;

  max_nm_unrolling = max_nm_unrolling / n_tmp_reg_blocks;

  if (m % vlen == 0) {
    /* If there is not remainder in M, then we block M in order to limit block size */
    if (m_chunks > 32) {
      libxsmm_compute_equalized_blocking(m_chunks, (m_chunks+1)/2, &m_range, &m_block_size, &foo1, &foo2);
      *m_blocking = m_range * vlen;
    } else {
      *m_blocking = m;
    }
  } else {
    /* If there is remainder we make sure we can fully unroll the kernel with masks */
    if (m_chunk_remainder > max_nm_unrolling) {
      m_chunk_remainder = max_nm_unrolling;
    }
    if (m_chunk_remainder >= m_chunks) {
      *m_blocking = m;
    } else {
      *m_blocking = (m_chunks - m_chunk_remainder) * vlen;
    }
  }

  /* FIXME: When we dont have nice N values AND we have more register unrolling oportunities, apply n blocking...
   * For now not any additional blocking in N */
  *n_blocking = n;
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_register_block_avx_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc) {
  libxsmm_matequation_gp_reg_mapping  l_gp_reg_mapping;
  libxsmm_matequation_kernel_config   l_kernel_config;
  libxsmm_loop_label_tracker          l_loop_label_tracker;
  libxsmm_matrix_eqn_arg              *arg_info;

  unsigned int eqn_idx = i_mateqn_desc->eqn_idx;
  unsigned int arg_id = 0;
  libxsmm_matrix_eqn *eqn = libxsmm_matrix_eqn_get_equation( eqn_idx );
  unsigned int m_blocking = 0, n_blocking = 0, cur_n = 0, cur_m = 0, n_microkernel = 0, m_microkernel = 0, adjusted_aux_vars = 0;
  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation doesn't exist... nothing to JIT,,,\n" );
    return;
  }

  /* Some basic initialization of the config kernel */
  libxsmm_generator_matequation_init_micro_kernel_config(io_generated_code, &l_kernel_config);

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif
  l_gp_reg_mapping.gp_reg_out = LIBXSMM_X86_GP_REG_RAX;
  l_gp_reg_mapping.temp_reg    = LIBXSMM_X86_GP_REG_RBX;
  l_gp_reg_mapping.temp_reg2   = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_R15;
  l_kernel_config.n_avail_gpr = 9;
  l_kernel_config.gpr_pool[0] = LIBXSMM_X86_GP_REG_RSI; l_kernel_config.gpr_pool[1] = LIBXSMM_X86_GP_REG_RDX; l_kernel_config.gpr_pool[2] = LIBXSMM_X86_GP_REG_R8; l_kernel_config.gpr_pool[3] = LIBXSMM_X86_GP_REG_R9;
  l_kernel_config.gpr_pool[4] = LIBXSMM_X86_GP_REG_R10; l_kernel_config.gpr_pool[5] = LIBXSMM_X86_GP_REG_R11; l_kernel_config.gpr_pool[6] = LIBXSMM_X86_GP_REG_R12; l_kernel_config.gpr_pool[7] = LIBXSMM_X86_GP_REG_R13; l_kernel_config.gpr_pool[8] = LIBXSMM_X86_GP_REG_R14;

  /** Open asm */
  libxsmm_x86_instruction_open_stream_matequation( io_generated_code, l_gp_reg_mapping.gp_reg_param_struct);

  /* Setup the stack */
  libxsmm_generator_matequation_setup_stack_frame( io_generated_code, i_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, eqn, 1);

  /* Iterate over the equation tree and copy the args ptrs in the auxiliary scratch */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      l_kernel_config.alu_mov_instruction,
      l_gp_reg_mapping.gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      LIBXSMM_X86_GP_REG_R15,
      0 );
  arg_info = (libxsmm_matrix_eqn_arg*) malloc(l_kernel_config.n_args * sizeof(libxsmm_matrix_eqn_arg));
  libxsmm_generator_copy_input_args(io_generated_code, &l_gp_reg_mapping, &l_kernel_config, eqn->eqn_root, &arg_id, arg_info, LIBXSMM_X86_GP_REG_R15);
  l_kernel_config.arg_info = arg_info;

  /* Setup output reg */
  libxsmm_x86_instruction_alu_mem( io_generated_code, l_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_param_struct, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, l_gp_reg_mapping.gp_reg_out, 0 );

  /* FIXME: Configure vlens... */
  l_kernel_config.vlen_in = 16;
  l_kernel_config.vlen_comp = 16;
  l_kernel_config.vlen_out = 16;

  /* FIXME: Assign reserved zmms by parsing the equation */
  l_kernel_config.reserved_zmms = 0;
  l_kernel_config.reserved_mask_regs = 1;

  /* Configure M and N blocking factors */
  libxsmm_generator_matequation_configure_M_N_blocking(eqn, i_mateqn_desc->m, i_mateqn_desc->n, l_kernel_config.vlen_in, &m_blocking, &n_blocking);

  cur_n = 0;
  while (cur_n != i_mateqn_desc->n) {
    cur_m = 0;
    adjusted_aux_vars = 0;
    n_microkernel = (cur_n < n_blocking) ? n_blocking : i_mateqn_desc->n - cur_n;
    while (cur_m != i_mateqn_desc->m) {
      m_microkernel = (cur_m < m_blocking) ? m_blocking : i_mateqn_desc->m - cur_m;
      unsigned int skip_n_loop_reg_cleanup = ((cur_n + n_microkernel == i_mateqn_desc->n) && (cur_m + m_microkernel == i_mateqn_desc->m)) ? 1 : 0 ;
      libxsmm_generator_mateqn_2d_microkernel(io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateqn_desc, eqn, m_microkernel, n_microkernel, skip_n_loop_reg_cleanup);
      cur_m += m_microkernel;
      if (cur_m != i_mateqn_desc->m) {
        adjusted_aux_vars = 1;
        libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, l_kernel_config.alu_add_instruction, m_microkernel, M_ADJUSTMENT, arg_info);
      }
    }
    if (adjusted_aux_vars == 1) {
      libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, l_kernel_config.alu_sub_instruction, m_microkernel, M_ADJUSTMENT, arg_info);
    }
    cur_n += n_microkernel;
    if (cur_n != i_mateqn_desc->n) {
      libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, l_kernel_config.alu_add_instruction, n_microkernel, N_ADJUSTMENT,  arg_info);
    }
  }

  /* Destroy stack frame */
  libxsmm_generator_matequation_destroy_stack_frame(  io_generated_code,  &l_kernel_config );

  /* Close asm */
  libxsmm_x86_instruction_close_stream_matequation( io_generated_code );

  /* Free aux data structure */
  free(arg_info);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_avx_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc) {
  /* TODO: add logic to select code generation strategy of matrix equation  */
#if 0
  libxsmm_generator_matequation_tmp_stack_scratch_avx_avx512_kernel(io_generated_code, i_mateqn_desc);
#else
  libxsmm_generator_matequation_tmp_register_block_avx_avx512_kernel(io_generated_code, i_mateqn_desc);
#endif
}

