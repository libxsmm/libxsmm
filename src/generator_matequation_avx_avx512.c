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
void libxsmm_generator_matequation_tmp_stack_scratch_setup_stack_frame( libxsmm_generated_code*   io_generated_code,
                                              const libxsmm_meqn_descriptor*                      i_mateqn_desc,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matrix_eqn*                                 i_eqn ) {

  unsigned int temp_reg                     = LIBXSMM_X86_GP_REG_R10;
  unsigned int skip_pushpops_callee_gp_reg  = 0;
  libxsmm_blasint n_tmp         = i_eqn->eqn_root->reg_score;
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
    /*TODO: Now we allocate tmps with dsize float */
    unsigned int tmp_size = i_eqn->eqn_root->max_tmp_size * 4;
    unsigned int scratch_size = tmp_size * n_tmp;
    i_micro_kernel_config->tmp_size = tmp_size;

    /* make scratch size multiple of 64b */
    scratch_size = (scratch_size % 64 == 0) ? scratch_size : ((scratch_size + 63)/64) * 64;

    /* Now align RSP to 64 byte boundary  */
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, i_micro_kernel_config->alu_mov_instruction, temp_reg, 0xFFFFFFFFFFFFFFC0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ANDQ, temp_reg, LIBXSMM_X86_GP_REG_RSP);

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, scratch_size );
    libxsmm_generator_meltw_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
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
void libxsmm_generator_matequation_tmp_stack_scratch_destroy_stack_frame( libxsmm_generated_code*   io_generated_code,
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
  unsigned int temp_reg = LIBXSMM_X86_GP_REG_R10;

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
  libxsmm_generator_matequation_tmp_stack_scratch_setup_stack_frame( io_generated_code, i_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, eqn);

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
  libxsmm_generator_matequation_tmp_stack_scratch_destroy_stack_frame(  io_generated_code,  &l_kernel_config );

  /* Close asm */
  libxsmm_x86_instruction_close_stream_matequation( io_generated_code );
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_register_block_avx_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc) {
  LIBXSMM_UNUSED( io_generated_code );
  LIBXSMM_UNUSED( i_mateqn_desc );

  printf("Inside matequation generator that uses register blocks as tmps..\n");
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_avx_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc) {
  /* TODO: add logic to select code generation strategy of matrix equation  */
  libxsmm_generator_matequation_tmp_stack_scratch_avx_avx512_kernel(io_generated_code, i_mateqn_desc);
}

