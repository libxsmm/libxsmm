/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_COMMON_H
#define GENERATOR_COMMON_H

#include "libxsmm_generator.h"

/* defining register mappings */
#define LIBXSMM_X86_GP_REG_RAX        0
#define LIBXSMM_X86_GP_REG_RCX        1
#define LIBXSMM_X86_GP_REG_RDX        2
#define LIBXSMM_X86_GP_REG_RBX        3
#define LIBXSMM_X86_GP_REG_RSP        4
#define LIBXSMM_X86_GP_REG_RBP        5
#define LIBXSMM_X86_GP_REG_RSI        6
#define LIBXSMM_X86_GP_REG_RDI        7
#define LIBXSMM_X86_GP_REG_R8         8
#define LIBXSMM_X86_GP_REG_R9         9
#define LIBXSMM_X86_GP_REG_R10       10
#define LIBXSMM_X86_GP_REG_R11       11
#define LIBXSMM_X86_GP_REG_R12       12
#define LIBXSMM_X86_GP_REG_R13       13
#define LIBXSMM_X86_GP_REG_R14       14
#define LIBXSMM_X86_GP_REG_R15       15
#define LIBXSMM_X86_GP_REG_UNDEF    127

#define LIBXSMM_X86_SSE3           1000
#define LIBXSMM_X86_AVX            1001
#define LIBXSMM_X86_AVX2           1002
#define LIBXSMM_X86_AVX512         1003

#define LIBXSMM_X86_MOVAPD        10000
#define LIBXSMM_X86_MOVUPD        10001
#define LIBXSMM_X86_MOVAPS        10002
#define LIBXSMM_X86_MOVUPS        10003
#define LIBXSMM_X86_BCASTSD       10004
#define LIBXSMM_X86_SHUFFPS       10005
#define LIBXSMM_X86_MOVSD         10006
#define LIBXSMM_X86_MOVSS         10007

/* micro kernel config */
typedef struct libxsmm_micro_kernel_config_struct {
  unsigned int instruction_set;
  unsigned int vector_length;
  unsigned int datatype_size;
  char         vector_name[16];
  char         a_vmove_instruction[16];
  char         b_vmove_instruction[16];
  char         b_shuff_instruction[16];
  char         c_vmove_instruction[16];
  unsigned int use_masking_a_c;
  char         prefetch_instruction[16];
  char         vxor_instruction[16];
  char         vmul_instruction[16];
  char         vadd_instruction[16];
  char         alu_add_instruction[16];
  char         alu_sub_instruction[16];
  char         alu_cmp_instruction[16];
  char         alu_jmp_instruction[16];
  char         alu_mov_instruction[16];
} libxsmm_micro_kernel_config; 

/* struct for storing the current gp reg mapping */
typedef struct libxsmm_gp_reg_mapping_struct {
  unsigned int gp_reg_a;
  unsigned int gp_reg_b;
  unsigned int gp_reg_c;
  unsigned int gp_reg_a_prefetch;
  unsigned int gp_reg_b_prefetch;
  unsigned int gp_reg_mloop;
  unsigned int gp_reg_nloop;
  unsigned int gp_reg_kloop;
  unsigned int gp_reg_help_0;
  unsigned int gp_reg_help_1;
  unsigned int gp_reg_help_2;
  unsigned int gp_reg_help_3;
  unsigned int gp_reg_help_4;
  unsigned int gp_reg_help_5;
} libxsmm_gp_reg_mapping;

void libxsmm_get_x86_gp_reg_name( const unsigned int i_gp_reg_number,
                                  char*              i_gp_reg_name ); 

void libxsmm_reset_x86_gp_reg_mapping( libxsmm_gp_reg_mapping* i_gp_reg_mapping );

/* some string manipulation helper needed to 
   generated code */
char* libxsmm_empty_string();

void libxsmm_append_code_as_string( libxsmm_generated_code* io_generated_code, 
                                    const char*             i_code_to_append );

void libxsmm_close_function( libxsmm_generated_code* io_generated_code );

#endif /* GENERATOR_COMMON_H */
