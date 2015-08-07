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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#include "generator_common.h"

char* libxsmm_empty_string() {
  char* l_string = (char*) malloc( sizeof(char) );
  l_string[0] = '\0';
  return l_string;
}

void libxsmm_append_code_as_string( libxsmm_generated_code* io_generated_code, 
                                    const char*             i_code_to_append ) {
  size_t l_length_1 = 0;
  size_t l_length_2 = 0;
  char* l_new_string = NULL;
  char* current_code = (char*)io_generated_code->generated_code;

  /* check if end up here accidentally */
  if ( io_generated_code->code_type > 1 ) {
    fprintf(stderr, "LIBXSMM ERROR libxsmm_append_code_as_string was called although jiting code was requested!" );
    exit(-3);
  }

  /* some safety checks */
  if (current_code != NULL) {
    l_length_1 = strlen(current_code);
  } else {
    /* nothing to do */
    l_length_1 = 0;
  }
  if (i_code_to_append != NULL) {
    l_length_2 = strlen(i_code_to_append);
  } else {
    fprintf(stderr, "LIBXSMM WARNING libxsmm_append_code_as_string was called with an empty string for appending code" );
  }

  /* allocate new string */
  l_new_string = (char*) malloc( (l_length_1+l_length_2+1)*sizeof(char) );
  if (l_new_string == NULL) {
    fprintf(stderr, "LIBXSMM ERROR libxsmm_append_code_as_string failed to allocate new code string buffer!" );
    exit(-1);
  }

  /* copy old content */
  if (l_length_1 > 0) {
    strcpy(l_new_string, current_code);
  } else {
    l_new_string[0] = '\0';
  }

  /* append new string */
  strcat(l_new_string, i_code_to_append);

  /* free old memory and overwrite pointer */
  if (l_length_1 > 0)
    free(current_code);
  
  io_generated_code->generated_code = (void*)l_new_string;

  /* update counters */
  io_generated_code->code_size = (unsigned int)(l_length_1+l_length_2);
  io_generated_code->buffer_size = (io_generated_code->code_size) + 1;
}

void libxsmm_close_function( libxsmm_generated_code* io_generated_code ) {
  if ( io_generated_code->code_type != 0 )
    return;

  libxsmm_append_code_as_string(io_generated_code, "}\n\n");
}

unsigned int libxsmm_check_x86_gp_reg_name_callee_save( const unsigned int i_gp_reg_number ) {
  if ( (i_gp_reg_number == LIBXSMM_X86_GP_REG_RBX) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_RBP) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_R12) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_R13) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_R14) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_R15)    ) {
    return 1;
  } else {
    return 0;
  }
}

void libxsmm_get_x86_gp_reg_name( const unsigned int i_gp_reg_number,
                                  char*              o_gp_reg_name ) {
  switch (i_gp_reg_number) {
    case LIBXSMM_X86_GP_REG_RAX: 
      strcpy(o_gp_reg_name, "rax");
      break;
    case LIBXSMM_X86_GP_REG_RCX:
      strcpy(o_gp_reg_name, "rcx");
      break;
    case LIBXSMM_X86_GP_REG_RDX:
      strcpy(o_gp_reg_name, "rdx");
      break;
    case LIBXSMM_X86_GP_REG_RBX:
      strcpy(o_gp_reg_name, "rbx");
      break;
    case LIBXSMM_X86_GP_REG_RSP: 
      strcpy(o_gp_reg_name, "rsp");
      break;
    case LIBXSMM_X86_GP_REG_RBP:
      strcpy(o_gp_reg_name, "rbp");
      break;
    case LIBXSMM_X86_GP_REG_RSI:
      strcpy(o_gp_reg_name, "rsi");
      break;
    case LIBXSMM_X86_GP_REG_RDI:
      strcpy(o_gp_reg_name, "rdi");
      break;
    case LIBXSMM_X86_GP_REG_R8: 
      strcpy(o_gp_reg_name, "r8");
      break;
    case LIBXSMM_X86_GP_REG_R9:
      strcpy(o_gp_reg_name, "r9");
      break;
    case LIBXSMM_X86_GP_REG_R10:
      strcpy(o_gp_reg_name, "r10");
      break;
    case LIBXSMM_X86_GP_REG_R11:
      strcpy(o_gp_reg_name, "r11");
      break;
    case LIBXSMM_X86_GP_REG_R12: 
      strcpy(o_gp_reg_name, "r12");
      break;
    case LIBXSMM_X86_GP_REG_R13:
      strcpy(o_gp_reg_name, "r13");
      break;
    case LIBXSMM_X86_GP_REG_R14:
      strcpy(o_gp_reg_name, "r14");
      break;
    case LIBXSMM_X86_GP_REG_R15:
      strcpy(o_gp_reg_name, "r15");
      break;
    default:
      fprintf(stderr, " LIBXSMM ERROR: libxsmm_get_x86_64_gp_req_name i_gp_reg_number is out of range!\n");
      exit(-1);
  }
}

void libxsmm_get_x86_instr_name( const unsigned int i_instr_number,
                                 char*              o_instr_name ) {
  switch (i_instr_number) {
    /* AVX vector moves */
    case LIBXSMM_X86_INSTR_VMOVAPD:
      strcpy(o_instr_name, "vmovapd");
      break;
    case LIBXSMM_X86_INSTR_VMOVUPD:
      strcpy(o_instr_name, "vmovupd");
      break;
    case LIBXSMM_X86_INSTR_VMOVAPS:
      strcpy(o_instr_name, "vmovaps");
      break;
    case LIBXSMM_X86_INSTR_VMOVUPS:
      strcpy(o_instr_name, "vmovups");
      break;
    case LIBXSMM_X86_INSTR_VBROADCASTSD:
      strcpy(o_instr_name, "vbroadcastsd");
      break;
    case LIBXSMM_X86_INSTR_VBROADCASTSS:
      strcpy(o_instr_name, "vbroadcastss");
      break;
    case LIBXSMM_X86_INSTR_VMOVDDUP:
      strcpy(o_instr_name, "vmovddup");
      break;
    case LIBXSMM_X86_INSTR_VMOVSD:
      strcpy(o_instr_name, "vmovsd");
      break;
    case LIBXSMM_X86_INSTR_VMOVSS:
      strcpy(o_instr_name, "vmovss");
      break;
    /* SSE vector moves */
    case LIBXSMM_X86_INSTR_MOVAPD:
      strcpy(o_instr_name, "movapd");
      break;
    case LIBXSMM_X86_INSTR_MOVUPD:
      strcpy(o_instr_name, "movupd");
      break;
    case LIBXSMM_X86_INSTR_MOVAPS:
      strcpy(o_instr_name, "movaps");
      break;
    case LIBXSMM_X86_INSTR_MOVUPS:
      strcpy(o_instr_name, "movups");
      break;
    case LIBXSMM_X86_INSTR_MOVDDUP:
      strcpy(o_instr_name, "movddup");
      break;
    case LIBXSMM_X86_INSTR_MOVSD:
      strcpy(o_instr_name, "movsd");
      break;
    case LIBXSMM_X86_INSTR_MOVSS:
      strcpy(o_instr_name, "movss");
      break;
    case LIBXSMM_X86_INSTR_SHUFPS:
      strcpy(o_instr_name, "shufps");
      break;
    /* IMCI special */
    case LIBXSMM_X86_INSTR_VLOADUNPACKLPD:
      strcpy(o_instr_name, "vloadunpacklpd");
      break;
    case LIBXSMM_X86_INSTR_VLOADUNPACKHPD:
      strcpy(o_instr_name, "vloadunpackhpd");
      break;
    case LIBXSMM_X86_INSTR_VLOADUNPACKLPS:
      strcpy(o_instr_name, "vloadunpacklps");
      break;
    case LIBXSMM_X86_INSTR_VLOADUNPACKHPS:
      strcpy(o_instr_name, "vloadunpackhps");
      break;
    case LIBXSMM_X86_INSTR_VPACKSTORELPD:
      strcpy(o_instr_name, "vpackstorelpd");
      break;
    case LIBXSMM_X86_INSTR_VPACKSTOREHPD:
      strcpy(o_instr_name, "vpackstorehpd");
      break;
    case LIBXSMM_X86_INSTR_VPACKSTORELPS:
      strcpy(o_instr_name, "vpackstorelps");
      break;
    case LIBXSMM_X86_INSTR_VPACKSTOREHPS:
      strcpy(o_instr_name, "vpackstorehps");
      break;
    case LIBXSMM_X86_INSTR_VPREFETCH1:
      strcpy(o_instr_name, "vprefetch1");
      break;
    case LIBXSMM_X86_INSTR_VPREFETCH0:
      strcpy(o_instr_name, "vprefetch0");
      break;
    /* AVX double precision */
    case LIBXSMM_X86_INSTR_VXORPD:
      strcpy(o_instr_name, "vxorpd");
      break;
    case LIBXSMM_X86_INSTR_VMULPD:
      strcpy(o_instr_name, "vmulpd");
      break;
    case LIBXSMM_X86_INSTR_VADDPD:
      strcpy(o_instr_name, "vaddpd");
      break;
    case LIBXSMM_X86_INSTR_VFMADD231PD:
      strcpy(o_instr_name, "vfmadd231pd");
      break;
    case LIBXSMM_X86_INSTR_VMULSD:
      strcpy(o_instr_name, "vmulsd");
      break;
    case LIBXSMM_X86_INSTR_VADDSD:
      strcpy(o_instr_name, "vaddsd");
      break;
    case LIBXSMM_X86_INSTR_VFMADD231SD:
      strcpy(o_instr_name, "vfmadd231sd");
      break;
    /* AVX single precision */
    case LIBXSMM_X86_INSTR_VXORPS:
      strcpy(o_instr_name, "vxorps");
      break;
    case LIBXSMM_X86_INSTR_VMULPS:
      strcpy(o_instr_name, "vmulps");
      break;
    case LIBXSMM_X86_INSTR_VADDPS:
      strcpy(o_instr_name, "vaddps");
      break;
    case LIBXSMM_X86_INSTR_VFMADD231PS:
      strcpy(o_instr_name, "vfmadd231ps");
      break;
    case LIBXSMM_X86_INSTR_VMULSS:
      strcpy(o_instr_name, "vmulss");
      break;
    case LIBXSMM_X86_INSTR_VADDSS:
      strcpy(o_instr_name, "vaddss");
      break;
    case LIBXSMM_X86_INSTR_VFMADD231SS:
      strcpy(o_instr_name, "vfmadd231ss");
      break;
    /* SSE double precision */
    case LIBXSMM_X86_INSTR_XORPD:
      strcpy(o_instr_name, "xorpd");
      break;
    case LIBXSMM_X86_INSTR_MULPD:
      strcpy(o_instr_name, "mulpd");
      break;
    case LIBXSMM_X86_INSTR_ADDPD:
      strcpy(o_instr_name, "addpd");
      break;
    case LIBXSMM_X86_INSTR_MULSD:
      strcpy(o_instr_name, "mulsd");
      break;
    case LIBXSMM_X86_INSTR_ADDSD:
      strcpy(o_instr_name, "addsd");
      break;
    /* SSE single precision */
    case LIBXSMM_X86_INSTR_XORPS:
      strcpy(o_instr_name, "xorps");
      break;
    case LIBXSMM_X86_INSTR_MULPS:
      strcpy(o_instr_name, "mulps");
      break;
    case LIBXSMM_X86_INSTR_ADDPS:
      strcpy(o_instr_name, "addps");
      break;
    case LIBXSMM_X86_INSTR_MULSS:
      strcpy(o_instr_name, "mulss");
      break;
    case LIBXSMM_X86_INSTR_ADDSS:
      strcpy(o_instr_name, "addss");
      break;
    /* GP instructions */
    case LIBXSMM_X86_INSTR_ADDQ:
      strcpy(o_instr_name, "addq");
      break;
    case LIBXSMM_X86_INSTR_SUBQ:
      strcpy(o_instr_name, "subq");
      break;
    case LIBXSMM_X86_INSTR_MOVQ:
      strcpy(o_instr_name, "movq");
      break;
    case LIBXSMM_X86_INSTR_CMPQ:
      strcpy(o_instr_name, "cmpq");
      break;
    case LIBXSMM_X86_INSTR_JL:
      strcpy(o_instr_name, "jl");
      break;
    case LIBXSMM_X86_INSTR_PREFETCH1: 
      strcpy(o_instr_name, "prefetch1");
      break;
    case LIBXSMM_X86_INSTR_KMOV: 
      strcpy(o_instr_name, "kmov");
      break;
    case LIBXSMM_X86_INSTR_KMOVW: 
      strcpy(o_instr_name, "kmovw");
      break;
    /* default, we didn't had a match */
    default:
      fprintf(stderr, " LIBXSMM ERROR: libxsmm_get_x86_64_instr_name i_instr_number (%i) is out of range!\n", i_instr_number);
      exit(-1);
  }
}

unsigned int libxsmm_is_x86_vec_instr_single_precision( const unsigned int i_instr_number ) {
  unsigned int l_return = 0;
  
  switch (i_instr_number) {
    case LIBXSMM_X86_INSTR_VMOVAPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMOVUPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMOVAPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VMOVUPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VBROADCASTSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VBROADCASTSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VMOVDDUP:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMOVSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMOVSS:
      l_return = 1;
      break;
    /* SSE vector moves */
    case LIBXSMM_X86_INSTR_MOVAPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MOVUPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MOVAPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_MOVUPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_MOVDDUP:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MOVSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MOVSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_SHUFPS:
      l_return = 1;
      break;
    /* IMCI special */
    case LIBXSMM_X86_INSTR_VLOADUNPACKLPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VLOADUNPACKHPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VLOADUNPACKLPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VLOADUNPACKHPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VPACKSTORELPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VPACKSTOREHPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VPACKSTORELPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VPACKSTOREHPS:
      l_return = 1;
      break;
    /* AVX double precision */
    case LIBXSMM_X86_INSTR_VXORPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMULPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VADDPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VFMADD231PD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMULSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VADDSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VFMADD231SD:
      l_return = 0;
      break;
    /* AVX single precision */
    case LIBXSMM_X86_INSTR_VXORPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VMULPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VADDPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VFMADD231PS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VMULSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VADDSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VFMADD231SS:
      l_return = 1;
      break;
    /* SSE double precision */
    case LIBXSMM_X86_INSTR_XORPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MULPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_ADDPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MULSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_ADDSD:
      l_return = 0;
      break;
    /* SSE single precision */
    case LIBXSMM_X86_INSTR_XORPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_MULPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_ADDPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_MULSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_ADDSS:
      l_return = 1;
      break;
    /* default, we didn't had a match */
    default:
      fprintf(stderr, " LIBXSMM ERROR: libxsmm_is_x86_vec_instr_single_precision i_instr_number (%i) is not a x86 FP vector instruction!\n", i_instr_number);
      exit(-1);
  }

  return l_return;
}


void libxsmm_reset_x86_gp_reg_mapping( libxsmm_gp_reg_mapping* i_gp_reg_mapping ) {
  i_gp_reg_mapping->gp_reg_a = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_b = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_c = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_mloop = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_nloop = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_kloop = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  i_gp_reg_mapping->gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
}

