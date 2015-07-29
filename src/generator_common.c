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

  /* check if end up here accidentally */
  if ( io_generated_code->generate_binary_code != 0 ) {
    fprintf(stderr, "LIBXSMM ERROR libxsmm_append_code_as_string was called although jiting code was requested!" );
    exit(-3);
  }

  /* some safety checks */
  if (io_generated_code->generated_code != NULL) {
    l_length_1 = strlen(io_generated_code->generated_code);
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
    strcpy(l_new_string, io_generated_code->generated_code);
  } else {
    l_new_string[0] = '\0';
  }

  /* append new string */
  strcat(l_new_string, i_code_to_append);

  /* free old memory and overwrite pointer */
  if (l_length_1 > 0)
    free(io_generated_code->generated_code);
  
  io_generated_code->generated_code = l_new_string;

  /* update counters */
  io_generated_code->code_size = (unsigned int)(l_length_1+l_length_2);
  io_generated_code->buffer_size = (io_generated_code->code_size) + 1;
}

void libxsmm_close_function( libxsmm_generated_code* io_generated_code ) {
  if ( io_generated_code->generate_binary_code != 0 )
    return;

  libxsmm_append_code_as_string(io_generated_code, "}\n\n");
}

void libxsmm_get_x86_gp_reg_name( const unsigned int i_gp_reg_number,
                                  char*              i_gp_reg_name ) {
  switch (i_gp_reg_number) {
    case LIBXSMM_X86_GP_REG_RAX: 
      strcpy(i_gp_reg_name, "rax");
      break;
    case LIBXSMM_X86_GP_REG_RCX:
      strcpy(i_gp_reg_name, "rcx");
      break;
    case LIBXSMM_X86_GP_REG_RDX:
      strcpy(i_gp_reg_name, "rdx");
      break;
    case LIBXSMM_X86_GP_REG_RBX:
      strcpy(i_gp_reg_name, "rbx");
      break;
    case LIBXSMM_X86_GP_REG_RSP: 
      strcpy(i_gp_reg_name, "rsp");
      break;
    case LIBXSMM_X86_GP_REG_RBP:
      strcpy(i_gp_reg_name, "rbp");
      break;
    case LIBXSMM_X86_GP_REG_RSI:
      strcpy(i_gp_reg_name, "rsi");
      break;
    case LIBXSMM_X86_GP_REG_RDI:
      strcpy(i_gp_reg_name, "rdi");
      break;
    case LIBXSMM_X86_GP_REG_R8: 
      strcpy(i_gp_reg_name, "r8");
      break;
    case LIBXSMM_X86_GP_REG_R9:
      strcpy(i_gp_reg_name, "r9");
      break;
    case LIBXSMM_X86_GP_REG_R10:
      strcpy(i_gp_reg_name, "r10");
      break;
    case LIBXSMM_X86_GP_REG_R11:
      strcpy(i_gp_reg_name, "r11");
      break;
    case LIBXSMM_X86_GP_REG_R12: 
      strcpy(i_gp_reg_name, "r12");
      break;
    case LIBXSMM_X86_GP_REG_R13:
      strcpy(i_gp_reg_name, "r13");
      break;
    case LIBXSMM_X86_GP_REG_R14:
      strcpy(i_gp_reg_name, "r14");
      break;
    case LIBXSMM_X86_GP_REG_R15:
      strcpy(i_gp_reg_name, "r15");
      break;
    default:
      fprintf(stderr, " LIBXSMM ERROR: libxsmm_get_x86_64_gp_req_name i_gp_reg_number is out of range!\n");
      exit(-1);
  }
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

