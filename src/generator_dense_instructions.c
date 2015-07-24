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

/* TODO change this */
int libxsmm_jit_code = 0;

void libxsmm_instruction_vec_move( char**             io_generated_code, 
                                   const char*        i_vmove_instr, 
                                   const unsigned int i_gp_reg_number,
                                   const int          i_displacement,
                                   const char*        i_vector_name,
                                   const unsigned int i_vec_reg_number_0,
                                   const unsigned int i_is_store ) {
  /* @TODO add checks in debug mode */
  if ( libxsmm_jit_code == 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    l_new_code[0] = '\0';
    char l_gp_reg_name[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name );

    /* build vmovpd/ps/sd/ss instruction, load use */
    if ( i_is_store == 0 ) {
      sprintf(l_new_code, "                       \"%s %i(%%%%%s), %%%%%s%i\\n\\t\"\n", i_vmove_instr, i_displacement, l_gp_reg_name, i_vector_name, i_vec_reg_number_0 );
    } else {
      sprintf(l_new_code, "                       \"%s %%%%%s%i, %i(%%%%%s)\\n\\t\"\n", i_vmove_instr, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_name );
    }
    libxsmm_append_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_vec_compute_reg( char**             io_generated_code, 
                                          const char*        i_vec_instr,
                                          const char*        i_vector_name,                                
                                          const unsigned int i_vec_reg_number_0,
                                          const unsigned int i_vec_reg_number_1,
                                          const unsigned int i_vec_reg_number_2 ) {
  /* @TODO add checks in debug mode */
  if ( libxsmm_jit_code == 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    l_new_code[0] = '\0';

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    sprintf(l_new_code, "                       \"%s %%%%%s%i, %%%%%s%i, %%%%%s%i\\n\\t\"\n", i_vec_instr, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2 );
    libxsmm_append_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_prefetch( char**             io_generated_code,
                                   const char*        i_prefetch_instr, 
                                   const unsigned int i_gp_reg_number,
                                   const int          i_displacement) {
  /* @TODO add checks in debug mode */
  if ( libxsmm_jit_code == 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    l_new_code[0] = '\0';
    char l_gp_reg_name[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name );

    sprintf(l_new_code, "                       \"%s %i(%%%%%s)\\n\\t\"\n", i_prefetch_instr, i_displacement, l_gp_reg_name );
    libxsmm_append_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_alu_imm( char**             io_generated_code,
                                  const char*        i_alu_instr,
                                  const unsigned int i_gp_reg_number,
                                  const unsigned int i_immediate) {
  /* @TODO add checks in debug mode */
  if ( libxsmm_jit_code == 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    l_new_code[0] = '\0';
    char l_gp_reg_name[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name );

    sprintf(l_new_code, "                       \"%s $%i, %%%%%s\\n\\t\"\n", i_alu_instr, i_immediate, l_gp_reg_name );
    libxsmm_append_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_register_jump_label( char**      io_generated_code,
                                              const char* i_jmp_label ) {
  /* @TODO add checks in debug mode */
  if ( libxsmm_jit_code == 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    l_new_code[0] = '\0';

    sprintf(l_new_code, "                       \"%s:\\n\\t\"\n", i_jmp_label );
    libxsmm_append_string( io_generated_code, l_new_code );
  }  
}

void libxsmm_instruction_jump_to_label( char**      io_generated_code,
                                        const char* i_jmp_instr,
                                        const char* i_jmp_label ) {
  /* @TODO add checks in debug mode */
  if ( libxsmm_jit_code == 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    l_new_code[0] = '\0';

    sprintf(l_new_code, "                       \"%s %s\\n\\t\"\n", i_jmp_instr, i_jmp_label );
    libxsmm_append_string( io_generated_code, l_new_code );
  }
}

void libxsmm_generator_dense_sse_avx_open_instruction_stream( char**                        io_generated_code,
                                                              const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                              const char*                   i_prefetch) { 
  /* @TODO add checks in debug mode */
  if ( libxsmm_jit_code == 1 ) {
    /* TODO-Greg: how do we interface here? */
    /* this is start of the xGEMM kernel, the registers are in the variables */
  } else {
    char l_new_code[512];
    char l_gp_reg_name[4];
    
    /* loading b pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_name );
    sprintf( l_new_code, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_string( io_generated_code, l_new_code );

    /* loading a pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_name );
    sprintf( l_new_code, "                       \"movq %%1, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_string( io_generated_code, l_new_code );

    /* loading c pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_name );
    sprintf( l_new_code, "                       \"movq %%2, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_string( io_generated_code, l_new_code );

    /* loading b prefetch pointer in assembly */
    if ( ( strcmp(i_prefetch, "BL2viaC") == 0 ) || 
         ( strcmp(i_prefetch, "BL1viaC") == 0 )    ) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_name );
      sprintf( l_new_code, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_string( io_generated_code, l_new_code );
    /* loading a prefetch pointer in assembly */
    } else if ( ( strcmp(i_prefetch, "AL1viaC") == 0 ) ||
                ( strcmp(i_prefetch, "AL2") == 0 )        ) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_name );
      sprintf( l_new_code, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_string( io_generated_code, l_new_code );
    } else {}
  }

  /* reset loop counters */
  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_mapping->gp_reg_mloop, 0);
  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_mapping->gp_reg_nloop, 0);
  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_mapping->gp_reg_kloop, 0);
}

void libxsmm_generator_dense_sse_avx_close_instruction_stream( char**                        io_generated_code,
                                                               const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                               const char*                   i_prefetch) {
  /* @TODO add checks in debug mode */
  if ( libxsmm_jit_code == 1 ) {
    /* TODO-Greg: how do we interface here? */
  } else {
    char l_new_code[512];
    char l_gp_reg_a[4];
    char l_gp_reg_b[4];
    char l_gp_reg_c[4];
    char l_gp_reg_pre_a[4];
    char l_gp_reg_pre_b[4];
    char l_gp_reg_mloop[4];
    char l_gp_reg_nloop[4];
    char l_gp_reg_kloop[4];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_a );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_b );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_c );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_pre_a );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_pre_b );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_mloop, l_gp_reg_mloop );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_nloop, l_gp_reg_nloop );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_kloop );

    if ( ( strcmp(i_prefetch, "BL2viaC") == 0 ) || 
         ( strcmp(i_prefetch, "BL1viaC") == 0 )    ) {
      sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      libxsmm_append_string( io_generated_code, l_new_code );
    } else if ( ( strcmp(i_prefetch, "AL1viaC") == 0 ) ||
                ( strcmp(i_prefetch, "AL2") == 0 )        ) {
      sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      libxsmm_append_string( io_generated_code, l_new_code );
    } else {
      sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      libxsmm_append_string( io_generated_code, l_new_code );
    }
  }
}

