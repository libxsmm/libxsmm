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

/* defining register mappings */
#define LIBXSMM_X86_GP_REG_RAX     0
#define LIBXSMM_X86_GP_REG_RCX     1
#define LIBXSMM_X86_GP_REG_RDX     2
#define LIBXSMM_X86_GP_REG_RBX     3
#define LIBXSMM_X86_GP_REG_RSP     4
#define LIBXSMM_X86_GP_REG_RBP     5
#define LIBXSMM_X86_GP_REG_RSI     6
#define LIBXSMM_X86_GP_REG_RDI     7
#define LIBXSMM_X86_GP_REG_R8      8
#define LIBXSMM_X86_GP_REG_R9      9
#define LIBXSMM_X86_GP_REG_R10    10
#define LIBXSMM_X86_GP_REG_R11    11
#define LIBXSMM_X86_GP_REG_R12    12
#define LIBXSMM_X86_GP_REG_R13    13
#define LIBXSMM_X86_GP_REG_R14    14
#define LIBXSMM_X86_GP_REG_R15    15
#define LIBXSMM_X86_GP_REG_UNDEF 127

/* micro kernel config */
/*struct libxsmm_micro_kernel_config {
  char* 
*/

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

char* libxsmm_empty_string();

int libxsmm_append_string( char** io_string_1, const char* i_string_2 );

void libxsmm_close_function( char** io_generated_code );

void libxsmm_get_x86_gp_reg_name( const unsigned int i_gp_reg_number,
                                  char*              i_gp_reg_name ); 

void libxsmm_reset_x86_gp_reg_mapping( libxsmm_gp_reg_mapping* i_gp_reg_mapping );

#endif /* GENERATOR_COMMON_H */
