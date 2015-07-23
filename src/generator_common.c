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

char* libxsmm_empty_string() {
  char* l_string = (char*) malloc( sizeof(char) );
  l_string[0] = '\0';
  return l_string;
}

int libxsmm_append_string(char** io_string_1, const char* i_string_2) {
  size_t l_length_1 = 0;
  size_t l_length_2 = 0;
  char* l_new_string = NULL;

  /* some safety checks */
  if (*io_string_1 != NULL) {
    l_length_1 = strlen(*io_string_1);
  } else {
    /* nothing to do */
    l_length_1 = 0;
  }
  if (i_string_2 != NULL) {
    l_length_2 = strlen(i_string_2);
  } else {
    /* nothing to do */
    return -2;
  }

  /* allocate new string */
  l_new_string = (char*) malloc( (l_length_1+l_length_2+1)*sizeof(char) );
  if (l_new_string == NULL)
    return -1;

  /* copy old content */
  if (l_length_1 > 0) {
    strcpy(l_new_string, *io_string_1);
  } else {
    l_new_string[0] = '\0';
  }

  /* append new string */
  strcat(l_new_string, i_string_2);

  /* free old memory and overwrite pointer */
  if (l_length_1 > 0)
    free(*io_string_1);
  
  *io_string_1 = l_new_string;

  return (int)(l_length_1+l_length_2); /* no error */
}

void libxsmm_close_function(char** io_generated_code) {
  libxsmm_append_string(io_generated_code, "}\n\n");
}

void libxsmm_get_x86_64_gp_reg_name( const unsigned int i_gp_reg_number,
                                     char*              i_gp_reg_name ) {
  switch (i_gp_reg_number) {
    case 0: 
      strcpy(i_gp_reg_name, "rax");
      break;
    case 1:
      strcpy(i_gp_reg_name, "rcx");
      break;
    case 2:
      strcpy(i_gp_reg_name, "rdx");
      break;
    case 3:
      strcpy(i_gp_reg_name, "rbx");
      break;
    case 4: 
      strcpy(i_gp_reg_name, "rsp");
      break;
    case 5:
      strcpy(i_gp_reg_name, "rbp");
      break;
    case 6:
      strcpy(i_gp_reg_name, "rsi");
      break;
    case 7:
      strcpy(i_gp_reg_name, "rdi");
      break;
    case 8: 
      strcpy(i_gp_reg_name, "r8");
      break;
    case 9:
      strcpy(i_gp_reg_name, "r9");
      break;
    case 10:
      strcpy(i_gp_reg_name, "r10");
      break;
    case 11:
      strcpy(i_gp_reg_name, "r11");
      break;
    case 12: 
      strcpy(i_gp_reg_name, "r12");
      break;
    case 13:
      strcpy(i_gp_reg_name, "r13");
      break;
    case 14:
      strcpy(i_gp_reg_name, "r14");
      break;
    case 15:
      strcpy(i_gp_reg_name, "r15");
      break;
    default:
      fprintf(stderr, " LIBXSMM ERROR: libxsmm_get_x86_64_gp_req_name i_gp_reg_number is out of range!\n");
      exit(-1);
  }
}

