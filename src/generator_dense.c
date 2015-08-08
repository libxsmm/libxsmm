/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
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
#include "generator_dense_common.h"
#include "generator_dense_sse3_avx_avx2.h"
#include "generator_dense_imci_avx512.h"

/* @TODO change int based architecture value */
void libxsmm_generator_dense_kernel( libxsmm_generated_code*         io_generated_code,
                                     const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                     const char*                     i_arch ) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_dense_add_isa_check_header( io_generated_code, i_arch );
  
  /* apply the alignement override */
  libxsmm_xgemm_descriptor l_xgemm_desc_mod = *i_xgemm_desc;
  unsigned int l_vector_length = 1;

  /* determining vector length depending on architecture and precision */
  /* @TODO fix me */
  if ( (strcmp(i_arch, "wsm") == 0) && (l_xgemm_desc_mod.single_precision == 0) ) {
    l_vector_length = 2;
  } else if ( (strcmp(i_arch, "wsm") == 0) && (l_xgemm_desc_mod.single_precision == 1) ) {
    l_vector_length = 4;
  } else if ( (strcmp(i_arch, "snb") == 0) && (l_xgemm_desc_mod.single_precision == 0) ) {
    l_vector_length = 4;
  } else if ( (strcmp(i_arch, "snb") == 0) && (l_xgemm_desc_mod.single_precision == 1) ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "hsw") == 0) && (l_xgemm_desc_mod.single_precision == 0) ) {
    l_vector_length = 4;
  } else if ( (strcmp(i_arch, "hsw") == 0) && (l_xgemm_desc_mod.single_precision == 1) ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "knc") == 0) && (l_xgemm_desc_mod.single_precision == 0) ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "knc") == 0) && (l_xgemm_desc_mod.single_precision == 1) ) {
    l_vector_length = 16;
  } else if ( (strcmp(i_arch, "knl") == 0) && (l_xgemm_desc_mod.single_precision == 0) ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "knl") == 0) && (l_xgemm_desc_mod.single_precision == 1) ) {
    l_vector_length = 16;
  } else if ( (strcmp(i_arch, "skx") == 0) && (l_xgemm_desc_mod.single_precision == 0) ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "skx") == 0) && (l_xgemm_desc_mod.single_precision == 1) ) {
    l_vector_length = 16;
  } else if ( (strcmp(i_arch, "noarch") == 0) ) {
    /* Nothing to do */
  } else {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_kernel: received invalid arch and precision\n");
    exit(-1);
  }
 
  /* derive if alignment is possible */
  if ( (l_xgemm_desc_mod.lda % l_vector_length) == 0 ) {
    l_xgemm_desc_mod.aligned_a = 1;
  } else {
    l_xgemm_desc_mod.aligned_a = 0;
  }
  if ( (l_xgemm_desc_mod.ldc % l_vector_length) == 0 ) {
    l_xgemm_desc_mod.aligned_c = 1;
  } else {
    l_xgemm_desc_mod.aligned_c = 0;
  }

  /* enforce possible external overwrite */
  l_xgemm_desc_mod.aligned_a = l_xgemm_desc_mod.aligned_a && i_xgemm_desc->aligned_a;
  l_xgemm_desc_mod.aligned_c = l_xgemm_desc_mod.aligned_c && i_xgemm_desc->aligned_c;

  if ( (strcmp(i_arch, "wsm") == 0) ||
       (strcmp(i_arch, "snb") == 0) ||
       (strcmp(i_arch, "hsw") == 0)    ) {
    /* call actual kernel generation with revised parameters */
    libxsmm_generator_dense_sse3_avx_avx2_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch );
  } else if ( (strcmp(i_arch, "knc") == 0) || 
       (strcmp(i_arch, "knl") == 0) || 
       (strcmp(i_arch, "skx") == 0)    ) {
    /* call actual kernel generation with revised parameters */
    libxsmm_generator_dense_imci_avx512_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch );
  } else if ( (strcmp(i_arch, "noarch") == 0) ) {
    /* call actual kernel generation with revised parameters */
    libxsmm_generator_dense_noarch_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch ); 
  } else {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_kernel: received invalid arch\n");
    exit(-1);
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_dense_add_isa_check_footer( io_generated_code, i_arch );

  /* add flop counter for debug compilation */
  libxsmm_generator_dense_add_flop_counter( io_generated_code, i_xgemm_desc );
}

void libxsmm_generator_dense_inlineasm(const char*                     i_file_out,
                                       const char*                     i_routine_name,
                                       const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                       const char*                     i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 0;
  
  /* add signature to code string */
  libxsmm_generator_dense_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_dense_kernel( &l_generated_code, i_xgemm_desc, i_arch );

  /* close current function */
  libxsmm_close_function( &l_generated_code );

  /* append code to source file */
  FILE *l_file_handle = fopen( i_file_out, "a" );
  if ( l_file_handle != NULL ) {
    fputs( l_generated_code.generated_code, l_file_handle );
    fclose( l_file_handle );
  } else {
    fprintf(stderr, "LIBXSMM ERROR libxsmm_generator_dense_inlineasm could not write to into destination source file\n");
    exit(-1);
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

void libxsmm_generator_dense_directasm(const char*                     i_file_out,
                                       const char*                     i_routine_name,
                                       const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                       const char*                     i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 1;

  /* check if we are not noarch */
  if ( strcmp( i_arch, "noarch" ) == 0 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_direct: we cannot create ASM when noarch is specified!\n");
    exit(-1);
  }

  /* add signature to code string */
  libxsmm_generator_dense_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_dense_kernel( &l_generated_code, i_xgemm_desc, i_arch );

  /* append code to source file */
  FILE *l_file_handle = fopen( i_file_out, "w" );
  if ( l_file_handle != NULL ) {
    fputs( l_generated_code.generated_code, l_file_handle );
    fclose( l_file_handle );
  } else {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_direct: could not write to into destination source file!\n");
    exit(-1);
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

