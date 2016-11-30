/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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

#include <libxsmm_generator.h>
#include <libxsmm_macros.h>
#include "generator_common.h"
#include "generator_gemm_common.h"
#include "generator_gemm_sse3_avx_avx2_avx512.h"
#include "generator_gemm_imci_avx512.h"
#include "generator_gemm_noarch.h"

/* @TODO change int based architecture value */
void libxsmm_generator_gemm_kernel( libxsmm_generated_code*         io_generated_code,
                                    const libxsmm_gemm_descriptor* i_xgemm_desc,
                                    const char*                     i_arch ) {
  /* apply the alignement override */
  libxsmm_gemm_descriptor l_xgemm_desc_mod = *i_xgemm_desc;
  unsigned int l_vector_length = 1;

  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( io_generated_code, i_arch );

  /* determining vector length depending on architecture and precision */
  /* @TODO fix me */
  if ( (strcmp(i_arch, "wsm") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) == 0 ) {
    l_vector_length = 2;
  } else if ( (strcmp(i_arch, "wsm") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) != 0 ) {
    l_vector_length = 4;
  } else if ( (strcmp(i_arch, "snb") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) == 0 ) {
    l_vector_length = 4;
  } else if ( (strcmp(i_arch, "snb") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) != 0 ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "hsw") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) == 0 ) {
    l_vector_length = 4;
  } else if ( (strcmp(i_arch, "hsw") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) != 0 ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "knc") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) == 0 ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "knc") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) != 0 ) {
    l_vector_length = 16;
  } else if ( (strcmp(i_arch, "knl") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) == 0 ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "knl") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) != 0 ) {
    l_vector_length = 16;
  } else if ( (strcmp(i_arch, "skx") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) == 0 ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "skx") == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & l_xgemm_desc_mod.flags) != 0 ) {
    l_vector_length = 16;
  } else if ( (strcmp(i_arch, "noarch") == 0) ) {
    /* Nothing to do */
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  /* check LDA */
  if ( l_xgemm_desc_mod.lda < l_xgemm_desc_mod.m ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* check LDB */
  if ( l_xgemm_desc_mod.ldb < l_xgemm_desc_mod.k ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_LDB );
    return;
  }

  /* check LDC */
  if ( l_xgemm_desc_mod.ldc < l_xgemm_desc_mod.m ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_LDC );
    return;
  }

  /* check if alignment is not possible */
  if ( 0 != (l_xgemm_desc_mod.lda % l_vector_length) ) {
    l_xgemm_desc_mod.flags &= ~LIBXSMM_GEMM_FLAG_ALIGN_A;
  }
  if ( 0 != (l_xgemm_desc_mod.ldc % l_vector_length) ) {
    l_xgemm_desc_mod.flags &= ~LIBXSMM_GEMM_FLAG_ALIGN_C;
  }

  if ( (strcmp(i_arch, "wsm") == 0) ||
       (strcmp(i_arch, "snb") == 0) ||
       (strcmp(i_arch, "hsw") == 0)    ) {
    /* call actual kernel generation with revised parameters */
    libxsmm_generator_gemm_sse3_avx_avx2_avx512_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch );
  } else if ( (strcmp(i_arch, "knc") == 0) ||
              (strcmp(i_arch, "knl") == 0)    ) {
    /* call actual kernel generation with revised parameters */
    libxsmm_generator_gemm_imci_avx512_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch );
  } else if ( (strcmp(i_arch, "skx") == 0) ) {
    /* call actual kernel generation with revised parameters */
    if ( (l_vector_length == 16  && (l_xgemm_desc_mod.m == 32 || l_xgemm_desc_mod.m == 48 || l_xgemm_desc_mod.m == 64)) ||
         (l_vector_length == 8   && (l_xgemm_desc_mod.m == 16 || l_xgemm_desc_mod.m == 24 || l_xgemm_desc_mod.m == 32))    ) {
      libxsmm_generator_gemm_sse3_avx_avx2_avx512_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch );
    } else {
      libxsmm_generator_gemm_imci_avx512_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch );
    }
  } else if ( (strcmp(i_arch, "noarch") == 0) ) {
    /* call actual kernel generation with revised parameters */
    libxsmm_generator_gemm_noarch_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch );
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( io_generated_code, i_arch );

  /* add flop counter for debug compilation */
  libxsmm_generator_gemm_add_flop_counter( io_generated_code, i_xgemm_desc );
}

void libxsmm_generator_gemm_inlineasm(const char*                     i_file_out,
                                       const char*                     i_routine_name,
                                       const libxsmm_gemm_descriptor* i_xgemm_desc,
                                       const char*                     i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 0;
  l_generated_code.last_error = 0;

#ifdef LIBXSMM_GENERATOR_MKL_BLAS_FALLBACK
  if ( (strcmp(i_arch, "noarch") == 0) ) {
    char l_new_code[512];
    const int l_max_code_length = sizeof(l_new_code) - 1;
    const int l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#include <mkl.h>\n");
    libxsmm_append_code_as_string( &l_generated_code, l_new_code, l_code_length );
  }
#endif
  /* add signature to code string */
  libxsmm_mmfunction_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_gemm_kernel( &l_generated_code, i_xgemm_desc, i_arch );

  /* close current function */
  libxsmm_close_function( &l_generated_code );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    fprintf(stderr, "LIBXSMM ERROR there was an error generating code. Last known error is:\n%s\n",
      libxsmm_strerror(l_generated_code.last_error));
    return;
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "a" );
    if ( l_file_handle != NULL ) {
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR libxsmm_generator_gemm_inlineasm could not write to into destination source file\n");
      return;
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

void libxsmm_generator_gemm_directasm(const char*                     i_file_out,
                                       const char*                     i_routine_name,
                                       const libxsmm_gemm_descriptor* i_xgemm_desc,
                                       const char*                     i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 1;
  l_generated_code.last_error = 0;

  /* check if we are not noarch */
  if ( strcmp( i_arch, "noarch" ) == 0 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_gemm_direct: we cannot create ASM when noarch is specified!\n");
    return;
  }

  /* add signature to code string */
  libxsmm_mmfunction_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_gemm_kernel( &l_generated_code, i_xgemm_desc, i_arch );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    fprintf(stderr, "LIBXSMM ERROR there was an error generating code. Last known error is:\n%s\n",
      libxsmm_strerror(l_generated_code.last_error));
    return;
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "w" );
    if ( l_file_handle != NULL ) {
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_gemm_direct: could not write to into destination source file!\n");
      return;
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

