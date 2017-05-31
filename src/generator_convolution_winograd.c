/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/

#include <libxsmm_generator.h>
#include <libxsmm_macros.h>
#include "generator_common.h"
#include "generator_convolution_winograd_forward_avx512.h"
#include "generator_convolution_winograd_weight_update_avx512.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

/* @TODO change int based architecture value */
void libxsmm_generator_convolution_winograd_weight_update_kernel( libxsmm_generated_code*                        io_generated_code,
                                                                  const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                                  const char*                                    i_arch ) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( io_generated_code, i_arch );

  if ( (strcmp(i_arch, "knm") == 0) ||
       (strcmp(i_arch, "knl") == 0) ||
       (strcmp(i_arch, "skx") == 0) ) {
    libxsmm_generator_convolution_winograd_weight_update_avx512( io_generated_code, i_conv_desc, i_arch );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( io_generated_code, i_arch );
}

void libxsmm_generator_convolution_winograd_weight_update_inlineasm(const char*                                    i_file_out,
                                                                    const char*                                    i_routine_name,
                                                                    const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                                    const char*                                    i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 0;
  l_generated_code.last_error = 0;

  /* add signature to code string */
  libxsmm_convfunction_signature_fp32( &l_generated_code, i_routine_name );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_convolution_winograd_weight_update_kernel( &l_generated_code, i_conv_desc, i_arch );

  /* close current function */
  libxsmm_close_function( &l_generated_code );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    LIBXSMM_HANDLE_ERROR_VERBOSE( &l_generated_code, l_generated_code.last_error );
    exit(-1);
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "a" );
    if ( l_file_handle != NULL ) {
      assert(0 != l_generated_code.generated_code);
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR libxsmm_generator_gemm_inlineasm could not write to into destination source file\n");
      exit(-1);
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

void libxsmm_generator_convolution_winograd_weight_update_directasm(const char*                                    i_file_out,
                                                                    const char*                                    i_routine_name,
                                                                    const libxsmm_convolution_winograd_descriptor* i_xgemm_desc,
                                                                    const char*                                    i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 1;
  l_generated_code.last_error = 0;

  /* check if we are not noarch */
  if ( strcmp( i_arch, "noarch" ) == 0 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_convolution_winograd_weight_update_directasm: we cannot create ASM when noarch is specified!\n");
    exit(-1);
  }

  /* add signature to code string */
  libxsmm_convfunction_signature_fp32( &l_generated_code, i_routine_name );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_convolution_winograd_weight_update_kernel( &l_generated_code, i_xgemm_desc, i_arch );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    LIBXSMM_HANDLE_ERROR_VERBOSE( &l_generated_code, l_generated_code.last_error );
    exit(-1);
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "w" );
    if ( l_file_handle != NULL ) {
      assert(0 != l_generated_code.generated_code);
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_gemm_direct: could not write to into destination source file!\n");
      exit(-1);
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

/* @TODO change int based architecture value */
void libxsmm_generator_convolution_winograd_forward_kernel( libxsmm_generated_code*                        io_generated_code,
                                                            const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                            const char*                                    i_arch ) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( io_generated_code, i_arch );

  if ( (strcmp(i_arch, "knm") == 0) ||
       (strcmp(i_arch, "knl") == 0) ||
       (strcmp(i_arch, "skx") == 0) ) {
    libxsmm_generator_convolution_winograd_forward_avx512( io_generated_code, i_conv_desc, i_arch );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( io_generated_code, i_arch );
}

void libxsmm_generator_convolution_winograd_forward_inlineasm( const char*                                    i_file_out,
                                                               const char*                                    i_routine_name,
                                                               const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                               const char*                                    i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 0; /* 0 */
  l_generated_code.last_error = 0;

  /* add signature to code string */
  libxsmm_convfunction_signature_fp32( &l_generated_code, i_routine_name );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_convolution_winograd_forward_kernel( &l_generated_code, i_conv_desc, i_arch );

  /* close current function */
  libxsmm_close_function( &l_generated_code );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    LIBXSMM_HANDLE_ERROR_VERBOSE( &l_generated_code, l_generated_code.last_error );
    exit(-1);
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "a" );
    if ( l_file_handle != NULL ) {
      assert(0 != l_generated_code.generated_code);
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR libxsmm_generator_gemm_inlineasm could not write to into destination source file\n");
      exit(-1);
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

void libxsmm_generator_convolution_winograd_forward_directasm( const char*                                    i_file_out,
                                                               const char*                                    i_routine_name,
                                                               const libxsmm_convolution_winograd_descriptor* i_xgemm_desc,
                                                               const char*                                    i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 1; /* 1 */
  l_generated_code.last_error = 0;

  /* check if we are not noarch */
  if ( strcmp( i_arch, "noarch" ) == 0 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_convolution_winograd_forward_directasm: we cannot create ASM when noarch is specified!\n");
    exit(-1);
  }

  /* add signature to code string */
  libxsmm_convfunction_signature_fp32( &l_generated_code, i_routine_name );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_convolution_winograd_forward_kernel( &l_generated_code, i_xgemm_desc, i_arch );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    LIBXSMM_HANDLE_ERROR_VERBOSE( &l_generated_code, l_generated_code.last_error );
    exit(-1);
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "w" );
    if ( l_file_handle != NULL ) {
      assert(0 != l_generated_code.generated_code);
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_gemm_direct: could not write to into destination source file!\n");
      exit(-1);
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}
