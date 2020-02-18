/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_common.h"
#include "generator_gemm_common.h"
#include "generator_gemm_sse3_avx_avx2_avx512.h"
#include "generator_gemm_noarch.h"
#include "libxsmm_main.h"

LIBXSMM_API
void libxsmm_generator_gemm_kernel( libxsmm_generated_code*        io_generated_code,
                                    const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  /* apply the alignment override */
  libxsmm_gemm_descriptor l_xgemm_desc_mod = *i_xgemm_desc;
  unsigned int l_vector_length = 1;

  /* determining vector length depending on architecture and precision */
  /* @TODO fix me */
  if ( io_generated_code->arch <= LIBXSMM_X86_GENERIC ) {
    /* nothing todo */
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_SSE4 ) && LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 2;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_SSE4 ) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_AVX2 ) && LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_AVX2 ) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 8;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) && LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 8;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 16;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) && ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) &&
              ( io_generated_code->arch != LIBXSMM_X86_AVX512_MIC ) && ( LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    l_vector_length = 16;
    /* some checks as we cannot mask everything */
    if ( (l_xgemm_desc_mod.k % 8 != 0) && (io_generated_code->arch == LIBXSMM_X86_AVX512_KNM) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    } else if (l_xgemm_desc_mod.k % 2 != 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    l_xgemm_desc_mod.k = l_xgemm_desc_mod.k/2;
    l_xgemm_desc_mod.ldb = l_xgemm_desc_mod.ldb/2;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE ) && ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    l_vector_length = 16;
    /* some checks as we cannot mask everything */
    if ( (l_xgemm_desc_mod.k % 4 != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    l_xgemm_desc_mod.k = l_xgemm_desc_mod.k/4;
    l_xgemm_desc_mod.ldb = l_xgemm_desc_mod.ldb/4;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE ) && LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 16;
    /* some checks as we cannot mask everything */
    if ( (l_xgemm_desc_mod.k % 2 != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    l_xgemm_desc_mod.k = l_xgemm_desc_mod.k/2;
    l_xgemm_desc_mod.ldb = l_xgemm_desc_mod.ldb/2;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  /* check LDA */
  if ( l_xgemm_desc_mod.lda < l_xgemm_desc_mod.m ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* check LDB */
  if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    if ( l_xgemm_desc_mod.ldb < l_xgemm_desc_mod.n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB_TRANS );
      return;
    }
  } else {
    if ( l_xgemm_desc_mod.ldb < l_xgemm_desc_mod.k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
  }

  /* check LDC */
  if ( l_xgemm_desc_mod.ldc < l_xgemm_desc_mod.m ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
    return;
  }

  /* check for trans B cases which are not supported in the generator */
  if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    if ( (LIBXSMM_GEMM_PRECISION_I16  == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) ||
         (LIBXSMM_GEMM_PRECISION_I8   == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) ||
         (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ))    ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
      return;
    } else {
      /* we are fine, we have transpose support */
    }
  }

  /* check for VNNI flag being set in case of low precision GEMM */
  if ( ( LIBXSMM_GEMM_PRECISION_I16  == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ||
       ( LIBXSMM_GEMM_PRECISION_I8   == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ||
       ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) )    ) {
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      return;
    }
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
      return;
    }
  }

  /* check if alignment is not possible */
  if ( 0 != (l_xgemm_desc_mod.lda % l_vector_length) ) {
    l_xgemm_desc_mod.flags &= ~LIBXSMM_GEMM_FLAG_ALIGN_A;
  }
  if ( 0 != (l_xgemm_desc_mod.ldc % l_vector_length) ) {
    l_xgemm_desc_mod.flags &= ~LIBXSMM_GEMM_FLAG_ALIGN_C;
  }

  if ( io_generated_code->arch <= LIBXSMM_X86_GENERIC ) {
    /* call actual kernel generation with revised parameters */
    libxsmm_generator_gemm_noarch_kernel( io_generated_code, &l_xgemm_desc_mod );
  } else if ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) {
    /* call actual kernel generation with revised parameters */
    libxsmm_generator_gemm_sse3_avx_avx2_avx512_kernel( io_generated_code, &l_xgemm_desc_mod );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}


LIBXSMM_API
void libxsmm_generator_gemm_inlineasm( const char*                    i_file_out,
                                       const char*                    i_routine_name,
                                       const libxsmm_gemm_descriptor* i_xgemm_desc,
                                       const char*                    i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 0;
  l_generated_code.last_error = 0;
  l_generated_code.arch = 0;
  l_generated_code.sf_size = 0;

  /* set arch */
  if ( strcmp(i_arch, "wsm") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_SSE4;
  } else if ( strcmp(i_arch, "snb") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX;
  } else if ( strcmp(i_arch, "hsw") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX2;
  } else if ( strcmp(i_arch, "knl") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_MIC;
  } else if ( strcmp(i_arch, "knm") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_KNM;
  } else if ( strcmp(i_arch, "skx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CORE;
  } else if ( strcmp(i_arch, "clx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CLX;
  } else if ( strcmp(i_arch, "cpx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CPX;
  } else {
    l_generated_code.arch = LIBXSMM_X86_GENERIC;
  }

  /* add signature to code string */
  libxsmm_mmfunction_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( &l_generated_code );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_gemm_kernel( &l_generated_code, i_xgemm_desc );

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( &l_generated_code );

  /* add flop counter for debug compilation */
  libxsmm_generator_gemm_add_flop_counter( &l_generated_code, i_xgemm_desc );

  /* close current function */
  libxsmm_close_function( &l_generated_code );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    LIBXSMM_HANDLE_ERROR_VERBOSE( &l_generated_code, l_generated_code.last_error );
    return;
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "a" );
    if ( l_file_handle != NULL ) {
      assert(l_generated_code.generated_code != NULL);
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


LIBXSMM_API
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
  l_generated_code.arch = 0;
  l_generated_code.sf_size = 0;

  /* set arch */
  if ( strcmp(i_arch, "wsm") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_SSE4;
  } else if ( strcmp(i_arch, "snb") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX;
  } else if ( strcmp(i_arch, "hsw") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX2;
  } else if ( strcmp(i_arch, "knl") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_MIC;
  } else if ( strcmp(i_arch, "knm") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_KNM;
  } else if ( strcmp(i_arch, "skx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CORE;
  } else if ( strcmp(i_arch, "clx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CLX;
  } else if ( strcmp(i_arch, "cpx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CPX;
  } else {
    l_generated_code.arch = LIBXSMM_X86_GENERIC;
  }

  /* check if we are not noarch */
  if ( strcmp( i_arch, "noarch" ) == 0 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_gemm_direct: we cannot create ASM when noarch is specified!\n");
    return;
  }

  /* add signature to code string */
  libxsmm_mmfunction_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_gemm_kernel( &l_generated_code, i_xgemm_desc );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    LIBXSMM_HANDLE_ERROR_VERBOSE( &l_generated_code, l_generated_code.last_error );
    return;
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "w" );
    if ( l_file_handle != NULL ) {
      assert(l_generated_code.generated_code != NULL);
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

