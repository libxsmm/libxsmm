/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry, Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_generator.h>
#include "generator_packed_getrf_avx_avx512.h"
#include "generator_packed_trsm_avx_avx512.h"
#include "generator_packed_trmm_avx_avx512.h"
#include "generator_packed_gemm_avx_avx512.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdarg.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API
void libxsmm_generator_pgemm_kernel( libxsmm_generated_code*          io_generated_code,
                                     const libxsmm_pgemm_descriptor*  i_packed_pgemm_desc,
                                     int                              i_arch, ... ) {
  const char *const cpuid = libxsmm_cpuid_name( i_arch );

  /* generate kernel */
  if ( LIBXSMM_X86_AVX <= i_arch ) {
#if defined(GARBAGE_PARAMETERS)
    unsigned int iunroll, junroll, loopi, loopj;
    va_list args;
    va_start(args, i_arch);
    iunroll = va_arg(args, unsigned int);
    junroll = va_arg(args, unsigned int);
    loopi = va_arg(args, unsigned int);
    loopj = va_arg(args, unsigned int);
    va_end(args);
    libxsmm_generator_packed_gemm_avx_avx512_kernel( io_generated_code, i_packed_pgemm_desc, cpuid, iunroll, junroll, loopi, loopj );
#else
    libxsmm_generator_packed_gemm_avx_avx512_kernel( io_generated_code, i_packed_pgemm_desc, cpuid );
#endif
  } else { /* TODO fix this error */
    LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_ARCH);
    return;
  }
}


LIBXSMM_API
void libxsmm_generator_getrf_kernel( libxsmm_generated_code*          io_generated_code,
                                     const libxsmm_getrf_descriptor*  i_packed_getrf_desc,
                                     int                              i_arch ) {
  const char *const cpuid = libxsmm_cpuid_name( i_arch );

  /* generate kernel */
  if ( LIBXSMM_X86_AVX <= i_arch ) {
    libxsmm_generator_packed_getrf_avx_avx512_kernel( io_generated_code, i_packed_getrf_desc, cpuid );
  } else { /* TODO fix this error */
    LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_ARCH);
    return;
  }
}


/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_trsm_kernel( libxsmm_generated_code*         io_generated_code,
                                    const libxsmm_trsm_descriptor*  i_packed_trsm_desc,
                                    const char*                     i_arch ) {
  /* generate kernel */
  if ( (strcmp(i_arch, "skx") == 0) ||
       (strcmp(i_arch, "knm") == 0) ||
       (strcmp(i_arch, "knl") == 0) ||
       (strcmp(i_arch, "hsw") == 0) ||
       (strcmp(i_arch, "snb") == 0)    ) {
    libxsmm_generator_packed_trsm_avx_avx512_kernel( io_generated_code, i_packed_trsm_desc, i_arch );
  } else {
    /* TODO fix this error */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}


/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_trmm_kernel(libxsmm_generated_code*         io_generated_code,
                                   const libxsmm_trmm_descriptor*  i_packed_trmm_desc,
                                   const char*                     i_arch) {
  /* generate kernel */
  if ( (strcmp(i_arch, "skx") == 0) ||
       (strcmp(i_arch, "knm") == 0) ||
       (strcmp(i_arch, "knl") == 0) ||
       (strcmp(i_arch, "hsw") == 0) ||
       (strcmp(i_arch, "snb") == 0)    ) {
    libxsmm_generator_packed_trmm_avx_avx512_kernel( io_generated_code, i_packed_trmm_desc, i_arch );
  } else {
    /* TODO fix this error */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}

