/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
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

#include <libxsmm_generator.h>
#include "generator_common.h"
#include "generator_matcopy_avx_avx512.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_matcopy_kernel( libxsmm_generated_code*                        io_generated_code,
                                       const libxsmm_matcopy_descriptor*              i_matcopy_desc,
                                       const char*                                    i_arch ) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( io_generated_code, i_arch );

  /* generate kernel */
  if ( (strcmp(i_arch, "knl") == 0) ||
       (strcmp(i_arch, "knm") == 0) ||
       (strcmp(i_arch, "skx") == 0) ||
       (strcmp(i_arch, "hsw") == 0) ||
       (strcmp(i_arch, "snb") == 0)    ) {
    libxsmm_generator_matcopy_avx_avx512_kernel( io_generated_code, i_matcopy_desc, i_arch );
  } else {
    /* TODO fix this error */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( io_generated_code, i_arch );
}

