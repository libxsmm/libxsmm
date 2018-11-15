/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Alexander Heinecke, Greg Henry, Hans Pabst, Timothy Costa (Intel Corp.)
******************************************************************************/
#include "generator_packed_getrf_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_packed_aux.h"
#include "generator_packed_gemmnn.h"
#include "generator_common.h"
#include "libxsmm_main.h"

#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if 0
# define GENERATOR_PACKED_GETRF_DEBUG
#endif


LIBXSMM_API_INTERN
void libxsmm_generator_packed_getrf_avx_avx512_kernel( libxsmm_generated_code*        io_code,
                                                       const libxsmm_trsm_descriptor* i_packed_trsm_desc,
                                                       const char*                    i_arch )
{
  unsigned char *const buf = (unsigned char *) io_code->generated_code;
  libxsmm_loop_label_tracker l_loop_label_tracker /*= { 0 }*/;
  /* avx512 just represents whether we want to use zmm registers or not     *
   *      A value of 0 says not, a value of 1 targets AVX512_CORE, a value  *
   *      of 2 targets AVX512_MIC                                           */
  int avx512;
#if 0 /* TOD: introduce/use register mapping rather than directly/hard-coding registers */
  /* Just reuse transpose gp mapping */
  libxsmm_getrf_gp_reg_mapping l_gp_reg_mapping = { 0/*avoid warning "maybe used uninitialized" */ };
  /* define gp register mapping */
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_R9;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_RSI;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_R9;
#endif
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  /* Actually, the logic is this: we need a, lda, and b. We don't need ldb  *
   * If n>=6, we need rbx                                                   *
   * If n>=8, we need rbp                                                   *
   * If LIBXSMM_MIN(n,REGSIZE)>=5 and m%REGSIZE==1, we need r12             *
   * If LIBXSMM_MIN(n,REGSIZE)>=6 and m%REGSIZE==1, we need r13             *
   * If LIBXSMM_MIN(n,REGSIZE)>=7 and m%REGSIZE==1, we need r14             *
   * If LIBXSMM_MIN(n,REGSIZE)>=8 and m%REGSIZE==1, we need r15             *
   * Otherwise, we get by with registers that don't require pushing/popping */
#endif
  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define transposition kernel config */
  if (strcmp(i_arch, "skx") == 0) {
    avx512 = 1;
  } else if (strcmp(i_arch, "knl") == 0 || strcmp(i_arch, "knm") == 0) {
    avx512 = 2;
  } else if (strcmp(i_arch, "snb") == 0 || strcmp(i_arch, "hsw") == 0) {
    avx512 = 0;
  } else {
    LIBXSMM_HANDLE_ERROR( io_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  /* @Greg add more fields here */

  /* @Greg add generator code here, please use functions defined in generator_x86_instructions.h */
  /* Todo-> I first want this code to work, and verify it works, then I can
   *        convert one instruction at a time to those in
   *        generator_x86_instructions.h. Or add to the existing instructions */

  if ( io_code->code_type > 1 )
  {
     unsigned int i = io_code->code_size;
     unsigned int m = i_packed_trsm_desc->m;
     unsigned int n = i_packed_trsm_desc->n;
     unsigned int lda = i_packed_trsm_desc->lda;
     /*unsigned int ldb = i_packed_trsm_desc->ldb;*/
     /*char trans = i_packed_trsm_desc->transa;*/
#if 0
     char side = i_packed_trsm_desc->side;
     char uplo = i_packed_trsm_desc->uplo;
#endif
     /*char diag = i_packed_trsm_desc->diag;*/
     const unsigned int lay = (unsigned int)i_packed_trsm_desc->layout;
     unsigned int datasz = (unsigned int)i_packed_trsm_desc->typesize;
     const double alpha = (8 == datasz ? i_packed_trsm_desc->alpha.d : ((double)i_packed_trsm_desc->alpha.s));
     /*const double beta = 1.0;*/
     unsigned int m1=m, n1=n, mn;
     unsigned int j, k, ii;
     /*int REGSIZE;*/
     int numb = 0;
     unsigned int bot, /*dis,*/ fincol;
     /*int scalealpha = 0;*/
     /*int nounit=0;*/
     unsigned int /*mb,*/ nb;
     /*int iun, jun;*/
     char regset = 'y';
     double one = 1.0;
     double none = -1.0;

     /* Register mapping: */
     int a0 = 0, a1 = 1, a2 = 2;
     int b0 = 3/*, b1 = 4, b2 = 5, b3*/;
     /*int c00 = 6, c01 = 7, c02 = 8, c03;*/
     /*int c10 = 9, c11 = 10, c12 = 11, c13;*/
     /*int c20 = 12, c21 = 13, c22 = 14, c23;*/
     /*int c30, c31, c32, c33;*/
     /*int c40, c41, c42, c43;*/
     /*int c0, c2, c3, c4;*/

     int onereg = 15;

     if ( lay == 101 )
     {
#if 0
        if (i_packed_trsm_desc->side == 'L' || i_packed_trsm_desc->side == 'l' ) side = 'R';
        else side = 'L';
        if (i_packed_trsm_desc->uplo == 'L' || i_packed_trsm_desc->uplo == 'l' ) uplo = 'U';
        else uplo = 'L';
#endif
        m1 = n; n1 = m;
     }
#if defined(GENERATOR_PACKED_GETRF_DEBUG)
printf("Inside libxsmm_generator_packed_getrf_avx_avx512_kernel: m=%d n=%d lay=%d lda=%d datasz=%d\n",m,n,lay,lda,datasz);
#endif
     if ( ( datasz !=4 ) && (datasz != 8) )
     {
        fprintf(stderr,"Expecting a datasize of 4 or 8 but got %u\n",datasz);
        exit(-1);
     }
     if ( avx512 < 0 || avx512 > 2 )
     {
        fprintf(stderr,"Expecting an integer between 0 and 2 for avx512, got %i\n",avx512);
        exit(-1);
     }
     if ( datasz == 4 && avx512 == 0 )
     {
        numb = 8;
        regset = 'y';
     } else if ( datasz == 8 && avx512 == 0 )
     {
        numb = 4;
        regset = 'y';
     } else if ( datasz == 4 && avx512 > 0 )
     {
        numb = 16;
        regset = 'z';
     } else if ( datasz == 8 && avx512 > 0 )
     {
        numb = 8;
        regset = 'z';
     }

     if ( LIBXSMM_FEQ(0, alpha) )
     {
        compact_set_zero_ ( io_code, 0, numb, datasz, regset );
        for ( j = 1; j <= n1; j++ )
        {
           for ( i = 1; i <= m1; i++ )
           {
              compact_store_matrix1_ ( io_code, lda, i, j, 0, numb, datasz, regset );
           }
        }
        i = io_code->code_size;
        buf[i++] = 0xc3; /* retq */
        io_code->code_size = i;
        return ;
     }

#if 0
     if ( LIBXSMM_NEQ(1, alpha) )
     {
        compact_load_parameter_ ( io_code, alpha, 2, numb, regset );
     }
     nounit = ( (diag=='N') || (diag=='n') );
#endif

     /* Determine ideal blocksizes: */
     nb = 2;
     if ( m1 <= 3 ) nb = 1;
     if ( n1 <= 2 ) nb = 1;
     mn = LIBXSMM_MIN(m1,n1);
     if ( mn >= 6 ) nb = 3;
     /*iun = 3;*/
     /*jun = 3;*/

/* Insert code here */
     compact_set_one_ ( io_code, onereg, numb, datasz, regset );
     for ( ii = 1 ; ii <= mn ; ii += nb ) {
        bot = LIBXSMM_MIN(ii+nb-1,mn);
        /*dis = bot - ii + 1;*/

        for ( j = ii ; j <= bot ; j++ ) {
           for ( i = j+1 ; i <= m1 ; i++ ) {
              if ( i == j+1 ) {
                 compact_load_matrix1_ ( io_code, lda, j, j, a0, numb, datasz, regset );
                 compact_divide_two_nums_ ( io_code, onereg, a0, a0, numb, regset );
              }
              compact_load_matrix1_ ( io_code, lda, i, j, a1, numb, datasz, regset );
              compact_mult_two_nums_ ( io_code, a0, a1, a1, numb, regset );
              fincol = bot;
              if ( i <= bot ) fincol = n1;
              for ( k = j+1 ; k <= fincol; k++ ) {
                 compact_load_matrix1_ ( io_code, lda, i, k, a2, numb, datasz, regset );
                 compact_load_matrix1_ ( io_code, lda, j, k, b0, numb, datasz, regset );
                 compact_fms_cminusab_ ( io_code, a2, a1, b0, numb, regset );
                 compact_store_matrix1_ ( io_code, lda, i, k, a2, numb, datasz, regset );
              }
              compact_store_matrix1_ ( io_code, lda, i, j, a1, numb, datasz, regset );
           }
        }
        if ( (bot < m1) && (bot < n1) ) {
/*
 *       Solve bottom right A22 part with a DGEMM("Notrans","Notrans",m-bot,n-bot,dis,-1.0,A(bot+1,ii),lda,A(ii,bot+1),lda,1.0,A(bot+1,bot+1),lda)
 *       A(bot+1:m,bot+1:n) = A(bot+1:m,bot+1:n) - A(bot+1:m,ii:bot)*A(ii:bot,bot+1:n);
 *       */
           compact_gemmnn_(0,0,bot+1,m1,ii,bot,ii,bot,bot+1,n1,bot+1,m1,bot+1,n1,none,LIBXSMM_X86_GP_REG_RDI,lda,LIBXSMM_X86_GP_REG_RDI,lda,one,LIBXSMM_X86_GP_REG_RDI,lda,io_code,numb,regset,3,3,0,0);
        }      /* Nonempty DGEMM conditional */
     }        /* Main loop for LU */

  }

  { int i = io_code->code_size;
    buf[i++] = 0xc3; /* retq */
    io_code->code_size = i;
  }
  /*  close asm: note that we really didn't need to push everything */
/*
  libxsmm_x86_instruction_close_stream_transpose( io_code, i_arch );
*/

#if 0
#define DEBUG_GIVE_BYTE_CODE_OUTPUT
#endif
#ifdef DEBUG_GIVE_BYTE_CODE_OUTPUT
   buf = (unsigned char *) io_code->generated_code;
   printf("#Final Routine: \n");
   for ( i = 0 ; i < io_code->code_size ; i+=8 )
   {
      printf("#\tBytes %d-%d\n",i,i+7);
      printf(".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",buf[i],buf[i+1],buf[i+2],buf[i+3],buf[i+4],buf[i+5],buf[i+6],buf[i+7]);
   }
#endif

}
