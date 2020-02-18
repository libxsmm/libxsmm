/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry, Hans Pabst, Timothy Costa (Intel Corp.)
******************************************************************************/
#include "generator_packed_gemm_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_packed_aux.h"
#include "generator_packed_gemmnn.h"
#include "generator_common.h"
#include "libxsmm_main.h"

#if 0
# define GENERATOR_PACKED_GEMM_DEBUG
#endif


/* TODO: Remove the extra garbage parameters from this calling sequence: */
#define GARBAGE_PARAMETERS

LIBXSMM_API_INTERN
void libxsmm_generator_packed_gemm_avx_avx512_kernel( libxsmm_generated_code*        io_code,
                                                      const libxsmm_pgemm_descriptor* i_packed_pgemm_desc,
                                                      const char*                    i_arch
#ifdef GARBAGE_PARAMETERS
                                                    , unsigned int                   iunroll,
                                                      unsigned int                   junroll,
                                                      unsigned int                   loopi,
                                                      unsigned int                   loopj
#endif
                                                      )
{
  unsigned char *const buf = (unsigned char *) io_code->generated_code;
  libxsmm_loop_label_tracker l_loop_label_tracker /*= { 0 }*/;
  /* avx512 just represents whether we want to use zmm registers or not     *
   *      A value of 0 says not, a value of 1 targets AVX512_CORE, a value  *
   *      of 2 targets AVX512_MIC                                           */
  int avx512;
#if 0 /* TOD: introduce/use register mapping rather than directly/hard-coding registers */
  /* Just reuse transpose gp mapping */
  libxsmm_gemm_gp_reg_mapping l_gp_reg_mapping = { 0/*avoid warning "maybe used uninitialized" */ };
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
     unsigned int m = i_packed_pgemm_desc->m;
     unsigned int n = i_packed_pgemm_desc->n;
     unsigned int k = i_packed_pgemm_desc->k;
     unsigned int lda = i_packed_pgemm_desc->lda;
     unsigned int ldb = i_packed_pgemm_desc->ldb;
     unsigned int ldc = i_packed_pgemm_desc->ldc;
     char transa = i_packed_pgemm_desc->transa;
     char transb = i_packed_pgemm_desc->transb;
     unsigned layout = (unsigned int) i_packed_pgemm_desc->layout;
     unsigned int datasz = (unsigned int)i_packed_pgemm_desc->typesize;
#if 0
     const double alpha = (8 == datasz ? i_packed_pgemm_desc->alpha.d : ((double)i_packed_pgemm_desc->alpha.s));
#else
     double alpha=1.0;
#endif
#if defined(_WIN32) || defined(__CYGWIN__)
     unsigned int areg = LIBXSMM_X86_GP_REG_RCX;
     unsigned int breg = LIBXSMM_X86_GP_REG_RDX;
     unsigned int creg = LIBXSMM_X86_GP_REG_R8;
#else
     unsigned int areg = LIBXSMM_X86_GP_REG_RDI;
     unsigned int breg = LIBXSMM_X86_GP_REG_RSI;
     unsigned int creg = LIBXSMM_X86_GP_REG_RDX;
#endif
     const double beta = 1.0;
     unsigned int m1=m, n1=n, k1=k;
     unsigned int j;
     /*int REGSIZE;*/
     int numb = 0;
     /*int scalealpha = 0;*/
     /*int nounit=0;*/
     int tra, trb, trc;
     char regset = 0;

     if ( i_packed_pgemm_desc->alpha_val == 0 ) {
        alpha = 1.0;
     } else if ( i_packed_pgemm_desc->alpha_val == 1 ) {
        alpha = -1.0;
     } else {
        printf("Warning: libxsmm_generator_packed_gemm_avx_avx512 has unknown alpha, using 1.0\n");
     }
#if defined(GENERATOR_PACKED_GEMM_DEBUG)
printf("Inside libxsmm_generator_packed_gemm_avx_avx512_kernel: transa=%c transb=%c m=%d n=%d k=%d lda=%d ldb=%d ldc=%d alpha=%g beta=%g datasz=%d avx512=%d lay=%d\n",transa,transb,m,n,k,lda,ldb,ldc,alpha,beta,datasz,avx512,layout);
printf("Extra parameters: iunroll=%d junroll=%d loopi=%d loopj=%d\n",iunroll,junroll,loopi,loopj);
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
        return;
     }

#if 0
     if ( LIBXSMM_NEQ(1, alpha) )
     {
        compact_load_parameter_ ( io_code, alpha, 2, numb, regset );
     }
     nounit = ( (diag=='N') || (diag=='n') );
#endif

     if ( transa == 'T' || transa == 't' ) tra = 1; else tra = 0;
     if ( transb == 'T' || transb == 't' ) trb = 1; else trb = 0;
     trc = 0;
     if ( layout == 101 ) { /* Row-major swaps tra/trb/trc */
        if ( tra ) tra = 0; else tra = 1;
        if ( trb ) trb = 0; else trb = 1;
#if !defined(NDEBUG) /* TODO: code protected by !defined(NDEBUG) is logically dead */
        LIBXSMM_ASSERT(0 == trc);
        /* coverity[dead_error_line] */
        if ( trc ) trc = 0; else
#endif
        trc = 1;
     }

     /* Change which registers to use for windows builds */
#if defined(GENERATOR_PACKED_GEMM_DEBUG)
     printf("Using compact_gemmnn header file\n");
#endif
     compact_gemmnn_ ( tra, trb, trc, 1, m1, 1, k1, 1, k1, 1, n1, 1, m1, 1, n1, alpha, areg, lda, breg, ldb, beta, creg, ldc, io_code, numb, regset, iunroll, junroll, loopi, loopj );
#if defined(GENERATOR_PACKED_GEMM_DEBUG)
     printf("Done using compact_gemmnn header file\n");
#endif

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
  for ( i = 0; i < io_code->code_size; i+=8 ) {
    printf("#\tBytes %d-%d\n",i,i+7);
    printf(".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",buf[i],buf[i+1],buf[i+2],buf[i+3],buf[i+4],buf[i+5],buf[i+6],buf[i+7]);
  }
#endif
}

