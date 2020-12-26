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
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


LIBXSMM_INLINE
void dfill_matrix ( double *matrix, unsigned int ld, unsigned int m, unsigned int n )
{
  unsigned int i, j;
  double dtmp;

  if ( ld < m )
  {
     fprintf(stderr,"Error is dfill_matrix: ld=%u m=%u mismatched!\n",ld,m);
     exit(EXIT_FAILURE);
  }
  for ( j = 1; j <= n; j++ )
  {
     /* Fill through the leading dimension */
     for ( i = 1; i <= ld; i++ )
     {
        dtmp = 1.0 - 2.0*libxsmm_rng_f64();
        matrix [ (j-1)*ld + (i-1) ] = dtmp;
     }
  }
}


LIBXSMM_INLINE
void sfill_matrix ( float *matrix, unsigned int ld, unsigned int m, unsigned int n )
{
  unsigned int i, j;
  double dtmp;

  if ( ld < m )
  {
     fprintf(stderr,"Error is sfill_matrix: ld=%u m=%u mismatched!\n",ld,m);
     exit(EXIT_FAILURE);
  }
  for ( j = 1; j <= n; j++ )
  {
     /* Fill through the leading dimension */
     for ( i = 1; i <= ld; i++ )
     {
        dtmp = 1.0 - 2.0*libxsmm_rng_f64();
        matrix [ (j-1)*ld + (i-1) ] = (float) dtmp;
     }
  }
}


LIBXSMM_INLINE
double residual_stranspose ( float *A, unsigned int lda, unsigned int m, unsigned int n, float *out,
                             unsigned int ld_out, unsigned int *nerrs )
{
  unsigned int i, j;
  double dtmp, derror;

  *nerrs = 0;
  derror = 0.0;
  for ( j = 1; j <= n; j++ )
  {
     for ( i = 1; i <= m; i++ )
     {
         dtmp = A[ (j-1)*lda + (i-1) ] - out [ (i-1)*ld_out + (j-1) ];
         if ( dtmp < 0.0 ) dtmp = -dtmp;
         if ( dtmp > 0.0 )
         {
            *nerrs = *nerrs + 1;
            if ( *nerrs < 10 ) printf("Err #%u: A(%u,%u)=%g B(%u,%u)=%g Diff=%g\n",*nerrs,i,j,A[(j-1)*lda+(i-1)],j,i,out[(i-1)*ld_out+(j-1)],dtmp);
         }
         derror += (double) dtmp;
     }
  }
  return ( derror );
}


LIBXSMM_INLINE
double residual_dtranspose ( double *A, unsigned int lda, unsigned int m, unsigned int n, double *out,
                             unsigned int ld_out, unsigned int *nerrs )
{
  unsigned int i, j;
  double dtmp, derror;
  static int ntimes = 0;

  *nerrs = 0;
  derror = 0.0;
  for ( j = 1; j <= n; j++ )
  {
     for ( i = 1; i <= m; i++ )
     {
         dtmp = A[ (j-1)*lda + (i-1) ] - out [ (i-1)*ld_out + (j-1) ];
         if ( dtmp < 0.0 ) dtmp = -dtmp;
         if ( dtmp > 0.0 ) {
            if ( ++ntimes < 5 ) printf("FP64 Position (%u,%u) is %g and not %g\n",i,j,out [ (i-1)*ld_out + (j-1) ], A[ (j-1)*lda + (i-1) ]);
            *nerrs = *nerrs + 1;
         }
         derror += dtmp;
     }
  }
  return ( derror );
}


/* Comment 1 of the following lines to compare to an ass. code byte-for-byte */
/* #define COMPARE_TO_A_FP32_ASSEMBLY_CODE */
/* #define COMPARE_TO_A_FP64_ASSEMBLY_CODE */

#if defined(COMPARE_TO_A_FP32_ASSEMBLY_CODE) || defined(COMPARE_TO_A_FP64_ASSEMBLY_CODE)
# ifndef COMPARE_TO_AN_ASSEMBLY_CODE
#   define COMPARE_TO_AN_ASSEMBLY_CODE
# endif
#endif
#if defined(COMPARE_TO_A_FP32_ASSEMBLY_CODE) && defined(COMPARE_TO_A_FP64_ASSEMBLY_CODE)
# error Define a comparison to either FP32 or FP64 code, not both at once
#endif

/* Use these lines to dump the real*4 or real*8 assembly files for the kernel */
/*
#define DUMP_FP32_ASSEMBLY_FILE
#define DUMP_FP64_ASSEMBLY_FILE
*/

int main(int argc, char* argv[])
{
  unsigned int m = 16, n = 16, ld_in = 16, ld_out = 16, nerrs;
  const unsigned char* cptr;
  double *dinp, *dout, dtmp;
  float  *sinp, *sout;
#if defined(DUMP_FP32_ASSEMBLY_FILE) || defined(DUMP_FP64_ASSEMBLY_FILE)
  FILE *fp;
  char buffer[80];
  int stop_dumping = 0;
  unsigned int i;
#endif
#ifdef COMPARE_TO_AN_ASSEMBLY_CODE
  unsigned int nbest, istop;
  unsigned char *cptr2;
  extern void myro_();
#endif
  union { libxsmm_xtransfunction f; const void* p; } skernel, dkernel;
  const libxsmm_trans_descriptor* desc = 0;
  libxsmm_descriptor_blob blob;

  if ( argc <= 3 )
  {
     printf("\nUSAGE: %s m n ld_in ld_out\n",argv[0]);
     printf("Out-of-place transpose a mxn matrix of leading dimension ld_in\n");
     printf("Defaults: m=n=ld_in=ld_out=16\n");
     printf("Note: ld_in is NOT needed for dispatching. Code works for any valid (>=m) ld_in\n");
     printf("Note: ld_out is now needed for dispatching. Code will only work for a fixed value, like m and n.\n");
  }
  if ( argc > 1 ) m = atoi(argv[1]);
  if ( argc > 2 ) n = atoi(argv[2]);
  if ( argc > 3 ) ld_in = atoi(argv[3]);
  if ( argc > 4 ) ld_out = atoi(argv[4]);
  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  ld_in = LIBXSMM_MAX(ld_in,m);
  ld_out = LIBXSMM_MAX(ld_out,n);

  printf("This is a tester for JIT transpose kernels! (m=%u n=%u ld_in=%u ld_out=%u)\n",m,n,ld_in,ld_out);

  /* test dispatch call */
  desc = libxsmm_trans_descriptor_init(&blob, sizeof(float), m, n, ld_out);
  skernel.f = libxsmm_dispatch_trans(desc);
  desc = libxsmm_trans_descriptor_init(&blob, sizeof(double), m, n, ld_out);
  dkernel.f = libxsmm_dispatch_trans(desc);

  printf("address of FP32 kernel: %p\n", skernel.p);
  printf("address of FP64 kernel: %p\n", dkernel.p);

#ifndef DUMP_FP64_ASSEMBLY_FILE
  cptr = (const unsigned char*)dkernel.p;
  printf("First few bytes/opcodes: 0x%02x 0x%02x 0x%02x 0x%02x 0x%02x 0x%02x\n",cptr[0],cptr[1],cptr[2],cptr[3],cptr[4],cptr[5]);
#else
  printf("Dumping FP64 assembly file\n");
  cptr = (const unsigned char*)dkernel.p;
  fp = fopen("foo.s","w");
  fputs("\t.text\n",fp);
  fputs("\t.align 256\n",fp);
  fputs("\t.globl trans_\n",fp);
  fputs("trans_:\n",fp);
  i = 0;
  stop_dumping = 0;
  while ( (i < 7000) && (stop_dumping == 0) ) {
     if ( (i >= 0) && (cptr[i  ]==0x5c) && (cptr[i+1]==0x5d) && (cptr[i+2]==0x5b) && (cptr[i+3]==0xc3) ) stop_dumping = 1;
     if ( (i >= 1) && (cptr[i-1]==0x5c) && (cptr[i  ]==0x5d) && (cptr[i+1]==0x5b) && (cptr[i+2]==0xc3) ) stop_dumping = 1;
     if ( (i >= 2) && (cptr[i-2]==0x5c) && (cptr[i-1]==0x5d) && (cptr[i  ]==0x5b) && (cptr[i+1]==0xc3) ) stop_dumping = 1;
     if ( (i >= 3) && (cptr[i-3]==0x5c) && (cptr[i-2]==0x5d) && (cptr[i-1]==0x5b) && (cptr[i  ]==0xc3) ) stop_dumping = 1;

     sprintf(buffer,".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",cptr[i],cptr[i+1],cptr[i+2],cptr[i+3]);
     fputs(buffer,fp);
     i += 4;
  }
  fputs("\tretq\n",fp);
  fputs("\t.type trans_,@function\n",fp);
  fputs("\t.size trans_,.-trans_\n",fp);
  fclose(fp);
  printf("Dumped FP64 %u bytes\n",i);
#endif

#ifdef DUMP_FP32_ASSEMBLY_FILE
  printf("Dumping FP32 assembly file\n");
  cptr = (const unsigned char*)skernel.p;
  fp = fopen("soo.s","w");
  fputs("\t.text\n",fp);
  fputs("\t.align 256\n",fp);
  fputs("\t.globl strans_\n",fp);
  fputs("strans_:\n",fp);
  i = 0;
  stop_dumping = 0;
  while ( (i < 7000) && (stop_dumping == 0) ) {
     if ( (i >= 0) && (cptr[i  ]==0x5c) && (cptr[i+1]==0x5d) && (cptr[i+2]==0x5b) && (cptr[i+3]==0xc3) ) stop_dumping = 1;
     if ( (i >= 1) && (cptr[i-1]==0x5c) && (cptr[i  ]==0x5d) && (cptr[i+1]==0x5b) && (cptr[i+2]==0xc3) ) stop_dumping = 1;
     if ( (i >= 2) && (cptr[i-2]==0x5c) && (cptr[i-1]==0x5d) && (cptr[i  ]==0x5b) && (cptr[i+1]==0xc3) ) stop_dumping = 1;
     if ( (i >= 3) && (cptr[i-3]==0x5c) && (cptr[i-2]==0x5d) && (cptr[i-1]==0x5b) && (cptr[i  ]==0xc3) ) stop_dumping = 1;

     sprintf(buffer,".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",cptr[i],cptr[i+1],cptr[i+2],cptr[i+3]);
     fputs(buffer,fp);
     i += 4;
  }
  fputs("\tretq\n",fp);
  fputs("\t.type strans_,@function\n",fp);
  fputs("\t.size strans_,.-strans_\n",fp);
  fclose(fp);
  printf("Dumped FP32 %u bytes\n",i);
#endif

#ifdef COMPARE_TO_AN_ASSEMBLY_CODE
# ifdef COMPARE_TO_A_FP64_ASSEMBLY_CODE
     cptr = (const unsigned char*)dkernel.p;
# else
     cptr = (const unsigned char*)skernel.p;
# endif
  cptr2 = (unsigned char *) &myro_;
  i = 0;
  nbest = 0;
  istop = 0;
  while ( istop == 0 )
  {
     if ( cptr[i] != cptr2[i] )
     {
        printf("Byte: %u=0x%x differs. We generated: 0x%02x. Should be: 0x%02x\n",
               i,i,cptr[i],cptr2[i]);
     } else {
        ++nbest;
     }
     if ( i >= 208 ) istop = 1;
     if ( i >= 2 )
     {
        if ( (cptr2[i]==0xc3) && (cptr2[i-1]==0x5b) && (cptr2[i-2]==0x5d) )
           istop = 1;
#if 0
        if ( i == 114 ) printf("cptr2=0x%02x 0x%02x 0x%02x istop=%u\n",cptr2[i],cptr2[i-1],cptr2[i-2],istop);
#endif
     }
     ++i;
  }
  printf("Bytes agree: %u\n",nbest);
#endif

  sinp  = (float  *) malloc ( ld_in*n*sizeof(float) );
  dinp  = (double *) malloc ( ld_in*n*sizeof(double) );
  sout = (float  *) malloc ( ld_out*m*sizeof(float) );
  dout = (double *) malloc ( ld_out*m*sizeof(double) );

  /* Fill matrices with random data: */
  sfill_matrix ( sinp, ld_in, m, n );
  dfill_matrix ( dinp, ld_in, m, n );
  sfill_matrix ( sout, ld_out, n, m );
  dfill_matrix ( dout, ld_out, n, m );

/*
  if ( ld_out != n )
  {
     fprintf(stderr,"Final warning: This code only works for ld_out=n (n=%u,ld_out=%u)\n",n,ld_out);
     exit(EXIT_FAILURE);
  }
*/

#ifdef COMPARE_TO_A_FP64_ASSEMBLY_CODE
  printf("Calling myro_: \n");
  myro_ ( dinp, &ld_in, dout, &ld_out );
  dtmp = residual_dtranspose ( dinp, ld_in, m, n, dout, ld_out, &nerrs );
  printf("Myro_ FP64 error: %g number of errors: %u\n",dtmp,nerrs);
  dfill_matrix ( dout, ld_out, n, m );
#endif
#ifdef COMPARE_TO_A_FP32_ASSEMBLY_CODE
  printf("Calling myro_: \n");
  myro_ ( sinp, &ld_in, sout, &ld_out );
  dtmp = residual_stranspose ( sinp, ld_in, m, n, sout, ld_out, &nerrs );
  printf("Myro_ FP32 error: %g number of errors: %u\n",dtmp,nerrs);
  sfill_matrix ( sout, ld_out, n, m );
#endif

  /* let's call */
#if 1
  printf("calling skernel\n");
  skernel.f( sinp, &ld_in, sout, &ld_out );
  printf("calling dkernel\n");
  dkernel.f( dinp, &ld_in, dout, &ld_out );
#endif

  /* Did it transpose correctly? */
  dtmp = residual_stranspose ( sinp, ld_in, m, n, sout, ld_out, &nerrs );
  printf("Single precision m=%u n=%u ld_in=%u ld_out=%u error: %g number of errors: %u",m,n,ld_in,ld_out,dtmp,nerrs);
  if ( nerrs > 0 ) printf(" ->FAILED at %ux%u real*4 case",m,n);
  printf("\n");

  dtmp = residual_dtranspose ( dinp, ld_in, m, n, dout, ld_out, &nerrs );
  printf("Double precision m=%u n=%u ld_in=%u ld_out=%u error: %g number of errors: %u\n",m,n,ld_in,ld_out,dtmp,nerrs);
  if ( nerrs > 0 ) printf(" ->FAILED at %ux%u real*8 case",m,n);
  printf("\n");

  free(dout);
  free(sout);
  free(dinp);
  free(sinp);

  return EXIT_SUCCESS;
}

