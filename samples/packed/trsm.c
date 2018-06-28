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
/* Alexander Heinecke, Greg Henry (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define BUFSIZE 32*32
#define BUFSIZE2 64000

#if 0
#define TRIANGLE_IS_IDENTITY
#endif

LIBXSMM_INLINE
void dcopy_to_temp ( int layout, double *A, int lda, int m, int n, double *Atemp,
                     unsigned int VLEN )
{
    int i, j;

    if ( lda*n > BUFSIZE )
    {
       printf("Reference routine not set up for matrices so large\n");
       exit(-1);
    }
    if ( layout == 102 )
    {
       /* printf("Column major\n"); */
       for ( j = 0 ; j < n ; j++ )
       {
          for ( i = 0 ; i < m ; i++ )
          {
             Atemp[i+j*m] = A[i*VLEN+j*lda*VLEN];
          }
       }
#if EVENTUALLY_USE_THIS_LOOP_IT_SHOULD_BE_FASTER
       for ( j = 0 ; j < n ; j++ )
       {
          for ( i = 0, ia = 0 ; i < m ; i++, ia+=VLEN )
          {
             Atemp[i+j*m] = A[ ia+j*lda*VLEN ];
          }
       }
#endif
    } else {
       /* printf("Row major\n"); */
       for ( j = 0 ; j < n ; j++ )
       {
          for ( i = 0 ; i < m ; i++ )
          {
             /* Transpose the data */
             Atemp[i+j*m] = A[j*VLEN+i*lda*VLEN];
          }
       }
    }
}

LIBXSMM_INLINE
void scopy_to_temp ( int layout, float *A, int lda, int m, int n, float *Atemp,
                     unsigned int VLEN )
{
    int i, j;

    if ( lda*n > BUFSIZE )
    {
       printf("Reference routine not set up for matrices so large\n");
       exit(-1);
    }
    if ( layout == 102 )
    {
       /* printf("Column major\n"); */
       for ( j = 0 ; j < n ; j++ )
       {
          for ( i = 0 ; i < m ; i++ )
          {
             Atemp[i+j*m] = A[i*VLEN+j*lda*VLEN];
          }
       }
    } else {
       /* printf("Row major\n"); */
       for ( j = 0 ; j < n ; j++ )
       {
          for ( i = 0 ; i < m ; i++ )
          {
             /* Transpose the data */
             Atemp[i+j*m] = A[j*VLEN+i*lda*VLEN];
          }
       }
    }
}

LIBXSMM_INLINE
void dcopy_from_temp ( int layout, double *A, int lda, int m, int n, double *Atemp,
                       unsigned int VLEN )
{
    int i, j, ia;

    if ( lda*n > BUFSIZE )
    {
       printf("Reference routine not set up for matrices so large\n");
    }
    if ( layout == 102 )
    {
       for ( j = 0 ; j < n ; j++ )
       {
          for ( i = 0, ia = 0 ; i < m ; i++, ia+=VLEN )
          {
             A[ia+j*lda*VLEN] = Atemp[i+j*m];
          }
       }
    } else {
       for ( j = 0 ; j < n ; j++ )
       {
          for ( i = 0 ; i < m ; i++ )
          {
             /* Transpose the data */
             A[j*VLEN+i*lda*VLEN] = Atemp[i+j*m];
          }
       }
    }
}

LIBXSMM_INLINE
void scopy_from_temp ( int layout, float *A, int lda, int m, int n, float *Atemp,
                       unsigned int VLEN )
{
    int i, j, ia;

    if ( lda*n > BUFSIZE )
    {
       printf("Reference routine not set up for matrices so large\n");
    }
    if ( layout == 102 )
    {
       for ( j = 0 ; j < n ; j++ )
       {
          for ( i = 0, ia = 0 ; i < m ; i++, ia+=VLEN )
          {
             A[ia+j*lda*VLEN] = Atemp[i+j*m];
          }
       }
    } else {
       for ( j = 0 ; j < n ; j++ )
       {
          for ( i = 0 ; i < m ; i++ )
          {
             /* Transpose the data */
             A[j*VLEN+i*lda*VLEN] = Atemp[i+j*m];
          }
       }
    }
}

#if !defined(USE_MKL_FOR_REFERENCE) && !defined(LIBXSMM_NOFORTRAN)
extern void dtrsm_();

/* Reference code for compact dtrsm. Note that this just copies data into
   a buffer from the compact storage and calls the regular dtrsm code. This
   is very naive reference code just used for testing purposes */
/* Note: if layout==101 (row major), then this code is known to only work when
 *        nmat == VLEN. To check for accuracy otherwise, transpose everything */
LIBXSMM_INLINE
void compact_dtrsm_ ( unsigned int *layout, char *side, char *uplo,
                      char *transa, char *diag, unsigned int *m,
                      unsigned int *n, double *alpha, double *A,
                      unsigned int *lda, double *B, unsigned int *ldb,
                      unsigned int *nmat, unsigned int *VLEN )
{
    int i, j, num, asize, offseta, offsetb;
    double *Ap, *Bp, Atemp[BUFSIZE], Btemp[BUFSIZE];
    static int ntimes = 0;

    if ( ++ntimes < 3 ) printf("Inside reference compact_dtrsm_()\n");
    if ( *layout == 102 )
    {
       if ( (*side == 'L') || (*side == 'l') ) asize = *m;
       else asize = *n;
       offsetb = (*ldb)*(*n)*(*VLEN);
    } else {
       if ( (*side == 'L') || (*side == 'l') ) asize = *n;
       else asize = *m;
       offsetb = (*ldb)*(*m)*(*VLEN);
    }
    offseta = (*lda)*asize*(*VLEN);
    if ( ++ntimes < 3 ) printf("m/n=%u,%u layout=%u asize=%i VLEN=%u nmat=%u offseta=%i offsetb=%i\n",*m,*n,*layout, asize, *VLEN, *nmat, offseta, offsetb );
    for ( i = 0, num = 0 ; i < (int)(*nmat) ; i+= *VLEN, num++ )
    {
       for ( j = 0 ; j < (int)*VLEN ; j++ )
       {
           /* Unpack the data, call a reference DTRSM, repack the data */
           Ap = &A[j+num*offseta];
           Bp = &B[j+num*offsetb];
if (++ntimes < 15 ) printf("Doing a dtrsm at place i=%d j=%d num=%d Ap[%d]=%g Bp[%d]=%g\n",i,j,num,j+num*offseta,Ap[0],j+num*offsetb,Bp[0]);
           dcopy_to_temp ( *layout, Ap, *lda, asize, asize, Atemp, *VLEN );
           dcopy_to_temp ( *layout, Bp, *ldb, *m, *n, Btemp, *VLEN );
           dtrsm_ ( side, uplo, transa, diag, m, n, alpha, Atemp, &asize, Btemp, m);
           dcopy_from_temp ( *layout, Bp, *ldb, *m, *n, Btemp, *VLEN );
       }
    }
}

#endif /*!defined(USE_MKL_FOR_REFERENCE) && !defined(LIBXSMM_NOFORTRAN)*/

extern void strsm_();

/* Reference code for compact strsm. Note that this just copies data into
   a buffer from the compact storage and calls the regular strsm code. This
   is very naive reference code just used for testing purposes */
/* Note: if layout==101 (row major), then this code is known to only work when
 *        nmat == VLEN. To check for accuracy otherwise, transpose everything */
LIBXSMM_INLINE
void compact_strsm_ ( unsigned int *layout, char *side, char *uplo,
                      char *transa, char *diag, unsigned int *m,
                      unsigned int *n, float *alpha, float *A,
                      unsigned int *lda, float *B, unsigned int *ldb,
                      unsigned int *nmat, unsigned int *VLEN )
{
    int i, j, num, asize;
    float *Ap, *Bp, Atemp[BUFSIZE], Btemp[BUFSIZE];

    if ( (*side == 'L') || (*side == 'l') ) asize = *m;
    else asize = *n;
    for ( i = 0, num = 0 ; i < (int)(*nmat) ; i+= *VLEN, num++ )
    {
       for ( j = 0 ; j < (int)*VLEN ; j++ )
       {
           /* Unpack the data, call a reference DTRSM, repack the data */
           Ap = &A[j+num*(*lda)*asize*(*VLEN)];
           Bp = &B[j+num*(*ldb)*(*n)*(*VLEN)];
           scopy_to_temp ( *layout, Ap, *lda, asize, asize, Atemp, *VLEN );
           scopy_to_temp ( *layout, Bp, *ldb, *m, *n, Btemp, *VLEN );
           strsm_ ( side, uplo, transa, diag, m, n, alpha, Atemp, &asize, Btemp, m);
           scopy_from_temp ( *layout, Bp, *ldb, *m, *n, Btemp, *VLEN );
       }
    }
}

LIBXSMM_INLINE
void dfill_matrix ( double *matrix, unsigned int ld, unsigned int m, unsigned int n )
{
  unsigned int i, j;
  double dtmp;

  if ( ld < m )
  {
     fprintf(stderr,"Error is dfill_matrix: ld=%u m=%u mismatched!\n",ld,m);
     exit(-1);
  }
  for ( j = 1 ; j <= n ; j++ )
  {
     /* Fill through the leading dimension */
     for ( i = 1; i <= ld; i++ )
     {
        dtmp = 1.0 - 2.0*libxsmm_rand_f64();
        matrix [ (j-1)*ld + (i-1) ] = dtmp;
     }
  }
}

LIBXSMM_INLINE
void dfill_identity ( double *matrix, unsigned int ld, unsigned int m, unsigned int n, int VLEN, int number_of_cases )
{
  unsigned int h, i, j, k, ia;
  double dtmp;

  if ( ld < m ) {
     fprintf(stderr,"Error is dfill_identity: ld=%u m=%u mismatched!\n",ld,m);
     exit(-1);
  }
  for ( h = 0; h < (unsigned int)number_of_cases ; h++ ) {
     ia = h*ld*n*VLEN;
     for ( j = 1 ; j <= n ; j++ ) {
        for ( i = 1 ; i <= ld; i++ ) {
           if ( i == j ) dtmp = 1.0; else dtmp = 0.0;
           for ( k = 0 ; k < (unsigned int)VLEN ; k++ ) matrix[ia++] = dtmp;
        }
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
     exit(-1);
  }
  for ( j = 1 ; j <= n ; j++ )
  {
     /* Fill through the leading dimension */
     for ( i = 1; i <= ld; i++ )
     {
        dtmp = 1.0 - 2.0*libxsmm_rand_f64();
        matrix [ (j-1)*ld + (i-1) ] = (float) dtmp;
     }
  }
}

LIBXSMM_INLINE
double residual_d ( double *A, unsigned int lda, unsigned int m, unsigned int n,
                    double *B, unsigned int ldb, unsigned int *nerrs,
                    unsigned int *ncorr )
{
   unsigned int i, j;
   double atmp, btmp, dtmp, ref, derror;
   static int ntimes = 0;

   *nerrs = 0;
   *ncorr = 0;
   derror = 0.0;
   for ( j = 1 ; j<= n; j++ )
   {
      for ( i = 1; i <= m; i++ )
      {
         atmp = A[ (j-1)*lda + (i-1)];
         btmp = B[ (j-1)*ldb + (i-1)];
         ref  = LIBXSMM_MAX(atmp,-atmp);
         if ( atmp >= btmp ) {
             dtmp = atmp - btmp;
         } else {
             dtmp = btmp - atmp;
         }
         if ( isnan(dtmp) || isinf(dtmp) )
         {
             if ( ++ntimes < 15 )
             {
                printf("Denormal bug: A(%u,%u) is %g B(%u,%u) is %g\n",i,j,atmp,i,j,btmp);
             }
         }
         if ( dtmp / ref > 1.0e-12 )
         {
             *nerrs = *nerrs + 1;
             if ( ++ntimes < 15 )
             {
                printf("Bug #%i: A[%u]=A(%u,%u) expected=%g instead=%g err=%g\n",ntimes,(j-1)*lda+(i-1),i,j,atmp,btmp,dtmp);
             }
         } else {
             if ( (*nerrs > 0) && (ntimes < 10) && (*ncorr < 40) )
             {
                printf("Cor #%u: A[%u]=A(%u,%u) expected=%g\n",*ncorr+1,(j-1)*lda+(i-1),i,j,atmp);
             }
             *ncorr = *ncorr + 1;
         }
         derror += dtmp;
      }
   }
   return ( derror );
}

LIBXSMM_INLINE
double residual_s ( float *A, unsigned int lda, unsigned int m, unsigned int n,
                    float *B, unsigned int ldb, unsigned int *nerrs,
                    unsigned int *ncorr )
{
   unsigned int i, j;
   double atmp, btmp, dtmp, ref, derror;
   static int ntimes = 0;

   *nerrs = 0;
   *ncorr = 0;
   derror = 0.0;
   for ( j = 1 ; j<= n; j++ )
   {
      for ( i = 1; i <= m; i++ )
      {
         atmp = (double) A[ (j-1)*lda + (i-1)];
         btmp = (double) B[ (j-1)*ldb + (i-1)];
         ref  = LIBXSMM_MAX(atmp,-atmp);
         if ( atmp >= btmp ) {
             dtmp = atmp - btmp;
         } else {
             dtmp = btmp - atmp;
         }
         if ( isnan(dtmp) || isinf(dtmp) )
         {
             if ( ++ntimes < 15 )
             {
                printf("Denormal bug: A(%u,%u) is %g B(%u,%u) is %g\n",i,j,atmp,i,j,btmp);
             }
         }
         if ( dtmp / ref > 1.0e-4 )
         {
             *nerrs = *nerrs + 1;
             if ( ++ntimes < 15 )
             {
                printf("Bug #%d: A(%u,%u) expected=%g instead=%g err=%g\n",ntimes,i,j,atmp,btmp,dtmp);
             }
         } else {
             if ( (*nerrs > 0) && (ntimes < 10) && (*ncorr < 40) )
             {
                printf("Cor #%u: A(%u,%u) expected=%g\n",*ncorr+1,i,j,atmp);
             }
             *ncorr = *ncorr + 1;
         }
         derror += dtmp;
      }
   }
   return ( derror );
}

#if !defined(USE_PREDEFINED_ASSEMBLY) && !defined(USE_XSMM_GENERATED) && !defined(USE_KERNEL_GENERATION_DIRECTLY) && !defined(TIME_MKL)
  #define USE_XSMM_GENERATED
#endif

#if 0
  #define USE_PREDEFINED_ASSEMBLY
  #define USE_XSMM_GENERATED
  #define USE_KERNEL_GENERATION_DIRECTLY
  #define TIME_MKL
#endif

#ifdef USE_PREDEFINED_ASSEMBLY
extern void trsm_xct_();
#endif
#ifdef MKL_TIMER
extern double dsecnd_();
#endif

int main(int argc, char* argv[])
{
  unsigned int m=8, n=8, lda=8, ldb=8, nerrs, num, nmat, nmats, nmatd, ntest;
  unsigned int layout, asize, VLEND=4, VLENS=8, bsize;
  unsigned int ncorr;
  int i, j;
  char side, uplo, trans, diag;
  unsigned int typesize8 = 8;
  unsigned int typesize4 = 4;
  float  *sa, *sb, *sc, *sd;
  double *da, *db, *dc, *dd, *tmpbuf;
  double dalpha = 1.0;
  float  salpha;
  double dtmp;
  const unsigned char *cptr;
  unsigned long op_count;
  const libxsmm_trsm_descriptor* desc8 = NULL;
  const libxsmm_trsm_descriptor* desc4 = NULL;
  libxsmm_descriptor_blob blob;
  union {
    libxsmm_xtrsmfunction dp;
    libxsmm_xtrsmfunction sp;
    const void* pv;
  } mykernel = { 0 };
#ifdef USE_KERNEL_GENERATION_DIRECTLY
  void (*opcode_routine)();
#endif
#ifdef USE_KERNEL_GENERATION_DIRECTLY
  #include <unistd.h>
  #include <signal.h>
  #include <malloc.h>
  #include <sys/mman.h>
  #include "../../src/generator_packed_trsm_avx_avx512.h"
  unsigned char *routine_output;
  libxsmm_generated_code io_generated_code;
  int pagesize = sysconf(_SC_PAGE_SIZE);
  if (pagesize == -1) fprintf(stderr,"sysconf pagesize\n");
  routine_output = (unsigned char *) mmap(NULL,
                      BUFSIZE2, PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANONYMOUS, 0,0);
  if (mprotect(routine_output, BUFSIZE2,
                PROT_EXEC | PROT_READ | PROT_WRITE ) == -1)
      fprintf(stderr,"mprotect\n");
  printf("Routine ready\n");
  io_generated_code.generated_code = &routine_output[0];
  io_generated_code.buffer_size = BUFSIZE2;
  io_generated_code.code_size = 0;
  io_generated_code.code_type = 2;
  io_generated_code.last_error = 0;
#endif

  if ( argc <= 3 )
  {
     printf("\nUSAGE: %s m n lda ldb nmat side uplo trans diag layout ntest alpha\n",argv[0]);
     printf("Compact TRSM a mxn matrix of leading dimension ldb\n");
     printf("This will test the jit of 1 VLEN work of nmat at a time\n");
     printf("Defaults: m=n=lda=ldb=nmat=8, alpha=1.0, side=uplo='L',trans=diag='N',layout=102,ntest=1\n");
  }
  if ( argc > 1 ) m = atoi(argv[1]); else m = 8;
  if ( argc > 2 ) n = atoi(argv[2]); else n = 8;
  if ( argc > 3 ) lda= atoi(argv[3]); else lda = 8;
  if ( argc > 4 ) ldb = atoi(argv[4]); else ldb = 8;
  if ( argc > 5 ) nmat = atoi(argv[5]); else nmat = 8;
  if ( argc > 6 ) side = argv[6][0]; else side = 'L';
  if ( argc > 7 ) uplo = argv[7][0]; else uplo = 'L';
  if ( argc > 8 ) trans = argv[8][0]; else trans = 'N';
  if ( argc > 9 ) diag = argv[9][0]; else diag = 'N';
  if ( argc > 10 ) layout = atoi(argv[10]); else layout=102;
  if ( argc > 11 ) ntest = atoi(argv[11]); else ntest = 1;
  if ( argc > 12 ) dalpha = atof(argv[12]); else dalpha = 1.0;
  salpha = (float)dalpha;

  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  /* A is either mxm or nxn depending on side */
  if ( (side == 'L') || (side=='l') ) asize = m; else asize = n;

  lda = LIBXSMM_MAX(lda,asize);
  if ( layout == 102 )
  {
      /* Column major: B is mxn, and stored in B format */
      ldb = LIBXSMM_MAX(ldb,m);
      bsize = ldb*n;
  } else {
      /* Row major: B is mxn, and stored in B^T format */
      ldb = LIBXSMM_MAX(ldb,n);
      bsize = ldb*m;
  }
  nmats = LIBXSMM_MAX(VLENS,nmat - (nmat%VLENS));
  nmatd = LIBXSMM_MAX(VLEND,nmat - (nmat%VLEND));
  nmat = LIBXSMM_MAX(nmats,nmatd);

  op_count = n * m * asize;

  printf("This is a real*%u tester for JIT compact TRSM kernels! (%c%c%c%c m=%u n=%u lda=%u ldb=%u layout=%u nmat=%u)\n",typesize8,side,uplo,trans,diag,m,n,lda,ldb,layout,nmat);
#ifdef USE_XSMM_GENERATED
  printf("This code tests the LIBXSMM generated kernels\n");
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
  printf("This code tests some predefined assembly kenrel\n");
#endif
#ifdef USE_KERNEL_GENERATION_DIRECTLY
  printf("This code tests kernel generation directly\n");
#endif
#ifdef TIME_MKL
  printf("This code tests MKL compact batch directly\n");
#endif

  desc8 = libxsmm_trsm_descriptor_init(&blob, typesize8, m, n, lda, ldb, &dalpha, trans, diag, side, uplo, layout);
  desc4 = libxsmm_trsm_descriptor_init(&blob, typesize4, m, n, lda, ldb, &salpha, trans, diag, side, uplo, layout);
#ifdef USE_XSMM_GENERATED
  printf("calling libxsmm_dispatch_trsm: typesize8=%u\n",typesize8);
  mykernel.dp = libxsmm_dispatch_trsm(desc8);
  printf("done calling libxsmm_dispatch_trsm: typesize8=%u\n",typesize8);
  if ( mykernel.dp == NULL ) printf("R8 Kernel after the create call is null\n");
  mykernel.sp = libxsmm_dispatch_trsm(desc4);
  if ( mykernel.sp == NULL ) printf("R4 kernel after the create call is null\n");
#endif

#ifdef USE_KERNEL_GENERATION_DIRECTLY
  libxsmm_generator_trsm_kernel ( &io_generated_code, &desc8, "hsw" );
#endif

#ifndef NO_ACCURACY_CHECK
  printf("mallocing matrices\n");
#endif
  sa  = (float  *) malloc ( lda*asize*nmats*sizeof(float) );
  da  = (double *) malloc ( lda*asize*nmatd*sizeof(double) );
  sb  = (float  *) malloc ( bsize*nmats*sizeof(float) );
  db  = (double *) malloc ( bsize*nmatd*sizeof(double) );
  sc  = (float  *) malloc ( bsize*nmats*sizeof(float) );
  dc  = (double *) malloc ( bsize*nmatd*sizeof(double) );
  sd  = (float  *) malloc ( bsize*nmats*sizeof(float) );
  dd  = (double *) malloc ( bsize*nmatd*sizeof(double) );
  tmpbuf = (double *) malloc ( asize*VLEND*sizeof(double) );

#ifndef NO_ACCURACY_CHECK
  printf("filling matrices\n");
#endif
  sfill_matrix ( sa, lda, asize, asize*nmats );
#ifdef TRIANGLE_IS_IDENTITY
  printf("Warning: setting triangular matrix to identity. Not good for accuracy testing\n");
  dfill_identity ( da, lda, asize, asize, VLEND, nmatd/VLEND );
#else
  dfill_matrix ( da, lda, asize, asize*nmatd );
#endif
  sfill_matrix ( sb, bsize, bsize, nmats );
  dfill_matrix ( db, bsize, bsize, nmatd );

#ifndef NO_ACCURACY_CHECK
  for ( i = 0 ; i < (int)(bsize*nmats) ; i++ ) sc[i]=sb[i];
  for ( i = 0 ; i < (int)(bsize*nmatd) ; i++ ) dc[i]=db[i];
  for ( i = 0 ; i < (int)(bsize*nmats) ; i++ ) sd[i]=sb[i];
  for ( i = 0 ; i < (int)(bsize*nmatd) ; i++ ) dd[i]=db[i];
  printf("Pointing at the kernel now\n");
#endif

#ifdef USE_XSMM_GENERATED
  cptr = (const unsigned char*) mykernel.pv;
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
  cptr = (const unsigned char*) trsm_xct_;
#endif
#ifdef USE_KERNEL_GENERATION_DIRECTLY
  cptr = (const unsigned char*) &routine_output[0];
  opcode_routine = (void *) &cptr[0];
#endif

#ifndef TIME_MKL
  #define DUMP_ASSEMBLY_FILE
#endif

#ifdef DUMP_ASSEMBLY_FILE
  printf("Dumping assembly file\n");
  FILE *fp = fopen("foo.s","w");
  char buffer[80];
  fputs("\t.text\n",fp);
  fputs("\t.align 256\n",fp);
  fputs("\t.globl trsm_xct_\n",fp);
  fputs("trsm_xct_:\n",fp);
  for (i = 0 ; i < 4000; i+=4 )
  {
     sprintf(buffer,".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",cptr[i],cptr[i+1],cptr[i+2],cptr[i+3]);
     fputs(buffer,fp);
  }
  fputs("\tretq\n",fp);
  fputs("\t.type trsm_xct_,@function\n",fp);
  fputs("\t.size trsm_xct_,.-trsm_xct_\n",fp);
  fclose(fp);
#endif

#if defined(USE_MKL_FOR_REFERENCE) || defined(TIME_MKL)
  #include "mkl.h"
  MKL_LAYOUT CLAYOUT = (layout == 101) ? MKL_ROW_MAJOR : MKL_COL_MAJOR;
  MKL_SIDE SIDE = (side == 'R' || side == 'r') ? MKL_RIGHT : MKL_LEFT;
  MKL_UPLO UPLO = (uplo == 'U' || uplo == 'u') ? MKL_UPPER : MKL_LOWER;
  MKL_TRANSPOSE TRANSA = (trans == 'N' || trans == 'n') ? MKL_NOTRANS : MKL_TRANS;
  MKL_DIAG DIAG = (diag == 'N' || diag == 'n') ? MKL_NONUNIT : MKL_UNIT;
  MKL_COMPACT_PACK CMP_FORMAT = mkl_get_format_compact();
#if 0
  MKL_COMPACT_PACK CMP_FORMAT = MKL_COMPACT_AVX;
#endif
#endif

#ifndef NO_ACCURACY_CHECK
  printf("Before routine, initial B(1,1)=%g B[256]=%g\n",db[0],db[256]);
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
  double one = 1.0;
#endif
  double timer;
#ifdef MKL_TIMER
  double tmptimer;
  tmptimer = dsecnd_();
#else
  unsigned long long l_start, l_end;
#endif

  timer = 0.0;
  for ( j = 0 ; j < (int)ntest ; j++ )
  {
#ifndef TRIANGLE_IS_IDENTITY
  for ( i = 0 ; i < (int)(bsize*nmatd) ; i++ ) db[i]=dd[i];
#endif
  for ( i = 0 , num = 0; i < (int)nmatd ; i+= (int)VLEND, num++ )
  {
     double *Ap = &da[num*lda*asize*VLEND];
     double *Bp = &db[num*bsize*VLEND];
#ifdef MKL_TIMER
     tmptimer = dsecnd_();
#else
     l_start = libxsmm_timer_tick();
#endif

#ifdef USE_XSMM_GENERATED
     mykernel.dp ( Ap, Bp, tmpbuf );
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
     trsm_xct_ ( Ap, Bp, &one );
#endif
#ifdef USE_KERNEL_GENERATION_DIRECTLY
     (*opcode_routine)( Ap, Bp );
#endif
#ifdef TIME_MKL
     mkl_dtrsm_compact ( CLAYOUT, SIDE, UPLO, TRANSA, DIAG, m, n, dalpha, da, lda, db, ldb, CMP_FORMAT, nmatd );
     i+=nmatd; /* Because MKL will do everything */
#endif
#ifdef MKL_TIMER
     timer += dsecnd_() - tmptimer;
#else
     l_end = libxsmm_timer_tick();
     timer += libxsmm_timer_duration(l_start,l_end);
#endif
  }
  }
  timer /= ((double)ntest);

#ifndef NO_ACCURACY_CHECK
  printf("Average time to get through %u matrices: %g\n",nmatd,timer);
  printf("Gflops: %g\n",(double)(op_count*nmatd)/(timer*1.0e9));
  printf("after routine, new      B(1,1)=%g B[256]=%g\n",db[0],db[256]);
#endif

#ifdef TEST_SINGLE
  printf("Before r4 routine, initial B(1,1)=%g B[256]=%g\n",sb[0],sb[256]);
  for ( i = 0 , num = 0; i < nmats ; i+= VLENS, num++ )
  {
     float *Ap = &sa[num*lda*asize*VLENS];
     float *Bp = &sb[num*bsize*VLENS];
#ifdef USE_XSMM_GENERATED
     mykernel.sp ( Ap, Bp, NULL );
#endif
  }
  printf("after r4 routine, new      B(1,1)=%g B]256]=%g\n",db[0],db[256]);
#endif

#ifndef NO_ACCURACY_CHECK
  /* Call some reference code now on a copy of the B matrix (C) */
  double timer2 = 0.0;
  for ( j = 0 ; j < (int)ntest ; j++ )
  {
#ifndef TRIANGLE_IS_IDENTITY
  for ( i = 0 ; i < (int)(bsize*nmatd) ; i++ ) dc[i]=dd[i];
#endif

#ifdef MKL_TIMER
  tmptimer = dsecnd_();
#else
  l_start = libxsmm_timer_tick();
#endif

#ifdef USE_MKL_FOR_REFERENCE
  mkl_dtrsm_compact ( CLAYOUT, SIDE, UPLO, TRANSA, DIAG, m, n, dalpha, da, lda, dc, ldb, CMP_FORMAT, nmatd );
#elif !defined(LIBXSMM_NOFORTRAN)
  if ( (layout == 101) && (nmatd!=VLEND) )
  {
     unsigned int lay = 102, m1 = n, n1 = m;
     char side1='L', uplo1='L';
     if ( side == 'L' || side == 'l' ) side1 = 'R';
     if ( uplo == 'L' || uplo == 'l' ) uplo1 = 'U';
     compact_dtrsm_ ( &lay, &side1, &uplo1, &trans, &diag, &m1, &n1, &dalpha, da, &lda, dc, &ldb, &nmatd, &VLEND );
  } else {
     compact_dtrsm_ ( &layout, &side, &uplo, &trans, &diag, &m, &n, &dalpha, da, &lda, dc, &ldb, &nmatd, &VLEND );
  }
#endif

#ifdef MKL_TIMER
  timer2 += dsecnd_() - tmptimer;
#else
  l_end = libxsmm_timer_tick();
  timer2 += libxsmm_timer_duration(l_start,l_end);
#endif

  }
  timer2 /= ((double)ntest);
  printf("Reference time=%g Reference Gflops=%g\n",timer2,(op_count*nmatd)/(timer2*1.0e9));

  /* Compute the residual between B and C */
  dtmp = residual_d ( dc, bsize, bsize, nmatd, db, bsize, &nerrs, &ncorr );
  printf("R8 %c%c%c%c m=%u n=%u lda=%u ldb=%u error: %g number of errors: %u corrects: %u",side,uplo,trans,diag,m,n,lda,ldb,dtmp,nerrs,ncorr);
  if ( nerrs > 0 ) printf(" ->FAILED at %ux%u real*8 %u case",m,n,layout);
  printf("\n");

#ifdef TEST_SINGLE
  /* Call some reference code now on a copy of the B matrix (C) */
  compact_strsm_ ( &layout, &side, &uplo, &trans, &diag, &m, &n, &salpha, sa, &lda, sc, &ldb, &nmats, &VLENS );
  /* Compute the residual between B and C */
  dtmp = residual_s ( sc, bsize, bsize, nmats, sb, bsize, &nerrs, &ncorr );
  printf("R4 %c%c%c%c m=%u n=%u lda=%u ldb=%u error: %g number of errors: %u corrects: %u\n",side,uplo,trans,diag,m,n,lda,ldb,dtmp,nerrs,ncorr);
  if ( nerrs > 0 ) printf(" ->FAILED at %ux%u real*4 case",m,n);
  printf("\n");
#endif

#else
  for ( j = 0, nerrs = 0 ; j < bsize*nmatd ; j++ )
  {
     if ( isnan(db[j]) || isinf(db[j]) )
     {
        if ( ++nerrs < 10 )
        {
           printf("WARNING: db[%d]=%g\n",j,db[j]);
        }
     }
  }
  printf("%g,real*8 %c%c%c%c m=%u n=%u lda=%u ldb=%u Denormals=%u Time=%g Gflops=%g",(op_count*nmatd)/(timer*1.0e9),side,uplo,trans,diag,m,n,lda,ldb,nerrs,timer,(op_count*nmatd)/(timer*1.0e9));
  if ( nerrs > 0 ) printf(" -> FAILED at %ux%u real*8 case",m,n);
  printf("\n");
#endif

  free(dd);
  free(sd);
  free(dc);
  free(sc);
  free(db);
  free(sb);
  free(da);
  free(sa);

  return 0;
}

