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

#if !defined(USE_MKL_FOR_REFERENCE) && !defined(LIBXSMM_NOFORTRAN) && (!defined(__BLAS) || (0 != __BLAS))
extern void dgetrf_();
extern void mkl_dgetrfnpi_();

/* Reference code for compact dgetrf. Note that this just copies data into
   a buffer from the compact storage and calls the regular dgetrf code. This
   is very naive reference code just used for testing purposes */
LIBXSMM_INLINE
void compact_dgetrf_ ( unsigned int *layout, unsigned int *m,
                      unsigned int *n, double *A, unsigned int *lda,
                      unsigned int *nmat, unsigned int *VLEN )
{
    int i, j, num, info, nfact=LIBXSMM_MIN(*m,*n);
    double *Ap, Atemp[BUFSIZE];
    static int ntimes = 0;
    extern void mkl_dgetrfnpi();

    if ( ++ntimes < 3 ) printf("Inside reference compact_dgetrf_()\n");
    if ( ++ntimes < 3 ) printf("layout=%d m=%d n=%d lda=%d nmat=%d VLEN=%d\n",*layout,*m,*n,*lda,*nmat,*VLEN);
    for ( i = 0, num = 0 ; i < (*nmat) ; i+= *VLEN, num++ )
    {
       for ( j = 0 ; j < *VLEN ; j++ )
       {
           /* Unpack the data, call a reference DGETRF, repack the data */
           Ap = &A[j+num*(*lda)*(*n)*(*VLEN)];
if (++ntimes < 3 ) printf("Doing a dgetrf at place i=%d j=%d num=%d Ap[%d]=%g\n",i,j,num,j+num*(*lda)*(*n)*(*VLEN),Ap[0]);
           dcopy_to_temp ( *layout, Ap, *lda, *m, *n, Atemp, *VLEN );
           mkl_dgetrfnpi_ ( m, n, &nfact, Atemp, m, &info );
           if ( info != 0 ) printf("Bad news reference code got info=%d\n",info);
           dcopy_from_temp ( *layout, Ap, *lda, *m, *n, Atemp, *VLEN );
       }
    }
}

extern void sgetrf_();
extern void mkl_sgetrfnpi_();

/* Reference code for compact sgetrf. Note that this just copies data into
   a buffer from the compact storage and calls the regular sgetrf code. This
   is very naive reference code just used for testing purposes */
/* Note: if layout==101 (row major), then this code is known to only work when
 *        nmat == VLEN. To check for accuracy otherwise, transpose everything */
LIBXSMM_INLINE
void compact_sgetrf_ ( unsigned int *layout, unsigned int *m,
                      unsigned int *n, float *A, unsigned int *lda,
                      unsigned int *nmat, unsigned int *VLEN )
{
    int i, j, num, info, nfact=LIBXSMM_MIN(*m,*n);
    float *Ap, Atemp[BUFSIZE];
    static int ntimes = 0;
    extern void mkl_sgetrfnpi();

    if ( ++ntimes < 3 ) printf("Inside reference compact_sgetrf_()\n");
    if ( ++ntimes < 3 ) printf("layout=%d VLEN=%d nmat=%d\n",*layout, *VLEN, *nmat );
    for ( i = 0, num = 0 ; i < (*nmat) ; i+= *VLEN, num++ )
    {
       for ( j = 0 ; j < *VLEN ; j++ )
       {
           /* Unpack the data, call a reference SGETRF, repack the data */
           Ap = &A[j+num*(*lda)*(*n)*(*VLEN)];
if (++ntimes < 3 ) printf("Doing a sgetrf at place i=%d j=%d num=%d Ap[%d]=%g\n",i,j,num,j+num*(*lda)*(*n)*(*VLEN),Ap[0]);
           scopy_to_temp ( *layout, Ap, *lda, *m, *n, Atemp, *VLEN );
           mkl_sgetrfnpi_ ( m, n, n, Atemp, m, &info );
           if ( info != 0 ) printf("Bad news! Serial reference got info=%d\n",info);
           scopy_from_temp ( *layout, Ap, *lda, *m, *n, Atemp, *VLEN );
       }
    }
}

#endif

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
     fprintf(stderr,"Error in dfill_identity: ld=%u m=%u mismatched!\n",ld,m);
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
         if ( (dtmp / ref > 1.0e-12) && (dtmp > 1.0e-15) ) {
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
         if ( (dtmp / ref > 1.0e-4) && (dtmp > 1.0e-7) )
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

#if !defined(USE_PREDEFINED_ASSEMBLY) && !defined(USE_XSMM_GENERATED) && !defined(USE_KERNEL_GENERATION_DIRECTLY) && !defined(TIME_MKL) && defined(__linux__)
  #define USE_KERNEL_GENERATION_DIRECTLY
#endif

#if 0
  #define USE_PREDEFINED_ASSEMBLY
  #define USE_XSMM_GENERATED
  #define USE_KERNEL_GENERATION_DIRECTLY
  #define TIME_MKL
#endif

#ifdef USE_PREDEFINED_ASSEMBLY
extern void getrf_();
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
  char side='L', uplo='L', trans='N', diag='N';
  float  *sa, *sb, *sc, *sd;
  double *da, *db, *dc, *dd, *tmpbuf;
  double dalpha = 1.0;
  float  salpha;
  double dtmp;
  const unsigned char *cptr;
  unsigned long op_count;
  unsigned int typesize8 = 8;
  const libxsmm_trsm_descriptor* desc8 = NULL;
#ifdef TEST_SINGLE
  unsigned int typesize4 = 4;
  const libxsmm_trsm_descriptor* desc4 = NULL;
#endif
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
  /* #include "../../src/generator_packed_trsm_avx_avx512.h" */
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

  printf("\nUSAGE: %s m n lda nmat layout ntest\n",argv[0]);
  if ( argc <= 3 )
  {
     printf("Compact LU (GETRF, no pivots) a mxn matrix of leading dim lda\n");
     printf("This will test the jit of 1 VLEN work of nmat at a time\n");
     printf("Defaults: m=n=lda=nmat=8, layout=102 (col major), ntest=1\n");
  }
  if ( argc > 1 ) m = atoi(argv[1]); else m = 8;
  if ( argc > 2 ) n = atoi(argv[2]); else n = 8;
  if ( argc > 3 ) lda= atoi(argv[3]); else lda = 8;
  if ( argc > 4 ) nmat = atoi(argv[4]); else nmat = 8;
  if ( argc > 5 ) layout = atoi(argv[5]); else layout=102;
  if ( argc > 6 ) ntest = atoi(argv[6]); else ntest = 1;
  salpha = (float)dalpha;

  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  ntest = LIBXSMM_MAX(ntest,1);
  nmat = LIBXSMM_MAX(VLEND,nmat - (nmat%VLEND));
  layout = LIBXSMM_MAX(LIBXSMM_MIN(layout,102),101);

  lda = LIBXSMM_MAX(lda,m);

  if ( m >= n ) {
     op_count = nmat * (double)n * (double)n * (3.0*(double)m-(double)n) / 3.0;
  } else {
     op_count = nmat * (double)m * (double)m * (3.0*(double)n-(double)m) / 3.0;
  }

  nmats = LIBXSMM_MAX(VLENS,nmat - (nmat%VLENS));
  nmatd = LIBXSMM_MAX(VLEND,nmat - (nmat%VLEND));
  nmat = LIBXSMM_MAX(nmats,nmatd);

  printf("This is a real*%d tester for JIT compact DGETRF kernels! (m=%u n=%u lda=%u layout=%d nmat=%d)\n",typesize8,m,n,lda,layout,nmat);
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
#ifdef TEST_SINGLE
  desc4 = libxsmm_trsm_descriptor_init(&blob, typesize4, m, n, lda, ldb, &salpha, trans, diag, side, uplo, layout);
#endif
#ifdef USE_XSMM_GENERATED
  printf("calling libxsmm_dispatch_trmm: typesize8=%u\n",typesize8);
  mykernel.dp = libxsmm_dispatch_getrf(desc8);
  printf("done calling libxsmm_dispatch_trmm: typesize8=%u\n",typesize8);
  if ( mykernel.dp == NULL ) printf("R8 Kernel after the create call is null\n");
#ifdef TEST_SINGLE
  mykernel.sp = libxsmm_dispatch_getrf(desc4);
  if ( mykernel.sp == NULL ) printf("R4 kernel after the create call is null\n");
#endif
#endif

#ifdef USE_KERNEL_GENERATION_DIRECTLY
  libxsmm_generator_packed_getrf_avx_avx512_kernel ( &io_generated_code, desc8, "hsw" );
#endif

#ifndef NO_ACCURACY_CHECK
  printf("mallocing matrices\n");
#endif
  sa  = (float  *) malloc ( lda*n*nmat*sizeof(float) );
  da  = (double *) malloc ( lda*n*nmat*sizeof(double) );
  sc  = (float  *) malloc ( lda*n*nmat*sizeof(float) );
  dc  = (double *) malloc ( lda*n*nmat*sizeof(double) );
  sd  = (float  *) malloc ( lda*n*nmat*sizeof(float) );
  dd  = (double *) malloc ( lda*n*nmat*sizeof(double) );

#ifndef NO_ACCURACY_CHECK
  printf("filling matrices\n");
#endif
  sfill_matrix ( sa, lda, m, n*nmat );
#ifdef TRIANGLE_IS_IDENTITY
  printf("Warning: setting triangular matrix to identity. Not good for accuracy testing\n");
  dfill_identity ( da, lda, m, m, VLEND, nmatd/VLEND );
#else
  dfill_matrix ( da, lda, m, n*nmatd );
#endif

#ifndef NO_ACCURACY_CHECK
  for ( i = 0 ; i < lda*n*nmat ; i++ ) sc[i]=sa[i];
  for ( i = 0 ; i < lda*n*nmat ; i++ ) dc[i]=da[i];
  for ( i = 0 ; i < lda*n*nmat ; i++ ) sd[i]=sa[i];
  for ( i = 0 ; i < lda*n*nmat ; i++ ) dd[i]=da[i];
  printf("Pointing at the kernel now\n");
#endif

#ifdef USE_XSMM_GENERATED
  cptr = (const unsigned char*) mykernel.pv;
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
  cptr = (const unsigned char*) getrf_;
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
  fputs("\t.globl getrf_\n",fp);
  fputs("getrf_:\n",fp);
  for (i = 0 ; i < 4000; i+=4 )
  {
     sprintf(buffer,".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",cptr[i],cptr[i+1],cptr[i+2],cptr[i+3]);
     fputs(buffer,fp);
  }
  fputs("\tretq\n",fp);
  fputs("\t.type getrf_,@function\n",fp);
  fputs("\t.size getrf_,.-getrf_\n",fp);
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
  printf("Before routine, initial A(1,1)=%g A[256]=%g\n",da[0],da[256]);
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
  double one = 1.0;
#endif
  double timer, firsttime;
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
  for ( i = 0 ; i < (int)(lda*n*nmat); i++ ) dc[i]=da[i];
#endif
  for ( i = 0 , num = 0; i < (int)nmat ; i+= (int)VLEND, num++ )
  {
     double *Ap = &dc[num*lda*n*VLEND];
#ifdef MKL_TIMER
     tmptimer = dsecnd_();
#else
     l_start = libxsmm_timer_tick();
#endif

#if !defined(USE_XSMM_GENERATED) && !defined(USE_PREDEFINED_ASSEMBLY) && !defined(USE_KERNEL_GENERATION_DIRECTLY) && !defined(TIME_MKL) && !defined(USE_PREDEFINED_ASSEMBLY_XCT)
     gen_compact_dgetrf_ ( &layout, &m, &n, Ap, &lda, &VLEND );
#endif
#ifdef USE_XSMM_GENERATED
     mykernel.dp ( Ap, Ap, NULL );
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
     getrf_ ( Ap, Ap, &one );
#endif
#ifdef USE_KERNEL_GENERATION_DIRECTLY
     (*opcode_routine)( Ap );
#endif
#ifdef TIME_MKL
     mkl_dgetrfnp_compact ( CLAYOUT, m, n, dc, lda, info, CMP_FORMAT, nmat );
     i+=nmatd; /* Because MKL will do everything */
#endif
#ifdef USE_PREDEFINED_ASSEMBLY_XCT
     getrf_xct_ ( Ap, &one );
#endif
#ifdef MKL_TIMER
     dtmp = dsecnd_() - tmptimer;
#else
     l_end = libxsmm_timer_tick();
     dtmp = libxsmm_timer_duration(l_start,l_end);
#endif
     if ( j == 0 ) firsttime=dtmp;
     timer += dtmp;
  }
  }
  if ( ntest >= 100 ) {
      /* Skip the first timing: super necessary if using MKL */
      timer = (timer-firsttime)/((double)(ntest-1));
  } else {
      timer /= ((double)ntest);
  }

#ifndef NO_ACCURACY_CHECK
  printf("Average time to get through %u matrices: %g\n",nmat,timer);
  printf("Gflops: %g\n",(double)op_count/(timer*1.0e9));
  printf("after routine, new      C(1,1)=%g C[256]=%g\n",dc[0],dc[256]);
#endif

#ifdef TEST_SINGLE
  printf("Before r4 routine, initial C(1,1)=%g C[256]=%g\n",sc[0],sc[256]);

  for ( i = 0 , num = 0; i < nmats ; i+= VLENS, num++ )
  {
     float *Ap = &sc[num*lda*n*VLENS];
#ifdef USE_XSMM_GENERATED
     mykernel.sp ( Ap );
#endif
  }
  printf("after r4 routine, new      C(1,1)=%g C]256]=%g\n",dc[0],dc[256]);
#endif

#ifndef NO_ACCURACY_CHECK
  /* Call some reference code now on a copy of the B matrix (C) */
  double timer2 = 0.0;
  for ( j = 0 ; j < (int)ntest ; j++ )
  {
#ifndef TRIANGLE_IS_IDENTITY
  for ( i = 0 ; i < (int)(lda*n*nmat) ; i++ ) dd[i]=da[i];
#endif

#ifdef MKL_TIMER
  tmptimer = dsecnd_();
#else
  l_start = libxsmm_timer_tick();
#endif

#ifndef USE_MKL_FOR_REFERENCE
  compact_dgetrf_ ( &layout, &m, &n, dd, &lda, &nmat, &VLEND );
#else
  mkl_dgetrfnp_compact ( CLAYOUT, m, n, dd, lda, info, CMP_FORMAT, nmat );
#endif

#ifdef MKL_TIMER
  timer2 += dsecnd_() - tmptimer;
#else
  l_end = libxsmm_timer_tick();
  timer2 += libxsmm_timer_duration(l_start,l_end);
#endif

  }
  timer2 /= ((double)ntest);
  printf("Reference time=%g Reference Gflops=%g\n",timer2,op_count/(timer2*1.0e9));

  /* Compute the residual between B and C */
  dtmp = residual_d ( dc, lda, m, n*nmat, dd, lda, &nerrs, &ncorr );
  printf("R8 m=%u n=%u lda=%u error: %g number of errors: %u corrects: %u",m,n,lda,dtmp,nerrs,ncorr);
  if ( nerrs > 0 ) printf(" ->FAILED at %ux%u real*8 %u case",m,n,layout);
  printf("\n");

#ifdef TEST_SINGLE
  /* Call some reference code now on a copy of the B matrix (C) */
  for ( i = 0 ; i < lda*n*nmat ; i++ ) sd[i]=sa[i];
  compact_sgetrf_ ( &layout, &m, &n, sd, &lda, &nmat, &VLENS );
  /* Compute the residual between C and D */
  dtmp = residual_s ( sc, lda, m, n*nmat, sd, lda, &nerrs, &ncorr );
  printf("float m=%u n=%u lda=%u error: %g number of errors: %u corrects: %u\n",m,n,lda,dtmp,nerrs,ncorr);
  if ( nerrs > 0 ) printf(" ->FAILED at %ux%u real*4 case",m,n);
  printf("\n");
#endif

#else
  for ( j = 0, nerrs = 0 ; j < lda*n*nmat; j++ )
  {
     if ( isnan(dc[j]) || isinf(dc[j]) )
     {
        if ( ++nerrs < 10 )
        {
           printf("WARNING: dc[%d]=%g\n",j,dc[j]);
        }
     }
  }
  printf("%g,real*8 m=%u n=%u lda=%u Denormals=%u Time=%g Gflops=%g",op_count/(timer*1.0e9),m,n,lda,nerrs,timer,op_count/(timer*1.0e9));
  if ( nerrs > 0 ) printf(" -> FAILED at %ux%u real*8 case",m,n);
  printf("\n");
#endif

  free(dd);
  free(sd);
  free(dc);
  free(sc);
  free(da);
  free(sa);

  return 0;
}
