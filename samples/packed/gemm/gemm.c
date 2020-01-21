/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Greg Henry, Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#if 0
#define USE_KERNEL_GENERATION_DIRECTLY
#endif
#if 0
#define USE_PREDEFINED_ASSEMBLY
#define USE_XSMM_GENERATED
#define TIME_MKL
#endif
#if 0
#define TEST_SINGLE
#endif

#if !defined(USE_PREDEFINED_ASSEMBLY) && !defined(USE_XSMM_GENERATED) && !defined(TIME_MKL) && \
   (!defined(__linux__) || !defined(USE_KERNEL_GENERATION_DIRECTLY))
# define USE_XSMM_GENERATED
# include <libxsmm.h>
#else
# include <libxsmm_source.h>
# include <unistd.h>
# include <signal.h>
# include <malloc.h>
# include <sys/mman.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define BUFSIZE 32*32
#define BUFSIZE2 64000

#if 0
#define TRIANGLE_IS_IDENTITY
#endif

#if 1
#define AVX2_TESTING
#endif
#if 0
#define AVX512_TESTING
#endif
#if !defined(AVX2_TESTING) && !defined(AVX512_TESTING)
  #define AVX2_TESTING
#endif
#if defined(AVX2_TESTING) && defined(AVX512_TESTING)
  #error Compile with either AVX2_TESTING or AVX512_TESTING never both
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
       for ( j = 0; j < n; j++ )
       {
          for ( i = 0; i < m; i++ )
          {
             Atemp[i+j*m] = A[i*VLEN+j*lda*VLEN];
          }
       }
#if EVENTUALLY_USE_THIS_LOOP_IT_SHOULD_BE_FASTER
       for ( j = 0; j < n; j++ )
       {
          for ( i = 0, ia = 0; i < m; i++, ia+=VLEN )
          {
             Atemp[i+j*m] = A[ ia+j*lda*VLEN ];
          }
       }
#endif
    } else {
       /* printf("Row major\n"); */
       for ( j = 0; j < n; j++ )
       {
          for ( i = 0; i < m; i++ )
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
       for ( j = 0; j < n; j++ )
       {
          for ( i = 0; i < m; i++ )
          {
             Atemp[i+j*m] = A[i*VLEN+j*lda*VLEN];
          }
       }
    } else {
       /* printf("Row major\n"); */
       for ( j = 0; j < n; j++ )
       {
          for ( i = 0; i < m; i++ )
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
       for ( j = 0; j < n; j++ )
       {
          for ( i = 0, ia = 0; i < m; i++, ia+=VLEN )
          {
             A[ia+j*lda*VLEN] = Atemp[i+j*m];
          }
       }
    } else {
       for ( j = 0; j < n; j++ )
       {
          for ( i = 0; i < m; i++ )
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
       for ( j = 0; j < n; j++ )
       {
          for ( i = 0, ia = 0; i < m; i++, ia+=VLEN )
          {
             A[ia+j*lda*VLEN] = Atemp[i+j*m];
          }
       }
    } else {
       for ( j = 0; j < n; j++ )
       {
          for ( i = 0; i < m; i++ )
          {
             /* Transpose the data */
             A[j*VLEN+i*lda*VLEN] = Atemp[i+j*m];
          }
       }
    }
}

#if !defined(USE_MKL_FOR_REFERENCE) && !defined(LIBXSMM_NOFORTRAN) && (!defined(__BLAS) || (0 != __BLAS))
extern void dgemm_();

/* Reference code for compact dgemm. Note that this just copies data into
   a buffer from the compact storage and calls the regular dgemm code. This
   is very naive reference code just used for testing purposes */
LIBXSMM_INLINE
void compact_dgemm_ ( unsigned int *layout, char *transa, char *transb,
                      unsigned int *m, unsigned int *n,
                      unsigned int *k, double *alpha, double *A,
                      unsigned int *lda, double *B, unsigned int *ldb,
                      double *beta, double *C, unsigned int *ldc,
                      unsigned int *nmat, unsigned int *VLEN )
{
    unsigned int i, j, num, info;
    double *Ap, Atemp[BUFSIZE];
    double *Bp, Btemp[BUFSIZE];
    double *Cp, Ctemp[BUFSIZE];
    static int ntimes = 0;
    char ntrans='N';

    if ( ++ntimes < 3 ) printf("Inside reference compact_dgemm_()\n");
    if ( ++ntimes < 3 ) printf("layout=%d m/n/k=%d %d %d lda/b/c=%d %d %d nmat=%d VLEN=%d\n",*layout,*m,*n,*k,*lda,*ldb,*ldc,*nmat,*VLEN);
    for ( i = 0, num = 0; i < (*nmat); i+= *VLEN, num++ )
    {
       for ( j = 0; j < *VLEN; j++ )
       {
           /* Unpack the data, call a reference DGEMM, repack the data */
           Ap = &A[j+num*(*lda)*(*k)*(*VLEN)];
           Bp = &B[j+num*(*ldb)*(*n)*(*VLEN)];
           Cp = &C[j+num*(*ldc)*(*n)*(*VLEN)];
if (++ntimes < 3 ) printf("Doing a dgemm at place i=%d j=%d num=%d Ap[%d]=%g\n",i,j,num,j+num*(*lda)*(*k)*(*VLEN),Ap[0]);
           dcopy_to_temp ( *layout, Ap, *lda, *m, *k, Atemp, *VLEN );
           dcopy_to_temp ( *layout, Bp, *ldb, *k, *n, Btemp, *VLEN );
           dcopy_to_temp ( *layout, Cp, *ldc, *m, *n, Ctemp, *VLEN );
           dgemm_ ( transa, transb, m, n, k, alpha, Atemp, m, Btemp, k, beta, Ctemp, m );
           dcopy_from_temp ( *layout, Cp, *ldc, *m, *n, Ctemp, *VLEN );
       }
    }
}

extern void sgemm_();

/* Reference code for compact sgemm. Note that this just copies data into
   a buffer from the compact storage and calls the regular sgemm code. This
   is very naive reference code just used for testing purposes */
/* Note: if layout==101 (row major), then this code is known to only work when
 *        nmat == VLEN. To check for accuracy otherwise, transpose everything */
LIBXSMM_INLINE
void compact_sgemm_ ( char *transa, char *transb,
                      unsigned int *layout, unsigned int *m, unsigned int *n,
                      unsigned int *k, float *alpha, float *A,
                      unsigned int *lda, float *B, unsigned int *ldb,
                      float *beta, float *C, unsigned int *ldc,
                      unsigned int *nmat, unsigned int *VLEN )
{
    unsigned int i, j, num, info;
    float *Ap, Atemp[BUFSIZE];
    float *Bp, Btemp[BUFSIZE];
    float *Cp, Ctemp[BUFSIZE];
    static int ntimes = 0;
    char ntrans='N';

    if ( ++ntimes < 3 ) printf("Inside reference compact_sgemm_()\n");
    if ( ++ntimes < 3 ) printf("layout=%d m/n/k=%d %d %d lda/b/c=%d %d %d nmat=%d VLEN=%d\n",*layout,*m,*n,*k,*lda,*ldb,*ldc,*nmat,*VLEN);
    for ( i = 0, num = 0; i < (*nmat); i+= *VLEN, num++ )
    {
       for ( j = 0; j < *VLEN; j++ )
       {
           /* Unpack the data, call a reference DGEMM, repack the data */
           Ap = &A[j+num*(*lda)*(*k)*(*VLEN)];
           Bp = &B[j+num*(*ldb)*(*n)*(*VLEN)];
           Cp = &C[j+num*(*ldc)*(*n)*(*VLEN)];
if (++ntimes < 3 ) printf("Doing a sgemm at place i=%d j=%d num=%d Ap[%d]=%g\n",i,j,num,j+num*(*lda)*(*k)*(*VLEN),Ap[0]);
           scopy_to_temp ( *layout, Ap, *lda, *m, *k, Atemp, *VLEN );
           scopy_to_temp ( *layout, Bp, *ldb, *k, *n, Btemp, *VLEN );
           scopy_to_temp ( *layout, Cp, *ldc, *m, *n, Ctemp, *VLEN );
           sgemm_ ( transa, transb, m, n, k, alpha, Atemp, m, Btemp, k, beta, Ctemp, m );
           scopy_from_temp ( *layout, Cp, *ldc, *m, *n, Ctemp, *VLEN );
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
     fprintf(stderr,"Error in dfill_matrix: ld=%u m=%u mismatched!\n",ld,m);
     exit(-1);
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
void dfill_identity ( double *matrix, unsigned int ld, unsigned int m, unsigned int n, int VLEN, int number_of_cases )
{
  unsigned int h, i, j, k, ia;
  double dtmp;

  if ( ld < m ) {
     fprintf(stderr,"Error in dfill_identity: ld=%u m=%u mismatched!\n",ld,m);
     exit(-1);
  }
  for ( h = 0; h < (unsigned int)number_of_cases; h++ ) {
     ia = h*ld*n*VLEN;
     for ( j = 1; j <= n; j++ ) {
        for ( i = 1; i <= ld; i++ ) {
           if ( i == j ) dtmp = 1.0; else dtmp = 0.0;
           for ( k = 0; k < (unsigned int)VLEN; k++ ) matrix[ia++] = dtmp;
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
double residual_d ( double *A, unsigned int lda, unsigned int m, unsigned int n,
                    double *B, unsigned int ldb, unsigned int *nerrs,
                    unsigned int *ncorr )
{
   unsigned int i, j, address, i4, j4, k4, i8, j8, k8;
   double atmp, btmp, dtmp, ref, derror;
   static int ntimes = 0;

   *nerrs = 0;
   *ncorr = 0;
   derror = 0.0;
   for ( j = 1; j<= n; j++ )
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
                address = (j-1)*lda + (i-1);
                j4 = (int)(address/(lda*4)) + 1;
                i4 = (int)((address-(j4-1)*lda*4) / 4) + 1;
                k4 = (address-(j4-1)*lda*4 - (i4-1)*4) + 1;
                j8 = (int)(address/(lda*8)) + 1;
                i8 = (int)((address-(j8-1)*lda*8) / 8) + 1;
                k8 = (address-(j8-1)*lda*8 - (i8-1)*8) + 1;
                printf("Bug #%i: A[%u]=A(%u,%u)=A4(%u,%u,%u)=A8(%u,%u,%u) expected=%g instead=%g err=%g\n",ntimes,address,i,j,i4,j4,k4,i8,j8,k8,atmp,btmp,dtmp);
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
   unsigned int i, j, address, i4, j4, k4, i8, j8, k8;
   double atmp, btmp, dtmp, ref, derror;
   static int ntimes = 0;

   *nerrs = 0;
   *ncorr = 0;
   derror = 0.0;
   for ( j = 1; j<= n; j++ )
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
                address = (j-1)*lda + (i-1);
                j4 = (int)(address/(lda*4)) + 1;
                i4 = (int)((address-(j4-1)*lda*4) / 4) + 1;
                k4 = (address-(j4-1)*lda*4 - (i4-1)*4) + 1;
                j8 = (int)(address/(lda*8)) + 1;
                i8 = (int)((address-(j8-1)*lda*8) / 8) + 1;
                k8 = (address-(j8-1)*lda*8 - (i8-1)*8) + 1;
                printf("Bug #%i: A[%u]=A(%u,%u)=A4(%u,%u,%u)=A8(%u,%u,%u) expected=%g instead=%g err=%g\n",ntimes,address,i,j,i4,j4,k4,i8,j8,k8,atmp,btmp,dtmp);
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

#ifdef USE_PREDEFINED_ASSEMBLY
extern void gemm_();
#endif
#ifdef MKL_TIMER
extern double dsecnd_();
#endif

int main(int argc, char* argv[])
{
  unsigned int m=8, n=8, k=8, lda=8, ldb=8, ldc=8, nerrs, num, nmat;
  unsigned int layout, asize, bsize, ntest, ncorr;
#ifdef AVX512_TESTING
  unsigned int VLEND=8, VLENS=16;
  int arch=LIBXSMM_X86_AVX512_CORE;
#else
  unsigned int VLEND=4, VLENS=8;
  int arch=LIBXSMM_X86_AVX2;
#endif
  unsigned int nmats, nmatd;
  unsigned int i, j, l, iunroll, junroll, loopi, loopj;
  char side='L', uplo='U', transa='N', transb='N', diag='N';
  unsigned int typesize8 = 8;
  unsigned int typesize4 = 4;
  float  *sa, *sb, *sc, *sd, *sc1;
  double *da, *db, *dc, *dd, *dc1;
  double dalpha = 1.0;
  float  salpha = (float)dalpha;
  double dbeta = 1.0;
  float  sbeta = (float)dbeta;
  double dtmp;
  const unsigned char *cptr = NULL;
  unsigned long op_count;
  const libxsmm_pgemm_descriptor* desc8 = NULL;
  const libxsmm_pgemm_descriptor* desc4 = NULL;
#ifdef USE_XSMM_GENERATED
  libxsmm_descriptor_blob blob;
  libxsmm_pgemm_xfunction mykernel = NULL;
#endif
#if defined(USE_KERNEL_GENERATION_DIRECTLY) && defined(__linux__)
  void (*opcode_routine)();
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
  io_generated_code.sf_size = 0;
#endif

  printf("\nUSAGE: %s m n k lda ldb ldc nmat layout ntest transa transb iunroll junroll loopj loopi\n",argv[0]);
  if ( argc <= 3 )
  {
#ifdef TEST_SINGLE
     printf("Compact SGEMM a C_mxn<-C_mxn+A_mxk*B_kxn matrix of leading dims lda/b/c\n");
     printf("This will test the jit of 1 VLEN=%d ",VLENS);
     if ( VLENS==8 ) printf("(AVX2)");
     else            printf("(AVX512)");
#else
     printf("Compact DGEMM a C_mxn<-C_mxn+A_mxk*B_kxn matrix of leading dims lda/b/c\n");
     printf("This will test the jit of 1 VLEN=%d ",VLEND);
     if ( VLEND==4 ) printf("(AVX2)");
     else            printf("(AVX512)");
#endif
     printf(" work of nmat at a time\n");
     printf("Configurable: M-loop controlled by iunroll & loopi. N-loop by junroll & loopj\n");
     printf("Defaults: m=n=k=lda=ldb=ldc=nmat=8, layout=102 (col major), transa=/b='N', ntest=1\n");
  }
  if ( argc > 1 ) m = atoi(argv[1]); else m = 8;
  if ( argc > 2 ) n = atoi(argv[2]); else n = 8;
  if ( argc > 3 ) k = atoi(argv[3]); else k = 8;
  if ( argc > 4 ) lda= atoi(argv[4]); else lda = 8;
  if ( argc > 5 ) ldb= atoi(argv[5]); else ldb = 8;
  if ( argc > 6 ) ldc= atoi(argv[6]); else ldc = 8;
  if ( argc > 7 ) nmat = atoi(argv[7]); else nmat = 8;
  if ( argc > 8 ) layout = atoi(argv[8]); else layout=102;
  if ( argc > 9 ) ntest = atoi(argv[9]); else ntest = 1;
  if ( argc > 10 ) transa = argv[10][0]; else transa = 'N';
  if ( argc > 11 ) transb = argv[11][0]; else transb = 'N';
  if ( argc > 12 ) iunroll=atoi(argv[12]); else iunroll=0;
  if ( argc > 13 ) junroll=atoi(argv[13]); else junroll=0;
  if ( argc > 14 ) loopj=atoi(argv[14]); else loopj=0;
  if ( argc > 15 ) loopi=atoi(argv[15]); else loopi=0;

  salpha = (float)dalpha;
  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  k = LIBXSMM_MAX(k,1);
  ntest = LIBXSMM_MAX(ntest,1);
  nmat = LIBXSMM_MAX(nmat,VLEND);
  layout = LIBXSMM_MAX(LIBXSMM_MIN(layout,102),101);

  if ( transa!='N' && transa!='n' && transa!='T' && transa!='t' ) transa='N';
  if ( transb!='N' && transb!='n' && transb!='T' && transb!='t' ) transb='N';

  lda = LIBXSMM_MAX(lda,m);
  ldb = LIBXSMM_MAX(ldb,k);
  ldc = LIBXSMM_MAX(ldc,m);

  nmats = LIBXSMM_MAX(VLENS,nmat - (nmat%VLENS));
  nmatd = LIBXSMM_MAX(VLEND,nmat - (nmat%VLEND));
#ifdef TEST_SINGLE
  nmat = nmats;
#else
  nmat = nmatd;
#endif

  op_count = (unsigned long)(nmat * 2.0 * (double)m * (double)n * (double)k);

#ifdef TEST_SINGLE
printf("This is a real*%d tester for JIT compact SGEMM %c%c kernels! (m=%u n=%u k=%u lda=%u ldb=%u ldc=%u layout=%d nmat=%d alpha=%g beta=%g iun=%d jun=%d loopi=%d loopj=%d VLEN=%d)\n",typesize4,transa,transb,m,n,k,lda,ldb,ldc,layout,nmat,dalpha,dbeta,iunroll,junroll,loopi,loopj,VLENS);
#else
printf("This is a real*%d tester for JIT compact DGEMM %c%c kernels! (m=%u n=%u k=%u lda=%u ldb=%u ldc=%u layout=%d nmat=%d alpha=%g beta=%g iun=%d jun=%d loopi=%d loopj=%d VLEN=%d)\n",typesize8,transa,transb,m,n,k,lda,ldb,ldc,layout,nmat,dalpha,dbeta,iunroll,junroll,loopi,loopj,VLEND);
#endif

#ifdef USE_XSMM_GENERATED
  printf("This code tests the LIBXSMM generated kernels\n");
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
  printf("This code tests some predefined assembly kernel\n");
#endif
#if defined(USE_KERNEL_GENERATION_DIRECTLY) && defined(__linux__)
  printf("This code tests kernel generation directly\n");
#endif
#ifdef TIME_MKL
  printf("This code tests MKL compact batch directly\n");
#endif
#ifdef AVX512_TESTING
  printf("This tests AVX512 binaries\n");
#endif
#ifdef AVX2_TESTING
  printf("This tests AVX2 binaries\n");
#endif

  desc8 = libxsmm_pgemm_descriptor_init(&blob, typesize8, m, n, k, lda, ldb, ldc, &dalpha, transa, transb, layout );
#ifdef TEST_SINGLE
  desc4 = libxsmm_pgemm_descriptor_init(&blob, typesize4, m, n, k, lda, ldb, ldc, &dalpha, transa, transb, layout );
#endif

  printf("Descriptor set\n");


#ifdef USE_XSMM_GENERATED
  printf("calling libxsmm_dispatch_pgemm: typesize8=%u\n",typesize8);
  mykernel = libxsmm_dispatch_pgemm(desc8);
  printf("done calling libxsmm_dispatch_pgemm: typesize8=%u\n",typesize8);
  if ( mykernel == NULL ) printf("R8 Kernel after the create call is null\n");
#ifdef TEST_SINGLE
  mykernel = libxsmm_dispatch_pgemm(desc4);
  if ( mykernel == NULL ) printf("R4 kernel after the create call is null\n");
#endif
#endif

#if defined(USE_KERNEL_GENERATION_DIRECTLY) && defined(__linux__)
  libxsmm_generator_pgemm_kernel( &io_generated_code, desc8, arch, iunroll, junroll, loopi, loopj );
#endif

#ifndef NO_ACCURACY_CHECK
  printf("mallocing matrices\n");
#endif
  sa  = (float  *) malloc ( lda*k*nmat*sizeof(float) );
  da  = (double *) malloc ( lda*k*nmat*sizeof(double) );
  sb  = (float  *) malloc ( ldb*n*nmat*sizeof(float) );
  db  = (double *) malloc ( ldb*n*nmat*sizeof(double) );
  sc1 = (float  *) malloc ( ldc*n*nmat*sizeof(float) );
  dc1 = (double *) malloc ( ldc*n*nmat*sizeof(double) );
  sc  = (float  *) malloc ( ldc*n*nmat*sizeof(float) );
  dc  = (double *) malloc ( ldc*n*nmat*sizeof(double) );
  sd  = (float  *) malloc ( ldc*n*nmat*sizeof(float) );
  dd  = (double *) malloc ( ldc*n*nmat*sizeof(double) );

#ifndef NO_ACCURACY_CHECK
  printf("filling matrices\n");
#endif
  sfill_matrix ( sa, lda, m, k*nmat );
  sfill_matrix ( sb, ldb, k, n*nmat );
  sfill_matrix ( sc, ldc, m, n*nmat );
  dfill_matrix ( da, lda, m, k*nmat );
  dfill_matrix ( db, ldb, k, n*nmat );
  dfill_matrix ( dc, ldc, m, n*nmat );

#ifndef NO_ACCURACY_CHECK
  for ( i = 0; i < ldc*n*nmat; i++ ) sd[i]=sc[i];
  for ( i = 0; i < ldc*n*nmat; i++ ) dd[i]=dc[i];
  for ( i = 0; i < ldc*n*nmat; i++ ) sc1[i]=sc[i];
  for ( i = 0; i < ldc*n*nmat; i++ ) dc1[i]=dc[i];
  printf("Pointing at the kernel now\n");
#endif

#ifdef USE_XSMM_GENERATED
  cptr = (const unsigned char*) mykernel;
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
  cptr = (const unsigned char*) gemm_;
#endif
#if defined(USE_KERNEL_GENERATION_DIRECTLY) && defined(__linux__)
  cptr = (const unsigned char*) &routine_output[0];
  opcode_routine = (void *) &cptr[0];
#endif

#ifndef TIME_MKL
# define DUMP_ASSEMBLY_FILE
#endif

#ifdef DUMP_ASSEMBLY_FILE
  printf("Dumping assembly file\n");
  FILE *fp = fopen("foo.s","w");
  char buffer[80];
  fputs("\t.text\n",fp);
  fputs("\t.align 256\n",fp);
  fputs("\t.globl gemm_\n",fp);
  fputs("gemm_:\n",fp);
  for (i = 0; i < 7000; i+=4 )
  {
     sprintf(buffer,".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",cptr[i],cptr[i+1],cptr[i+2],cptr[i+3]);
     fputs(buffer,fp);
  }
  fputs("\tretq\n",fp);
  fputs("\t.type gemm_,@function\n",fp);
  fputs("\t.size gemm_,.-gemm_\n",fp);
  fclose(fp);
#endif

#if defined(USE_MKL_FOR_REFERENCE) || defined(TIME_MKL)
# include <mkl.h>
  MKL_LAYOUT CLAYOUT = (layout == 101) ? MKL_ROW_MAJOR : MKL_COL_MAJOR;
  MKL_SIDE SIDE = (side == 'R' || side == 'r') ? MKL_RIGHT : MKL_LEFT;
  MKL_UPLO UPLO = (uplo == 'U' || uplo == 'u') ? MKL_UPPER : MKL_LOWER;
  MKL_TRANSPOSE TRANSA = (transa == 'N' || transa == 'n') ? MKL_NOTRANS : MKL_TRANS;
  MKL_TRANSPOSE TRANSB = (transb == 'N' || transb == 'n') ? MKL_NOTRANS : MKL_TRANS;
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
  double timer, firsttime = 0;
#ifdef MKL_TIMER
  double tmptimer;
  tmptimer = dsecnd_();
#else
  unsigned long long l_start, l_end;
#endif

  timer = 0.0;
  for ( j = 0; j < (int)ntest; j++ )
  {
  for ( i = 0; i < ldc*n*nmat; i++ ) dc[i]=dc1[i];
  for ( i = 0 , num = 0; i < (int)nmat; i+= (int)VLEND, num++ )
  {
     double *Ap = &da[num*lda*k*VLEND];
     double *Bp = &db[num*ldb*n*VLEND];
     double *Cp = &dc[num*ldc*n*VLEND];
#ifdef MKL_TIMER
     tmptimer = dsecnd_();
#else
     l_start = libxsmm_timer_tick();
#endif

#if !defined(USE_XSMM_GENERATED) && !defined(USE_PREDEFINED_ASSEMBLY) && !defined(USE_KERNEL_GENERATION_DIRECTLY) && !defined(TIME_MKL) && !defined(USE_PREDEFINED_ASSEMBLY_XCT)
     gen_compact_dgemm_ ( &layout, &m, &n, &k, &dalpha, Ap, &lda, Bp, &ldb, &dbeta, Cp, &ldc, &VLEND );
#endif
#ifdef USE_XSMM_GENERATED
     mykernel ( Ap, Bp, Cp );
#endif
#ifdef USE_PREDEFINED_ASSEMBLY
     gemm_ ( Ap, Bp, Cp );
#endif
#ifdef USE_KERNEL_GENERATION_DIRECTLY
     (*opcode_routine)( Ap, Bp, Cp );
#endif
#ifdef TIME_MKL
     mkl_dgemm_compact ( CLAYOUT, TRANSA, TRANSB, m, n, k, dalpha, da, lda, db, ldb, dbeta, dc, ldc, CMP_FORMAT, nmat );
     i+=nmatd; /* Because MKL will do everything */
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

  for ( i = 0 , num = 0; i < nmats; i+= VLENS, num++ )
  {
     float *Ap = &sa[num*lda*k*VLENS];
     float *Bp = &sb[num*ldb*n*VLENS];
     float *Cp = &sc[num*ldc*n*VLENS];
#ifdef USE_XSMM_GENERATED
     mykernel ( Ap, Bp, Cp );
#endif
  }
  printf("after r4 routine, new      C(1,1)=%g C]256]=%g\n",dc[0],dc[256]);
#endif

#ifndef NO_ACCURACY_CHECK
  /* Call some reference code now on a copy of the B matrix (C) */
  double timer2 = 0.0;
  for ( j = 0; j < (int)ntest; j++ )
  {
  for ( i = 0; i < ldc*n*nmat; i++ ) dd[i]=dc1[i];
#ifdef MKL_TIMER
  tmptimer = dsecnd_();
#else
  l_start = libxsmm_timer_tick();
#endif

#ifndef USE_MKL_FOR_REFERENCE
  compact_dgemm_ ( &layout, &transa, &transb, &m, &n, &k, &dalpha, da, &lda, db, &ldb, &dbeta, dd, &ldc, &nmat, &VLEND );
#else
  mkl_dgemm_compact ( CLAYOUT, TRANSA, TRANSB, m, n, k, dalpha, da, lda, db, ldb, dbeta, dd, ldc, CMP_FORMAT, nmat );
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
  dtmp = residual_d ( dc, ldc, m, n*nmat, dd, ldc, &nerrs, &ncorr );
  printf("R8 mnk=%u %u %u ldabc=%u %u %u error: %g number of errors: %u corrects: %u",m,n,k,lda,ldb,ldc,dtmp,nerrs,ncorr);
  if ( nerrs > 0 ) printf(" ->FAILED at %ux%u real*8 %u case",m,n,layout);
  printf("\n");

#ifdef TEST_SINGLE
  /* Call some reference code now on a copy of the B matrix (C) */
  compact_dgemm_ ( &layout, &transa, &transb, &m, &n, &k, &salpha, sa, &lda, sb, &ldb, &sbeta, sd, &ldc, &nmat, &VLENS );
  /* Compute the residual between C and D */
  dtmp = residual_s ( sc, ldc, m, n*nmat, sd, ldc, &nerrs, &ncorr );
  printf("R4 mnk=%u %u %u ldabc=%u %u %u error: %g number of errors: %u corrects: %u",m,n,k,lda,ldb,ldc,dtmp,nerrs,ncorr);
  if ( nerrs > 0 ) printf(" ->FAILED at %ux%u real*4 case",m,n);
  printf("\n");
#endif

#else
  for ( j = 0, nerrs = 0; j < lda*n*nmat; j++ )
  {
     if ( isnan(dc[j]) || isinf(dc[j]) )
     {
        if ( ++nerrs < 10 )
        {
           printf("WARNING: dc[%d]=%g\n",j,dc[j]);
        }
     }
  }
  printf("%g,real*8 m/n/k=%u %u %u lda-c=%u %u %u Denormals=%u Time=%g Gflops=%g",op_count/(timer*1.0e9),m,n,k,lda,ldb,ldc,nerrs,timer,op_count/(timer*1.0e9));
  if ( nerrs > 0 ) printf(" -> FAILED at %ux%u real*8 case",m,n);
  printf("\n");
#endif

  free(dd);
  free(sd);
  free(dc);
  free(sc);
  free(dc1);
  free(sc1);
  free(db);
  free(sb);
  free(da);
  free(sa);

  return 0;
}

