/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
#include <libxsmm_timer.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <errno.h>
/* this is linux specific */
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/mman.h>

#include <immintrin.h>

/*@TODO remove:*/
#define LIBXSMM_BUILD_PAGESIZE sysconf(_SC_PAGESIZE)

#define REPS 100000

void print_help() {
  printf("\n\n");
  printf("Usage (dense*dense=dense):\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    alpha: -1 or 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    ARCH: snb, hsw, knl, skx\n");
  printf("    PREFETCH: nopf (none), pfsigonly, BL2viaC, AL2, curAL2, AL2jpst, AL2_BL2viaC, curAL2_BL2viaC, AL2jpst_BL2viaC\n");
  printf("    PRECISION: SP, DP\n");
  printf("\n\n");
}

void init_double( double*                         io_a,
                  double*                         io_b,
                  double*                         io_c,
                  double*                         io_c_gold,
                  const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;

  /* touch A */
  for ( l_i = 0; l_i < i_xgemm_desc->lda; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->k; l_j++) {
      io_a[(l_j * i_xgemm_desc->lda) + l_i] = (double)drand48();
    }
  }
  /* touch B */
  for ( l_i = 0; l_i < i_xgemm_desc->ldb; l_i++ ) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++ ) {
      io_b[(l_j * i_xgemm_desc->ldb) + l_i] = (double)drand48();
    }
  }
  /* touch C */
  for ( l_i = 0; l_i < i_xgemm_desc->ldc; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++) {
      io_c[(l_j * i_xgemm_desc->ldc) + l_i] = (double)0.0;
      io_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] = (double)0.0;
    }
  }
}

void init_float( float*                          io_a,
                 float*                          io_b,
                 float*                          io_c,
                 float*                          io_c_gold,
                 const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;

  /* touch A */
  for ( l_i = 0; l_i < i_xgemm_desc->lda; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->k; l_j++) {
      io_a[(l_j * i_xgemm_desc->lda) + l_i] = (float)drand48();
    }
  }
  /* touch B */
  for ( l_i = 0; l_i < i_xgemm_desc->ldb; l_i++ ) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++ ) {
      io_b[(l_j * i_xgemm_desc->ldb) + l_i] = (float)drand48();
    }
  }
  /* touch C */
  for ( l_i = 0; l_i < i_xgemm_desc->ldc; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++) {
      io_c[(l_j * i_xgemm_desc->ldc) + l_i] = (float)0.0;
      io_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] = (float)0.0;
    }
  }
}

void run_gold_double( const double*                   i_a,
                      const double*                   i_b,
                      double*                         o_c,
                      const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_m, l_n, l_k, l_t;
  double l_runtime = 0.0;

  const unsigned long long l_start = libxsmm_timer_tick();

  for ( l_t = 0; l_t < REPS; l_t++  ) {
    for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++  ) {
      for ( l_k = 0; l_k < i_xgemm_desc->k; l_k++  ) {
        for ( l_m = 0; l_m < i_xgemm_desc->m; l_m++ ) {
          o_c[(l_n * i_xgemm_desc->ldc) + l_m] += i_a[(l_k * i_xgemm_desc->lda) + l_m] * i_b[(l_n * i_xgemm_desc->ldb) + l_k];
        }
      }
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for C\n", l_runtime);
  printf("%f GFLOPS for C\n", ((double)((double)REPS * (double)i_xgemm_desc->m * (double)i_xgemm_desc->n * (double)i_xgemm_desc->k) * 2.0) / (l_runtime * 1.0e9));
}


void run_gold_float( const float*                   i_a,
                     const float*                   i_b,
                     float*                         o_c,
                     const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_m, l_n, l_k, l_t;
  double l_runtime = 0.0;

  const unsigned long long l_start = libxsmm_timer_tick();

  for ( l_t = 0; l_t < REPS; l_t++  ) {
    for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++  ) {
      for ( l_k = 0; l_k < i_xgemm_desc->k; l_k++  ) {
        for ( l_m = 0; l_m < i_xgemm_desc->m; l_m++ ) {
          o_c[(l_n * i_xgemm_desc->ldc) + l_m] += i_a[(l_k * i_xgemm_desc->lda) + l_m] * i_b[(l_n * i_xgemm_desc->ldb) + l_k];
        }
      }
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for C\n", l_runtime);
  printf("%f GFLOPS for C\n", ((double)((double)REPS * (double)i_xgemm_desc->m * (double)i_xgemm_desc->n * (double)i_xgemm_desc->k) * 2.0) / (l_runtime * 1.0e9));
}

void run_jit_double( const double*                   i_a,
                     const double*                   i_b,
                     double*                         o_c,
                     const libxsmm_gemm_descriptor* i_xgemm_desc,
                     const char*                     i_arch ) {

  /* define function pointer */
  typedef void (*jitfun)(const double* a, const double* b, double* c);
  typedef void (*jitfun_pf)(const double* a, const double* b, double* c, const double* a_pf, const double* b_pf, const double* c_pf);
  jitfun l_test_jit;
  jitfun_pf l_test_jit_pf;

  int l_code_pages, l_code_page_size, l_fd, error;
  unsigned char* l_code;
  unsigned int l_t;
  void* p;

  double l_jittime = 0.0, l_runtime = 0.0;
  unsigned long long l_start = libxsmm_timer_tick();
  
  /* allocate buffer for code */
  unsigned char* l_gen_code = (unsigned char*) malloc( 131072 * sizeof(unsigned char) );
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = (void*)l_gen_code;
  l_generated_code.buffer_size = 131072;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 2;
  l_generated_code.last_error = 0;

  /* generate kernel */
  libxsmm_generator_dense_kernel( &l_generated_code,
                                  i_xgemm_desc,
                                  i_arch );

  /* handle evetl. errors */
  if ( l_generated_code.last_error != 0 ) {
    fprintf(stderr, "%s\n", libxsmm_strerror( l_generated_code.last_error ) );
    exit(-1);
  }

  /* create executable buffer */
  l_code_pages = (((l_generated_code.code_size-1)*sizeof(unsigned char))/LIBXSMM_BUILD_PAGESIZE)+1;
  l_code_page_size = LIBXSMM_BUILD_PAGESIZE*l_code_pages;
  l_fd = open("/dev/zero", O_RDWR);
  p = mmap(0, l_code_page_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, l_fd, 0);
  close(l_fd);
  /* explicitly disable THP for this memory region, kernel 2.6.38 or higher! */
  madvise(p, l_code_page_size, MADV_NOHUGEPAGE); 
  if (p == MAP_FAILED) {
    fprintf(stderr, "something bad happend in mmap!\n");
  }
  l_code = (unsigned char*)p;
  memset( l_code, 0, l_code_page_size );
  memcpy( l_code, l_gen_code, l_generated_code.code_size );
  error = mprotect( (void*)l_code, l_code_page_size, PROT_EXEC | PROT_READ );
  if (error == -1) {
    int errsv = errno;
    if (errsv == EINVAL) {
      fprintf(stderr, "mprotect failed: addr is not a valid pointer, or not a multiple of the system page size!\n");
    } else if (errsv == ENOMEM) {
      fprintf(stderr, "mprotect failed: Internal kernel structures could not be allocated!\n");
    } else if (errsv == EACCES) {
      fprintf(stderr, "mprotect failed: The memory cannot be given the specified access!\n");
    } else {
      fprintf(stderr, "mprotect failed: Unknown Error!\n");
    }
    exit(-1);
  }

  /* print function point address */
  printf("function pointer address: %llx\n", (size_t)l_code);

  /* set function pointer and jitted code */
  if ( 0 == (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) ) {
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_NONE ) {
      l_test_jit = (jitfun)l_code;
    } else {
      l_test_jit_pf = (jitfun_pf)l_code;
    }
  }

  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("size of generated code: %i\n", l_generated_code.code_size );

  { /* write buffer for manual decode as binary to a file */
    FILE *l_byte_code;
    char l_objdump_name[128];
    sprintf( l_objdump_name, "kernel_%i_%i_%i.bin", i_xgemm_desc->m, i_xgemm_desc->n, i_xgemm_desc->k ); 
    l_byte_code = fopen( l_objdump_name, "wb");

    if ( l_byte_code != NULL ){
      fwrite( (const void*)l_gen_code, 1, l_generated_code.code_size, l_byte_code);
      fclose( l_byte_code );
    } else {
      /* error */
    }
  }

  l_start = libxsmm_timer_tick();

  if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < REPS; l_t++ ) {
      l_test_jit(i_a, i_b, o_c);
    }
  } else {
    for ( l_t = 0; l_t < REPS; l_t++ ) {
      l_test_jit_pf(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for creating jit\n", l_jittime);
  printf("%fs for executing jit\n", l_runtime);
  printf("%f GFLOPS for jit\n", ((double)((double)REPS * (double)i_xgemm_desc->m * (double)i_xgemm_desc->n * (double)i_xgemm_desc->k) * 2.0) / (l_runtime * 1.0e9));

  free(l_gen_code);
}

void run_jit_float( const float*                    i_a,
                    const float*                    i_b,
                    float*                          o_c,
                    const libxsmm_gemm_descriptor* i_xgemm_desc,
                    const char*                     i_arch ) {

  /* define function pointer */
  typedef void (*jitfun)(const float* a, const float* b, float* c);
  typedef void (*jitfun_pf)(const float* a, const float* b, float* c, const float* a_pf, const float* b_pf, const float* c_pf);
  jitfun l_test_jit;
  jitfun_pf l_test_jit_pf;

  int l_code_pages, l_code_page_size, l_fd, error;
  unsigned char* l_code;
  unsigned int l_t;
  void* p;

  double l_jittime = 0.0, l_runtime = 0.0;
  unsigned long long l_start = libxsmm_timer_tick();
  
  /* allocate buffer for code */
  unsigned char* l_gen_code = (unsigned char*) malloc( 32768 * sizeof(unsigned char) );
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = (void*)l_gen_code;
  l_generated_code.buffer_size = 32768;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 2;
  l_generated_code.last_error = 0;

  /* generate kernel */
  libxsmm_generator_dense_kernel( &l_generated_code,
                                  i_xgemm_desc,
                                  i_arch );

  /* handle evetl. errors */
  if ( l_generated_code.last_error != 0 ) {
    fprintf(stderr, "%s\n", libxsmm_strerror( l_generated_code.last_error ) );
    exit(-1);
  }

  /* create executable buffer */
  l_code_pages = (((l_generated_code.code_size-1)*sizeof(unsigned char))/LIBXSMM_BUILD_PAGESIZE)+1;
  l_code_page_size = LIBXSMM_BUILD_PAGESIZE*l_code_pages;
  l_fd = open("/dev/zero", O_RDWR);
  p = mmap(0, l_code_page_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, l_fd, 0);
  close(l_fd);
  /* explicitly disable THP for this memory region, kernel 2.6.38 or higher! */
  madvise(p, l_code_page_size, MADV_NOHUGEPAGE); 
  if (p == MAP_FAILED) {
    fprintf(stderr, "LIBXSMM: something bad happend in mmap, couldn't allocate code buffer!\n");
    exit(-1);
  }
  l_code = (unsigned char*)p;
  memset( l_code, 0, l_code_page_size );
  memcpy( l_code, l_gen_code, l_generated_code.code_size );
  error = mprotect( (void*)l_code, l_code_page_size, PROT_EXEC | PROT_READ );
  if (error == -1) {
    int errsv = errno;
    if (errsv == EINVAL) {
      fprintf(stderr, "mprotect failed: addr is not a valid pointer, or not a multiple of the system page size!\n");
    } else if (errsv == ENOMEM) {
      fprintf(stderr, "mprotect failed: Internal kernel structures could not be allocated!\n");
    } else if (errsv == EACCES) {
      fprintf(stderr, "mprotect failed: The memory cannot be given the specified access!\n");
    } else {
      fprintf(stderr, "mprotect failed: Unknown Error!\n");
    }
    exit(-1);
  }
  
  /* print function point address */
  printf("function pointer address: %llx\n", (size_t)l_code);
  printf("code pointer address: %llx\n", (size_t)l_gen_code); 

  /* set function pointer and jitted code */
  if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_NONE ) {
    l_test_jit = (jitfun)l_code;
  } else {
    l_test_jit_pf = (jitfun_pf)l_code;
  }

  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("size of generated code: %i\n", l_generated_code.code_size );

  { /* write buffer for manual decode as binary to a file */
    FILE *l_byte_code;
    char l_objdump_name[128];
    sprintf( l_objdump_name, "kernel_%i_%i_%i.bin", i_xgemm_desc->m, i_xgemm_desc->n, i_xgemm_desc->k ); 
    l_byte_code = fopen( l_objdump_name, "wb");

    if ( l_byte_code != NULL ){
      fwrite( (const void*)l_gen_code, 1, l_generated_code.code_size, l_byte_code);
      fclose( l_byte_code );
    } else {
      /* error */
    }
  }

  l_start = libxsmm_timer_tick();

  if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < REPS; l_t++ ) {
      l_test_jit(i_a, i_b, o_c);
    }
  } else {
    for ( l_t = 0; l_t < REPS; l_t++ ) {
      l_test_jit_pf(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for creating jit\n", l_jittime);
  printf("%fs for executing jit\n", l_runtime);
  printf("%f GFLOPS for jit\n", ((double)((double)REPS * (double)i_xgemm_desc->m * (double)i_xgemm_desc->n * (double)i_xgemm_desc->k) * 2.0) / (l_runtime * 1.0e9));

  free(l_gen_code);
}

void max_error_double( const double*                   i_c,
                       const double*                   i_c_gold,
                       const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;
  double l_max_error = 0.0;

  for ( l_i = 0; l_i < i_xgemm_desc->m; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++) {
#if 0
      printf("Entries in row %i, column %i, gold: %f, jit: %f\n", l_i+1, l_j+1, i_c_gold[(l_j*i_xgemm_desc->ldc)+l_i], i_c[(l_j*i_xgemm_desc->ldc)+l_i]);
#endif
      if (l_max_error < fabs( i_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] - i_c[(l_j * i_xgemm_desc->ldc) + l_i]))
        l_max_error = fabs( i_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] - i_c[(l_j * i_xgemm_desc->ldc) + l_i]);
    }
  }

  printf("max. error: %f\n", l_max_error);
}

void max_error_float( const float*                    i_c,
                      const float*                    i_c_gold,
                      const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;
  double l_max_error = 0.0;

  for ( l_i = 0; l_i < i_xgemm_desc->m; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++) {
#if 0
      printf("Entries in row %i, column %i, gold: %f, jit: %f\n", l_i+1, l_j+1, i_c_gold[(l_j*i_xgemm_desc->ldc)+l_i], i_c[(l_j*i_xgemm_desc->ldc)+l_i]);
#endif
      if (l_max_error < fabs( (double)i_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] - (double)i_c[(l_j * i_xgemm_desc->ldc) + l_i]))
        l_max_error = fabs( (double)i_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] - (double)i_c[(l_j * i_xgemm_desc->ldc) + l_i]);
    }
  }

  printf("max. error: %f\n", l_max_error);
}

int main(int argc, char* argv []) {
  char* l_arch = NULL;
  char* l_precision = NULL;
  int l_m = 0;
  int l_n = 0;
  int l_k = 0;
  int l_lda = 0;
  int l_ldb = 0;
  int l_ldc = 0;
  int l_aligned_a = 0;
  int l_aligned_c = 0;
  int l_alpha = 0;
  int l_beta = 0;
  int l_single_precision = 0;
  int l_prefetch = 0;

  libxsmm_gemm_descriptor l_xgemm_desc;
  /* init data structures */
  double* l_a_d; 
  double* l_b_d; 
  double* l_c_d; 
  double* l_c_gold_d;
  float* l_a_f;
  float* l_b_f; 
  float* l_c_f; 
  float* l_c_gold_f;

  /* check argument count for a valid range */
  if ( argc != 14 ) {
    print_help();
    return -1;
  }

  /* xgemm sizes */
  l_m = atoi(argv[1]);
  l_n = atoi(argv[2]);
  l_k = atoi(argv[3]);
  l_lda = atoi(argv[4]);
  l_ldb = atoi(argv[5]);
  l_ldc = atoi(argv[6]);

  /* some sugar */
  l_alpha = atoi(argv[7]);
  l_beta = atoi(argv[8]);
  l_aligned_a = atoi(argv[9]);
  l_aligned_c = atoi(argv[10]);

  /* arch specific stuff */
  l_arch = argv[11];
  l_precision = argv[13];

  /* set value of prefetch flag */
  if (strcmp("nopf", argv[12]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_NONE;
  }
  else if (strcmp("pfsigonly", argv[12]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_SIGNATURE;
  }
  else if (strcmp("BL2viaC", argv[12]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_BL2_VIA_C;
  }
  else if (strcmp("curAL2", argv[12]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2_AHEAD;
  }
  else if (strcmp("curAL2_BL2viaC", argv[12]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD;
  }
  else if (strcmp("AL2", argv[12]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2;
  }
  else if (strcmp("AL2_BL2viaC", argv[12]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C;
  }
  else if (strcmp("AL2jpst", argv[12]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2_JPST;
  }
  else if (strcmp("AL2jpst_BL2viaC", argv[12]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST;
  }
  else {
    print_help();
    return -1;
  }  

  /* check value of arch flag */
  if ( (strcmp(l_arch, "snb") != 0)    && 
       (strcmp(l_arch, "hsw") != 0)    && 
       (strcmp(l_arch, "knl") != 0)    && 
       (strcmp(l_arch, "skx") != 0)       ) {
    print_help();
    return -1;
  }

  /* check and evaluate precison flag */
  if ( strcmp(l_precision, "SP") == 0 ) {
    l_single_precision = 1;
  } else if ( strcmp(l_precision, "DP") == 0 ) {
    l_single_precision = 0;
  } else {
    print_help();
    return -1;
  }

  /* check alpha */
  if ((l_alpha != -1) && (l_alpha != 1)) {
    print_help();
    return -1;
  }

  /* check beta */
  if ((l_beta != 0) && (l_beta != 1)) {
    print_help();
    return -1;
  }

  LIBXSMM_GEMM_DESCRIPTOR(l_xgemm_desc, 1,
    (0 == l_single_precision ? 0 : LIBXSMM_GEMM_FLAG_F32PREC)
      | (0 != l_aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0)
      | (0 != l_aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0),
    l_m, l_n, l_k, l_lda, l_ldb, l_ldc,
    l_alpha, l_beta, l_prefetch);

  if ( l_single_precision == 0 ) {
    l_a_d = (double*)_mm_malloc(l_xgemm_desc.lda * l_xgemm_desc.k * sizeof(double), 64);
    l_b_d = (double*)_mm_malloc(l_xgemm_desc.ldb * l_xgemm_desc.n * sizeof(double), 64);
    l_c_d = (double*)_mm_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(double), 64);
    l_c_gold_d = (double*)_mm_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(double), 64);
    init_double(l_a_d, l_b_d, l_c_d, l_c_gold_d, &l_xgemm_desc);
  } else {
    l_a_f = (float*)_mm_malloc(l_xgemm_desc.lda * l_xgemm_desc.k * sizeof(float), 64);
    l_b_f = (float*)_mm_malloc(l_xgemm_desc.ldb * l_xgemm_desc.n * sizeof(float), 64);
    l_c_f = (float*)_mm_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(float), 64);
    l_c_gold_f = (float*)_mm_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(float), 64);
    init_float(l_a_f, l_b_f, l_c_f, l_c_gold_f, &l_xgemm_desc);
  }

  /* print some output... */
  printf("------------------------------------------------\n");
  printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i)", l_xgemm_desc.m, l_xgemm_desc.k, l_xgemm_desc.k, l_xgemm_desc.n, l_xgemm_desc.m, l_xgemm_desc.n);
  if ( l_single_precision == 0 ) {
    printf(", DP\n");
  } else {
    printf(", SP\n");
  }
  printf("------------------------------------------------\n");

  /* run C */
  if ( l_single_precision == 0 ) {
    run_gold_double( l_a_d, l_b_d, l_c_gold_d, &l_xgemm_desc );
  } else {
    run_gold_float( l_a_f, l_b_f, l_c_gold_f, &l_xgemm_desc );
  }  

  /* run jit */
  if ( l_single_precision == 0 ) {
    run_jit_double( l_a_d, l_b_d, l_c_d, &l_xgemm_desc, l_arch );
  } else {
    run_jit_float( l_a_f, l_b_f, l_c_f, &l_xgemm_desc, l_arch );
  }  
 
  /* test result */
  if ( l_single_precision == 0 ) {
    max_error_double( l_c_d, l_c_gold_d, &l_xgemm_desc );
  } else {
    max_error_float( l_c_f, l_c_gold_f, &l_xgemm_desc );
  }

  /* free */
  if ( l_single_precision == 0 ) {
    _mm_free(l_a_d);
    _mm_free(l_b_d);
    _mm_free(l_c_d);
    _mm_free(l_c_gold_d);
  } else {
    _mm_free(l_a_f);
    _mm_free(l_b_f);
    _mm_free(l_c_f);
    _mm_free(l_c_gold_f);
  }

  printf("------------------------------------------------\n");
  return 0;
}

