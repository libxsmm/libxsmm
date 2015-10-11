/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
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
#ifndef GENERATOR_EXTERN_TYPEDEFS_H
#define GENERATOR_EXTERN_TYPEDEFS_H

#define LIBXSMM_XGEMM_DESCRIPTOR(DESCRIPTOR, PRECISION, TRANSA, TRANSB, ALPHA, BETA, M, N, K, LDA, LDB, LDC) \
  (DESCRIPTOR); (DESCRIPTOR).trans_a = (TRANSA); (DESCRIPTOR).trans_b = (TRANSB); \
  (DESCRIPTOR).alpha = (ALPHA); (DESCRIPTOR).beta = (BETA); \
  (DESCRIPTOR).m = (M); (DESCRIPTOR).n = (N); (DESCRIPTOR).k = (K); \
  (DESCRIPTOR).lda = (LDA); (DESCRIPTOR).ldb = (LDB); (DESCRIPTOR).ldc = (LDC); \
  (DESCRIPTOR).aligned_a = 0; (DESCRIPTOR).aligned_c = 0; \
  (DESCRIPTOR).single_precision = (PRECISION); \
  strcpy((DESCRIPTOR).prefetch, "nopf")

/* struct for storing the current xgemm description
   which should be generated */
typedef struct libxsmm_xgemm_descriptor_struct {
  unsigned int m;
  unsigned int n;
  unsigned int k;
  unsigned int lda;
  unsigned int ldb;
  unsigned int ldc;
  int alpha;
  int beta;
  char trans_a;
  char trans_b;
  unsigned int aligned_a;
  unsigned int aligned_c;
  unsigned int single_precision;
  char prefetch[32]; /* TODO do this with ints as well */
} libxsmm_xgemm_descriptor;

/* struct for storing the generated code
   and some information attached to it */
typedef struct libxsmm_generated_code_struct {
  void* generated_code;              /* pointer to memory which can contain strings or binary code */
  unsigned int buffer_size;          /* total size if the buffer generated_code */
  unsigned int code_size;            /* size of bytes used in generated_code */
  unsigned int code_type;            /*   0: generated code contains inline assembly in a C 
                                             function which can be dumped into into a *.c/cc/cpp file
                                          1: generated code contains assembly which can be 
                                             dumped into a *.s file
                                         >1: generated code contains a function in binary code which can be 
                                             called, when the buffer is copied to executable memory */ 
  unsigned int last_error;           /* 0    no error occured
                                        > 0  the occured error code */ 
} libxsmm_generated_code;

/* function to translate LIBXSMM Generator error codes
   to error messages */
const char* libxsmm_strerror( const unsigned int      i_error_code );

#endif /* GENERATOR_EXTERN_TYPEDEFS_H */

