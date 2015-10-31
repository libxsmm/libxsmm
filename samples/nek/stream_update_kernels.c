/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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

#include <stddef.h>

#if defined(__SSE3__)
#include <immintrin.h>
#endif

void stream_update_axhm( const double* i_g1,
                         const double* i_g2,
                         const double* i_g3, 
                         const double* i_tm1,
                         const double* i_tm2,
                         const double* i_tm3,
                         const double* i_a,
                         const double* i_b,
                         double*       io_c,
                         const double  i_h1,
                         const double  i_h2,
                         const int     i_length) {
  int l_n = 0;
  int l_trip_prolog = 0;
  int l_trip_stream = 0;
  int l_trip_end = i_length;
  
  /* let's calculate the prolog until C is cachline aligned */ 
  /* @TODO check for shifts by the compiler */
  if ( ((size_t)io_c % 64) != 0 ) {
    l_trip_prolog = (64 - ((size_t)io_c % 64))/sizeof(double);
  }
  
  /* let's calculate the end of the streaming part */
  /* @TODO check for shifts by the compiler */
  l_trip_stream = (((i_length-l_trip_prolog)/sizeof(double))*sizeof(double))+l_trip_prolog;

  /* some bound checks */
  l_trip_prolog = (l_trip_prolog > i_length) ? i_length      : l_trip_prolog;
  l_trip_stream = (l_trip_stream > i_length) ? l_trip_prolog : l_trip_stream;
  
  /* run the prologue */
  for ( ; l_n < l_trip_prolog;  l_n++ ) {
    io_c[l_n] =   i_h1*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n]) 
                + i_h2*(i_b[l_n]*i_a[l_n]);
  }
  /* run the bulk, hopefully using streaming stores */
  /* @TODO, check if different archs are needed as well */
#if defined(__AVX__)
  __m256d vec_h1 = _mm256_broadcast_sd(&i_h1);
  __m256d vec_h2 = _mm256_broadcast_sd(&i_h2);
  for ( ; l_n < l_trip_stream;  l_n+=4 ) {
    __m256d vec_g1 = _mm256_loadu_pd(&(i_g1[l_n]));
    __m256d vec_tm1 = _mm256_loadu_pd(&(i_tm1[l_n]));
    vec_g1 = _mm256_mul_pd(vec_g1, vec_tm1);
    __m256d vec_g2 = _mm256_loadu_pd(&(i_g2[l_n]));
    __m256d vec_tm2 = _mm256_loadu_pd(&(i_tm2[l_n]));
    vec_g2 = _mm256_mul_pd(vec_g2, vec_tm2);
    __m256d vec_g3 = _mm256_loadu_pd(&(i_g3[l_n]));
    __m256d vec_tm3 = _mm256_loadu_pd(&(i_tm3[l_n]));
    vec_g3 = _mm256_mul_pd(vec_g3, vec_tm3);
    __m256d vec_a = _mm256_loadu_pd(&(i_a[l_n]));
    __m256d vec_b = _mm256_loadu_pd(&(i_b[l_n]));    
    vec_a = _mm256_mul_pd(vec_a, vec_b);
    vec_g1 = _mm256_add_pd(vec_g1, vec_g2);
    vec_a = _mm256_mul_pd(vec_a, vec_h2);
    vec_g1 = _mm256_add_pd(vec_g1, vec_g3);
    vec_g1 = _mm256_mul_pd(vec_g1, vec_h1);
    _mm256_stream_pd( &(io_c[l_n]), _mm256_add_pd( vec_g1, vec_a ) );
  }
#else
  for ( ; l_n < l_trip_stream;  l_n++ ) {
    io_c[l_n] =   i_h1*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n]) 
                + i_h2*(i_b[l_n]*i_a[l_n]);
  }
#endif
  /* run the epilogue */
  for ( ; l_n < i_length;  l_n++ ) {
    io_c[l_n] =   i_h1*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n]) 
                + i_h2*(i_b[l_n]*i_a[l_n]);
  }
}
