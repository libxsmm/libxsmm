/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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
#include <libxsmm.h>
#include <stddef.h>

#if defined(__SSE3__)
#include <immintrin.h>
#endif

/*#define DISABLE_NONTEMPORAL_STORES*/

LIBXSMM_INLINE_ALWAYS LIBXSMM_RETARGETABLE
void stream_init( int    i_length,
                  size_t i_start_address,
                  int*   o_trip_prolog,
                  int*   o_trip_stream ) {
  /* let's calculate the prolog until C is cachline aligned */
  /* @TODO we need to add shifts */
  if ( (i_start_address % 64) != 0 ) {
    *o_trip_prolog = (64 - (i_start_address % 64))/sizeof(double);
  }

  /* let's calculate the end of the streaming part */
  /* @TODO we need to add shifts */
  *o_trip_stream = (((i_length-(*o_trip_prolog))/sizeof(double))*sizeof(double))+(*o_trip_prolog);

  /* some bound checks */
  *o_trip_prolog = ((*o_trip_prolog) > i_length) ?   i_length       : (*o_trip_prolog);
  *o_trip_stream = ((*o_trip_stream) > i_length) ? (*o_trip_prolog) : (*o_trip_stream);
}

/* avoid warning about external function definition with no prior declaration */
LIBXSMM_API void stream_vector_copy(const double* i_a, double* io_c, const int i_length);
LIBXSMM_API_DEFINITION void stream_vector_copy(  const double* i_a,
                                      double*       io_c,
                                      const int     i_length) {
  int l_n = 0;
  int l_trip_prolog = 0;
  int l_trip_stream = 0;

  /* init the trip counts */
  stream_init( i_length, (size_t)io_c, &l_trip_prolog, &l_trip_stream );

  /* run the prologue */
  for ( ; l_n < l_trip_prolog;  l_n++ ) {
    io_c[l_n] = i_a[l_n];
  }
  /* run the bulk, hopefully using streaming stores */
#if defined(__SSE3__) && defined(__AVX__) && !defined(__AVX512F__)
  {
    /* we need manual unrolling as the compiler otherwise generates
       too many dependencies */
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm256_store_pd(  &(io_c[l_n]),   _mm256_loadu_pd(&(i_a[l_n]))   );
      _mm256_store_pd(  &(io_c[l_n+4]), _mm256_loadu_pd(&(i_a[l_n+4])) );
#else
      _mm256_stream_pd( &(io_c[l_n]),   _mm256_loadu_pd(&(i_a[l_n]))   );
      _mm256_stream_pd( &(io_c[l_n+4]), _mm256_loadu_pd(&(i_a[l_n+4])) );
#endif
    }
  }
#elif defined(__SSE3__) && defined(__AVX__) && defined(__AVX512F__)
  {
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm512_store_pd(  &(io_c[l_n]),   _mm512_loadu_pd(&(i_a[l_n]))   );
#else
      _mm512_stream_pd( &(io_c[l_n]),   _mm512_loadu_pd(&(i_a[l_n]))   );
#endif
    }
  }
#else
  for ( ; l_n < l_trip_stream;  l_n++ ) {
    io_c[l_n] = i_a[l_n];
  }
#endif
  /* run the epilogue */
  for ( ; l_n < i_length;  l_n++ ) {
    io_c[l_n] = i_a[l_n];
  }
}

LIBXSMM_API
void stream_vector_set( const double i_scalar,
                        double*       io_c,
                        const int     i_length) {
  int l_n = 0;
  int l_trip_prolog = 0;
  int l_trip_stream = 0;

  /* init the trip counts */
  stream_init( i_length, (size_t)io_c, &l_trip_prolog, &l_trip_stream );

  /* run the prologue */
  for ( ; l_n < l_trip_prolog;  l_n++ ) {
    io_c[l_n] = i_scalar;
  }
  /* run the bulk, hopefully using streaming stores */
#if defined(__SSE3__) && defined(__AVX__) && !defined(__AVX512F__)
  {
    /* we need manual unrolling as the compiler otherwise generates
       too many dependencies */
    const __m256d vec_scalar = _mm256_broadcast_sd(&i_scalar);
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm256_store_pd(  &(io_c[l_n]),   vec_scalar );
      _mm256_store_pd(  &(io_c[l_n+4]), vec_scalar );
#else
      _mm256_stream_pd( &(io_c[l_n]),   vec_scalar );
      _mm256_stream_pd( &(io_c[l_n+4]), vec_scalar );
#endif
    }
  }
#elif defined(__SSE3__) && defined(__AVX__) && defined(__AVX512F__)
  {
    const __m512d vec_scalar = _mm512_broadcastsd_pd(_mm_load_sd(&i_scalar));
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm512_store_pd(  &(io_c[l_n]), vec_scalar );
#else
      _mm512_stream_pd( &(io_c[l_n]), vec_scalar );
#endif
    }
  }
#else
  for ( ; l_n < l_trip_stream;  l_n++ ) {
    io_c[l_n] = i_scalar;
  }
#endif
  /* run the epilogue */
  for ( ; l_n < i_length;  l_n++ ) {
    io_c[l_n] = i_scalar;
  }
}


LIBXSMM_API
void stream_vector_compscale( const double* i_a,
                              const double* i_b,
                              double*       io_c,
                              const int     i_length) {
  int l_n = 0;
  int l_trip_prolog = 0;
  int l_trip_stream = 0;

  /* init the trip counts */
  stream_init( i_length, (size_t)io_c, &l_trip_prolog, &l_trip_stream );

  /* run the prologue */
  for ( ; l_n < l_trip_prolog;  l_n++ ) {
    io_c[l_n] = i_a[l_n]*i_b[l_n];
  }
  /* run the bulk, hopefully using streaming stores */
#if defined(__SSE3__) && defined(__AVX__) && !defined(__AVX512F__)
  {
    /* we need manual unrolling as the compiler otherwise generates
       too many dependencies */
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
      __m256d vec_a_1, vec_b_1;
      __m256d vec_a_2, vec_b_2;

      vec_a_1 = _mm256_loadu_pd(&(i_a[l_n]));
      vec_a_2 = _mm256_loadu_pd(&(i_a[l_n+4]));
      vec_b_1 = _mm256_loadu_pd(&(i_b[l_n]));
      vec_b_2 = _mm256_loadu_pd(&(i_b[l_n+4]));

#ifdef DISABLE_NONTEMPORAL_STORES
      _mm256_store_pd(  &(io_c[l_n]),   _mm256_mul_pd( vec_a_1, vec_b_1 ) );
      _mm256_store_pd(  &(io_c[l_n+4]), _mm256_mul_pd( vec_a_2, vec_b_2 ) );
#else
      _mm256_stream_pd( &(io_c[l_n]),   _mm256_mul_pd( vec_a_1, vec_b_1 ) );
      _mm256_stream_pd( &(io_c[l_n+4]), _mm256_mul_pd( vec_a_2, vec_b_2 ) );
#endif
    }
  }
#elif defined(__SSE3__) && defined(__AVX__) && defined(__AVX512F__)
  {
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
      __m512d vec_a, vec_b;

      vec_a = _mm512_loadu_pd(&(i_a[l_n]));
      vec_b = _mm512_loadu_pd(&(i_b[l_n]));

#ifdef DISABLE_NONTEMPORAL_STORES
      _mm512_store_pd(  &(io_c[l_n]), _mm512_mul_pd( vec_a, vec_b ) );
#else
      _mm512_stream_pd( &(io_c[l_n]), _mm512_mul_pd( vec_a, vec_b ) );
#endif
    }
  }
#else
  for ( ; l_n < l_trip_stream;  l_n++ ) {
    io_c[l_n] = i_a[l_n]*i_b[l_n];
  }
#endif
  /* run the epilogue */
  for ( ; l_n < i_length;  l_n++ ) {
    io_c[l_n] = i_a[l_n]*i_b[l_n];
  }
}

LIBXSMM_API
void stream_update_helmholtz( const double* i_g1,
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

  /* init the trip counts */
  stream_init( i_length, (size_t)io_c, &l_trip_prolog, &l_trip_stream );

  /* run the prologue */
/*
#if !defined(__SSE3__)
*/
  {
    for ( ; l_n < l_trip_prolog;  l_n++ ) {
      io_c[l_n] =   i_h1*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n])
                  + i_h2*(i_b[l_n]*i_a[l_n]);
    }
  }
/*
#else
  {
    const __m128d vec_h1 = _mm_loaddup_pd(&i_h1);
    const __m128d vec_h2 = _mm_loaddup_pd(&i_h2);
    const __m128i mask   = _mm_set_epi32(0,0,-1,-1);
    for ( ; l_n < l_trip_prolog;  l_n++ ) {
      __m128d vec_g1, vec_g2, vec_g3, vec_tm1, vec_tm2, vec_tm3, vec_a, vec_b;
      vec_g1 = _mm_load_sd(&(i_g1[l_n]));
      vec_tm1 = _mm_load_sd(&(i_tm1[l_n]));
      vec_g1 = _mm_mul_sd(vec_g1, vec_tm1);
      vec_g2 = _mm_load_sd(&(i_g2[l_n]));
      vec_tm2 = _mm_load_sd(&(i_tm2[l_n]));
      vec_g2 = _mm_mul_sd(vec_g2, vec_tm2);
      vec_g3 = _mm_load_sd(&(i_g3[l_n]));
      vec_tm3 = _mm_load_sd(&(i_tm3[l_n]));
      vec_g3 = _mm_mul_sd(vec_g3, vec_tm3);
      vec_a = _mm_load_sd(&(i_a[l_n]));
      vec_b = _mm_load_sd(&(i_b[l_n]));
      vec_a = _mm_mul_sd(vec_a, vec_b);
      vec_g1 = _mm_add_sd(vec_g1, vec_g2);
      vec_a = _mm_mul_sd(vec_a, vec_h2);
      vec_g1 = _mm_add_sd(vec_g1, vec_g3);
      vec_g1 = _mm_mul_sd(vec_g1, vec_h1);
      _mm_maskmoveu_si128(_mm_castpd_si128(_mm_add_pd( vec_g1, vec_a )), mask, (char*)(&(io_c[l_n])));
    }
  }
#endif
*/
  /* run the bulk, hopefully using streaming stores */
#if defined(__SSE3__) && defined(__AVX__) && !defined(__AVX512F__)
  {
    const __m256d vec_h1 = _mm256_broadcast_sd(&i_h1);
    const __m256d vec_h2 = _mm256_broadcast_sd(&i_h2);
    /* we need manual unrolling as the compiler otherwise generates
       too many dependencies */
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
      __m256d vec_g1_1, vec_g2_1, vec_g3_1, vec_tm1_1, vec_tm2_1, vec_tm3_1, vec_a_1, vec_b_1;
      __m256d vec_g1_2, vec_g2_2, vec_g3_2, vec_tm1_2, vec_tm2_2, vec_tm3_2, vec_a_2, vec_b_2;

      vec_g1_1 = _mm256_loadu_pd(&(i_g1[l_n]));
      vec_tm1_1 = _mm256_loadu_pd(&(i_tm1[l_n]));
      vec_g1_2 = _mm256_loadu_pd(&(i_g1[l_n+4]));
      vec_tm1_2 = _mm256_loadu_pd(&(i_tm1[l_n+4]));

      vec_g1_1 = _mm256_mul_pd(vec_g1_1, vec_tm1_1);
      vec_g2_1 = _mm256_loadu_pd(&(i_g2[l_n]));
      vec_g1_2 = _mm256_mul_pd(vec_g1_2, vec_tm1_2);
      vec_g2_2 = _mm256_loadu_pd(&(i_g2[l_n+4]));

      vec_tm2_1 = _mm256_loadu_pd(&(i_tm2[l_n]));
      vec_g2_1 = _mm256_mul_pd(vec_g2_1, vec_tm2_1);
      vec_tm2_2 = _mm256_loadu_pd(&(i_tm2[l_n+4]));
      vec_g2_2 = _mm256_mul_pd(vec_g2_2, vec_tm2_2);

      vec_g3_1 = _mm256_loadu_pd(&(i_g3[l_n]));
      vec_tm3_1 = _mm256_loadu_pd(&(i_tm3[l_n]));
      vec_g3_2 = _mm256_loadu_pd(&(i_g3[l_n+4]));
      vec_tm3_2 = _mm256_loadu_pd(&(i_tm3[l_n+4]));

      vec_g3_1 = _mm256_mul_pd(vec_g3_1, vec_tm3_1);
      vec_a_1 = _mm256_loadu_pd(&(i_a[l_n]));
      vec_g3_2 = _mm256_mul_pd(vec_g3_2, vec_tm3_2);
      vec_a_2 = _mm256_loadu_pd(&(i_a[l_n+4]));

      vec_b_1 = _mm256_loadu_pd(&(i_b[l_n]));
      vec_a_1 = _mm256_mul_pd(vec_a_1, vec_b_1);
      vec_b_2 = _mm256_loadu_pd(&(i_b[l_n+4]));
      vec_a_2 = _mm256_mul_pd(vec_a_2, vec_b_2);

      vec_g1_1 = _mm256_add_pd(vec_g1_1, vec_g2_1);
      vec_a_1 = _mm256_mul_pd(vec_a_1, vec_h2);
      vec_g1_2 = _mm256_add_pd(vec_g1_2, vec_g2_2);
      vec_a_2 = _mm256_mul_pd(vec_a_2, vec_h2);

      vec_g1_1 = _mm256_add_pd(vec_g1_1, vec_g3_1);
      vec_g1_1 = _mm256_mul_pd(vec_g1_1, vec_h1);
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm256_store_pd(  &(io_c[l_n]), _mm256_add_pd( vec_g1_1, vec_a_1 ) );
#else
      _mm256_stream_pd( &(io_c[l_n]), _mm256_add_pd( vec_g1_1, vec_a_1 ) );
#endif

      vec_g1_2 = _mm256_add_pd(vec_g1_2, vec_g3_2);
      vec_g1_2 = _mm256_mul_pd(vec_g1_2, vec_h1);
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm256_store_pd(  &(io_c[l_n+4]), _mm256_add_pd( vec_g1_2, vec_a_2 ) );
#else
      _mm256_stream_pd( &(io_c[l_n+4]), _mm256_add_pd( vec_g1_2, vec_a_2 ) );
#endif
    }
  }
#elif defined(__SSE3__) && defined(__AVX__) && defined(__AVX512F__)
  {
    const __m512d vec_h1 = _mm512_broadcastsd_pd(_mm_load_sd(&i_h1));
    const __m512d vec_h2 = _mm512_broadcastsd_pd(_mm_load_sd(&i_h2));
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
      __m512d vec_g1, vec_g2, vec_g3, vec_tm1, vec_tm2, vec_tm3, vec_a, vec_b;
      vec_g1 = _mm512_loadu_pd(&(i_g1[l_n]));
      vec_tm1 = _mm512_loadu_pd(&(i_tm1[l_n]));
      vec_g1 = _mm512_mul_pd(vec_g1, vec_tm1);
      vec_g2 = _mm512_loadu_pd(&(i_g2[l_n]));
      vec_tm2 = _mm512_loadu_pd(&(i_tm2[l_n]));
      vec_g2 = _mm512_mul_pd(vec_g2, vec_tm2);
      vec_g3 = _mm512_loadu_pd(&(i_g3[l_n]));
      vec_tm3 = _mm512_loadu_pd(&(i_tm3[l_n]));
      vec_g3 = _mm512_mul_pd(vec_g3, vec_tm3);
      vec_a = _mm512_loadu_pd(&(i_a[l_n]));
      vec_b = _mm512_loadu_pd(&(i_b[l_n]));
      vec_a = _mm512_mul_pd(vec_a, vec_b);
      vec_g1 = _mm512_add_pd(vec_g1, vec_g2);
      vec_a = _mm512_mul_pd(vec_a, vec_h2);
      vec_g1 = _mm512_add_pd(vec_g1, vec_g3);
      vec_g1 = _mm512_mul_pd(vec_g1, vec_h1);
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm512_store_pd(  &(io_c[l_n]), _mm512_add_pd( vec_g1, vec_a ) );
#else
      _mm512_stream_pd( &(io_c[l_n]), _mm512_add_pd( vec_g1, vec_a ) );
#endif
    }
  }
#else
  for ( ; l_n < l_trip_stream;  l_n++ ) {
    io_c[l_n] =   i_h1*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n])
                + i_h2*(i_b[l_n]*i_a[l_n]);
  }
#endif
  /* run the epilogue */
/*
#if !defined(__SSE3__)
*/
  {
    for ( ; l_n < i_length;  l_n++ ) {
      io_c[l_n] =   i_h1*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n])
                  + i_h2*(i_b[l_n]*i_a[l_n]);
    }
  }
/*
#else
  {
    const __m128d vec_h1 = _mm_loaddup_pd(&i_h1);
    const __m128d vec_h2 = _mm_loaddup_pd(&i_h2);
    const __m128i mask   = _mm_set_epi32(0,0,-1,-1);
    for ( ; l_n < i_length;  l_n++ ) {
      __m128d vec_g1, vec_g2, vec_g3, vec_tm1, vec_tm2, vec_tm3, vec_a, vec_b;
      vec_g1 = _mm_load_sd(&(i_g1[l_n]));
      vec_tm1 = _mm_load_sd(&(i_tm1[l_n]));
      vec_g1 = _mm_mul_sd(vec_g1, vec_tm1);
      vec_g2 = _mm_load_sd(&(i_g2[l_n]));
      vec_tm2 = _mm_load_sd(&(i_tm2[l_n]));
      vec_g2 = _mm_mul_sd(vec_g2, vec_tm2);
      vec_g3 = _mm_load_sd(&(i_g3[l_n]));
      vec_tm3 = _mm_load_sd(&(i_tm3[l_n]));
      vec_g3 = _mm_mul_sd(vec_g3, vec_tm3);
      vec_a = _mm_load_sd(&(i_a[l_n]));
      vec_b = _mm_load_sd(&(i_b[l_n]));
      vec_a = _mm_mul_sd(vec_a, vec_b);
      vec_g1 = _mm_add_sd(vec_g1, vec_g2);
      vec_a = _mm_mul_sd(vec_a, vec_h2);
      vec_g1 = _mm_add_sd(vec_g1, vec_g3);
      vec_g1 = _mm_mul_sd(vec_g1, vec_h1);
      _mm_maskmoveu_si128(_mm_castpd_si128(_mm_add_pd( vec_g1, vec_a )), mask, (char*)(&(io_c[l_n])));
    }
  }
#endif
*/
}

LIBXSMM_API
void stream_update_helmholtz_no_h2( const double* i_g1,
                                    const double* i_g2,
                                    const double* i_g3,
                                    const double* i_tm1,
                                    const double* i_tm2,
                                    const double* i_tm3,
                                    double*       io_c,
                                    const double  i_h1,
                                    const int     i_length) {
  int l_n = 0;
  int l_trip_prolog = 0;
  int l_trip_stream = 0;

  /* init the trip counts */
  stream_init( i_length, (size_t)io_c, &l_trip_prolog, &l_trip_stream );

  /* run the prologue */
  for ( ; l_n < l_trip_prolog;  l_n++ ) {
    io_c[l_n] =   i_h1*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n]);
  }
  /* run the bulk, hopefully using streaming stores */
#if defined(__SSE3__) && defined(__AVX__) && !defined(__AVX512F__)
  {
    const __m256d vec_h1 = _mm256_broadcast_sd(&i_h1);
    /* we need manual unrolling as the compiler otherwise generates
       too many dependencies */
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
      __m256d vec_g1_1, vec_g2_1, vec_g3_1, vec_tm1_1, vec_tm2_1, vec_tm3_1;
      __m256d vec_g1_2, vec_g2_2, vec_g3_2, vec_tm1_2, vec_tm2_2, vec_tm3_2;

      vec_g1_1 = _mm256_loadu_pd(&(i_g1[l_n]));
      vec_tm1_1 = _mm256_loadu_pd(&(i_tm1[l_n]));
      vec_g1_2 = _mm256_loadu_pd(&(i_g1[l_n+4]));
      vec_tm1_2 = _mm256_loadu_pd(&(i_tm1[l_n+4]));

      vec_g1_1 = _mm256_mul_pd(vec_g1_1, vec_tm1_1);
      vec_g2_1 = _mm256_loadu_pd(&(i_g2[l_n]));
      vec_g1_2 = _mm256_mul_pd(vec_g1_2, vec_tm1_2);
      vec_g2_2 = _mm256_loadu_pd(&(i_g2[l_n+4]));

      vec_tm2_1 = _mm256_loadu_pd(&(i_tm2[l_n]));
      vec_g2_1 = _mm256_mul_pd(vec_g2_1, vec_tm2_1);
      vec_tm2_2 = _mm256_loadu_pd(&(i_tm2[l_n+4]));
      vec_g2_2 = _mm256_mul_pd(vec_g2_2, vec_tm2_2);

      vec_g3_1 = _mm256_loadu_pd(&(i_g3[l_n]));
      vec_tm3_1 = _mm256_loadu_pd(&(i_tm3[l_n]));
      vec_g3_2 = _mm256_loadu_pd(&(i_g3[l_n+4]));
      vec_tm3_2 = _mm256_loadu_pd(&(i_tm3[l_n+4]));

      vec_g3_1 = _mm256_mul_pd(vec_g3_1, vec_tm3_1);
      vec_g3_2 = _mm256_mul_pd(vec_g3_2, vec_tm3_2);
      vec_g1_1 = _mm256_add_pd(vec_g1_1, vec_g2_1);
      vec_g1_2 = _mm256_add_pd(vec_g1_2, vec_g2_2);

      vec_g1_1 = _mm256_add_pd(vec_g1_1, vec_g3_1);
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm256_store_pd(  &(io_c[l_n]), _mm256_mul_pd(vec_g1_1, vec_h1) );
#else
      _mm256_stream_pd( &(io_c[l_n]), _mm256_mul_pd(vec_g1_1, vec_h1) );
#endif
      vec_g1_2 = _mm256_add_pd(vec_g1_2, vec_g3_2);
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm256_store_pd(  &(io_c[l_n+4]), _mm256_mul_pd(vec_g1_2, vec_h1) );
#else
      _mm256_stream_pd( &(io_c[l_n+4]), _mm256_mul_pd(vec_g1_2, vec_h1) );
#endif
    }
  }
#elif defined(__SSE3__) && defined(__AVX__) && defined(__AVX512F__)
  {
    const __m512d vec_h1 = _mm512_broadcastsd_pd(_mm_load_sd(&i_h1));
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
      __m512d vec_g1, vec_g2, vec_g3, vec_tm1, vec_tm2, vec_tm3;
      vec_g1 = _mm512_loadu_pd(&(i_g1[l_n]));
      vec_tm1 = _mm512_loadu_pd(&(i_tm1[l_n]));
      vec_g1 = _mm512_mul_pd(vec_g1, vec_tm1);
      vec_g2 = _mm512_loadu_pd(&(i_g2[l_n]));
      vec_tm2 = _mm512_loadu_pd(&(i_tm2[l_n]));
      vec_g2 = _mm512_mul_pd(vec_g2, vec_tm2);
      vec_g3 = _mm512_loadu_pd(&(i_g3[l_n]));
      vec_tm3 = _mm512_loadu_pd(&(i_tm3[l_n]));
      vec_g3 = _mm512_mul_pd(vec_g3, vec_tm3);
      vec_g1 = _mm512_add_pd(vec_g1, vec_g2);
      vec_g1 = _mm512_add_pd(vec_g1, vec_g3);
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm512_store_pd(  &(io_c[l_n]), _mm512_mul_pd(vec_g1, vec_h1) );
#else
      _mm512_stream_pd( &(io_c[l_n]), _mm512_mul_pd(vec_g1, vec_h1) );
#endif
    }
  }
#else
  for ( ; l_n < l_trip_stream;  l_n++ ) {
    io_c[l_n] =   i_h1*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n]);
  }
#endif
  /* run the epilogue */
  for ( ; l_n < i_length;  l_n++ ) {
    io_c[l_n] =   i_h1*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n]);
  }
}

LIBXSMM_API
void stream_update_var_helmholtz( const double* i_g1,
                                  const double* i_g2,
                                  const double* i_g3,
                                  const double* i_tm1,
                                  const double* i_tm2,
                                  const double* i_tm3,
                                  const double* i_a,
                                  const double* i_b,
                                  double*       io_c,
                                  const double* i_h1,
                                  const double* i_h2,
                                  const int     i_length) {
  int l_n = 0;
  int l_trip_prolog = 0;
  int l_trip_stream = 0;

  /* init the trip counts */
  stream_init( i_length, (size_t)io_c, &l_trip_prolog, &l_trip_stream );

  /* run the prologue */
/*
#if !defined(__SSE3__)
*/
  {
    for ( ; l_n < l_trip_prolog;  l_n++ ) {
      io_c[l_n] =   i_h1[l_n]*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n])
                  + i_h2[l_n]*(i_b[l_n]*i_a[l_n]);
    }
  }
/*
#else
  {
    const __m128i mask   = _mm_set_epi32(0,0,-1,-1);
    for ( ; l_n < l_trip_prolog;  l_n++ ) {
      __m128d vec_g1, vec_g2, vec_g3, vec_tm1, vec_tm2, vec_tm3, vec_a, vec_b, vec_h1, vec_h2;
      vec_g1 = _mm_load_sd(&(i_g1[l_n]));
      vec_tm1 = _mm_load_sd(&(i_tm1[l_n]));
      vec_g1 = _mm_mul_sd(vec_g1, vec_tm1);
      vec_g2 = _mm_load_sd(&(i_g2[l_n]));
      vec_tm2 = _mm_load_sd(&(i_tm2[l_n]));
      vec_g2 = _mm_mul_sd(vec_g2, vec_tm2);
      vec_g3 = _mm_load_sd(&(i_g3[l_n]));
      vec_tm3 = _mm_load_sd(&(i_tm3[l_n]));
      vec_g3 = _mm_mul_sd(vec_g3, vec_tm3);
      vec_a = _mm_load_sd(&(i_a[l_n]));
      vec_b = _mm_load_sd(&(i_b[l_n]));
      vec_a = _mm_mul_sd(vec_a, vec_b);
      vec_g1 = _mm_add_sd(vec_g1, vec_g2);
      vec_h2 = _mm_load_sd(&(i_h2[l_n]));
      vec_a = _mm_mul_sd(vec_a, vec_h2);
      vec_g1 = _mm_add_sd(vec_g1, vec_g3);
      vec_h1 = _mm_load_sd(&(i_h1[l_n]));
      vec_g1 = _mm_mul_sd(vec_g1, vec_h1);
      _mm_maskmoveu_si128(_mm_castpd_si128(_mm_add_pd( vec_g1, vec_a )), mask, (char*)(&(io_c[l_n])));
    }
  }
#endif
*/
  /* run the bulk, hopefully using streaming stores */
#if defined(__SSE3__) && defined(__AVX__) && !defined(__AVX512F__)
  {
    /* we need manual unrolling as the compiler otherwise generates
       too many dependencies */
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
      __m256d vec_g1_1, vec_g2_1, vec_g3_1, vec_tm1_1, vec_tm2_1, vec_tm3_1, vec_a_1, vec_b_1, vec_h1_1, vec_h2_1;
      __m256d vec_g1_2, vec_g2_2, vec_g3_2, vec_tm1_2, vec_tm2_2, vec_tm3_2, vec_a_2, vec_b_2, vec_h1_2, vec_h2_2;

      vec_g1_1 = _mm256_loadu_pd(&(i_g1[l_n]));
      vec_tm1_1 = _mm256_loadu_pd(&(i_tm1[l_n]));
      vec_g1_2 = _mm256_loadu_pd(&(i_g1[l_n+4]));
      vec_tm1_2 = _mm256_loadu_pd(&(i_tm1[l_n+4]));

      vec_g1_1 = _mm256_mul_pd(vec_g1_1, vec_tm1_1);
      vec_g2_1 = _mm256_loadu_pd(&(i_g2[l_n]));
      vec_g1_2 = _mm256_mul_pd(vec_g1_2, vec_tm1_2);
      vec_g2_2 = _mm256_loadu_pd(&(i_g2[l_n+4]));

      vec_tm2_1 = _mm256_loadu_pd(&(i_tm2[l_n]));
      vec_g2_1 = _mm256_mul_pd(vec_g2_1, vec_tm2_1);
      vec_tm2_2 = _mm256_loadu_pd(&(i_tm2[l_n+4]));
      vec_g2_2 = _mm256_mul_pd(vec_g2_2, vec_tm2_2);

      vec_g3_1 = _mm256_loadu_pd(&(i_g3[l_n]));
      vec_tm3_1 = _mm256_loadu_pd(&(i_tm3[l_n]));
      vec_g3_2 = _mm256_loadu_pd(&(i_g3[l_n+4]));
      vec_tm3_2 = _mm256_loadu_pd(&(i_tm3[l_n+4]));

      vec_g3_1 = _mm256_mul_pd(vec_g3_1, vec_tm3_1);
      vec_a_1 = _mm256_loadu_pd(&(i_a[l_n]));
      vec_g3_2 = _mm256_mul_pd(vec_g3_2, vec_tm3_2);
      vec_a_2 = _mm256_loadu_pd(&(i_a[l_n+4]));

      vec_b_1 = _mm256_loadu_pd(&(i_b[l_n]));
      vec_a_1 = _mm256_mul_pd(vec_a_1, vec_b_1);
      vec_b_2 = _mm256_loadu_pd(&(i_b[l_n+4]));
      vec_a_2 = _mm256_mul_pd(vec_a_2, vec_b_2);

      vec_h2_1 = _mm256_loadu_pd(&(i_h2[l_n]));
      vec_g1_1 = _mm256_add_pd(vec_g1_1, vec_g2_1);
      vec_a_1 = _mm256_mul_pd(vec_a_1, vec_h2_1);
      vec_h2_2 = _mm256_loadu_pd(&(i_h2[l_n+4]));
      vec_g1_2 = _mm256_add_pd(vec_g1_2, vec_g2_2);
      vec_a_2 = _mm256_mul_pd(vec_a_2, vec_h2_2);

      vec_h1_1 = _mm256_loadu_pd(&(i_h1[l_n]));
      vec_g1_1 = _mm256_add_pd(vec_g1_1, vec_g3_1);
      vec_g1_1 = _mm256_mul_pd(vec_g1_1, vec_h1_1);
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm256_store_pd(  &(io_c[l_n]), _mm256_add_pd( vec_g1_1, vec_a_1 ) );
#else
      _mm256_stream_pd( &(io_c[l_n]), _mm256_add_pd( vec_g1_1, vec_a_1 ) );
#endif

      vec_h1_2 = _mm256_loadu_pd(&(i_h1[l_n+4]));
      vec_g1_2 = _mm256_add_pd(vec_g1_2, vec_g3_2);
      vec_g1_2 = _mm256_mul_pd(vec_g1_2, vec_h1_2);
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm256_store_pd(  &(io_c[l_n+4]), _mm256_add_pd( vec_g1_2, vec_a_2 ) );
#else
      _mm256_stream_pd( &(io_c[l_n+4]), _mm256_add_pd( vec_g1_2, vec_a_2 ) );
#endif
    }
  }
#elif defined(__SSE3__) && defined(__AVX__) && defined(__AVX512F__)
  {
    for ( ; l_n < l_trip_stream;  l_n+=8 ) {
      __m512d vec_g1, vec_g2, vec_g3, vec_tm1, vec_tm2, vec_tm3, vec_a, vec_b, vec_h1, vec_h2;
      vec_g1 = _mm512_loadu_pd(&(i_g1[l_n]));
      vec_tm1 = _mm512_loadu_pd(&(i_tm1[l_n]));
      vec_g1 = _mm512_mul_pd(vec_g1, vec_tm1);
      vec_g2 = _mm512_loadu_pd(&(i_g2[l_n]));
      vec_tm2 = _mm512_loadu_pd(&(i_tm2[l_n]));
      vec_g2 = _mm512_mul_pd(vec_g2, vec_tm2);
      vec_g3 = _mm512_loadu_pd(&(i_g3[l_n]));
      vec_tm3 = _mm512_loadu_pd(&(i_tm3[l_n]));
      vec_g3 = _mm512_mul_pd(vec_g3, vec_tm3);
      vec_a = _mm512_loadu_pd(&(i_a[l_n]));
      vec_b = _mm512_loadu_pd(&(i_b[l_n]));
      vec_a = _mm512_mul_pd(vec_a, vec_b);
      vec_g1 = _mm512_add_pd(vec_g1, vec_g2);
      vec_h2 = _mm512_loadu_pd(&(i_h2[l_n]));
      vec_a = _mm512_mul_pd(vec_a, vec_h2);
      vec_g1 = _mm512_add_pd(vec_g1, vec_g3);
      vec_h1 = _mm512_loadu_pd(&(i_h1[l_n]));
      vec_g1 = _mm512_mul_pd(vec_g1, vec_h1);
#ifdef DISABLE_NONTEMPORAL_STORES
      _mm512_store_pd(  &(io_c[l_n]), _mm512_add_pd( vec_g1, vec_a ) );
#else
      _mm512_stream_pd( &(io_c[l_n]), _mm512_add_pd( vec_g1, vec_a ) );
#endif
    }
  }
#else
  for ( ; l_n < l_trip_stream;  l_n++ ) {
    io_c[l_n] =   i_h1[l_n]*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n])
                + i_h2[l_n]*(i_b[l_n]*i_a[l_n]);
  }
#endif
  /* run the epilogue */
/*
#if !defined(__SSE3__)
*/
  {
    for ( ; l_n < i_length;  l_n++ ) {
      io_c[l_n] =   i_h1[l_n]*(i_g1[l_n]*i_tm1[l_n] + i_g2[l_n]*i_tm2[l_n] + i_g3[l_n]*i_tm3[l_n])
                  + i_h2[l_n]*(i_b[l_n]*i_a[l_n]);
    }
  }
/*
#else
  {
    const __m128i mask   = _mm_set_epi32(0,0,-1,-1);
    for ( ; l_n < i_length;  l_n++ ) {
      __m128d vec_g1, vec_g2, vec_g3, vec_tm1, vec_tm2, vec_tm3, vec_a, vec_b;
      vec_g1 = _mm_load_sd(&(i_g1[l_n]));
      vec_tm1 = _mm_load_sd(&(i_tm1[l_n]));
      vec_g1 = _mm_mul_sd(vec_g1, vec_tm1);
      vec_g2 = _mm_load_sd(&(i_g2[l_n]));
      vec_tm2 = _mm_load_sd(&(i_tm2[l_n]));
      vec_g2 = _mm_mul_sd(vec_g2, vec_tm2);
      vec_g3 = _mm_load_sd(&(i_g3[l_n]));
      vec_tm3 = _mm_load_sd(&(i_tm3[l_n]));
      vec_g3 = _mm_mul_sd(vec_g3, vec_tm3);
      vec_a = _mm_load_sd(&(i_a[l_n]));
      vec_b = _mm_load_sd(&(i_b[l_n]));
      vec_a = _mm_mul_sd(vec_a, vec_b);
      vec_g1 = _mm_add_sd(vec_g1, vec_g2);
      vec_h2 = _mm_load_sd(&(i_h2[l_n]));
      vec_a = _mm_mul_sd(vec_a, vec_h2);
      vec_g1 = _mm_add_sd(vec_g1, vec_g3);
       vec_h1 = _mm_load_sd(&(i_h1[l_n]));
      vec_g1 = _mm_mul_sd(vec_g1, vec_h1);
      _mm_maskmoveu_si128(_mm_castpd_si128(_mm_add_pd( vec_g1, vec_a )), mask, (char*)(&(io_c[l_n])));
    }
  }
#endif
*/
}
