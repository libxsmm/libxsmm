#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 1000000


LIBXSMM_INLINE unsigned int isqrt_u32(unsigned int u32)
{
  const unsigned int r = (unsigned int)(sqrt((double)u32) + 0.5);
  return ((double)r * r) <= u32 ? r : (r - 1);
}


LIBXSMM_INLINE unsigned int isqrt_u64(unsigned long long u64)
{
  const unsigned long long r = (unsigned long long)(sqrtl((long double)u64) + 0.5);
  return (unsigned int)(((long double)r * r) <= u64 ? r : (r - 1));
}


LIBXSMM_INLINE unsigned int icbrt_u32(unsigned int u32)
{
  const unsigned int r = (unsigned int)(pow((double)u32, 1.0 / 3.0) + 0.5);
  return ((double)r * r * r) <= u32 ? r : (r - 1);
}


LIBXSMM_INLINE unsigned int icbrt_u64(unsigned long long u64)
{
  const unsigned long long r = (unsigned long long)(powl((long double)u64, 1.0 / 3.0) + 0.5);
  return (unsigned int)(((long double)r * r * r) <= u64 ? r : (r - 1));
}


int main(int argc, char* argv[])
{
  const int exp_maxiter = (1 < argc ? atoi(argv[1]) : 20);
  const unsigned long long scale64 = ((unsigned long long)-1) / (RAND_MAX) - 1;
  const unsigned int scale32 = ((unsigned int)-1) / (RAND_MAX) - 1;
  int warn_dsqrt = 0, warn_ssqrt = 0, i;

  for (i = 0; i < 256; ++i) {
    const float a = libxsmm_sexp2_u8((unsigned char)i);
    const float b = (float)pow(2.0, (double)i);
    if (LIBXSMM_NEQ(a, b)) exit(EXIT_FAILURE);
  }

  for (i = -128; i < 127; ++i) {
    const float a = libxsmm_sexp2_i8((signed char)i);
    const float b = (float)pow(2.0, (double)i);
    if (LIBXSMM_NEQ(a, b)) exit(EXIT_FAILURE);
  }

  for (i = 0; i < (N); ++i) {
    const int r1 = rand(), r2 = rand();
    const double rd = 2.0 * (r1 * (r2 - RAND_MAX / 2)) / RAND_MAX;
    const unsigned long long r64 = scale64 * r1;
    const unsigned int r32 = scale32 * r1;
    double d1, d2, e1, e2, e3;
    unsigned int a, b;

    d1 = libxsmm_sexp2_fast((float)rd, exp_maxiter);
    d2 = powf(2.f, (float)rd);
    e1 = fabs(d1 - d2); e2 = fabs(d2);
    e3 = 0 < e2 ? (e1 / e2) : 0.0;
    if (1E-4 < fmin(e1, e3)) exit(EXIT_FAILURE);

    a = LIBXSMM_SQRT2(r32);
    if ((r32 * 2.0) < ((double)a * a)) {
      exit(EXIT_FAILURE);
    }
    a = LIBXSMM_SQRT2(r64);
    if ((r64 * 2.0) < ((double)a * a)) {
      exit(EXIT_FAILURE);
    }

    a = libxsmm_isqrt_u32(r32);
    b = isqrt_u32(r32);
    if (a != b) exit(EXIT_FAILURE);
    a = libxsmm_isqrt_u64(r64);
    b = isqrt_u64(r64);
    if (a != b) exit(EXIT_FAILURE);
    d1 = libxsmm_ssqrt((float)fabs(rd));
    e1 = fabs(d1 * d1 - fabs(rd));
    d2 = sqrtf((float)fabs(rd));
    e2 = fabs(d2 * d2 - fabs(rd));
    if (e2 < e1) {
      e3 = 0 < e2 ? (e1 / e2) : 0.f;
      if (1E-2 > fmin(fabs(e1 - e2), e3)) {
        ++warn_ssqrt;
      }
      else {
        exit(EXIT_FAILURE);
      }
    }
    d1 = libxsmm_dsqrt(fabs(rd));
    e1 = fabs(d1 * d1 - fabs(rd));
    d2 = sqrt(fabs(rd));
    e2 = fabs(d2 * d2 - fabs(rd));
    if (e2 < e1) {
      e3 = 0 < e2 ? (e1 / e2) : 0.f;
      if (1E-11 > fmin(fabs(e1 - e2), e3)) {
        ++warn_dsqrt;
      }
      else {
        exit(EXIT_FAILURE);
      }
    }

    a = libxsmm_icbrt_u32(r32);
    b = icbrt_u32(r32);
    if (a != b) exit(EXIT_FAILURE);
    a = libxsmm_icbrt_u64(r64);
    b = icbrt_u64(r64);
    if (a != b) exit(EXIT_FAILURE);
  }

  if (0 < warn_ssqrt || 0 < warn_dsqrt) {
    fprintf(stderr, "missed bitwise exact result in %i of %i cases!\n", LIBXSMM_MAX(warn_ssqrt, warn_dsqrt), N);
  }

  { /* check prime factorization */
    const unsigned int test[] = { 0, 1, 2, 3, 5, 7, 12, 13, 2057, 120, 14, 997 };
    const int n = sizeof(test) / sizeof(*test);
    unsigned int fact[32];
    for (i = 0; i < n; ++i) {
      const int np = libxsmm_primes_u32(test[i], fact);
      int j; for (j = 1; j < np; ++j) fact[0] *= fact[j];
      if (0 < np && fact[0] != test[i]) {
        exit(EXIT_FAILURE);
      }
    }
  }
  /* check work division routine */
  if (libxsmm_split_work(12 * 5 * 7 * 11 * 13 * 17, 231) != (3 * 7 * 11)) exit(EXIT_FAILURE);
  if (libxsmm_split_work(12 * 5 * 7, 32) != (2 * 3 * 5)) exit(EXIT_FAILURE);
  if (libxsmm_split_work(12 * 13, 13) != 13) exit(EXIT_FAILURE);
  if (libxsmm_split_work(12, 6) != 6) exit(EXIT_FAILURE);

  return EXIT_SUCCESS;
}

