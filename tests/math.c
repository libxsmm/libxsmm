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
  int i;

  for (i = 0; i < 256; ++i) {
    const float a = libxsmm_sexp2_u8((unsigned char)i);
    const float b = (float)pow(2.0, (double)i);
    if (LIBXSMM_NEQ(a, b)) exit(EXIT_FAILURE);
  }

  for (i = 0; i < (N); ++i) {
    const int r1 = rand(), r2 = rand();
    const float rs = 2.f * (r1 * (r2 - RAND_MAX / 2)) / RAND_MAX;
    const unsigned long long r64 = scale64 * r1;
    const unsigned int r32 = scale32 * r1;
    unsigned int a, b;

    const float s1 = libxsmm_sexp2_fast(rs, exp_maxiter);
    const float s2 = powf(2.f, rs), sd = fabsf(s1 - s2);
    const float s3 = fabsf(s2), sr = 0 < s3 ? (sd / s3) : 0.f;
    if (1E-4 < fminf(sd, sr)) exit(EXIT_FAILURE);

    a = LIBXSMM_SQRT2(r32);
    if ((r32 * 2.0) < ((double)a * a)) {
      exit(EXIT_FAILURE);
    }
    a = LIBXSMM_SQRT2(r64);
    if ((r64 * 2.0) < ((double)a * a)) {
      exit(EXIT_FAILURE);
    }

    a = isqrt_u32(r32);
    b = libxsmm_sqrt_u32(r32);
    if (a != b) exit(EXIT_FAILURE);
    a = isqrt_u64(r64);
    b = libxsmm_sqrt_u64(r64);
    if (a != b) exit(EXIT_FAILURE);

    a = icbrt_u32(r32);
    b = libxsmm_cbrt_u32(r32);
    if (a != b) exit(EXIT_FAILURE);
    a = icbrt_u64(r64);
    b = libxsmm_cbrt_u64(r64);
    if (a != b) exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

