#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 10000000


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


int main(void)
{
  const unsigned long long scale64 = ((unsigned long long)-1) / (RAND_MAX) - 1;
  const unsigned int scale32 = ((unsigned int)-1) / (RAND_MAX) - 1;
  int i;
  for (i = 0; i < N; ++i) {
    const unsigned long long r64 = scale64 * rand();
    const unsigned int r32 = scale32 * rand();
    unsigned int a, b;
    a = LIBXSMM_SQRT2(r32);
    if ((r32 * 2.0) < ((double)a * a)) {
      exit(EXIT_FAILURE);
    }
    a = isqrt_u32(r32);
    b = libxsmm_sqrt_u32(r32);
    if (a != b) {
      exit(EXIT_FAILURE);
    }
    a = icbrt_u32(r32);
    b = libxsmm_cbrt_u32(r32);
    if (a != b) {
      exit(EXIT_FAILURE);
    }
    a = LIBXSMM_SQRT2(r64);
    if ((r64 * 2.0) < ((double)a * a)) {
      exit(EXIT_FAILURE);
    }
    a = isqrt_u64(r64);
    b = libxsmm_sqrt_u64(r64);
    if (a != b) {
      exit(EXIT_FAILURE);
    }
    a = icbrt_u64(r64);
    b = libxsmm_cbrt_u64(r64);
    if (a != b) {
      exit(EXIT_FAILURE);
    }
  }
  return EXIT_SUCCESS;
}

