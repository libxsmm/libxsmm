/* Optionally link-in the BLAS routines lsame_() and xerbla_() */
#if !defined(__BLAS) || (0 != __BLAS)

#include <stdio.h>

int lsame_(const char* ca, const char* cb)
{
  if ( *ca == *cb ) return 1;
  if ( (*cb >= 'a') && (*cb <= 'z') ) {
    if ( *ca == *cb + 32 ) return 1;
  }
  else if ( (*cb >= 'A') && (*cb <= 'Z') ) {
    if ( *ca == *cb - 32 ) return 1;
  }
  return 0;
}

void xerbla_(const char* c, const int* info)
{
  printf(" ** On entry to %s parameter number %02d had an illegal value\n", c, *info);
}

#endif

