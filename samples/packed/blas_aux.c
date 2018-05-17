#include <stdio.h>

/* Link in if we don't have the BLAS routines lsame_() and xerbla_() */
int lsame_ ( char *ca, char *cb )
{
   if ( *ca == *cb ) return 1;
   if ( (*cb >= 'a') && (*cb <= 'z') )
   {
      if ( *ca == *cb + 32 ) return 1;
   } else if ( (*cb >= 'A') && (*cb <= 'Z') ) {
      if ( *ca == *cb - 32 ) return 1;
   }
   return 0;
}

void xerbla_ ( char *c, int *info )
{
   printf(" ** On entry to %s parameter number %02d had an illegal value\n",c,*info);
}

