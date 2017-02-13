/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Kunal Banerjee, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <libxsmm_generator.h>
#include <libxsmm_macros.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

LIBXSMM_INLINE void print_help(void) {
  printf("\nwrong usage -> exit!\n\n\n");
  printf("Usage (batched-dense*batched-dense=batched-dense):\n");
  printf("    dense, dense_asm\n");
  printf("    filename to append\n");
  printf("    routine name\n");
  printf("    itiles\n");
  printf("    jtiles\n");
  printf("    bimg\n");
  printf("    ur_i\n");
  printf("    ur_j\n");
  printf("    ur_m\n");
  printf("    vratio\n");
  printf("    ARCH: knm, knl, skx\n");
  printf("    PREFETCH: nopf, all\n");
  printf("    PRECISION: SP\n");
  printf("\n\n\n\n");
}


/* This function finds the prime factors of a number */
void factors( unsigned int num,
              unsigned int num_factors[] )
{
  unsigned int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
  int i;
  unsigned int total_primes = 10;
  unsigned int index = 0;

  for ( i = total_primes-1; i >= 0; i-- ) {
    while((num % primes[i]) == 0) {
      num_factors[index] = primes[i];
      index++;
      num = num/primes[i];
    }
  }
}


/* This function finds the loop increments (ur_i, ur_j, ur_m) of (itiles, jtiles, bimg) */
/* such that ur_i*ur_j*ur_m <= max_acc                                                  */
/* The following loop may not give an optimal solution (knapsack problem)               */
/* Eg, 12 = 3*2*2, MAX_ACC = 4, this algorithm: 3, best: 2*2                            */
void factors_ijm( unsigned int  itiles,
                  unsigned int  jtiles,
                  unsigned int  bimg,
                  unsigned int* ur_i,
                  unsigned int* ur_j,
                  unsigned int* ur_m,
                  unsigned int  max_acc)
{
  unsigned int i;
  unsigned int j;
  unsigned int k;
  unsigned int index;
  int found;
  unsigned int cur_acc;
  unsigned int fact[10];
  unsigned int cur_fact[10];
  unsigned int fact_i[10];
  unsigned int fact_j[10];
  unsigned int fact_m[10];

  for ( i = 0; i < 10; i++ ) {
    fact[i] = 1;
    cur_fact[i] = 1;
  }
  factors(itiles*jtiles*bimg, fact);

  cur_acc = 1;
  index = 0;
  for ( i = 0; fact[i] != 1; i++ ) {
    if ( (fact[i] * cur_acc) <= max_acc ) {
      cur_acc = cur_acc*fact[i];
      cur_fact[index] = fact[i];
      index++;
    }
  }

  for ( i = 0; i < 10; i++ ) {
    fact_i[i] = 1;
    fact_j[i] = 1;
    fact_m[i] = 1;
  }
  factors(itiles, fact_i);
  factors(jtiles, fact_j);
  factors(bimg,   fact_m);

  *ur_i = 1;
  *ur_j = 1;
  *ur_m = 1;

  for ( i= 0; cur_fact[i] != 1; i++ ) {
    found = 0;
    for ( j = 0; fact_i[j] != 1; j++ ) {
      if ( cur_fact[i] == fact_i[j] ) {
        *ur_i = (*ur_i)*fact_i[j];
        found = 1;
        /* Remove this element from fact_i */
        for ( k = j; k < 9; k++ ) {
          fact_i[k] = fact_i[k+1];
        }
        break;
      }
    }
    if ( found == 1 )
      continue;

    for ( j = 0; fact_j[j] != 1; j++ ) {
      if ( cur_fact[i] == fact_j[j] ) {
        *ur_j = (*ur_j)*fact_j[j];
        found = 1;
        /* Remove this element from fact_j */
        for ( k = j; k < 9; k++ ) {
          fact_j[k] = fact_j[k+1];
        }
        break;
      }
    }
    if ( found == 1 )
      continue;

    for ( j = 0; fact_m[j] != 1; j++ ) {
      if ( cur_fact[i] == fact_m[j] ) {
        *ur_m = (*ur_m)*fact_m[j];
        found = 1;
        /* Remove this element from fact_m */
        for ( k = j; k < 9; k++ ) {
          fact_m[k] = fact_m[k+1];
        }
        break;
      }
    }
    if ( found == 1 ) {
      continue;
    }

    fprintf(stderr, "Error: Control should not reach here FACT=%u\n", cur_fact[i]);
  }
}


int main(int argc, char* argv []) {
  libxsmm_convolution_winograd_descriptor l_conv_desc;
  char* l_type;
  char* l_file_out;
  char* l_routine_name;
  char* l_arch;
  char* l_precision;
  int l_prefetch;
  int l_itiles = 0;
  int l_jtiles = 0;
  int l_bimg = 0;
  int l_ur_i = 0;
  int l_ur_j = 0;
  int l_ur_m = 0;
  int l_vratio = 0;
  int l_single_precision = 0;
  int flag_ur = 0;

  /* check argument count for a valid range */
  if (argc != 14) {
    print_help();
    return -1;
  }

  /* names of files and routines */
  l_type = argv[1];
  l_file_out = argv[2];
  l_routine_name = argv[3];

  /* xgemm sizes */
  l_itiles = atoi(argv[4]);
  l_jtiles = atoi(argv[5]);
  l_bimg = atoi(argv[6]);
  l_ur_i = atoi(argv[7]);
  l_ur_j = atoi(argv[8]);
  l_ur_m = atoi(argv[9]);
  l_vratio = atoi(argv[10]);

  /* arch specific stuff */
  l_arch = argv[11];
  l_precision = argv[13];

  /* some intial parameters checks */
  /* check for sparse / dense only */
  if ( (strcmp(l_type, "dense")          != 0) &&
       (strcmp(l_type, "dense_asm")      != 0) ) {
    print_help();
    return -1;
  }

  /* set value of prefetch flag */
  if (strcmp("nopf", argv[12]) == 0) {
    l_prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
  } else if (strcmp("all", argv[12]) == 0) {
    l_prefetch = LIBXSMM_CONVOLUTION_PREFETCH_ALL;
  } else {
    print_help();
    return -1;
  }

  /* check value of arch flag */
  if ( (strcmp(l_arch, "knm") != 0) &&
       (strcmp(l_arch, "knl") != 0) &&
       (strcmp(l_arch, "skx") != 0) ) {
    print_help();
    return -1;
  }

  /* check and evaluate precison flag */
  if ( strcmp(l_precision, "SP") == 0 ) {
    l_single_precision = 1;
    LIBXSMM_UNUSED(l_single_precision/*TODO*/);
  } else {
    print_help();
    return -1;
  }

  if ( 1 > l_vratio ) {
    printf("\nvratio %d must be greater than equal to 1\n", l_vratio);
    return -1;
  }

  if ( !flag_ur && (0 != (l_bimg % l_ur_m)) ) {
    printf("\nbimg %d must be perfectly divisible by ur_m %d\n", l_bimg, l_ur_m);
    return -1;
  }

  if ( !flag_ur && (0 != (l_jtiles % l_ur_j)) ) {
    printf("\njtiles %d must be perfectly divisible by ur_j %d\n", l_jtiles, l_ur_j);
    return -1;
  }

  if ( !flag_ur && (0 != (l_itiles % l_ur_i)) ) {
    printf("\nitiles %d must be perfectly divisible by ur_i %d\n", l_itiles, l_ur_i);
    return -1;
  }

  l_conv_desc.itiles = l_itiles;
  l_conv_desc.jtiles = l_jtiles;
  l_conv_desc.bimg = l_bimg;
  l_conv_desc.vratio = l_vratio;
  l_conv_desc.ur_i = l_ur_i;
  l_conv_desc.ur_j = l_ur_j;
  l_conv_desc.ur_m = l_ur_m;
  l_conv_desc.prefetch = l_prefetch;

  if ( (strcmp(l_type, "dense")     == 0) ||
       (strcmp(l_type, "dense_asm") == 0) ) {
    if ( flag_ur ) {
      factors_ijm( l_conv_desc.itiles, l_conv_desc.jtiles, l_conv_desc.bimg,
                   &(l_conv_desc.ur_i), &(l_conv_desc.ur_j), &(l_conv_desc.ur_m), 26 );
    }

    if ( strcmp(l_type, "dense")  == 0 ) {
       /* libxsmm_generator_convolution_winograd_weight_update_inlineasm( l_file_out, l_routine_name, &l_conv_desc, l_arch ); */
       libxsmm_generator_convolution_winograd_forward_inlineasm( l_file_out, l_routine_name, &l_conv_desc, l_arch );
    } else {
      /* libxsmm_generator_convolution_winograd_weight_update_directasm( l_file_out, l_routine_name, &l_conv_desc, l_arch ); */
      libxsmm_generator_convolution_winograd_forward_directasm( l_file_out, l_routine_name, &l_conv_desc, l_arch );
    }
  }

  return 0;
}

