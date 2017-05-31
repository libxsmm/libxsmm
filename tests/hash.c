/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_source.h>
#include <stdlib.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif


/**
 * This test case is NOT an example of how to use LIBXSMM
 * since INTERNAL functions are tested which are not part
 * of the LIBXSMM API.
 */
int main(void)
{
  const unsigned int size = 2507, seed = 1975;
  const unsigned int nbytes = size * sizeof(int);
  unsigned int hash1, hash2, hash3, i;
  int result = EXIT_SUCCESS;

  int *const data = (int*)libxsmm_malloc(nbytes);
  assert(0 != data);

  for (i = 0; i < size; ++i) {
    data[i] = rand() - ((RAND_MAX) >> 1);
  }

  hash1 = libxsmm_crc32(data, nbytes, seed);
  hash2 = libxsmm_crc32_sw(data, nbytes, seed);
  hash3 = libxsmm_hash(data, nbytes, seed);

  libxsmm_free(data);

  if (hash1 != hash2) {
#if defined(_DEBUG)
    fprintf(stderr, "(crc32=%u) != (crc32_sw=%u)\n", hash1, hash2);
#endif
    result = EXIT_FAILURE;
  }

  if (hash1 != hash3) {
#if defined(_DEBUG)
    fprintf(stderr, "(crc32=%u) != (hash=%u)\n", hash1, hash3);
#endif
    result = EXIT_FAILURE;
  }

  return result;
}

