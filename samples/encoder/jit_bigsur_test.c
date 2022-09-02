/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <sys/types.h>
# include <sys/mman.h>
# include <sys/stat.h>
# include <unistd.h>
# include <fcntl.h>
# ifdef __APPLE__
# include <libkern/OSCacheControl.h>
# endif
# include <pthread.h>

# include <generator_aarch64_instructions.h>

#if 0
/*__APPLE__*/
unsigned int libxsmm_ninit;
int libxsmm_verbosity;
#endif

typedef void (*reset_func)(float* in);

void* dynamic_reset_zero_create() {
  unsigned char* codebuffer = (unsigned char*)malloc( 4096*sizeof(unsigned char) );
  libxsmm_generated_code mycode;

  /* init generated code object */
  mycode.generated_code = codebuffer;
  mycode.buffer_size = 4096;
  mycode.arch = LIBXSMM_AARCH64_V81;
  mycode.code_size = 0;
  mycode.code_type = 2;
  mycode.last_error = 0;
  mycode.sf_size = 0;
  memset( mycode.generated_code, 0, mycode.buffer_size );

  libxsmm_aarch64_instruction_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                          LIBXSMM_AARCH64_GP_REG_X0, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                          0, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  libxsmm_aarch64_instruction_asimd_compute( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             0,
                                             0, 0,
                                             0,
                                             LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
  libxsmm_aarch64_instruction_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                          LIBXSMM_AARCH64_GP_REG_X0, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                          0, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

  {
    unsigned int code_head = mycode.code_size/4;
    unsigned int* code     = (unsigned int *)mycode.generated_code;

    /* insert ret instruction */
    code[code_head] = 0xd65f03c0;

    /* advance code head */
    mycode.code_size += 4;
  }

  printf(" codesize: %i\n", mycode.code_size );
  FILE *fp;
  char filename[255];
  memset( filename, 0, 255);
  strcat( filename, "jit_dump" );
  strcat( filename, ".bin" );

  fp = fopen( filename, "wb" );
  if (fp == NULL) {
    printf("Error opening binary dumping file!\n");
    exit(1);
  }
  fwrite(mycode.generated_code, sizeof(unsigned char), mycode.code_size, fp);
  fclose(fp);

  printf(" attempting to create executable buffer...\n");
  char *kernelptr;
#ifdef __APPLE__
  kernelptr = mmap( 0, mycode.code_size, PROT_WRITE | PROT_EXEC | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT, -1, 0);
  pthread_jit_write_protect_np(0/*false*/);
  { int i;
    for (i = 0; i < (int)mycode.code_size; ++i) kernelptr[i] = codebuffer[i];
  }
  pthread_jit_write_protect_np(1/*false*/);
#else
  kernelptr = mmap( 0, mycode.code_size, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  { int i;
    for (i = 0; i < (int)mycode.code_size; ++i) kernelptr[i] = codebuffer[i];
  }
#endif
  printf(" ...done!\n");

  return kernelptr;
}

void static_reset_zero( float* data ) {
  data[0] = 0.0f;
  data[1] = 0.0f;
  data[2] = 0.0f;
  data[3] = 0.0f;
}

int main( /*int argc, char* argv[]*/ ) {
  float data1[4];
  float data2[4];
  unsigned int i;

  for ( i = 0; i < 4; ++i ) {
    data1[i] = (float)(i*1);
    data2[i] = (float)(i*2);
  }

  printf("data1 : ");
  for ( i = 0; i < 4; ++i ) {
    printf("%f ", data1[i]);
  }
  printf("\n\n");

  printf("data2 : ");
  for ( i = 0; i < 4; ++i ) {
    printf("%f ", data2[i]);
  }
  printf("\n\n");

  static_reset_zero( data1 );

  printf("data1 : ");
  for ( i = 0; i < 4; ++i ) {
    printf("%f ", data1[i]);
  }
  printf("\n\n");

  printf("Attempt to JIT...\n");
  reset_func myfunc = (reset_func)dynamic_reset_zero_create();
  printf("...done!\n\n");
  printf("Attempt to execute JIT...\n");
  myfunc( data2 );
  printf("...done!\n\n");

  printf("data2 : ");
  for ( i = 0; i < 4; ++i ) {
    printf("%f ", data2[i]);
  }
  printf("\n\n");

  return 0;
}
