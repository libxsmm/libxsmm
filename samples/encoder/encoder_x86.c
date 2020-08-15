#include <stdio.h>
#include <stdlib.h>

#include <generator_x86_instructions.h>

int main( int argc, char* argv[] ) {
  unsigned char codebuffer[131072];
  unsigned int i;
  unsigned int instr;
  int displ;
  unsigned int basereg;
  unsigned int idxreg;
  unsigned int scale;
  unsigned int bcst;
  unsigned int arch;
  libxsmm_generated_code mycode;
  FILE *fp;

  arch = LIBXSMM_X86_AVX512_CPX;

  /* init generated code object */
  mycode.generated_code = codebuffer;
  mycode.buffer_size = 131072;
  mycode.code_size = 0;
  mycode.code_type = 2;
  mycode.last_error = 0;
  mycode.arch = arch;
  mycode.sf_size = 0;

  instr = LIBXSMM_X86_INSTR_VCVTNE2PS2BF16;
  displ = 0;
  displ = 0x11223344;
  basereg = LIBXSMM_X86_GP_REG_RSI;
  idxreg = LIBXSMM_X86_GP_REG_RDX;
  scale = 8;
  bcst = 0;

  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_convert ( &mycode, arch, instr, 'z', i, 0, 0, 0 );
  }

  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_convert ( &mycode, arch, instr, 'z', 0, i, 0, 0 );
  }

  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_convert ( &mycode, arch, instr, 'z', 0, 0, i, 0 );
  }

  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_mem( &mycode, arch, instr, bcst, basereg, idxreg, scale, displ, 'z', i, 0 );
  }

  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_mem( &mycode, arch, instr, bcst, basereg, idxreg, scale, displ, 'z', 0, i );
  }

  /* dump stream into binday file */
  fp = fopen("bytecode.bin", "wb");
  if (fp == NULL) {
    printf("Error opening binary dumping file!\n");
    exit(1);
  }
  fwrite(codebuffer, sizeof(unsigned char), mycode.code_size, fp);
  fclose(fp);

  return 0;
}
