#include <catch2/catch.hpp>
#include "generator_power_instructions.h"

TEST_CASE( "Tests libxsmm_power_instruction_b_conditional", "[power][b_conditional]" ) {
  unsigned int l_instr = 0;

  // bne cr2, 0
  l_instr = libxsmm_power_instruction_b_conditional( LIBXSMM_POWER_INSTR_B_BC,
                                                     4,
                                                     10,
                                                     0 );
  REQUIRE( l_instr == 0x408a0000 );

  // blt 8
  l_instr = libxsmm_power_instruction_b_conditional( LIBXSMM_POWER_INSTR_B_BC,
                                                     12,
                                                     0,
                                                     2 );
  REQUIRE( l_instr == 0x41800008 );

  // blt 48
  l_instr = libxsmm_power_instruction_b_conditional( LIBXSMM_POWER_INSTR_B_BC,
                                                     12,
                                                     0,
                                                     12 );
  REQUIRE( l_instr == 0x41800030 );

  // beq 12
  l_instr = libxsmm_power_instruction_b_conditional( LIBXSMM_POWER_INSTR_B_BC,
                                                     12,  // 011at with at=00
                                                     2,   // zero (eq)
                                                     3 ); // offset
  REQUIRE( l_instr == 0x4182000c );
}

TEST_CASE( "Tests libxsmm_power_instruction_fip_storage_access", "[power][fip_storage_access]" ) {
  unsigned int l_instr = 0;

  // ld 0, 0(0)
  l_instr = libxsmm_power_instruction_fip_storage_access( LIBXSMM_POWER_INSTR_FIP_LD,
                                                          LIBXSMM_POWER_GPR_R0,
                                                          LIBXSMM_POWER_GPR_R0,
                                                          0 );
  REQUIRE( l_instr == 0xe8000000 );

  // ld 13, 0(1)
  l_instr = libxsmm_power_instruction_fip_storage_access( LIBXSMM_POWER_INSTR_FIP_LD,
                                                          LIBXSMM_POWER_GPR_R13,
                                                          LIBXSMM_POWER_GPR_R1,
                                                          0 );
  REQUIRE( l_instr == 0xe9a10000 );

  // ld 17, 32(1)
  l_instr = libxsmm_power_instruction_fip_storage_access( LIBXSMM_POWER_INSTR_FIP_LD,
                                                          LIBXSMM_POWER_GPR_R17,
                                                          LIBXSMM_POWER_GPR_R1,
                                                          32 );
  REQUIRE( l_instr == 0xea210020 );

  // ld 31, 144(9)
  l_instr = libxsmm_power_instruction_fip_storage_access( LIBXSMM_POWER_INSTR_FIP_LD,
                                                          LIBXSMM_POWER_GPR_R31,
                                                          LIBXSMM_POWER_GPR_R9,
                                                          144 );
  REQUIRE( l_instr == 0xebe90090 );

  // std 0, 0(0)
  l_instr = libxsmm_power_instruction_fip_storage_access( LIBXSMM_POWER_INSTR_FIP_STD,
                                                          LIBXSMM_POWER_GPR_R0,
                                                          LIBXSMM_POWER_GPR_R0,
                                                          0 );
  REQUIRE( l_instr == 0xf8000000 );

  // std 13, 0(1)
  l_instr = libxsmm_power_instruction_fip_storage_access( LIBXSMM_POWER_INSTR_FIP_STD,
                                                          LIBXSMM_POWER_GPR_R13,
                                                          LIBXSMM_POWER_GPR_R1,
                                                          0 );
  REQUIRE( l_instr == 0xf9a10000 );

  // std 17, 32(1)
  l_instr = libxsmm_power_instruction_fip_storage_access( LIBXSMM_POWER_INSTR_FIP_STD,
                                                          LIBXSMM_POWER_GPR_R17,
                                                          LIBXSMM_POWER_GPR_R1,
                                                          32 );
  REQUIRE( l_instr == 0xfa210020 );

  // std 31, 144(9)
  l_instr = libxsmm_power_instruction_fip_storage_access( LIBXSMM_POWER_INSTR_FIP_STD,
                                                          LIBXSMM_POWER_GPR_R31,
                                                          LIBXSMM_POWER_GPR_R9,
                                                          144 );
  REQUIRE( l_instr == 0xfbe90090 );
}

TEST_CASE( "Tests libxsmm_power_instruction_fip_arithmetic", "[power][fip_arithmetic]" ) {
  unsigned int l_instr = 0;

  // addi 0, 0, 0
  l_instr = libxsmm_power_instruction_fip_arithmetic( LIBXSMM_POWER_INSTR_FIP_ADDI,
                                                      LIBXSMM_POWER_GPR_R0,
                                                      LIBXSMM_POWER_GPR_R0,
                                                      0 );
  REQUIRE( l_instr == 0x38000000 );

  // addi 17, 3, 512
  l_instr = libxsmm_power_instruction_fip_arithmetic( LIBXSMM_POWER_INSTR_FIP_ADDI,
                                                      LIBXSMM_POWER_GPR_R17,
                                                      LIBXSMM_POWER_GPR_R3,
                                                      512 );
  REQUIRE( l_instr == 0x3a230200 );

  // subi 5, 21, 1024
  l_instr = libxsmm_power_instruction_fip_arithmetic( LIBXSMM_POWER_INSTR_FIP_ADDI,
                                                      LIBXSMM_POWER_GPR_R5,
                                                      LIBXSMM_POWER_GPR_R21,
                                                      -1024 );
  REQUIRE( l_instr == 0x38b5fc00 );
}

TEST_CASE( "Tests libxsmm_power_instruction_fip_compare", "[power][fip_compare]" ) {
  unsigned int l_instr = 0;

  // cmpi 0, 0, 0, 0
  // cmpwi r0, 0
  l_instr = libxsmm_power_instruction_fip_compare( LIBXSMM_POWER_INSTR_FIP_CMPI,
                                                   0,
                                                   0,
                                                   0,
                                                   0 );
  REQUIRE( l_instr == 0x2c000000 );

  // cmpi 0, 0, r6, 0
  // cmpwi r6, 0
  l_instr = libxsmm_power_instruction_fip_compare( LIBXSMM_POWER_INSTR_FIP_CMPI,
                                                   0,
                                                   0,
                                                   LIBXSMM_POWER_GPR_R6,
                                                   0 );
  REQUIRE( l_instr == 0x2c060000 );

  // cmpi 0, 1, r8, -4
  // cmpdi r8, -4
  l_instr = libxsmm_power_instruction_fip_compare( LIBXSMM_POWER_INSTR_FIP_CMPI,
                                                   0,
                                                   1,
                                                   LIBXSMM_POWER_GPR_R8,
                                                   -4 );
  REQUIRE( l_instr == 0x2c28fffc );

  // cmpi 3, 0, r31, 16219
  // cmpwi cr3, r31, 16219
  l_instr = libxsmm_power_instruction_fip_compare( LIBXSMM_POWER_INSTR_FIP_CMPI,
                                                   3,
                                                   0,
                                                   LIBXSMM_POWER_GPR_R31,
                                                   16219 );
  REQUIRE( l_instr == 0x2d9f3f5b );
}

TEST_CASE( "Tests libxsmm_power_instruction_fip_logical", "[power][fip_logical]" ) {
  unsigned int l_instr = 0;

  // ori 0, 0, 0
  l_instr = libxsmm_power_instruction_fip_logical( LIBXSMM_POWER_INSTR_FIP_ORI,
                                                   LIBXSMM_POWER_GPR_R0,
                                                   LIBXSMM_POWER_GPR_R0,
                                                   0 );
  REQUIRE( l_instr == 0x60000000 );

  // ori 31, 31, 0
  l_instr = libxsmm_power_instruction_fip_logical( LIBXSMM_POWER_INSTR_FIP_ORI,
                                                   LIBXSMM_POWER_GPR_R31,
                                                   LIBXSMM_POWER_GPR_R31,
                                                  0 );
  REQUIRE( l_instr == 0x63ff0000 );

  // ori 5, 9, 0
  l_instr = libxsmm_power_instruction_fip_logical( LIBXSMM_POWER_INSTR_FIP_ORI,
                                                   LIBXSMM_POWER_GPR_R5,
                                                   LIBXSMM_POWER_GPR_R9,
                                                   0 );
  REQUIRE( l_instr == 0x61250000 );

  // ori 0, 0, 0
  l_instr = libxsmm_power_instruction_fip_logical( LIBXSMM_POWER_INSTR_FIP_ORI,
                                                   LIBXSMM_POWER_GPR_R0,
                                                   LIBXSMM_POWER_GPR_R0,
                                                   0 );
  REQUIRE( l_instr == 0x60000000 );

   // ori 7, 19, 21
  l_instr = libxsmm_power_instruction_fip_logical( LIBXSMM_POWER_INSTR_FIP_ORI,
                                                   LIBXSMM_POWER_GPR_R7,
                                                   LIBXSMM_POWER_GPR_R19,
                                                   21 );
  REQUIRE( l_instr == 0x62670015 );
}

TEST_CASE( "Tests libxsmm_power_instruction_fip_rotate", "[power][fip_rotate]" ) {
  unsigned int l_instr = 0;

  // rldicr r0, r0, 0, 0
  l_instr = libxsmm_power_instruction_fip_rotate( LIBXSMM_POWER_INSTR_FIP_RLDICR,
                                                  LIBXSMM_POWER_GPR_R0,
                                                  LIBXSMM_POWER_GPR_R0,
                                                  0,
                                                  0 );
  REQUIRE( l_instr == 0x78000004 );

  // rldicr r3, r6, 8, 63
  l_instr = libxsmm_power_instruction_fip_rotate( LIBXSMM_POWER_INSTR_FIP_RLDICR,
                                                  LIBXSMM_POWER_GPR_R3,
                                                  LIBXSMM_POWER_GPR_R6,
                                                  8,
                                                  63 );
  REQUIRE( l_instr == 0x78c347e4 );
}

TEST_CASE( "Tests libxsmm_power_instruction_flp_storage_access", "[power][flp_storage_access]" ) {
  unsigned int l_instr = 0;

  // lfd f14, 152(sp)
  l_instr = libxsmm_power_instruction_flp_storage_access( LIBXSMM_POWER_INSTR_FLP_LFD,
                                                          LIBXSMM_POWER_FPR_F14,
                                                          LIBXSMM_POWER_GPR_SP,
                                                          152 );
  REQUIRE( l_instr == 0xc9c10098 );

  // lfd f15, 160(sp)
  l_instr = libxsmm_power_instruction_flp_storage_access( LIBXSMM_POWER_INSTR_FLP_LFD,
                                                          LIBXSMM_POWER_FPR_F15,
                                                          LIBXSMM_POWER_GPR_SP,
                                                          160 );
  REQUIRE( l_instr == 0xc9e100a0 );

  // stfd f14, 152(sp)
  l_instr = libxsmm_power_instruction_flp_storage_access( LIBXSMM_POWER_INSTR_FLP_STFD,
                                                          LIBXSMM_POWER_FPR_F14,
                                                          LIBXSMM_POWER_GPR_SP,
                                                          152 );
  REQUIRE( l_instr == 0xd9c10098 );

  // stfd f15, 160(sp)
  l_instr = libxsmm_power_instruction_flp_storage_access( LIBXSMM_POWER_INSTR_FLP_STFD,
                                                          LIBXSMM_POWER_FPR_F15,
                                                          LIBXSMM_POWER_GPR_SP,
                                                          160 );
  REQUIRE( l_instr == 0xd9e100a0 );
}

TEST_CASE( "Tests libxsmm_power_instruction_vec_storage_access", "[power][vec_storage_access]" ) {
  unsigned int l_instr = 0;

  // lvx v31, 0, r11
  l_instr = libxsmm_power_instruction_vec_storage_access( LIBXSMM_POWER_INSTR_VEC_LVX,
                                                          LIBXSMM_POWER_VSR_VS31,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R11 );
  REQUIRE( l_instr == 0x7fe058ce );

  // lvx v30, 0, r11
  l_instr = libxsmm_power_instruction_vec_storage_access( LIBXSMM_POWER_INSTR_VEC_LVX,
                                                          LIBXSMM_POWER_VSR_VS30,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R11 );
  REQUIRE( l_instr == 0x7fc058ce );

  // stvx v30, 0, r11
  l_instr = libxsmm_power_instruction_vec_storage_access( LIBXSMM_POWER_INSTR_VEC_STVX,
                                                          LIBXSMM_POWER_VSR_VS30,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R11 );
  REQUIRE( l_instr == 0x7fc059ce );

  // stvx v31, 0, r11
  l_instr = libxsmm_power_instruction_vec_storage_access( LIBXSMM_POWER_INSTR_VEC_STVX,
                                                          LIBXSMM_POWER_VSR_VS31,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R11 );
  REQUIRE( l_instr == 0x7fe059ce );
}

TEST_CASE( "Tests libxsmm_power_instruction_vsx_storage_access", "[power][vsx_storage_access]" ) {
  unsigned int l_instr = 0;

  // lxvw4x 0, 0, 0
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_LXVW4X,
                                                          LIBXSMM_POWER_VSR_VS0,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R0 );
  REQUIRE( l_instr == 0x7c000618 );

  // lxvw4x 17, 0, 5
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_LXVW4X,
                                                          LIBXSMM_POWER_VSR_VS17,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R5 );
  REQUIRE( l_instr == 0x7e202e18 );

  // lxvw4x 53, 19, 21
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_LXVW4X,
                                                          LIBXSMM_POWER_VSR_VS53,
                                                          LIBXSMM_POWER_GPR_R19,
                                                          LIBXSMM_POWER_GPR_R21 );
  REQUIRE( l_instr == 0x7eb3ae19 );

  // stxvw4x 0, 0, 0
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_STXVW4X,
                                                          LIBXSMM_POWER_VSR_VS0,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R0 );
  REQUIRE( l_instr == 0x7c000718 );

  // stxvw4x 48, 21, 5
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_STXVW4X,
                                                          LIBXSMM_POWER_VSR_VS48,
                                                          LIBXSMM_POWER_GPR_R21,
                                                          LIBXSMM_POWER_GPR_R5 );
  REQUIRE( l_instr == 0x7e152f19 );

  // lxvwsx 0, 0, 0
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_LXVWSX,
                                                          LIBXSMM_POWER_VSR_VS0,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R0 );
  REQUIRE( l_instr == 0x7c0002d8 );

  // lxvwsx 20, 0, 4
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_LXVWSX,
                                                          LIBXSMM_POWER_VSR_VS20,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R4 );
  REQUIRE( l_instr == 0x7e8022d8 );

  // lxvwsx 43, 2, 4
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_LXVWSX,
                                                          LIBXSMM_POWER_VSR_VS43,
                                                          2,
                                                          LIBXSMM_POWER_GPR_R4 );
  REQUIRE( l_instr == 0x7d6222d9 );

  // lxvll vs0, 0, r0
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_LXVLL,
                                                          LIBXSMM_POWER_VSR_VS0,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R0 );
  REQUIRE( l_instr == 0x7c00025a );

  // stxvll vs0, 0, r0
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_STXVLL,
                                                          LIBXSMM_POWER_VSR_VS0,
                                                          0,
                                                          LIBXSMM_POWER_GPR_R0 );
  REQUIRE( l_instr == 0x7c00035a );

  // lxvll vs17, 8, 4
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_LXVLL,
                                                          LIBXSMM_POWER_VSR_VS17,
                                                          LIBXSMM_POWER_GPR_R8,
                                                          LIBXSMM_POWER_GPR_R4 );
  REQUIRE( l_instr == 0x7e28225a );

  // stxvll vs17, 8, 4
  l_instr = libxsmm_power_instruction_vsx_storage_access( LIBXSMM_POWER_INSTR_VSX_STXVLL,
                                                          LIBXSMM_POWER_VSR_VS17,
                                                          LIBXSMM_POWER_GPR_R8,
                                                          LIBXSMM_POWER_GPR_R4 );
  REQUIRE( l_instr == 0x7e28235a );
}

TEST_CASE( "Tests libxsmm_power_instruction_vsx_vector_bfp_madd", "[power][vsx_vector_bfp_madd]" ) {
  unsigned int l_instr = 0;

  // xvmaddasp 0, 0, 0
  l_instr = libxsmm_power_instruction_vsx_vector_bfp_madd( LIBXSMM_POWER_INSTR_VSX_XVMADDASP,
                                                           LIBXSMM_POWER_VSR_VS0,
                                                           LIBXSMM_POWER_VSR_VS0,
                                                           LIBXSMM_POWER_VSR_VS0 );
  REQUIRE( l_instr == 0xf0000208 );

  // xvmaddasp 3, 8, 4
  l_instr = libxsmm_power_instruction_vsx_vector_bfp_madd( LIBXSMM_POWER_INSTR_VSX_XVMADDASP,
                                                           LIBXSMM_POWER_VSR_VS3,
                                                           LIBXSMM_POWER_VSR_VS8,
                                                           LIBXSMM_POWER_VSR_VS4 );
  REQUIRE( l_instr == 0xf0682208 );

  // xvmaddasp 3, 48, 4
  l_instr = libxsmm_power_instruction_vsx_vector_bfp_madd( LIBXSMM_POWER_INSTR_VSX_XVMADDASP,
                                                           LIBXSMM_POWER_VSR_VS3,
                                                           LIBXSMM_POWER_VSR_VS48,
                                                           LIBXSMM_POWER_VSR_VS4 );
  REQUIRE( l_instr == 0xf070220c );

  // xvmaddasp 3, 48, 33
  l_instr = libxsmm_power_instruction_vsx_vector_bfp_madd( LIBXSMM_POWER_INSTR_VSX_XVMADDASP,
                                                           LIBXSMM_POWER_VSR_VS3,
                                                           LIBXSMM_POWER_VSR_VS48,
                                                           LIBXSMM_POWER_VSR_VS33 );
  REQUIRE( l_instr == 0xf0700a0e );

  // xvmaddasp 62, 48, 33
  l_instr = libxsmm_power_instruction_vsx_vector_bfp_madd( LIBXSMM_POWER_INSTR_VSX_XVMADDASP,
                                                           LIBXSMM_POWER_VSR_VS62,
                                                           LIBXSMM_POWER_VSR_VS48,
                                                           LIBXSMM_POWER_VSR_VS33 );
  REQUIRE( l_instr == 0xf3d00a0f );
}

TEST_CASE( "Tests libxsmm_power_instruction_vsx_vector_permute_byte_reverse", "[power][vsx_vector_permute_byte_reverse]" ) {
  unsigned int l_instr = 0;

  // xxbrd   vs0, vs0
  l_instr = libxsmm_power_instruction_vsx_vector_permute_byte_reverse( LIBXSMM_POWER_INSTR_VSX_XXBRD,
                                                                       LIBXSMM_POWER_VSR_VS0,
                                                                       LIBXSMM_POWER_VSR_VS0 );
  REQUIRE( l_instr == 0xf017076c );

  // xxbrd vs21, vs47
  l_instr = libxsmm_power_instruction_vsx_vector_permute_byte_reverse( LIBXSMM_POWER_INSTR_VSX_XXBRD,
                                                                       LIBXSMM_POWER_VSR_VS21,
                                                                       LIBXSMM_POWER_VSR_VS47 );
  REQUIRE( l_instr == 0xf2b77f6e );

  // xxbrd vs57, vs47
  l_instr = libxsmm_power_instruction_vsx_vector_permute_byte_reverse( LIBXSMM_POWER_INSTR_VSX_XXBRD,
                                                                       LIBXSMM_POWER_VSR_VS57,
                                                                       LIBXSMM_POWER_VSR_VS47 );
  REQUIRE( l_instr == 0xf3377f6f );

  // xxbrw   vs0, vs0
  l_instr = libxsmm_power_instruction_vsx_vector_permute_byte_reverse( LIBXSMM_POWER_INSTR_VSX_XXBRW,
                                                                       LIBXSMM_POWER_VSR_VS0,
                                                                       LIBXSMM_POWER_VSR_VS0 );
  REQUIRE( l_instr == 0xf00f076c );

  // xxbrw vs21, vs47
  l_instr = libxsmm_power_instruction_vsx_vector_permute_byte_reverse( LIBXSMM_POWER_INSTR_VSX_XXBRW,
                                                                       LIBXSMM_POWER_VSR_VS21,
                                                                       LIBXSMM_POWER_VSR_VS47 );
  REQUIRE( l_instr == 0xf2af7f6e );

  // xxbrw vs57, vs47
  l_instr = libxsmm_power_instruction_vsx_vector_permute_byte_reverse( LIBXSMM_POWER_INSTR_VSX_XXBRW,
                                                                       LIBXSMM_POWER_VSR_VS57,
                                                                       LIBXSMM_POWER_VSR_VS47 );
  REQUIRE( l_instr == 0xf32f7f6f );
}

TEST_CASE( "Tests libxsmm_power_instruction_generic", "[power][libxsmm_power_instruction_generic]" ) {
  unsigned int l_instr = 0;

  // beq 12
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_B_BC,
                                                 12,
                                                 2,
                                                 3 );
  REQUIRE( l_instr == 0x4182000c );

  // ld 17, 32(1)
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_FIP_LD,
                                                 LIBXSMM_POWER_GPR_R17,
                                                 LIBXSMM_POWER_GPR_R1,
                                                 32 );
  REQUIRE( l_instr == 0xea210020 );

  // std 31, 144(9)
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_FIP_STD,
                                                 LIBXSMM_POWER_GPR_R31,
                                                 LIBXSMM_POWER_GPR_R9,
                                                 144 );
  REQUIRE( l_instr == 0xfbe90090 );

  // addi 17, 3, 512
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_FIP_ADDI,
                                                 LIBXSMM_POWER_GPR_R17,
                                                 LIBXSMM_POWER_GPR_R3,
                                                 512 );
  REQUIRE( l_instr == 0x3a230200 );

  // ori 5, 9, 0
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_FIP_ORI,
                                                 LIBXSMM_POWER_GPR_R5,
                                                 LIBXSMM_POWER_GPR_R9,
                                                 0 );
  REQUIRE( l_instr == 0x61250000 );

  // lfd f15, 160(sp)
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_FLP_LFD,
                                                 LIBXSMM_POWER_FPR_F15,
                                                 LIBXSMM_POWER_GPR_SP,
                                                 160 );
  REQUIRE( l_instr == 0xc9e100a0 );

  // stfd f14, 152(sp)
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_FLP_STFD,
                                                 LIBXSMM_POWER_FPR_F14,
                                                 LIBXSMM_POWER_GPR_SP,
                                                 152 );
  REQUIRE( l_instr == 0xd9c10098 );

  // lvx v30, 0, r11
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_VEC_LVX,
                                                 LIBXSMM_POWER_VSR_VS30,
                                                 0,
                                                 LIBXSMM_POWER_GPR_R11 );
  REQUIRE( l_instr == 0x7fc058ce );

  // stvx v30, 0, r11
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_VEC_STVX,
                                                 LIBXSMM_POWER_VSR_VS30,
                                                 0,
                                                 LIBXSMM_POWER_GPR_R11 );
  REQUIRE( l_instr == 0x7fc059ce );


  // lxvw4x 53, 19, 21
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_VSX_LXVW4X,
                                                 LIBXSMM_POWER_VSR_VS53,
                                                 LIBXSMM_POWER_GPR_R19,
                                                 LIBXSMM_POWER_GPR_R21 );
  REQUIRE( l_instr == 0x7eb3ae19 );

  // stxvw4x 48, 21, 5
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_VSX_STXVW4X,
                                                 LIBXSMM_POWER_VSR_VS48,
                                                 LIBXSMM_POWER_GPR_R21,
                                                 LIBXSMM_POWER_GPR_R5 );
  REQUIRE( l_instr == 0x7e152f19 );

  // lxvwsx 43, 2, 4
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_VSX_LXVWSX,
                                                 LIBXSMM_POWER_VSR_VS43,
                                                 2,
                                                 LIBXSMM_POWER_GPR_R4 );
  REQUIRE( l_instr == 0x7d6222d9 );

  // xvmaddasp 3, 8, 4
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_VSX_XVMADDASP,
                                                 LIBXSMM_POWER_VSR_VS3,
                                                 LIBXSMM_POWER_VSR_VS8,
                                                 LIBXSMM_POWER_VSR_VS4 );
  REQUIRE( l_instr == 0xf0682208 );

  // lxvl vs17, 8, 4
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_VSX_LXVLL,
                                                 LIBXSMM_POWER_VSR_VS17,
                                                 LIBXSMM_POWER_GPR_R8,
                                                 LIBXSMM_POWER_GPR_R4 );
  REQUIRE( l_instr == 0x7e28225a );

  // stxvl vs17, 8, 4
  l_instr = libxsmm_power_instruction_generic_3( LIBXSMM_POWER_INSTR_VSX_STXVLL,
                                                 LIBXSMM_POWER_VSR_VS17,
                                                 LIBXSMM_POWER_GPR_R8,
                                                 LIBXSMM_POWER_GPR_R4 );
  REQUIRE( l_instr == 0x7e28235a );

  // xxbrd vs21, vs47
  l_instr = libxsmm_power_instruction_generic_2( LIBXSMM_POWER_INSTR_VSX_XXBRD,
                                                 LIBXSMM_POWER_VSR_VS21,
                                                 LIBXSMM_POWER_VSR_VS47 );
  REQUIRE( l_instr == 0xf2b77f6e );

  // xxbrw vs21, vs47
  l_instr = libxsmm_power_instruction_generic_2( LIBXSMM_POWER_INSTR_VSX_XXBRW,
                                                 LIBXSMM_POWER_VSR_VS21,
                                                 LIBXSMM_POWER_VSR_VS47 );
  REQUIRE( l_instr == 0xf2af7f6e );


  // cmpi 0, 0, r6, 0
  // cmpwi r6, 0
  l_instr = libxsmm_power_instruction_generic_4( LIBXSMM_POWER_INSTR_FIP_CMPI,
                                                 0,
                                                 0,
                                                 LIBXSMM_POWER_GPR_R6,
                                                 0 );
  REQUIRE( l_instr == 0x2c060000 );

  // rldicr r3, r6, 8, 63
  l_instr = libxsmm_power_instruction_generic_4( LIBXSMM_POWER_INSTR_FIP_RLDICR,
                                                 LIBXSMM_POWER_GPR_R3,
                                                 LIBXSMM_POWER_GPR_R6,
                                                 8,
                                                 63 );
  REQUIRE( l_instr == 0x78c347e4 );
}