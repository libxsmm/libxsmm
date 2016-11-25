/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_COMMON_H
#define GENERATOR_COMMON_H

#include <libxsmm_generator.h>

/* Fallback to MKL and not to triple nested loops in the generator (noarch)
 **/
#if 0
#define LIBXSMM_GENERATOR_MKL_BLAS_FALLBACK
#endif

/*@TODO check if we want to use enums here? Has this implications in the encoder? */
/* defining register mappings */
#define LIBXSMM_X86_GP_REG_RAX               0
#define LIBXSMM_X86_GP_REG_RCX               1
#define LIBXSMM_X86_GP_REG_RDX               2
#define LIBXSMM_X86_GP_REG_RBX               3
#define LIBXSMM_X86_GP_REG_RSP               4
#define LIBXSMM_X86_GP_REG_RBP               5
#define LIBXSMM_X86_GP_REG_RSI               6
#define LIBXSMM_X86_GP_REG_RDI               7
#define LIBXSMM_X86_GP_REG_R8                8
#define LIBXSMM_X86_GP_REG_R9                9
#define LIBXSMM_X86_GP_REG_R10              10
#define LIBXSMM_X86_GP_REG_R11              11
#define LIBXSMM_X86_GP_REG_R12              12
#define LIBXSMM_X86_GP_REG_R13              13
#define LIBXSMM_X86_GP_REG_R14              14
#define LIBXSMM_X86_GP_REG_R15              15
#define LIBXSMM_X86_GP_REG_UNDEF           127

/* define a place holder to handle AVX and SSE with a single encoder function
   using this values as the third operand means SSE */
#define LIBXSMM_X86_VEC_REG_UNDEF          255
#define LIBXSMM_X86_MASK_REG_UNDEF         255
#define LIBXSMM_X86_IMCI_AVX512_MASK         1  /* this specifies k1 */

/* special instruction */
#define LIBXSMM_X86_INSTR_UNDEF           9999

/* Load/Store/Move instructions */
/* AVX1,AVX2,AVX512 */
#define LIBXSMM_X86_INSTR_VMOVAPD        10000
#define LIBXSMM_X86_INSTR_VMOVUPD        10001
#define LIBXSMM_X86_INSTR_VMOVAPS        10002
#define LIBXSMM_X86_INSTR_VMOVUPS        10003
#define LIBXSMM_X86_INSTR_VBROADCASTSD   10004
#define LIBXSMM_X86_INSTR_VBROADCASTSS   10005
#define LIBXSMM_X86_INSTR_VMOVDDUP       10006
#define LIBXSMM_X86_INSTR_VMOVSD         10007
#define LIBXSMM_X86_INSTR_VMOVSS         10008
#define LIBXSMM_X86_INSTR_VPBROADCASTB   10025
#define LIBXSMM_X86_INSTR_VPBROADCASTW   10026
#define LIBXSMM_X86_INSTR_VPBROADCASTD   10027
#define LIBXSMM_X86_INSTR_VPBROADCASTQ   10028
#define LIBXSMM_X86_INSTR_VMOVDQA32      10029
#define LIBXSMM_X86_INSTR_VMOVDQA64      10030
#define LIBXSMM_X86_INSTR_VMOVDQU8       10031
#define LIBXSMM_X86_INSTR_VMOVDQU16      10032
#define LIBXSMM_X86_INSTR_VMOVDQU32      10033
#define LIBXSMM_X86_INSTR_VMOVDQU64      10034

/* SSE */
#define LIBXSMM_X86_INSTR_MOVAPD         10009
#define LIBXSMM_X86_INSTR_MOVUPD         10010
#define LIBXSMM_X86_INSTR_MOVAPS         10011
#define LIBXSMM_X86_INSTR_MOVUPS         10012
#define LIBXSMM_X86_INSTR_MOVSD          10013
#define LIBXSMM_X86_INSTR_MOVSS          10014
#define LIBXSMM_X86_INSTR_MOVDDUP        10015
#define LIBXSMM_X86_INSTR_SHUFPS         10016
/* IMCI */
#define LIBXSMM_X86_INSTR_VLOADUNPACKLPD 10017
#define LIBXSMM_X86_INSTR_VLOADUNPACKHPD 10018
#define LIBXSMM_X86_INSTR_VLOADUNPACKLPS 10019
#define LIBXSMM_X86_INSTR_VLOADUNPACKHPS 10020
#define LIBXSMM_X86_INSTR_VPACKSTORELPD  10021
#define LIBXSMM_X86_INSTR_VPACKSTOREHPD  10022
#define LIBXSMM_X86_INSTR_VPACKSTORELPS  10023
#define LIBXSMM_X86_INSTR_VPACKSTOREHPS  10024

/* Gather/Scatter instructions */
#define LIBXSMM_X86_INSTR_VGATHERDPS     11000
#define LIBXSMM_X86_INSTR_VGATHERDPD     11001
#define LIBXSMM_X86_INSTR_VGATHERQPS     11002
#define LIBXSMM_X86_INSTR_VGATHERQPD     11003
#define LIBXSMM_X86_INSTR_VSCATTERDPS    11004
#define LIBXSMM_X86_INSTR_VSCATTERDPD    11005
#define LIBXSMM_X86_INSTR_VSCATTERQPS    11006
#define LIBXSMM_X86_INSTR_VSCATTERQPD    11007

/* Vector compute instructions */
/* AVX1,AVX2,AVX512 */
#define LIBXSMM_X86_INSTR_VXORPD         20000
#define LIBXSMM_X86_INSTR_VMULPD         20001
#define LIBXSMM_X86_INSTR_VADDPD         20002
#define LIBXSMM_X86_INSTR_VSUBPD         20003
#define LIBXSMM_X86_INSTR_VFMADD231PD    20004
#define LIBXSMM_X86_INSTR_VFMSUB231PD    20005
#define LIBXSMM_X86_INSTR_VFNMADD231PD   20006
#define LIBXSMM_X86_INSTR_VFNMSUB231PD   20007
#define LIBXSMM_X86_INSTR_VMULSD         20008
#define LIBXSMM_X86_INSTR_VADDSD         20009
#define LIBXSMM_X86_INSTR_VSUBSD         20010
#define LIBXSMM_X86_INSTR_VFMADD231SD    20011
#define LIBXSMM_X86_INSTR_VFMSUB231SD    20012
#define LIBXSMM_X86_INSTR_VFNMADD231SD   20013
#define LIBXSMM_X86_INSTR_VFNMSUB231SD   20014
#define LIBXSMM_X86_INSTR_VXORPS         20015
#define LIBXSMM_X86_INSTR_VMULPS         20016
#define LIBXSMM_X86_INSTR_VADDPS         20017
#define LIBXSMM_X86_INSTR_VSUBPS         20018
#define LIBXSMM_X86_INSTR_VFMADD231PS    20019
#define LIBXSMM_X86_INSTR_VFMSUB231PS    20020
#define LIBXSMM_X86_INSTR_VFNMADD231PS   20021
#define LIBXSMM_X86_INSTR_VFNMSUB231PS   20022
#define LIBXSMM_X86_INSTR_VMULSS         20023
#define LIBXSMM_X86_INSTR_VADDSS         20024
#define LIBXSMM_X86_INSTR_VSUBSS         20025
#define LIBXSMM_X86_INSTR_VFMADD231SS    20026
#define LIBXSMM_X86_INSTR_VFMSUB231SS    20027
#define LIBXSMM_X86_INSTR_VFNMADD231SS   20028
#define LIBXSMM_X86_INSTR_VFNMSUB231SS   20029
/* SSE */
#define LIBXSMM_X86_INSTR_XORPD          20030
#define LIBXSMM_X86_INSTR_MULPD          20031
#define LIBXSMM_X86_INSTR_ADDPD          20032
#define LIBXSMM_X86_INSTR_SUBPD          20033
#define LIBXSMM_X86_INSTR_MULSD          20034
#define LIBXSMM_X86_INSTR_ADDSD          20035
#define LIBXSMM_X86_INSTR_SUBSD          20036
#define LIBXSMM_X86_INSTR_XORPS          20037
#define LIBXSMM_X86_INSTR_MULPS          20038
#define LIBXSMM_X86_INSTR_ADDPS          20039
#define LIBXSMM_X86_INSTR_SUBPS          20040
#define LIBXSMM_X86_INSTR_MULSS          20041
#define LIBXSMM_X86_INSTR_ADDSS          20042
#define LIBXSMM_X86_INSTR_SUBSS          20043
/* AVX512, IMCI Integer XOR as there is no FP */
#define LIBXSMM_X86_INSTR_VPXORD         20044
/* additional integer stuff */
#define LIBXSMM_X86_INSTR_VPADDQ         20045
#define LIBXSMM_X86_INSTR_VPADDD         20046
#define LIBXSMM_X86_INSTR_VPADDW         20047
#define LIBXSMM_X86_INSTR_VPADDB         20048
#define LIBXSMM_X86_INSTR_VPMADDWD       20049
#define LIBXSMM_X86_INSTR_VPMADDUBSW     20050
#define LIBXSMM_X86_INSTR_VPADDSW        20051
#define LIBXSMM_X86_INSTR_VPADDSB        20052

/* GP instructions */
#define LIBXSMM_X86_INSTR_ADDQ           30000
#define LIBXSMM_X86_INSTR_SUBQ           30001
#define LIBXSMM_X86_INSTR_MOVQ           30002
#define LIBXSMM_X86_INSTR_CMPQ           30003
#define LIBXSMM_X86_INSTR_JL             30004
#define LIBXSMM_X86_INSTR_VPREFETCH0     30005
#define LIBXSMM_X86_INSTR_VPREFETCH1     30006
#define LIBXSMM_X86_INSTR_PREFETCHT0     30007
#define LIBXSMM_X86_INSTR_PREFETCHT1     30008
#define LIBXSMM_X86_INSTR_PREFETCHT2     30009
#define LIBXSMM_X86_INSTR_PREFETCHNTA    30010

/* Mask move instructions */
#define LIBXSMM_X86_INSTR_KMOV           40000
#define LIBXSMM_X86_INSTR_KMOVW          40001

/* Mask compute instructions */
#define LIBXSMM_X86_INSTR_KXNORW         45000

/* define error codes */
#define LIBXSMM_ERR_GENERAL              90000
#define LIBXSMM_ERR_ARCH_PREC            90001
#define LIBXSMM_ERR_ARCH                 90002
#define LIBXSMM_ERR_LDA                  90003
#define LIBXSMM_ERR_LDB                  90004
#define LIBXSMM_ERR_LDC                  90005
#define LIBXSMM_ERR_SPGEMM_GEN           90006
#define LIBXSMM_ERR_CSC_INPUT            90007
#define LIBXSMM_ERR_CSC_READ_LEN         90008
#define LIBXSMM_ERR_CSC_READ_DESC        90009
#define LIBXSMM_ERR_CSC_READ_ELEMS       90010
#define LIBXSMM_ERR_CSC_LEN              90011
#define LIBXSMM_ERR_N_BLOCK              90012
#define LIBXSMM_ERR_M_BLOCK              90013
#define LIBXSMM_ERR_NO_IMCI              90014
#define LIBXSMM_ERR_REG_BLOCK            90015
#define LIBXSMM_ERR_VEC_MOVE_IMCI        90016
#define LIBXSMM_ERR_APPEND_STR           90017
#define LIBXSMM_ERR_ALLOC                90018
#define LIBXSMM_ERR_NO_IMCI_AVX512_BCAST 90019
#define LIBXSMM_ERR_CALLEE_SAVE_A        90020
#define LIBXSMM_ERR_CALLEE_SAVE_B        90021
#define LIBXSMM_ERR_CALLEE_SAVE_C        90022
#define LIBXSMM_ERR_CALLEE_SAVE_A_PREF   90023
#define LIBXSMM_ERR_CALLEE_SAVE_B_PREF   90024
#define LIBXSMM_ERR_NO_INDEX_SCALE_ADDR  90025
#define LIBXSMM_ERR_UNSUPPORTED_JUMP     90026
#define LIBXSMM_ERR_NO_JMPLBL_AVAIL      90027
#define LIBXSMM_ERR_EXCEED_JMPLBL        90028
#define LIBXSMM_ERR_CSC_ALLOC_DATA       90029
#define LIBXSMM_ERR_CSR_INPUT            90030
#define LIBXSMM_ERR_CSR_READ_LEN         90031
#define LIBXSMM_ERR_CSR_READ_DESC        90032
#define LIBXSMM_ERR_CSR_READ_ELEMS       90033
#define LIBXSMM_ERR_CSR_LEN              90034
#define LIBXSMM_ERR_CSR_ALLOC_DATA       90035
#define LIBXSMM_ERR_UNSUP_CONV_FORMAT    90036
#define LIBXSMM_ERR_INVALID_KW_UNROLL    90037
#define LIBXSMM_ERR_INVALID_KH_UNROLL    90038
#define LIBXSMM_ERR_INVALID_OFW_UNROLL   90039
#define LIBXSMM_ERR_INVALID_OFH_UNROLL   90040
#define LIBXSMM_ERR_INVALID_CONV_ACC     90041
#define LIBXSMM_ERR_UNSUP_ARCH           90042
#define LIBXSMM_ERR_CONV_IFM_VEC         90043
#define LIBXSMM_ERR_CONV_OFM_VEC         90044
#define LIBXSMM_ERR_CONV_CONT_STRIDE     90045
#define LIBXSMM_ERR_UNSUP_DATATYPE       90046
#define LIBXSMM_ERR_UNSUP_DT_FORMAT      90047
#define LIBXSMM_ERR_K_BLOCK              90048
#define LIBXSMM_ERR_INVALID_GEMM_CONFIG  90049

/* micro kernel config */
typedef struct libxsmm_micro_kernel_config_struct {
  unsigned int instruction_set;
  unsigned int vector_reg_count;
  unsigned int vector_length;
  unsigned int datatype_size;
  unsigned int a_vmove_instruction;
  unsigned int b_vmove_instruction;
  unsigned int b_shuff_instruction;
  unsigned int c_vmove_instruction;
  unsigned int use_masking_a_c;
  unsigned int prefetch_instruction;
  unsigned int vxor_instruction;
  unsigned int vmul_instruction;
  unsigned int vadd_instruction;
  unsigned int alu_add_instruction;
  unsigned int alu_sub_instruction;
  unsigned int alu_cmp_instruction;
  unsigned int alu_jmp_instruction;
  unsigned int alu_mov_instruction;
  char vector_name;
} libxsmm_micro_kernel_config;

/* struct for storing the current gp reg mapping */
typedef struct libxsmm_gp_reg_mapping_struct {
  unsigned int gp_reg_a;
  unsigned int gp_reg_b;
  unsigned int gp_reg_c;
  unsigned int gp_reg_a_prefetch;
  unsigned int gp_reg_b_prefetch;
  unsigned int gp_reg_c_prefetch;
  unsigned int gp_reg_mloop;
  unsigned int gp_reg_nloop;
  unsigned int gp_reg_kloop;
  unsigned int gp_reg_help_0;
  unsigned int gp_reg_help_1;
  unsigned int gp_reg_help_2;
  unsigned int gp_reg_help_3;
  unsigned int gp_reg_help_4;
  unsigned int gp_reg_help_5;
} libxsmm_gp_reg_mapping;

/* convolution kernel config */
typedef struct libxsmm_convolution_kernel_config_struct {
  unsigned int instruction_set;
  unsigned int vector_reg_count;
  unsigned int vector_length_in;
  unsigned int datatype_size_in;
  unsigned int vector_length_out;
  unsigned int datatype_size_out;
  unsigned int vector_length_wt;
  unsigned int datatype_size_wt;
  unsigned int vmove_instruction;
  unsigned int vbcst_instruction;
  unsigned int vfma_instruction;
  unsigned int vxor_instruction;
  unsigned int vadd_instruction;
  unsigned int prefetch_instruction;
  unsigned int alu_add_instruction;
  unsigned int alu_mul_instruction;
  unsigned int alu_sub_instruction;
  unsigned int alu_cmp_instruction;
  unsigned int alu_jmp_instruction;
  unsigned int alu_mov_instruction;
  char vector_name;
  unsigned int l_ld_ifm_act;
  unsigned int l_ld_ofm_act;
  unsigned int l_ld_ifm_fil;
  unsigned int l_ld_ofm_fil;
} libxsmm_convolution_kernel_config;

/* struct for storing the current gp reg mapping for convolution */
typedef struct libxsmm_convolution_forward_gp_reg_mapping_struct {
  unsigned int gp_reg_input;
  unsigned int gp_reg_weight;
  unsigned int gp_reg_output;
  unsigned int gp_reg_input_pf;
  unsigned int gp_reg_weight_pf;
  unsigned int gp_reg_output_pf;
  unsigned int gp_reg_kw_loop;
  unsigned int gp_reg_kh_loop;
  unsigned int gp_reg_ifmInner_loop;
  unsigned int gp_reg_help_0;
  unsigned int gp_reg_help_1;
  unsigned int gp_reg_help_2;
  unsigned int gp_reg_help_3;
  unsigned int gp_reg_help_4;
  unsigned int gp_reg_help_5;
  unsigned int gp_reg_help_6;
} libxsmm_convolution_forward_gp_reg_mapping;

/* struct for storing the current gp reg mapping for convolution */
typedef struct libxsmm_convolution_backward_gp_reg_mapping_struct {
  unsigned int gp_reg_input;
  unsigned int gp_reg_weight;
  unsigned int gp_reg_output;
  unsigned int gp_reg_input_pf;
  unsigned int gp_reg_weight_pf;
  unsigned int gp_reg_output_pf;
  unsigned int gp_reg_kw_loop;
  unsigned int gp_reg_oi_loop;
  unsigned int gp_reg_kh_loop;
  unsigned int gp_reg_ofmInner_loop;
  unsigned int gp_reg_help_0;
  unsigned int gp_reg_help_1;
  unsigned int gp_reg_help_2;
  unsigned int gp_reg_help_3;
  unsigned int gp_reg_help_4;
  unsigned int gp_reg_help_5;
  unsigned int gp_reg_help_6;
} libxsmm_convolution_backward_gp_reg_mapping;

/* struct for storing the current gp reg mapping for convolution */
typedef struct libxsmm_convolution_weight_update_gp_reg_mapping_struct {
  unsigned int gp_reg_input;
  unsigned int gp_reg_weight;
  unsigned int gp_reg_output;
  unsigned int gp_reg_input_pf;
  unsigned int gp_reg_weight_pf;
  unsigned int gp_reg_output_pf;
  unsigned int gp_reg_ifmInner_loop;
  unsigned int gp_reg_oj_loop;
  unsigned int gp_reg_oi_loop;
  unsigned int gp_reg_help_0;
  unsigned int gp_reg_help_1;
  unsigned int gp_reg_help_2;
  unsigned int gp_reg_help_3;
  unsigned int gp_reg_help_4;
  unsigned int gp_reg_help_5;
  unsigned int gp_reg_help_6;
} libxsmm_convolution_weight_update_gp_reg_mapping;

/* struct for tracking local labels in assembly
   we don't allow overlapping loops */
typedef struct libxsmm_loop_label_tracker_struct {
  unsigned int label_address[32];
  unsigned int label_count;
} libxsmm_loop_label_tracker;

LIBXSMM_INTERNAL_API
void libxsmm_reset_loop_label_tracker( libxsmm_loop_label_tracker* io_loop_label_tracker );

LIBXSMM_INTERNAL_API
void libxsmm_get_x86_gp_reg_name( const unsigned int i_gp_reg_number,
                                  char*              o_gp_reg_name,
                                  const int          i_gp_reg_name_max_length );

LIBXSMM_INTERNAL_API
unsigned int libxsmm_check_x86_gp_reg_name_callee_save( const unsigned int i_gp_reg_number );

LIBXSMM_INTERNAL_API
void libxsmm_get_x86_instr_name( const unsigned int i_instr_number,
                                 char*              o_instr_name,
                                 const int          i_instr_name_max_length );

LIBXSMM_INTERNAL_API
void libxsmm_reset_x86_gp_reg_mapping( libxsmm_gp_reg_mapping* io_gp_reg_mapping );

LIBXSMM_INTERNAL_API
unsigned int libxsmm_is_x86_vec_instr_single_precision( const unsigned int i_instr_number );

/* some string manipulation helper needed to
   generated code */
LIBXSMM_INTERNAL_API
void libxsmm_append_code_as_string( libxsmm_generated_code* io_generated_code,
                                    const char*             i_code_to_append,
                                    const int               i_append_length );

LIBXSMM_INTERNAL_API
void libxsmm_close_function( libxsmm_generated_code* io_generated_code );

LIBXSMM_INTERNAL_API
void libxsmm_mmfunction_signature( libxsmm_generated_code*       io_generated_code,
                                  const char*                    i_routine_name,
                                  const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_INTERNAL_API
void libxsmm_generator_isa_check_header( libxsmm_generated_code* io_generated_code,
                                         const char*             i_arch );

LIBXSMM_INTERNAL_API
void libxsmm_generator_isa_check_footer( libxsmm_generated_code* io_generated_code,
                                         const char*             i_arch );

LIBXSMM_INTERNAL_API
void libxsmm_handle_error( libxsmm_generated_code* io_generated_code,
                           const unsigned int      i_error_code );


LIBXSMM_INTERNAL_API
void libxsmm_strncpy( char*        o_dest,
                      const char*  i_src,
                      unsigned int i_dest_length,
                      unsigned int i_src_length );

LIBXSMM_INTERNAL_API
void libxsmm_convfunction_signature_fp32( libxsmm_generated_code*         io_generated_code,
                                          const char*                     i_routine_name     );

LIBXSMM_INTERNAL_API
void libxsmm_convfunction_signature_int16( libxsmm_generated_code*         io_generated_code,
                                           const char*                     i_routine_name     );

#endif /* GENERATOR_COMMON_H */

