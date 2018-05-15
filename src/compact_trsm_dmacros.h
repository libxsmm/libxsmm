#define XCT_NOUNIT_DIAG
#define N_UNROLL_AVX2 2
#define M_UNROLL_AVX2 2
#define ELE_IN_REGISTER_AVX2_F32 8
#define ELE_IN_REGISTER_AVX2_F64 4
#define P_UNROLL_AVX2_F32 ELE_IN_REGISTER_AVX2_F32
#define P_UNROLL_AVX2_F64 ELE_IN_REGISTER_AVX2_F64
#define trsm_ll_ap(i,j,lda,_is_row_) ((_is_row_) ? ((j)+((i)*(lda))) : ((i)+((j)*(lda))))
#define trsm_ll_bp(i,j,ldb,_is_row_) ((_is_row_) ? ((j)+((i)*(ldb))) : ((i)+((j)*(ldb))))
/* In the original XCT code, this was a #define that depended on what we want. But since we want this to work everywhere, I'm turning this on for everything. */
#define _XCT_NOUNIT_DIAG_

#define xct_ftype float /* Obsoleted in LIBXSMM, just a carry-over from XCT */
#define SET_ZERO_PACKED(x,y,z) do {\
    if (datasz==8) libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VXORPD, i_vector_name, (z), (y), (x) ); \
    else libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VXORPS, i_vector_name, (z), (y), (x) ); \
} while(0)
#define VMOVU_PACKED(reg, mat_ptr, mat_offset, load_store) do { \
    if (load_store && datasz==8) libxsmm_x86_instruction_vec_move ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VMOVUPD, mat_ptr, LIBXSMM_X86_GP_REG_UNDEF, 1, (mat_offset)*2, i_vector_name, (reg), 0, 1 ); \
    else if (load_store && datasz==4) libxsmm_x86_instruction_vec_move ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VMOVUPS, mat_ptr, LIBXSMM_X86_GP_REG_UNDEF, 1, (mat_offset), i_vector_name, (reg), 0, 1 ); \
    else if (datasz==8) libxsmm_x86_instruction_vec_move ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VMOVUPD, mat_ptr, LIBXSMM_X86_GP_REG_UNDEF, 1, (mat_offset)*2, i_vector_name, (reg), 0, 0 ); \
    else libxsmm_x86_instruction_vec_move ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VMOVUPS, mat_ptr, LIBXSMM_X86_GP_REG_UNDEF, 1, (mat_offset), i_vector_name, (reg), 0, 0 ); \
} while(0)
#define VFMADD231_PACKED(x,y,z) do { \
    if (datasz==8) libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VFMADD231PD, i_vector_name, (z), (y), (x) ); \
    else libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VFMADD231PS, i_vector_name, (z), (y), (x) ); \
} while(0)
#define VSUB_PACKED(x,y,z) do { \
    if (datasz==8) libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VSUBPD, i_vector_name, (x), (y), (z)); \
    else libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VSUBPS, i_vector_name, (z), (y), (x)); \
} while(0)
#define VDIV_PACKED(x,y,z) do { \
    if (datasz==8) libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VDIVPD, i_vector_name, (x), (y), (z)); \
    else libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VDIVPS, i_vector_name, (z), (y), (x)); \
} while(0)
#define VMUL_PACKED(x,y,z) do { \
    if (datasz==8) libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VMULPD, i_vector_name, (z), (y), (x)); \
    else libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VMULPS, i_vector_name, (z), (y), (x)); \
} while(0)

