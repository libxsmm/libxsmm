/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_MAIN_H
#define LIBXSMM_MAIN_H

#include <libxsmm.h>
/**
 * TF includes src/libxsmm_main.h and uses LIBXSMM's sync primitives
 * without including libxsmm_sync. However, libxsmm_sync.h shall be
 * an explicit include separate from including libxsmm.h.
 */
#include "libxsmm_sync.h"

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXSMM_CAPACITY_REGISTRY) /* must be POT */
# define LIBXSMM_CAPACITY_REGISTRY 131072
#endif
#if !defined(LIBXSMM_CAPACITY_CACHE) /* must be POT */
# define LIBXSMM_CAPACITY_CACHE 16
#endif

#if !defined(LIBXSMM_PAGE_MINSIZE)
# define LIBXSMM_PAGE_MINSIZE 4096 /* 4 KB */
#endif

#if !defined(LIBXSMM_BATCH_CHECK) && !defined(NDEBUG)
# define LIBXSMM_BATCH_CHECK
#endif

#if !defined(LIBXSMM_NTHREADS_MAX)
# if (0 != LIBXSMM_SYNC)
#   define LIBXSMM_NTHREADS_MAX 1024
# else
#   define LIBXSMM_NTHREADS_MAX 1
# endif
#endif
/* relies on LIBXSMM_NTHREADS_MAX */
#if !defined(LIBXSMM_NTHREADS_USE) && 0
# define LIBXSMM_NTHREADS_USE
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS)
# define LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS LIBXSMM_NTHREADS_MAX
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_SCALE)
# define LIBXSMM_MALLOC_SCRATCH_SCALE 1.0
#endif
#if !defined(LIBXSMM_MALLOC_LIMIT)
# define LIBXSMM_MALLOC_LIMIT (2U << 20) /* 2 MB */
#endif
/* map memory also for non-executable buffers */
#if !defined(LIBXSMM_MALLOC_MMAP) && 0
# define LIBXSMM_MALLOC_MMAP
#endif
/* map memory for hooked allocation */
#if !defined(LIBXSMM_MALLOC_MMAP_HOOK) && 1
# define LIBXSMM_MALLOC_MMAP_HOOK
#endif
/* map memory for scratch buffers */
#if !defined(LIBXSMM_MALLOC_MMAP_SCRATCH) && 1
# define LIBXSMM_MALLOC_MMAP_SCRATCH
#endif
/* align even if interceptor is disabled at runtime */
#if !defined(LIBXSMM_MALLOC_ALIGN_ALL) && 1
# define LIBXSMM_MALLOC_ALIGN_ALL
#endif
#if !defined(LIBXSMM_MALLOC_HOOK_INTRINSIC) && 1
# if defined(LIBXSMM_PLATFORM_X86) && defined(LIBXSMM_INTRINSICS_INCLUDE) && \
    !defined(LIBXSMM_INTRINSICS_DEBUG) && !defined(LIBXSMM_MALLOC_MMAP)
#   define LIBXSMM_MALLOC_HOOK_INTRINSIC
# endif
#endif
#if !defined(LIBXSMM_MALLOC_HOOK_REALLOC) && 1
# if !defined(LIBXSMM_MALLOC_HOOK_INTRINSIC)
#   define LIBXSMM_MALLOC_HOOK_REALLOC
# endif
#endif
#if !defined(LIBXSMM_MALLOC_HOOK_CALLOC) && 1
# define LIBXSMM_MALLOC_HOOK_CALLOC
#endif
#if !defined(LIBXSMM_MALLOC_INTERNAL_CALLER_ID)
# define LIBXSMM_MALLOC_INTERNAL_CALLER_ID ((uintptr_t)LIBXSMM_UNLIMITED)
#endif
#if !defined(LIBXSMM_MALLOC_INTERNAL_CALLER)
# define LIBXSMM_MALLOC_INTERNAL_CALLER ((const void*)(LIBXSMM_MALLOC_INTERNAL_CALLER_ID))
#endif

#if !defined(LIBXSMM_INTERCEPT_DYNAMIC) && defined(LIBXSMM_BUILD) && \
    (defined(__GNUC__) || defined(_CRAYC)) && !defined(_WIN32) && !defined(__CYGWIN__) && \
   !(defined(__APPLE__) && defined(__MACH__) && LIBXSMM_VERSION2(6, 1) >= \
      LIBXSMM_VERSION2(__clang_major__, __clang_minor__))
# define LIBXSMM_INTERCEPT_DYNAMIC
#endif

#if !defined(LIBXSMM_MALLOC_HOOK_STATIC) && \
    (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD))) /* GLIBC */ && \
   (!defined(_WIN32)) /* TODO */
# define LIBXSMM_MALLOC_HOOK_STATIC
#endif
#if !defined(LIBXSMM_MALLOC_HOOK_DYNAMIC) && defined(LIBXSMM_INTERCEPT_DYNAMIC) && \
     defined(LIBXSMM_MALLOC_HOOK_STATIC) && !defined(_CRAYC) && !defined(__TRACE)
# define LIBXSMM_MALLOC_HOOK_DYNAMIC
#endif
#if (defined(LIBXSMM_MALLOC_HOOK_STATIC) || defined(LIBXSMM_MALLOC_HOOK_DYNAMIC))
# define LIBXSMM_MALLOC_HOOK
#endif
#if !defined(LIBXSMM_DNN_CONVOLUTION_SETUP_USE_NTS) && defined(LIBXSMM_MALLOC_HOOK) && \
    (defined(LIBXSMM_MALLOC_ALIGN_ALL) || (defined(LIBXSMM_MALLOC) && (0 != LIBXSMM_MALLOC)))
# define LIBXSMM_DNN_CONVOLUTION_SETUP_USE_NTS
#endif

#if defined(LIBXSMM_INTERCEPT_DYNAMIC)
# if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
# endif
# include <dlfcn.h>
# if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload_attribute(pop)
# endif
# if !defined(RTLD_NEXT)
#   define LIBXSMM_RTLD_NEXT ((void*)-1l)
# else
#   define LIBXSMM_RTLD_NEXT RTLD_NEXT
# endif
#endif

#if !defined(LIBXSMM_VERBOSITY_HIGH)
# define LIBXSMM_VERBOSITY_HIGH 3 /* secondary warning or info-verbosity */
#endif
#if !defined(LIBXSMM_VERBOSITY_WARN)
# define LIBXSMM_VERBOSITY_WARN ((LIBXSMM_VERBOSITY_HIGH) - LIBXSMM_MIN(1, LIBXSMM_VERBOSITY_HIGH))
#endif

#if !defined(LIBXSMM_LOCK)
# define LIBXSMM_LOCK LIBXSMM_LOCK_DEFAULT
#endif

#if !defined(LIBXSMM_EXT_MIN_NTASKS)
# define LIBXSMM_MIN_NTASKS(NT) 1
#endif
#if !defined(LIBXSMM_OVERHEAD)
# define LIBXSMM_OVERHEAD(NT) 0
#endif
#if !defined(LIBXSMM_NOOP_ARGS)
# define LIBXSMM_NOOP_ARGS(...)
#endif
#if !defined(LIBXSMM_NOOP)
# define LIBXSMM_NOOP
#endif

/** Check if M, N, K, or LDx fits into the descriptor. */
#if (0 != LIBXSMM_ILP64)
# define LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K) (0xFFFFFFFF >= (M) && 0xFFFFFFFF >= (N) && 0xFFFFFFFF >= (K))
#else /* always fits */
# define LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K) 1
#endif

#if defined(LIBXSMM_ASSERT) /* assert available */
# define LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K) LIBXSMM_ASSERT(LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K))
#else
# define LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K)
#endif

#if defined(LIBXSMM_UNPACKED)
# define LIBXSMM_DESCRIPTOR_CLEAR_AUX(DST, SIZE) LIBXSMM_MEMSET127(DST, 0, SIZE)
#else
# define LIBXSMM_DESCRIPTOR_CLEAR_AUX(DST, SIZE)
#endif
#define LIBXSMM_DESCRIPTOR_CLEAR(BLOB) \
  LIBXSMM_ASSERT((LIBXSMM_DESCRIPTOR_MAXSIZE) == sizeof(*(BLOB))); \
  LIBXSMM_DESCRIPTOR_CLEAR_AUX(BLOB, LIBXSMM_DESCRIPTOR_MAXSIZE)

/** Low-level/internal GEMM descriptor initialization. */
#define LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(LDA, LDB, LDC); \
  LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K); \
  LIBXSMM_DESCRIPTOR_CLEAR_AUX(&(DESCRIPTOR), sizeof(DESCRIPTOR)); \
  (DESCRIPTOR).datatype = (unsigned char)(DATA_TYPE); (DESCRIPTOR).prefetch = (unsigned char)(PREFETCH); \
  (DESCRIPTOR).flags = (unsigned int)((FLAGS) \
    /*| (LIBXSMM_NEQ(0, ALPHA) ? 0 : LIBXSMM_GEMM_FLAG_ALPHA_0)*/ \
    | (LIBXSMM_NEQ(0, BETA) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0)); \
  (DESCRIPTOR).m   = (unsigned int)(M);   (DESCRIPTOR).n   = (unsigned int)(N);   (DESCRIPTOR).k   = (unsigned int)(K); \
  (DESCRIPTOR).lda = (unsigned int)(LDA); (DESCRIPTOR).ldb = (unsigned int)(LDB); (DESCRIPTOR).ldc = (unsigned int)(LDC); \
  (DESCRIPTOR).meltw_datatype_aux = 0; (DESCRIPTOR).c1 = 0; (DESCRIPTOR).c2 = 0; (DESCRIPTOR).c3 = 0; \
  (DESCRIPTOR).meltw_ldx = 0; (DESCRIPTOR).meltw_ldy = 0; (DESCRIPTOR).meltw_ldz = 0; \
  (DESCRIPTOR).meltw_param = 0; (DESCRIPTOR).meltw_flags = 0; \
  (DESCRIPTOR).meltw_operation = 0

/** Similar to LIBXSMM_GEMM_DESCRIPTOR, but separately taking the input-/output-precision. */
#define LIBXSMM_GEMM_DESCRIPTOR2(DESCRIPTOR, IPREC, OPREC, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, LIBXSMM_GETENUM(IPREC, OPREC), FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

/** Declare and construct a GEMM descriptor. */
#define LIBXSMM_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  libxsmm_gemm_descriptor DESCRIPTOR; LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, \
    FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

/** Similar to LIBXSMM_GEMM_DESCRIPTOR_TYPE, but separately taking the input-/output-precision. */
#define LIBXSMM_GEMM_DESCRIPTOR2_TYPE(DESCRIPTOR, IPREC, OPREC, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXSMM_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, LIBXSMM_GETENUM(IPREC, OPREC), FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

#define LIBXSMM_REGDESC_DEFAULT
#define LIBXSMM_REGDESC(START, MODIFIER) \
  START libxsmm_gemm_descriptor MODIFIER gemm; \
  START libxsmm_meltw_descriptor MODIFIER meltw; \
  START libxsmm_meqn_descriptor MODIFIER meqn

/**
* Packed structure, which stores the argument description of GEMM routines.
* The size of the structure is padded to LIBXSMM_DESCRIPTOR_MAXSIZE.
*/
LIBXSMM_EXTERN_C LIBXSMM_PACKED(struct LIBXSMM_RETARGETABLE) libxsmm_gemm_descriptor {
  /** Extents of the matrix. */
  unsigned int m, n, k;
  /** Leading dimensions. */
  unsigned int lda, ldb, ldc;
  /** Set of flags. */
  unsigned int flags;
  /** Prefetch strategy. */
  unsigned char prefetch;
  /** Denotes the data-type. */
  unsigned char datatype;
  /**
   * Do not reorder elements between above and below blocks!
   */
  /** Denotes of optional eltwise data-type */
  unsigned char meltw_datatype_aux;
  /** multipurpose 64-bit field, currently used for: a) stride_a in brgemm */
  unsigned long long c1;
  /** multipurpose 64-bit field, currently used for: a) stride_b in brgemm */
  unsigned long long c2;
  /** multipurpose 8-bit field, currently used for: a) unroll hint in brgemm */
  unsigned char c3;
  /** LDx, LDy, LDz,  additional meltw LDs */
  unsigned int meltw_ldx, meltw_ldy, meltw_ldz;
  /** optional param field */
  unsigned char meltw_param;
  /** Set of flags */
  unsigned short meltw_flags;
  /** operation specifier */
  unsigned char meltw_operation;
};

/** Packed structure storing the mateltw argument description. */
LIBXSMM_EXTERN_C LIBXSMM_PACKED(struct LIBXSMM_RETARGETABLE) libxsmm_meltw_descriptor {
  /** LDx, M, and N. */
  unsigned int m, n, ldi, ldo, ldi2, ldi3;
  /** Size of data element. */
  unsigned char datatype;
  unsigned char datatype2;
  /** Set of flags */
  unsigned short flags;
  /** optional param field */
  unsigned char param;
  /** operation specifier */
  unsigned char operation;
};

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_pspgemm_csr_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
  unsigned int packed_width;
} libxsmm_pspgemm_csr_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_pspgemm_csc_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* column_ptr;
  const unsigned int* row_idx;
  const void* values;
  unsigned int packed_width;
} libxsmm_pspgemm_csc_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_pgemm_ac_rm_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  unsigned int packed_width;
} libxsmm_pgemm_ac_rm_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_pgemm_bc_rm_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  unsigned int packed_width;
} libxsmm_pgemm_bc_rm_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_csr_reg_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxsmm_csr_reg_descriptor;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_xcopykernel {
  libxsmm_meltwfunction_unary function;
  const void* ptr;
} libxsmm_xcopykernel;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_code_pointer {
  void (*ptr_fn)(LIBXSMM_VARIADIC);
  const void* ptr_const;
  void* ptr;
  uintptr_t uval;
  intptr_t ival;
  libxsmm_xmmfunction xgemm; /* GEMM: smm, dmm, wimm, or void-function */
  libxsmm_xmeltwfunction xmateltw;
  libxsmm_matrix_eqn_function xmateqn;
} libxsmm_code_pointer;

/** Structure which describes all tensors in LIBXSMM's DNN module */
LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_tensor {
  libxsmm_dnn_tensor_datalayout* layout;           /* data-layout descriptor */
  void* data;                                      /* pointer to data */
  unsigned char scf;                               /* fix point scaling factor for this tensor */
};

/* Structure to record segment in stream of code */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE segment_t {
  int segment_type;
  int n_convs;
  int aux_index;
  int img;
  int ofm;
  int ifm;
} segment_t;

LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_layer {
  libxsmm_dnn_datatype datatype_in;
  libxsmm_dnn_datatype datatype_out;
  libxsmm_dnn_conv_desc desc;
  libxsmm_dnn_conv_algo algo;
  libxsmm_dnn_tensor_format buffer_format;
  libxsmm_dnn_tensor_format filter_format;
  libxsmm_dnn_conv_fuse_op fuse_ops;
  libxsmm_dnn_conv_option options;
  int target_archid;

  /* additional size for internal data types */
  int ifhp;
  int ifwp;
  int ofh;
  int ofw;
  int ofhp;
  int ofwp;
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  int fwd_ofw_rb;
  int fwd_ofh_rb;
  int bwd_ofw_rb;
  int bwd_ofh_rb;
  int upd_ofw_rb;
  int upd_ofh_rb;
  int fm_lp_block; /* additional blocking for low precision datatypes of feature maps */
  int blocksifm_blocking;
  int blocksofm_blocking;
  int avoid_acc_load;
  int avoid_acc_load_bwd;
  int pack_input;
  int pack_input_bwd;
  int spread_input_bwd;
  int weight_copies;
  int loop_order;
  int use_ofm_parallelization;
  int use_ifm_parallelization;
  int avoid_fmas_in_rim;
  int upd_use_batchreduce;
  int upd_pack_input;
  int upd_loop_order;
  int upd_linearized_tasklist;
  int upd_avoid_rim_fmas;
  int fwd_flags;
  int bwd_flags;
  int shuffle_filter_accesses;
  int use_fallback_fwd_loops;
  int use_fallback_bwd_loops;
  int fwd_gemm_pixels;
  int bwd_gemm_pixels;
  int input_pixels;
  int output_pixels;
  int n_used_pixels;
  int pixel_blocking;
  int use_intermediate_f32_wt_tensor;
  int upd_linearized_pixels;
  int ifwp_extended;
  int ofwp_extended;
  int batchreduce_h_pixels;
  int on_the_fly_input_packing;
  int upd_pack_input_upfront;
  int use_hybrid_imgofm_parallelization;
  int remainder_pixels;
  int pack_to_cnhw;
  int fuse_upd_transposes;
  int compute_pixels;
  int upd_trans_w_only;
  int fwd_padding_copy;
  int upd_padding_copy;
  int block_fwd_oj;
  int block_fwd_ifm;
  int block_fwd_ofm;
  int block_bwd_oj;
  int block_bwd_ifm;
  int block_bwd_ofm;
  int block_upd_ifm;
  int block_upd_ofm;

  libxsmm_meltwfunction_unary tr_kernel;
  libxsmm_meltwfunction_unary fwd_cvtfp32bf16_kernel;

  /* Hoisting the compute kernels for FWD  */
  libxsmm_bsmmfunction fwd_config_kernel;
  libxsmm_bsmmfunction_reducebatch_addr fwd_compute_kernel_addr;
  libxsmm_bsmmfunction_reducebatch_offs fwd_compute_kernel_offs_b;
  libxsmm_bmmfunction_reducebatch_offs fwd_compute_kernel_offs_a;
  libxsmm_bmmfunction_reducebatch_strd fwd_compute_kernel_strd;
  libxsmm_smmfunction_reducebatch_addr fwd_compute_kernel_addr_a_f32;
  libxsmm_smmfunction_reducebatch_addr fwd_compute_kernel_addr_b_f32;
  libxsmm_smmfunction_reducebatch_offs fwd_compute_kernel_offs_f32;
  libxsmm_smmfunction_reducebatch_strd fwd_compute_kernel_strd_f32;

  /* Hoisting the compute kernels for BWD  */
  libxsmm_bsmmfunction bwd_config_kernel;
  libxsmm_bsmmfunction_reducebatch_addr bwd_compute_kernel_addr;
  libxsmm_bsmmfunction_reducebatch_offs bwd_compute_kernel_offs;
  libxsmm_bsmmfunction_reducebatch_strd bwd_compute_kernel_strd;

  /* Hoisting the compute kernels for UPD  */
  libxsmm_bsmmfunction                  upd_config_kernel;
  libxsmm_bsmmfunction_reducebatch_strd upd_compute_kernel_brgemm_no_linearized_pixels;
  libxsmm_bsmmfunction_reducebatch_strd upd_compute_kernel_brgemm_linearized_pixels_hybrid_par_no_cnhw;
  libxsmm_bsmmfunction                  upd_compute_kernel_gemm_linearized_pixels_hybrid_par_cnhw;
  libxsmm_bsmmfunction                  upd_compute_kernel_gemm_linearized_pixels_no_hybrid_par;

  libxsmm_bsmmfunction tilerelease_kernel;

  unsigned long long *A_offsets;
  unsigned long long *B_offsets;
  unsigned long long *A_offsets_bwd;
  unsigned long long *B_offsets_bwd;

  /* AMX specific fields */
  int x_rows;
  int n_pixel_tiles;
  int n_ofm_tiles;
  int wrb_1;
  int wrb_2;
  int wrb_3;
  int wrb_4;
  int hrb_1;
  int hrb_2;
  int n_compute_pixels;
  int pixels;
  int linearize_pixels;
  int split_pixel;
  int reconfig;
  int zero_rim;
  char tc[64];
  char tc2[64];
  char tc_upd[64];
  int input_padded_pixels;
  int output_padded_pixels;
  int blocks_pixels;
  /* End of AMX specific fields  */

  /* internal data representation */
  libxsmm_dnn_tensor* reg_input;
  libxsmm_dnn_tensor* reg_output;
  libxsmm_dnn_tensor* reg_filter;
  libxsmm_dnn_tensor* grad_input;
  libxsmm_dnn_tensor* grad_output;
  libxsmm_dnn_tensor* grad_filter;
  libxsmm_dnn_tensor* reg_bias;
  libxsmm_dnn_tensor* grad_bias;
  /* internal data representations for copies of tensors */
  libxsmm_dnn_tensor* reg_input_tr;
  libxsmm_dnn_tensor* reg_filter_tr;
  /* batchnorm stats */
  libxsmm_dnn_tensor* batch_stats;
  /* maxstats used in low-precision kernels */
  libxsmm_dnn_tensor* maxstats_fwd;
  libxsmm_dnn_tensor* maxstats_bwd;
  libxsmm_dnn_tensor* maxstats_upd;

  /* barrier */
  libxsmm_barrier* barrier;

  /* scratch */
  size_t fwd_packing_padding_scratch_size;
  size_t fwd_lp_output_full_scratch_size;
  size_t fwd_lp_output_block_scratch_size;
  size_t fwd_packing_padding_scratch_offset;
  size_t fwd_lp_output_full_scratch_offset;
  size_t fwd_lp_output_block_scratch_offset;
  size_t fwd_scratch_size;

  size_t bwd_filter_trans_scratch_size;
  size_t bwd_packing_padding_scratch_size;
  size_t bwd_lp_input_full_scratch_size;
  size_t bwd_filter_trans_scratch_offset;
  size_t bwd_packing_padding_scratch_offset;
  size_t bwd_lp_input_full_scratch_offset;
  size_t bwd_scratch_size;

  size_t upd_packing_padding_scratch_size;
  size_t upd_lp_output_full_scratch_size;
  size_t upd_lp_input_full_scratch_size;
  size_t upd_filter_scratch_size;
  size_t upd_lp_filter_full_scratch_size;
  size_t upd_packing_padding_scratch_offset;
  size_t upd_lp_output_full_scratch_offset;
  size_t upd_lp_input_full_scratch_offset;
  size_t upd_lp_filter_full_scratch_offset;
  size_t upd_filter_scratch_offset;
  size_t upd_scratch_size;

  void* scratch;
  size_t scratch_size;

  libxsmm_code_pointer gemm_fwd;     /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd2;    /* ability to hoist forward GEMMs */

  /* JIT-generated convolution code */
  libxsmm_code_pointer code_fwd[3];
  libxsmm_code_pointer code_bwd[3];
  libxsmm_code_pointer code_upd[5];

  libxsmm_code_pointer matcopy_fwd[4];
  libxsmm_code_pointer matcopy_bwd[4];
  libxsmm_code_pointer matcopy_upd[3];
};

LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_fusedbatchnorm {
  libxsmm_dnn_fusedbatchnorm_desc desc;
  libxsmm_dnn_tensor* reg_input;      /* input tensor */
  libxsmm_dnn_tensor* reg_output;     /* output tensor */
  libxsmm_dnn_tensor* grad_input;     /* grad input tensor */
  libxsmm_dnn_tensor* grad_output;    /* grad output tensor */
  libxsmm_dnn_tensor* reg_add;        /* elementwise tensor */
  libxsmm_dnn_tensor* grad_add;       /* grad elementwise tensor */
  libxsmm_dnn_tensor* reg_beta;       /* beta tensor */
  libxsmm_dnn_tensor* reg_gamma;      /* gamma tensor */
  libxsmm_dnn_tensor* grad_beta;      /* grad beta tensor */
  libxsmm_dnn_tensor* grad_gamma;     /* grad gamma tensor */
  libxsmm_dnn_tensor* expvalue;       /* expected value */
  libxsmm_dnn_tensor* rcpstddev;      /* reciprocal of standard derivation */
  libxsmm_dnn_tensor* variance;       /* variance */
  libxsmm_dnn_tensor* relumask;       /* relumask */
  libxsmm_barrier* barrier;           /* barrier */
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  size_t scratch_size;
  void* scratch;
};

LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_softmaxloss {
  libxsmm_dnn_softmaxloss_desc desc;
  libxsmm_dnn_tensor* reg_input;      /* input tensor */
  libxsmm_dnn_tensor* reg_output;     /* output tensor */
  libxsmm_dnn_tensor* grad_input;     /* grad input tensor */
  libxsmm_dnn_tensor* label;          /* labels tensor */
  libxsmm_barrier* barrier;           /* barrier */
  int bc;
  int Bc;
  int bn;
  int Bn;
  float loss;
  size_t scratch_size;
  void* scratch;
};

LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_optimizer {
  libxsmm_dnn_optimizer_desc desc;
  libxsmm_dnn_tensor* reg_filter;      /* filter tensor */
  libxsmm_dnn_tensor* grad_filter;     /* grad filter tensor */
  libxsmm_dnn_tensor* master_filter;   /* master filter tensor */
  libxsmm_barrier* barrier;            /* barrier */
  int bc;
  int Bc;
  int bk;
  int Bk;
  int fm_lp_block;
  size_t scratch_size;
  void* scratch;
};

LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_fusedgroupnorm {
  libxsmm_dnn_fusedgroupnorm_desc desc;
  libxsmm_dnn_tensor* reg_input;      /* input tensor */
  libxsmm_dnn_tensor* reg_output;     /* output tensor */
  libxsmm_dnn_tensor* grad_input;     /* grad input tensor */
  libxsmm_dnn_tensor* grad_output;    /* grad output tensor */
  libxsmm_dnn_tensor* reg_add;        /* elementwise tensor */
  libxsmm_dnn_tensor* grad_add;       /* grad elementwise tensor */
  libxsmm_dnn_tensor* reg_beta;       /* beta tensor */
  libxsmm_dnn_tensor* reg_gamma;      /* gamma tensor */
  libxsmm_dnn_tensor* grad_beta;      /* grad beta tensor */
  libxsmm_dnn_tensor* grad_gamma;     /* grad gamma tensor */
  libxsmm_dnn_tensor* expvalue;       /* expected value */
  libxsmm_dnn_tensor* rcpstddev;      /* reciprocal of standard derivation */
  libxsmm_dnn_tensor* variance;       /* variance */
  libxsmm_dnn_tensor* relumask;       /* relumask */
  libxsmm_barrier* barrier;           /* barrier */
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  size_t scratch_size;
  void* scratch;
};

LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_fullyconnected {
  libxsmm_dnn_fullyconnected_desc desc;
  libxsmm_dnn_tensor* reg_input;      /* input tensor */
  libxsmm_dnn_tensor* reg_output;     /* output tensor */
  libxsmm_dnn_tensor* grad_input;     /* grad input tensor */
  libxsmm_dnn_tensor* grad_output;    /* grad output tensor */
  libxsmm_dnn_tensor* reg_filter;     /* filter tensor */
  libxsmm_dnn_tensor* grad_filter;    /* grad filter tensor */
  libxsmm_dnn_tensor* reg_bias;       /* bias tensor */
  libxsmm_dnn_tensor* grad_bias;      /* grad bais tensor */
  libxsmm_dnn_tensor* relumask;       /* relumask */
  libxsmm_barrier* barrier;           /* barrier */
  int target_archid;

  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  /* Parameters to tune/specialize FC algorithms */
  int fwd_2d_blocking;
  int bwd_2d_blocking;
  int upd_2d_blocking;
  int fwd_bf;
  int bwd_bf;
  int upd_bf;
  int fwd_row_teams;
  int fwd_column_teams;
  int bwd_row_teams;
  int bwd_column_teams;
  int upd_row_teams;
  int upd_column_teams;
  int ifm_subtasks;
  int ofm_subtasks;
  int compressed_A;
  int sparsity_factor_A;

  int fm_lp_block;
  int bn;
  int bk;
  int bc;
  size_t scratch_size;
  size_t doutput_scratch_mark;
  void* scratch;

  libxsmm_bsmmfunction fwd_config_kernel;
  libxsmm_bsmmfunction bwd_config_kernel;
  libxsmm_bsmmfunction upd_config_kernel;
  libxsmm_bsmmfunction tilerelease_kernel;

  libxsmm_meltwfunction_unary tr_kernel;
  libxsmm_code_pointer gemm_fwd;     /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd2;    /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd3;    /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd4;    /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd5;    /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd6;    /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd7;    /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd8;    /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd9;    /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd10;   /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd11;   /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd12;   /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd13;   /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd14;   /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd15;   /* ability to hoist forward GEMMs */
  libxsmm_code_pointer gemm_fwd16;   /* ability to hoist forward GEMMs */

  libxsmm_code_pointer gemm_bwd;     /* ability to hoist backward GEMMs */
  libxsmm_code_pointer gemm_bwd2;    /* ability to hoist backward GEMMs */
  libxsmm_code_pointer gemm_bwd3;    /* ability to hoist backward GEMMs */
  libxsmm_code_pointer gemm_upd;     /* ability to hoist update GEMMs */
  libxsmm_code_pointer gemm_upd2;    /* ability to hoist update GEMMs */
  libxsmm_code_pointer gemm_upd3;    /* ability to hoist update GEMMs */

  /* JITed eltwise kernels... */
  libxsmm_meltwfunction_unary fwd_cvtfp32bf16_kernel;
  libxsmm_meltwfunction_unary bwd_cvtfp32bf16_kernel;
  libxsmm_meltwfunction_unary bwd_relu_kernel;
  libxsmm_meltwfunction_unary fwd_cvtfp32bf16_relu_kernel;
  libxsmm_meltwfunction_unary fwd_sigmoid_cvtfp32bf16_kernel;
};

LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_pooling {
  libxsmm_dnn_pooling_desc desc;
  libxsmm_dnn_tensor* reg_input;      /* input tensor */
  libxsmm_dnn_tensor* reg_output;     /* output tensor */
  libxsmm_dnn_tensor* grad_input;     /* grad input tensor */
  libxsmm_dnn_tensor* grad_output;    /* grad output tensor */
  libxsmm_dnn_tensor* mask;           /* elementwise tensor */
  libxsmm_barrier* barrier;           /* barrier */
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  int ofh;
  int ofw;
  size_t scratch_size;
  void* scratch;
};

LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_rnncell {
  libxsmm_dnn_rnncell_desc desc;
  libxsmm_blasint T;                              /* sequence length, must be smaller than max sequence length in desc */
  libxsmm_blasint bk;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint lpb;

  /* external tensors */
  libxsmm_dnn_tensor* xt;
  libxsmm_dnn_tensor* csp;
  libxsmm_dnn_tensor* hp;
  libxsmm_dnn_tensor* w;
  libxsmm_dnn_tensor* wt;
  libxsmm_dnn_tensor* r;
  libxsmm_dnn_tensor* rt;
  libxsmm_dnn_tensor* b;
  libxsmm_dnn_tensor* cst;
  libxsmm_dnn_tensor* ht;
  libxsmm_dnn_tensor* dxt;
  libxsmm_dnn_tensor* dcsp;
  libxsmm_dnn_tensor* dhp;
  libxsmm_dnn_tensor* dw;
  libxsmm_dnn_tensor* dr;
  libxsmm_dnn_tensor* db;
  libxsmm_dnn_tensor* dcs;
  libxsmm_dnn_tensor* dht;
  libxsmm_dnn_tensor* it;
  libxsmm_dnn_tensor* ft;
  libxsmm_dnn_tensor* ot;
  libxsmm_dnn_tensor* cit;
  libxsmm_dnn_tensor* cot;
  float forget_bias;
  /* internal  state */
  void* internal_z;
  /* scratch pointers */
  void* scratch_base;
  void* scratch_wT;
  void* scratch_rT;
  void* scratch_w;
  void* scratch_r;
  void* scratch_xT;
  void* scratch_hT;
  void* scratch_deltat;
  void* scratch_di;
  void* scratch_df;
  void* scratch_do;
  void* scratch_dci;
  void* scratch_diB;
  void* scratch_dfB;
  void* scratch_dpB;
  void* scratch_dciB;
  void* scratch_dx;
  void* scratch_dhp;
  void* scratch_db;
  void* scratch_t1;
  void* scratch_t2;
  void* csp_scratch;
  void* cst_scratch;
  void* ht_scratch;
  void* it_scratch;
  void* ft_scratch;
  void* ot_scratch;
  void* cit_scratch;
  void* cot_scratch;
  /* options */
  int use_fwd_fused_impl;
  int fwd_block;
  int bwdupd_block;
  int fwd_generic;
  int bwdupd_generic;
  /* Ability to hoist GEMMs */
  libxsmm_bsmmfunction_reducebatch_strd fwd_kernela;
  libxsmm_bsmmfunction_reducebatch_strd fwd_kernelb;
  libxsmm_bsmmfunction_reducebatch_addr fwd_tileconfig;
  libxsmm_bsmmfunction_reducebatch_strd bwdupd_kernela;
  libxsmm_bsmmfunction_reducebatch_strd bwdupd_kernelb;
  libxsmm_bsmmfunction_reducebatch_strd bwdupd_kernelc;
  libxsmm_bsmmfunction_reducebatch_strd bwdupd_kerneld;
  libxsmm_bsmmfunction_reducebatch_addr bwdupd_tileconfig;
  libxsmm_bsmmfunction tilerelease_kernel;
  libxsmm_barrier* barrier; /* barrier */
};

struct LIBXSMM_RETARGETABLE libxsmm_dfsspmdm {
  int M;
  int N;
  int K;
  int ldb;
  int ldc;
  int N_chunksize;
  double* a_dense;
  libxsmm_dmmfunction kernel;
};

struct LIBXSMM_RETARGETABLE libxsmm_sfsspmdm {
  int M;
  int N;
  int K;
  int ldb;
  int ldc;
  int N_chunksize;
  float* a_dense;
  libxsmm_smmfunction kernel;
};

/** Packed structure storing the mateltw argument description. */
LIBXSMM_EXTERN_C LIBXSMM_PACKED(struct LIBXSMM_RETARGETABLE) libxsmm_meqn_descriptor {
  /** LDx, M, and N. */
  unsigned int m, n, ldo;
  /** Size of data element. */
  unsigned char datatype;
  /** Set of flags */
  unsigned int eqn_idx;
};

typedef enum libxsmm_build_kind {
  LIBXSMM_BUILD_KIND_GEMM       = LIBXSMM_KERNEL_KIND_MATMUL,
  LIBXSMM_BUILD_KIND_MELTW      = LIBXSMM_KERNEL_KIND_MELTW,
  LIBXSMM_BUILD_KIND_MEQN       = LIBXSMM_KERNEL_KIND_MEQN,
  LIBXSMM_BUILD_KIND_USER       = LIBXSMM_KERNEL_KIND_USER,
  LIBXSMM_BUILD_KIND_PGEMMRMAC  = LIBXSMM_KERNEL_UNREGISTERED,
  LIBXSMM_BUILD_KIND_PGEMMRMBC,
  LIBXSMM_BUILD_KIND_PSPGEMM_CSR,
  LIBXSMM_BUILD_KIND_PSPGEMM_CSC,
  LIBXSMM_BUILD_KIND_SREG
} libxsmm_build_kind;

/** Integral type (libxsmm_kernel_kind, libxsmm_build_kind). */
#if defined(LIBXSMM_UNPACKED)
# define LIBXSMM_DESCRIPTOR_BIG(KIND) ((libxsmm_descriptor_kind)((KIND) | 0x8000000000000000))
# define LIBXSMM_DESCRIPTOR_ISBIG(KIND) ((int)(((libxsmm_descriptor_kind)(KIND)) >> 63))
# define LIBXSMM_DESCRIPTOR_KIND(KIND) ((int)(((libxsmm_descriptor_kind)(KIND)) & 0x7FFFFFFFFFFFFFFF))
typedef uint64_t libxsmm_descriptor_kind;
#else
# define LIBXSMM_DESCRIPTOR_BIG(KIND) ((libxsmm_descriptor_kind)((KIND) | 0x80))
# define LIBXSMM_DESCRIPTOR_ISBIG(KIND) ((int)((KIND) >> 7))
# define LIBXSMM_DESCRIPTOR_KIND(KIND) ((int)((KIND) & 0x7F))
typedef unsigned char libxsmm_descriptor_kind;
#endif

/** All descriptor types, which are valid for code-registration. */
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_descriptor {
  char data[LIBXSMM_DESCRIPTOR_MAXSIZE];
  libxsmm_descriptor_kind kind; /* kind: must be the first member */
  LIBXSMM_REGDESC(LIBXSMM_PACKED(struct) { libxsmm_descriptor_kind /*repeated kind*/ pad; , desc; });
  LIBXSMM_PACKED(struct) { libxsmm_descriptor_kind /*repeated kind*/ pad; char desc[1]; } user;
} libxsmm_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_build_request {
  union {
    const void* ptr; /* raw content */
    LIBXSMM_REGDESC(LIBXSMM_REGDESC_DEFAULT, const*);
    const libxsmm_pspgemm_csr_descriptor* pspgemm_csr;
    const libxsmm_pspgemm_csc_descriptor* pspgemm_csc;
    const libxsmm_pgemm_ac_rm_descriptor* pgemmacrm;
    const libxsmm_pgemm_bc_rm_descriptor* pgemmbcrm;
    const libxsmm_csr_reg_descriptor* sreg;
  } descriptor;
  libxsmm_build_kind kind;
  /* used by user-kind */
  size_t user_size;
} libxsmm_build_request;

typedef enum libxsmm_malloc_flags {
  LIBXSMM_MALLOC_FLAG_DEFAULT = 0,
  LIBXSMM_MALLOC_FLAG_SCRATCH = 1,
  LIBXSMM_MALLOC_FLAG_PRIVATE = 2,
  LIBXSMM_MALLOC_FLAG_REALLOC = 4,
  LIBXSMM_MALLOC_FLAG_PHUGE   = 8,
  LIBXSMM_MALLOC_FLAG_PLOCK   = 16,
  LIBXSMM_MALLOC_FLAG_MMAP    = 32,
  LIBXSMM_MALLOC_FLAG_R       = 64,
  LIBXSMM_MALLOC_FLAG_W       = 128,
  LIBXSMM_MALLOC_FLAG_X       = 256,
  LIBXSMM_MALLOC_FLAG_RW  = LIBXSMM_MALLOC_FLAG_R | LIBXSMM_MALLOC_FLAG_W,
  LIBXSMM_MALLOC_FLAG_WX  = LIBXSMM_MALLOC_FLAG_X | LIBXSMM_MALLOC_FLAG_W,
  LIBXSMM_MALLOC_FLAG_RWX = LIBXSMM_MALLOC_FLAG_X | LIBXSMM_MALLOC_FLAG_RW,
  LIBXSMM_MALLOC_FLAG_VALID       = LIBXSMM_MALLOC_FLAG_SCRATCH |
      LIBXSMM_MALLOC_FLAG_PRIVATE | LIBXSMM_MALLOC_FLAG_REALLOC |
      LIBXSMM_MALLOC_FLAG_PHUGE   | LIBXSMM_MALLOC_FLAG_PLOCK |
      LIBXSMM_MALLOC_FLAG_MMAP    | LIBXSMM_MALLOC_FLAG_RWX
} libxsmm_malloc_flags;

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void* (*libxsmm_realloc_fun)(void* /*ptr*/, size_t /*size*/);

#if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_malloc_fntype {
  union { const void* dlsym; void* (*ptr)(size_t, size_t);  } alignmem;
  union { const void* dlsym; void* (*ptr)(size_t, size_t);  } memalign;
  union { const void* dlsym; libxsmm_malloc_fun ptr;        } malloc;
# if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
  union { const void* dlsym; void* (*ptr)(size_t, size_t);  } calloc;
# endif
# if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
  union { const void* dlsym; libxsmm_realloc_fun ptr;      } realloc;
# endif
  union { const void* dlsym; libxsmm_free_fun ptr;          } free;
} libxsmm_malloc_fntype;
LIBXSMM_APIVAR_PRIVATE(libxsmm_malloc_fntype libxsmm_malloc_fn);
#endif

#if (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD)))
/* prototypes for GLIBC internal implementation */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void* __libc_memalign(size_t alignment, size_t size);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void* __libc_malloc(size_t size);
#if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void* __libc_calloc(size_t num, size_t size);
#endif
#if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void* __libc_realloc(void* ptr, size_t size);
#endif
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void  __libc_free(void* ptr);
#endif /*(defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD)))*/

LIBXSMM_API_INTERN void* libxsmm_memalign_internal(size_t alignment, size_t size);

/* See https://sourceware.org/binutils/docs-2.34/ld/Options.html#index-_002d_002dwrap_003dsymbol */
LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void* __real_memalign(size_t alignment, size_t size);
LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void* __real_malloc(size_t size);
#if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void* __real_calloc(size_t num, size_t size);
#endif
#if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void* __real_realloc(void* ptr, size_t size);
#endif
LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void __real_free(void* ptr);

/** Retrieve internal information about a buffer (default memory domain). */
LIBXSMM_API int libxsmm_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra);

/** Initializes malloc hooks and other internals. */
LIBXSMM_API_INTERN void libxsmm_malloc_init(void);
LIBXSMM_API_INTERN void libxsmm_malloc_finalize(void);

/** Calculates an alignment depending on supposedly allocated size; alignment can be zero ("auto"). */
LIBXSMM_API_INTERN size_t libxsmm_alignment(size_t size, size_t alignment);

/** Same as libxsmm_set_default_allocator, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN int libxsmm_xset_default_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  const void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Same as libxsmm_get_default_allocator, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN int libxsmm_xget_default_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  const void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/** Same as libxsmm_set_scratch_allocator, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN int libxsmm_xset_scratch_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  const void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Same as libxsmm_get_scratch_allocator, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN int libxsmm_xget_scratch_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  const void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/**
 * Attribute memory allocation and protect with only the necessary flags.
 * This procedure is expected to run only one time per buffer, and may
 * relocate the given memory.
 */
LIBXSMM_API_INTERN int libxsmm_malloc_attrib(void** memory, int flags,
  /** If a name is given, an executable buffer will be dumped into a file. */
  const char* name);

/** Like libxsmm_release_scratch, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN void libxsmm_xrelease_scratch(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXSMM_API int libxsmm_xmalloc(void** memory, size_t size, size_t alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
/** Release memory, which was allocated using libxsmm_[*]malloc. */
LIBXSMM_API void libxsmm_xfree(const void* memory, int check);

/**
 * Format for instance an amount of Bytes like libxsmm_format_value(result, sizeof(result), nbytes, "KMGT", "B", 10).
 * The value returned is in requested/determined unit so that the user can decide about printing the buffer.
 */
LIBXSMM_API_INTERN size_t libxsmm_format_value(char buffer[32], int buffer_size, size_t nbytes, const char scale[], const char* unit, int base);

/** Returns the type-name of data-type (can be also libxsmm_gemm_precision). */
LIBXSMM_API_INTERN const char* libxsmm_typename(libxsmm_datatype datatype);

/** Dump data and (optionally) checks attempt to dump different data into an existing file (unique). */
LIBXSMM_API_INTERN int libxsmm_dump(const char* title, const char* name, const void* data, size_t size, int unique);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXSMM_CAPACITY_REGISTRY for unmanaged code). */
LIBXSMM_API_INTERN int libxsmm_build(const libxsmm_build_request* request, unsigned int regindex, libxsmm_code_pointer* code);

/** Returns the type-size of data-type (can be also libxsmm_gemm_precision). */
LIBXSMM_API unsigned char libxsmm_typesize(libxsmm_datatype datatype);

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_kernel_xinfo {
  /** Non-zero if kernel is registered. */
  unsigned int registered;
  /** Number of FLoating Point OPerationS (FLOPS). */
  unsigned int nflops;
} libxsmm_kernel_xinfo;

/** Receive information about JIT-generated code. */
LIBXSMM_API_INTERN const libxsmm_kernel_xinfo* libxsmm_get_kernel_xinfo(libxsmm_code_pointer code, const libxsmm_descriptor** desc, size_t* code_size);

/** Calculates duration in seconds from given RTC ticks. */
LIBXSMM_API_INTERN double libxsmm_timer_duration_rtc(libxsmm_timer_tickint tick0, libxsmm_timer_tickint tick1);
/** Returns the current tick of platform-specific real-time clock. */
LIBXSMM_API_INTERN libxsmm_timer_tickint libxsmm_timer_tick_rtc(void);
/** Returns the current tick of a (monotonic) platform-specific counter. */
LIBXSMM_API_INTERN libxsmm_timer_tickint libxsmm_timer_tick_tsc(void);

LIBXSMM_API_INTERN void libxsmm_memory_init(int target_arch);
LIBXSMM_API_INTERN void libxsmm_memory_finalize(void);

LIBXSMM_API_INTERN void libxsmm_dnn_init(int target_arch);
LIBXSMM_API_INTERN void libxsmm_dnn_finalize(void);

/** intern function to calculate blockings, that's private API hence it's in this function */
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_get_feature_map_blocks(
  int C, int K, int* C_block, int* K_block, int* fm_lp_block,
  libxsmm_dnn_datatype datatype_in, libxsmm_dnn_datatype datatype_out);

/** Global lock; create an own lock for an independent domain. */
LIBXSMM_APIVAR_PUBLIC(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK) libxsmm_lock_global);
/** Determines whether a threaded implementation is synchronized or not. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_nosync);

/** Function used to allocate default memory. */
LIBXSMM_APIVAR_PRIVATE(libxsmm_malloc_function libxsmm_default_malloc_fn);
/** Function used to allocate scratch memory. */
LIBXSMM_APIVAR_PRIVATE(libxsmm_malloc_function libxsmm_scratch_malloc_fn);
/** Function used to release default memory. */
LIBXSMM_APIVAR_PRIVATE(libxsmm_free_function libxsmm_default_free_fn);
/** Function used to release scratch memory. */
LIBXSMM_APIVAR_PRIVATE(libxsmm_free_function libxsmm_scratch_free_fn);
/** If non-NULL, this context is used by the context-form of memory allocation. */
LIBXSMM_APIVAR_PRIVATE(const void* libxsmm_default_allocator_context);
/** If non-NULL, this context is used by the context-form of memory allocation. */
LIBXSMM_APIVAR_PRIVATE(const void* libxsmm_scratch_allocator_context);
/** Number of scratch memory pools used; clamped against internal maximum. */
LIBXSMM_APIVAR_PRIVATE(unsigned int libxsmm_scratch_pools);
/** Growth factor used to scale the scratch memory in case of reallocation. */
LIBXSMM_APIVAR_PRIVATE(double libxsmm_scratch_scale);
/** Number of seconds per RDTSC-cycle (zero or negative if RDTSC invalid). */
LIBXSMM_APIVAR_PRIVATE(double libxsmm_timer_scale);
/** Counts the number of attempts to create an SPMDM-handle. */
LIBXSMM_APIVAR_PRIVATE(unsigned int libxsmm_statistic_num_spmdm);
/** Counts the maximum number of thread that have been active. */
LIBXSMM_APIVAR_PRIVATE(unsigned int libxsmm_thread_count);

#if (0 != LIBXSMM_SYNC)
LIBXSMM_APIVAR_PRIVATE(LIBXSMM_TLS_TYPE libxsmm_tlskey);
#endif

#endif /*LIBXSMM_MAIN_H*/

