/******************************************************************************
** Copyright (c) 2014-2018, Intel Corporation                                **
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
/* Hans Pabst (Intel Corp.), Rajkishore Barik (Intel Corp. )
******************************************************************************/
#ifndef LIBXSMM_MAIN_H
#define LIBXSMM_MAIN_H

#include <libxsmm_frontend.h>
#include <libxsmm_generator.h>
#include <libxsmm_malloc.h>
#include <libxsmm_sync.h>
#include <libxsmm_dnn.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXSMM_CAPACITY_REGISTRY) /* must be POT */
# define LIBXSMM_CAPACITY_REGISTRY 524288 /* 524287: Mersenne Prime number (2^19-1) */
#endif

#if !defined(LIBXSMM_MAX_NTHREADS)
# define LIBXSMM_MAX_NTHREADS 512
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS)
# define LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS LIBXSMM_MAX_NTHREADS
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_LIMIT)
# define LIBXSMM_MALLOC_SCRATCH_LIMIT (4ULL << 30) /* 4 GB */
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_MMAP) && 0
# define LIBXSMM_MALLOC_SCRATCH_MMAP
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_SCALE)
# if defined(LIBXSMM_MALLOC_SCRATCH_MMAP)
#   define LIBXSMM_MALLOC_SCRATCH_SCALE 1.3
# else
#   define LIBXSMM_MALLOC_SCRATCH_SCALE 1.0
# endif
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_INTERNAL_SITE)
# define LIBXSMM_MALLOC_SCRATCH_INTERNAL_SITE ((uintptr_t)-1)
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_INTERNAL)
# define LIBXSMM_MALLOC_SCRATCH_INTERNAL ((const void*)(LIBXSMM_MALLOC_SCRATCH_INTERNAL_SITE))
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

/* Helper macro to eventually (if defined) call libxsmm_init */
#if !defined(LIBXSMM_INIT) && !defined(LIBXSMM_CTOR)
# define LIBXSMM_INIT libxsmm_init();
#elif !defined(LIBXSMM_INIT)
# define LIBXSMM_INIT
#endif

/** Check if M, N, K, or LDx fits into the descriptor. */
#if (0 != LIBXSMM_ILP64)
# define LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K) ( \
    ((unsigned int)(-1)) >= ((unsigned int)(M)) && \
    ((unsigned int)(-1)) >= ((unsigned int)(N)) && \
    ((unsigned int)(-1)) >= ((unsigned int)(K)))
#else /* always fits */
# define LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K) 1
#endif

#if defined(LIBXSMM_ASSERT) /* assert available */
# define LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K) LIBXSMM_ASSERT(LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K))
#else
# define LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K)
#endif

#if (defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)) /* TODO: full support for Windows calling convention */
# define LIBXSMM_GEMM_DESCRIPTOR_PREFETCH(DESCRIPTOR, PREFETCH) LIBXSMM_UNUSED(PREFETCH); \
            (DESCRIPTOR).prefetch = (unsigned short)(LIBXSMM_GEMM_PREFETCH_NONE)
#else
# define LIBXSMM_GEMM_DESCRIPTOR_PREFETCH(DESCRIPTOR, PREFETCH) (DESCRIPTOR).prefetch = (unsigned short)(PREFETCH)
#endif

/**
* Construct a GEMM descriptor after it has been declared. The descriptor flags will sanitized to remove any
* alignment request which cannot be satisfied (avoids to build an unnecessary code version).
*/
#define LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K); LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(LDA, LDB, LDC); \
  (DESCRIPTOR).lda = (unsigned int)(LDA); (DESCRIPTOR).ldb = (unsigned int)(LDB); (DESCRIPTOR).ldc = (unsigned int)(LDC); \
  (DESCRIPTOR).m   = (unsigned int)(M);   (DESCRIPTOR).n   = (unsigned int)(N);   (DESCRIPTOR).k   = (unsigned int)(K); \
  (DESCRIPTOR).flags = (unsigned short)(FLAGS); LIBXSMM_GEMM_DESCRIPTOR_PREFETCH(DESCRIPTOR, PREFETCH); \
  (DESCRIPTOR).alpha = (signed char)(ALPHA); (DESCRIPTOR).beta = (signed char)(BETA); \
  (DESCRIPTOR).datatype = (unsigned char)(DATA_TYPE); (DESCRIPTOR).iflags = 0
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


/**
* Structure, which stores the argument description of GEMM routines.
* This structure must be ordered by the size of the members (packed).
* The size of the structure matches LIBXSMM_GEMM_DESCRIPTOR_SIZE.
*/
LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_gemm_descriptor {
  /** Leading dimensions are general offsets. */
  unsigned int lda, ldb, ldc;
  /** Extents of the matrix. */
  unsigned int m, n, k;
  /** Set of flags. */
  unsigned short flags;
  /** Prefetch strategy enumeration. */
  unsigned short prefetch;
  /** Integer value. */
  signed char alpha, beta;
  /** Denotes the data-type. */
  unsigned char datatype;
  /** INTERNAL (last member!) */
  unsigned char iflags;
};

/** Structure storing the matcopy argument description. */
LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_mcopy_descriptor { /* 20 Byte */
  /** LDx, M, and N. */
  unsigned int m, n, ldi, ldo;
  /** Size of data element. */
  unsigned char typesize;
  /** Level of unrolling. */
  unsigned char unroll_level;
  /** Boolean value (@TODO fix this). */
  unsigned char prefetch;
  /** Set of flags. */
  unsigned char flags;
};

/** Structure storing the transpose argument description. */
LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_trans_descriptor { /* 13 Byte */
  /** LD, M, and N. */
  unsigned int m, n, ldo;
  /** Size of data element. */
  unsigned char typesize;
};

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_csr_soa_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxsmm_csr_soa_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_csc_soa_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* column_ptr;
  const unsigned int* row_idx;
  const void* values;
} libxsmm_csc_soa_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_csr_reg_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxsmm_csr_reg_descriptor;

/** Function type used for convolutions (single-precision); the actual signature depends on the kind of convolution. */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_sconvfunction)(
  const float* input1, const float* input2, float* output,
  const float* ipf1, const float* ipf2, const float* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wconvfunction)(
  const short* input1, const short* input2, int* output,
  const short* ipf1, const short* ipf2, const int* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wsconvfunction)(
  const short* input1, const short* input2, float* output,
  const short* ipf1, const short* ipf2, const float* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_uwsconvfunction)(
  short* input1, float* input2, short* output,
  short* ipf1, float* ipf2, short* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bdbconvfunction)(
  unsigned char* input1, int* input2, unsigned char* output,
  unsigned char* ipf1, int* ipf2, unsigned char* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_busconvfunction)(
  const unsigned char* input1, const char* input2, short* output,
  const unsigned char* ipf1, const char* ipf2, const short* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_budconvfunction)(
  const unsigned char* input1, const char* input2, int* output,
  const unsigned char* ipf1, const char* ipf2, const int* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wconvfunction_bwd)(
  int* input1, const short* input2, const short* output,
  const int* ipf1, const short* ipf2, const short* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_busconvfunction_bwd)(
  const unsigned short* input1, const char* input2, const char* output,
  const unsigned short* ipf1, const char* ipf2, const char* opf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_budconvfunction_bwd)(
  const unsigned int* input1, const char* input2, const char* output,
  const unsigned int* ipf1, const char* ipf2, const char* opf, ...);

/** Function type which is either libxsmm_sconvfunction or libxsmm_wconvfunction (weak-typed). */
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_xconvfunction {
  libxsmm_sconvfunction sconv;
  libxsmm_wsconvfunction wsconv;
  libxsmm_uwsconvfunction uwsconv;
  libxsmm_wconvfunction wconv;
  libxsmm_bdbconvfunction bdbconv;
  libxsmm_busconvfunction busconv;
  libxsmm_budconvfunction budconv;
  libxsmm_wconvfunction_bwd wconvb;
  libxsmm_busconvfunction_bwd busconvb;
  libxsmm_budconvfunction_bwd budconvb;
} libxsmm_xconvfunction;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_code_pointer {
  void (*ptr_fn)(LIBXSMM_VARIADIC);
  const void* ptr_const;
  void* pmm;
  uintptr_t uval;
  intptr_t ival;
  libxsmm_xmmfunction xgemm; /* GEMM: smm, dmm, wimm, wsmm, or void-function */
  libxsmm_xmcopyfunction xmatcopy;
  libxsmm_xtransfunction xtrans;
  libxsmm_xconvfunction xconv;
} libxsmm_code_pointer;

/** Structure which describes all tensors in LIBXSMM's DNN module */
LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_dnn_tensor {
  libxsmm_dnn_tensor_datalayout* layout;           /* data-layout descriptor */
  void* data;                                      /* pointer to data */
  unsigned char scf;                               /* fix point scaling factor for this tensor */
};

/* Structure to record segment in stream of code  */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE segment_t {
  int segment_type;
  int n_convs;
  int aux_index;
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
  libxsmm_convolution_winograd_descriptor cwino_fwd;
  libxsmm_convolution_winograd_descriptor cwino_bwd;
  libxsmm_convolution_winograd_descriptor cwino_upd;
  libxsmm_dnn_internal_format custom_format_type;    /* Specifies internal LIBXSMM format to be used */

  /* additional size for internal data types */
  int ifhp;
  int ifwp;
  int ofh;
  int ofw;
  int ofhp;
  int ofwp;
  int ifmblock;
  int ifmblock_hp;
  int ofmblock;
  int ofmblock_lp;
  int blocksifm;
  int blocksofm;
  int blocksifm_lp;
  int blocksofm_lp;
  int fwd_ofw_rb;
  int fwd_ofw_rb_2;
  int fwd_ofh_rb;
  int fwd_ofh_rb_2;
  int bwd_ofw_rb;
  int bwd_ofh_rb;
  int upd_ofw_rb;
  int upd_ofh_rb;
  int fm_lp_block; /* additional blocking for low precision datatypes of feature maps */
  int upd_use_thread_fil;
  int upd_use_external_reduce;
  int filter_transposed;
  int nBImg;
  int nbImg;
  int blocksifm_blocking;
  int blocksofm_blocking;
  int blocksimg_blocking;
  int use_nts_fwd;
  int use_nts_bwd;
  int use_fwd_for_bwd;
  int exploit_duality;
  int qfma_input_pad;
  int resize_input;
  int ifhp_resized;
  int ifwp_resized;
  int use_fastpath;
  int use_hybrid_wu_parallelism;
  int weight_copies;
  int compute_batch_stats_in_kernel;
  int compute_max_in_kernel_fwd;
  int compute_max_in_kernel_bwd;
  int perform_relu_in_kernel;
  int use_lp_kernel;
  int output_lp_padding;
  int reduce_weights;
  int use_vperm_transposes;
  int avoid_output_trans;
  int avoid_input_trans;
  int enforce_sfma_kernel;
  int n_variants;
  int w_variants;
  int h_variants;

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
  /* maxstats used in low-recision kernels */
  libxsmm_dnn_tensor* maxstats_fwd;
  libxsmm_dnn_tensor* maxstats_bwd;
  libxsmm_dnn_tensor* maxstats_upd;

  /* barrier */
  libxsmm_barrier* barrier;

  /* scratch */
  void* scratch1;
  size_t scratch1_size;
  void* scratch3;
  size_t scratch3_size;
  void* scratch4;
  size_t scratch4_size;
  void* scratch5;             /* This scratch is used as a copy buffer when padding needs to be applied */
  void* scratch6;
  size_t scratch6_size;
  size_t minibatch_scratch_size;
  size_t fwdbwd_scratch_size;
  size_t max_scratch5_size;
  int padding_flag;           /* Flag that dictates if we should apply padding in the input */
  void* scratchIw;
  size_t scratchIw_size;
  void* scratchOw;
  size_t scratchOw_size;
  void* scratchVk;
  size_t scratchVk_size;

  /* JIT-generated convolution code */
  int use_fwd_generic;
  int use_bwd_generic;
  int use_upd_generic;
  /*
  libxsmm_convolution_forward_descriptor       fwd_desc;
  libxsmm_convolution_forward_descriptor       bwd_desc;
  libxsmm_convolution_weight_update_descriptor wu_desc;
  */
  libxsmm_code_pointer code_fwd[3];
  libxsmm_code_pointer code_bwd[3];
  libxsmm_code_pointer code_upd[2];

  libxsmm_code_pointer matcopy_fwd[4];
  libxsmm_code_pointer matcopy_bwd[4];
  libxsmm_code_pointer matcopy_upd[3];

  /* Data structures and metadata related to per-thread private JITing */
  int use_thread_private_jit;
  int use_thread_private_filter;
  int trans_ofw_ifm;

  int *n_entries_fwd;
  int **compute_fwd_indices_ptrs;
  int **bn_indices_ptrs;
  char **kernel_fwd_variant_ptrs;
  int block_fwd_oj;
  int block_fwd_oi;
  int block_fwd_ifm;
  int block_fwd_ofm;
  int *n_fwd_code_segments;
  segment_t **fwd_code_segments;
  int *ofh_fwd_start;
  int *ofh_fwd_end;

  int *n_entries_bwd;
  int **compute_bwd_indices_ptrs;
  char **kernel_bwd_variant_ptrs;
  int block_bwd_oj;
  int block_bwd_oi;
  int block_bwd_ifm;
  int block_bwd_ofm;
  int *n_bwd_code_segments;
  segment_t **bwd_code_segments;
  int *n_entries_trans_bwd;
  int **transpose_bwd_indices_ptrs;
  int *ofh_bwd_start;
  int *ofh_bwd_end;

  int *n_entries_upd;
  int block_upd_ifm;
  int block_upd_ofm;
  int **compute_upd_indices_ptrs;
  char **kernel_upd_variant_ptrs;
  int *n_upd_code_segments;
  segment_t **upd_code_segments;
  int *n_entries_init_upd;
  int **init_upd_indices_ptrs;
  int *n_entries_copy_upd;
  int **copy_upd_indices_ptrs;
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

typedef enum libxsmm_build_kind {
  LIBXSMM_BUILD_KIND_GEMM,
  LIBXSMM_BUILD_KIND_SRSOA,
  LIBXSMM_BUILD_KIND_SCSOA,
  LIBXSMM_BUILD_KIND_SREG,
  LIBXSMM_BUILD_KIND_CFWD,
  LIBXSMM_BUILD_KIND_CUPD,
  LIBXSMM_BUILD_KIND_CWFWD,
  LIBXSMM_BUILD_KIND_CWBWD,
  LIBXSMM_BUILD_KIND_CWUPD,
  LIBXSMM_BUILD_KIND_MCOPY,
  LIBXSMM_BUILD_KIND_TRANS
} libxsmm_build_kind;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_build_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const libxsmm_csr_soa_descriptor* srsoa;
  const libxsmm_csc_soa_descriptor* scsoa;
  const libxsmm_csr_reg_descriptor* sreg;
  const libxsmm_convolution_forward_descriptor* cfwd;
  const libxsmm_convolution_weight_update_descriptor* cupd;
  const libxsmm_convolution_winograd_descriptor* cwino;
  const libxsmm_mcopy_descriptor* matcopy;
  const libxsmm_trans_descriptor* trans;
} libxsmm_build_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_build_request {
  libxsmm_build_descriptor descriptor;
  libxsmm_build_kind kind;
} libxsmm_build_request;

typedef enum libxsmm_malloc_flags {
  LIBXSMM_MALLOC_FLAG_DEFAULT = 0,
  LIBXSMM_MALLOC_FLAG_SCRATCH = 1,
  LIBXSMM_MALLOC_FLAG_MMAP = 2,
  LIBXSMM_MALLOC_FLAG_R = 4,
  LIBXSMM_MALLOC_FLAG_W = 8,
  LIBXSMM_MALLOC_FLAG_X = 16,
  LIBXSMM_MALLOC_FLAG_RW  = LIBXSMM_MALLOC_FLAG_R | LIBXSMM_MALLOC_FLAG_W,
  LIBXSMM_MALLOC_FLAG_RWX = LIBXSMM_MALLOC_FLAG_RW | LIBXSMM_MALLOC_FLAG_X
} libxsmm_malloc_flags;

/** Greatest common divisor. */
LIBXSMM_API_INTERN size_t libxsmm_gcd(size_t a, size_t b);
/** Least common multiple. */
LIBXSMM_API_INTERN size_t libxsmm_lcm(size_t a, size_t b);
/** Calculates an alignment depending on supposedly allocated size; alignment can be zero ("auto"). */
LIBXSMM_API_INTERN size_t libxsmm_alignment(size_t size, size_t alignment);

/** Same as libxsmm_set_default_allocator, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN int libxsmm_xset_default_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Same as libxsmm_get_default_allocator, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN int libxsmm_xget_default_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/** Same as libxsmm_set_scratch_allocator, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN int libxsmm_xset_scratch_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Same as libxsmm_get_scratch_allocator, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN int libxsmm_xget_scratch_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/** Retrieve internal information about a buffer (default memory domain). */
LIBXSMM_API int libxsmm_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXSMM_API_INTERN int libxsmm_xmalloc(void** memory, size_t size, size_t alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
/** Release memory, which was allocated using libxsmm_[*]malloc. */
LIBXSMM_API_INTERN int libxsmm_xfree(const void* memory);

/**
 * Attribute memory allocation and protect with only the necessary flags.
 * This procedure is expected to run only one time per buffer, and may
 * relocate the given memory.
 */
LIBXSMM_API_INTERN int libxsmm_malloc_attrib(void** memory, int flags,
  /** If a name is given, an executable buffer will be dumped into a file. */
  const char* name);

LIBXSMM_API_INTERN unsigned char libxsmm_typesize(libxsmm_datatype datatype);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXSMM_CAPACITY_REGISTRY for unmanaged code). */
LIBXSMM_API_INTERN int libxsmm_build(const libxsmm_build_request* request, unsigned int regindex, libxsmm_code_pointer* code);

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_kernel_info {
  libxsmm_gemm_descriptor xgemm;
  libxsmm_mcopy_descriptor mcopy;
  libxsmm_trans_descriptor trans;
} libxsmm_kernel_info;

/** Attempts to receive information about JIT-generated code. */
LIBXSMM_API const libxsmm_kernel_info* libxsmm_get_kernel_info(libxsmm_code_pointer code, libxsmm_kernel_kind* kind, size_t* size);

/** Updates counters of the statistic, which is shown at program termination. */
LIBXSMM_API unsigned int libxsmm_update_mmstatistic(libxsmm_gemm_precision precision,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, unsigned int ntry, unsigned int ncol);

/** Returns the current tick of a (monotonic) platform-specific counter; not necessarily CPU cycles. */
LIBXSMM_API_INTERN unsigned long long libxsmm_timer_tick_rdtsc(void);

LIBXSMM_API_INTERN void libxsmm_dnn_init(int target_arch);
LIBXSMM_API_INTERN void libxsmm_dnn_finalize(void);

/** Code generation routine for a forward-convolution kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API_INTERN libxsmm_sconvfunction libxsmm_create_sconv_forward(const libxsmm_convolution_forward_descriptor* descriptor);

/** Code generation routine for a backward-convolution kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API_INTERN libxsmm_sconvfunction libxsmm_create_sconv_backward(const libxsmm_convolution_backward_descriptor* descriptor);

/** Code generation routine for a convolution kernel as specified by descriptor. */
LIBXSMM_API_INTERN libxsmm_sconvfunction libxsmm_create_sconv_update_weights(const libxsmm_convolution_weight_update_descriptor* descriptor);

/** Code generation routine for a forward-convolution kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API_INTERN void* libxsmm_create_xconv_forward(const libxsmm_convolution_forward_descriptor* descriptor);

/** Code generation routine for a backward-convolution kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API_INTERN void* libxsmm_create_xconv_backward(const libxsmm_convolution_backward_descriptor* descriptor);

/** Code generation routine for a convolution kernel as specified by descriptor. */
LIBXSMM_API_INTERN void* libxsmm_create_xconv_update_weights(const libxsmm_convolution_weight_update_descriptor* descriptor);

/** Code generation routine for a forward-convolution Winograd kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API_INTERN void* libxsmm_create_xconv_wino_forward(const libxsmm_convolution_winograd_descriptor* descriptor);

/** Code generation routine for a backward-convolution Winograd kernel. Call libxsmm_release_kernel in order to deallocate the JIT'ted code. */
LIBXSMM_API_INTERN void* libxsmm_create_xconv_wino_backward(const libxsmm_convolution_winograd_descriptor* descriptor);

/** Code generation routine for a weight-update-convolution Winograd kernel as specified by descriptor. */
LIBXSMM_API_INTERN void* libxsmm_create_xconv_wino_update_weights(const libxsmm_convolution_winograd_descriptor* descriptor);

/** Global lock; create an own lock for an independent domain. */
LIBXSMM_APIVAR_PUBLIC(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK) libxsmm_lock_global);
/** Verbosity level (0: quiet, 1: errors, 2: warnings, 3: info, neg.: all/dump). */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_verbosity);
/** Target architecture (libxsmm_get_target_archid, libxsmm_set_target_archid). */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_target_archid);
/** Determines whether a threaded implementation is synchronized or not. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_nosync);
/** Number of threads per core. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_nt);

/** Function used to allocate default memory. */
LIBXSMM_APIVAR(libxsmm_malloc_function libxsmm_default_malloc_fn);
/** Function used to allocate scratch memory. */
LIBXSMM_APIVAR(libxsmm_malloc_function libxsmm_scratch_malloc_fn);
/** Function used to release default memory. */
LIBXSMM_APIVAR(libxsmm_free_function libxsmm_default_free_fn);
/** Function used to release scratch memory. */
LIBXSMM_APIVAR(libxsmm_free_function libxsmm_scratch_free_fn);
/** If non-NULL, this context used for the context-form of the malloc/free function. */
LIBXSMM_APIVAR(void* libxsmm_default_allocator_context);
/** If non-NULL, this context used for the context-form of the malloc/free function. */
LIBXSMM_APIVAR(void* libxsmm_scratch_allocator_context);
/** Number of discovered threads (per libxsmm_get_tid) */
LIBXSMM_APIVAR(unsigned int libxsmm_threads_count);
/** Number of scratch memory pools used; clamped against internal maximum. */
LIBXSMM_APIVAR(unsigned int libxsmm_scratch_pools);
/** Maximum total size of the scratch memory domain. */
LIBXSMM_APIVAR(size_t libxsmm_scratch_limit);
/** Growth factor used to scale the scratch memory in case of reallocation. */
LIBXSMM_APIVAR(double libxsmm_scratch_scale);
/** Number of seconds per RDTSC-cycle (zero if RDTSC is not used for wall-clock) */
LIBXSMM_APIVAR(double libxsmm_timer_scale);

#endif /*LIBXSMM_MAIN_H*/

