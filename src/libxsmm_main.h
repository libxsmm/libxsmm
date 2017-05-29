/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
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

#include <libxsmm_typedefs.h>
#include <libxsmm_generator.h>
#include <libxsmm_malloc.h>
#include <libxsmm_sync.h>
#include <libxsmm_dnn.h>

#include <stddef.h>
#include <stdint.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXSMM_CAPACITY_REGISTRY) /* must be POT */
# define LIBXSMM_CAPACITY_REGISTRY 524288 /* 524287: Mersenne Prime number (2^19-1) */
#endif

#if !defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS)
# define LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS 16
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_SCALE)
# define LIBXSMM_MALLOC_SCRATCH_SCALE 1.4
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
#if !defined(LIBXSMM_CTOR) && !defined(LIBXSMM_INIT)
# define LIBXSMM_INIT libxsmm_init();
#elif !defined(LIBXSMM_INIT)
# define LIBXSMM_INIT
#endif

typedef union LIBXSMM_RETARGETABLE libxsmm_code_pointer {
  const void* const_pmm;
  void* pmm;
  uintptr_t uimm;
  intptr_t imm;
  libxsmm_xmmfunction xmm;
  libxsmm_smmfunction smm;
  libxsmm_wmmfunction wmm;
  void (*vmm)(const void* a, const void* b, void* c, ...);
#if defined(LIBXSMM_BUILD) || defined(LIBXSMM_DNN_INTERNAL_API)
  libxsmm_xconvfunction xconv;
#endif
  libxsmm_xmatcopyfunction xmatcopy;
  libxsmm_xtransfunction xtrans;
} libxsmm_code_pointer;

typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_csr_soa_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxsmm_csr_soa_descriptor;

typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_csr_reg_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxsmm_csr_reg_descriptor;

/** Structure which describes an activation layer. */
struct LIBXSMM_RETARGETABLE libxsmm_dnn_buffer {
  int N;                            /* number of images in mini-batch */
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int H;                            /* height of image */
  int W;                            /* width of image */
  int lpb;                          /* low precision blocking factor */
  int bimg;                         /* size of blocked images */
  libxsmm_dnn_tensor_format format; /* format of activation buffer */
  libxsmm_dnn_internal_format custom_format_type;
  libxsmm_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
  char exp;                         /* fix point exponent for this tensor */
};

/** Structure which describes a bias. */
struct LIBXSMM_RETARGETABLE libxsmm_dnn_bias {
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int lpb;                          /* low precision blocking factor */
  libxsmm_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
  char exp;                         /* fix point exponent for this tensor */
};

/** Structure which describes a filter */
struct LIBXSMM_RETARGETABLE libxsmm_dnn_filter {
  int ifmb;                         /* number of feature map blocks */
  int bifm;                         /* sized of blocked feature maps, in a block */
  int ofmb;                         /* number of feature map blocks */
  int bofm;                         /* sized of blocked feature maps, in a block */
  int R;                            /* height of filter kernel */
  int S;                            /* width of filter kernel */
  int lpb;                          /* low precision blocking factor */
  libxsmm_dnn_tensor_format format; /* format of filter buffer */
  libxsmm_dnn_internal_format custom_format_type;
  libxsmm_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
  char exp;                         /* fix point exponent for this tensor */
};

struct LIBXSMM_RETARGETABLE libxsmm_dnn_layer {
  libxsmm_dnn_datatype datatype;
  libxsmm_dnn_datatype datatype_itm;
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

  /* additional size for iternal data types */
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
  int fwd_ofw_rb_2;
  int fwd_ofh_rb;
  int bwd_ofw_rb;
  int bwd_ofh_rb;
  int upd_ofw_rb;
  int upd_ofh_rb;
  int fm_lp_block;              /* additional blocking for low precision datatypes of feature maps */
  int upd_use_thread_fil;
  int upd_use_external_reduce;
  int filter_transposed;
  int nBImg;
  int nbImg;

  /* internal data representation */
  libxsmm_dnn_buffer* reg_input;
  libxsmm_dnn_buffer* reg_output;
  libxsmm_dnn_filter* reg_filter;
  libxsmm_dnn_buffer* grad_input;
  libxsmm_dnn_buffer* grad_output;
  libxsmm_dnn_filter* grad_filter;
  libxsmm_dnn_bias* bias;

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
  size_t minibatch_scratch_size;
  size_t fwdbwd_scratch_size;
  size_t max_scratch5_size;
  int padding_flag;           /* Flag that dictates if we should apply padding in the input */
  void* scratch6;
  size_t scratch6_size;
  void* scratch7;         /* This scratch is used for low precision intermediate buffer for input in backward pass*/
  size_t scratch7_size;
  void* scratchIw;
  size_t scratchIw_size;
  void* scratchOw;
  size_t scratchOw_size;
  void* scratchVk;
  size_t scratchVk_size;
  void* scratchInput;
  size_t scratchInput_size;
  void* scratchTemp;
  int flag_reuseInput;        /* This flag is set to 1 when we want to reuse the input in Winograd domain between forward pass and weight update */

  /* JIT-generated convolution code */
  /*
  libxsmm_convolution_forward_descriptor       fwd_desc;
  libxsmm_convolution_forward_descriptor       bwd_desc;
  libxsmm_convolution_weight_update_descriptor wu_desc;
  */
  int avx512avx2fallback;
  libxsmm_code_pointer code_fwd[4];
  libxsmm_code_pointer code_bwd[4];
  libxsmm_code_pointer code_upd[6];

  libxsmm_code_pointer matcopy_fwd[1];
  libxsmm_code_pointer matcopy_bwd[2];
  libxsmm_code_pointer matcopy_upd[2];
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
  LIBXSMM_BUILD_KIND_SSOA,
  LIBXSMM_BUILD_KIND_SREG,
  LIBXSMM_BUILD_KIND_CFWD,
  LIBXSMM_BUILD_KIND_CBWD,
  LIBXSMM_BUILD_KIND_CUPD,
  LIBXSMM_BUILD_KIND_CWFWD,
  LIBXSMM_BUILD_KIND_CWBWD,
  LIBXSMM_BUILD_KIND_CWUPD,
  LIBXSMM_BUILD_KIND_MCOPY,
  LIBXSMM_BUILD_KIND_TRANS
} libxsmm_build_kind;

typedef union LIBXSMM_RETARGETABLE libxsmm_build_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const libxsmm_csr_soa_descriptor* ssoa;
  const libxsmm_csr_reg_descriptor* sreg;
  const libxsmm_convolution_forward_descriptor* cfwd;
  const libxsmm_convolution_backward_descriptor* cbwd;
  const libxsmm_convolution_weight_update_descriptor* cupd;
  const libxsmm_convolution_winograd_descriptor* cwino;
  const libxsmm_matcopy_descriptor* matcopy;
  const libxsmm_transpose_descriptor* trans;
} libxsmm_build_descriptor;

typedef struct LIBXSMM_RETARGETABLE libxsmm_build_request {
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
LIBXSMM_API size_t libxsmm_gcd(size_t a, size_t b);
/** Least common multiple. */
LIBXSMM_API size_t libxsmm_lcm(size_t a, size_t b);
/** Calculates an alignment depending on supposedly allocated size; alignment can be zero ("auto"). */
LIBXSMM_API size_t libxsmm_alignment(size_t size, size_t alignment);

/** Same as libxsmm_set_default_allocator, but takes a lock (can be NULL). */
LIBXSMM_API int libxsmm_xset_default_allocator(LIBXSMM_LOCK_TYPE* lock,
  void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Same as libxsmm_get_default_allocator, but takes a lock (can be NULL). */
LIBXSMM_API int libxsmm_xget_default_allocator(LIBXSMM_LOCK_TYPE* lock,
  void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/** Same as libxsmm_set_scratch_allocator, but takes a lock (can be NULL). */
LIBXSMM_API int libxsmm_xset_scratch_allocator(LIBXSMM_LOCK_TYPE* lock,
  void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Same as libxsmm_get_scratch_allocator, but takes a lock (can be NULL). */
LIBXSMM_API int libxsmm_xget_scratch_allocator(LIBXSMM_LOCK_TYPE* lock,
  void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/** Retrieve internal information about a buffer (default memory domain). */
LIBXSMM_API int libxsmm_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXSMM_API int libxsmm_xmalloc(void** memory, size_t size, size_t alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
/** Release memory, which was allocated using libxsmm_[*]malloc. */
LIBXSMM_API int libxsmm_xfree(const void* memory);

/**
 * Attribute memory allocation and protect with only the necessary flags.
 * This procedure is expected to run only one time per buffer, and may
 * relocate the given memory.
 */
LIBXSMM_API int libxsmm_malloc_attrib(void** memory, int flags,
  /** If a name is given, an executable buffer will be dumped into a file. */
  const char* name);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXSMM_CAPACITY_REGISTRY for unmanaged code). */
LIBXSMM_API int libxsmm_build(const libxsmm_build_request* request, unsigned regindex, libxsmm_code_pointer* code);

/** Updates counters of the statistic, which is shown at program termination. */
LIBXSMM_API unsigned int libxsmm_update_mmstatistic(int flags, int m, int n, int k, unsigned int ntry, unsigned int ncol);

LIBXSMM_API void libxsmm_dnn_init(int target_arch);
LIBXSMM_API void libxsmm_dnn_finalize(void);

LIBXSMM_API_INTERN LIBXSMM_LOCK_TYPE libxsmm_lock_global;
/** Function used to allocate default memory. */
LIBXSMM_API_INTERN libxsmm_malloc_function libxsmm_default_malloc_fn;
/** Function used to allocate scratch memory. */
LIBXSMM_API_INTERN libxsmm_malloc_function libxsmm_scratch_malloc_fn;
/** Function used to release default memory. */
LIBXSMM_API_INTERN libxsmm_free_function libxsmm_default_free_fn;
/** Function used to release scratch memory. */
LIBXSMM_API_INTERN libxsmm_free_function libxsmm_scratch_free_fn;
/** If non-NULL, this context used for the context-form of the malloc/free function. */
LIBXSMM_API_INTERN void* libxsmm_default_allocator_context;
/** If non-NULL, this context used for the context-form of the malloc/free function. */
LIBXSMM_API_INTERN void* libxsmm_scratch_allocator_context;
/** Number of scratch memory pools used; clamped against internal maximum. */
LIBXSMM_API_INTERN unsigned int libxsmm_scratch_pools;
/** Growth factor used to scale the scratch memory in case of reallocation. */
LIBXSMM_API_INTERN double libxsmm_scratch_scale;
/** Stores the verbosity level (libxsmm_get_verbosity, libxsmm_set_verbosity). */
LIBXSMM_API_INTERN int libxsmm_verbosity;
/** Target architecture (libxsmm_get_target_archid, libxsmm_set_target_archid). */
LIBXSMM_API_INTERN int libxsmm_target_archid;
/** Determines whether a threaded implementation is synchronized or not. */
LIBXSMM_API_INTERN int libxsmm_sync;
/** Number of threads per core. */
LIBXSMM_API_INTERN int libxsmm_nt;

#endif /*LIBXSMM_MAIN_H*/

