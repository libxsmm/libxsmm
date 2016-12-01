/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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

#include <stddef.h>
#include <stdint.h>

#include <libxsmm_macros.h>
#include <libxsmm_typedefs.h>
#include <libxsmm_generator.h>
#include <libxsmm_dnn.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXSMM_REGSIZE) /* must be POT */
# define LIBXSMM_REGSIZE 524288 /* 524287: Mersenne Prime number (2^19-1) */
#endif
#if !defined(LIBXSMM_CPU_DCACHESIZE)
# define LIBXSMM_CPU_DCACHESIZE 32768
#endif

/** Helper macro to account for libxsmm_init being already executed via GCC constructor attribute */
#if !defined(LIBXSMM_CTOR) && defined(__GNUC__) && !(defined(__INTEL_COMPILER) && !defined(LIBXSMM_BUILD))
# if defined(LIBXSMM_BUILD_EXT) && defined(__STATIC)
#   define LIBXSMM_INIT libxsmm_ext_init/*dummy*/ = libxsmm_init;
    /**
     * Global (dummy-)variable which is touched via LIBXSMM_INIT macro
     * in order to keep the libxsmm_init/libxsmm_finalize symbols
     * even when linking statically (or only linking libxsmmext).
     */
    LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void (*libxsmm_ext_init)(void);
# else
#   define LIBXSMM_INIT
# endif
# define LIBXSMM_CTOR_ATTRIBUTE LIBXSMM_ATTRIBUTE(constructor)
# define LIBXSMM_DTOR_ATTRIBUTE LIBXSMM_ATTRIBUTE(destructor)
# define LIBXSMM_CTOR
#else /* lazy initialization */
# define LIBXSMM_INIT libxsmm_init();
# define LIBXSMM_CTOR_ATTRIBUTE
# define LIBXSMM_DTOR_ATTRIBUTE
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


typedef union LIBXSMM_RETARGETABLE libxsmm_code_pointer {
  /*const*/void* pmm;
  uintptr_t imm;
#if defined(LIBXSMM_BUILD) || defined(LIBXSMM_DNN_INTERNAL_API)
  libxsmm_xconvfunction xconv;
#endif
  libxsmm_xmmfunction xmm;
} libxsmm_code_pointer;

typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_csr_soa_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxsmm_csr_soa_descriptor;

/** Structure which describes an activation layer. */
struct LIBXSMM_RETARGETABLE libxsmm_dnn_buffer {
  int N;                            /* number of images in mini-batch */
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int H;                            /* height of image */
  int W;                            /* width of image */
  int lpb;                          /* low precision blocking factor */
  libxsmm_dnn_conv_format format;   /* format of activation buffer */
  libxsmm_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
};

/** Structure which describes a bias. */
struct LIBXSMM_RETARGETABLE libxsmm_dnn_bias {
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int lpb;                          /* low precision blocking factor */
  libxsmm_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
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
  libxsmm_dnn_conv_format format;   /* format of filter buffer */
  libxsmm_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
};

struct LIBXSMM_RETARGETABLE libxsmm_dnn_conv_handle {
  libxsmm_dnn_datatype datatype_in;
  libxsmm_dnn_datatype datatype_out;
  libxsmm_dnn_conv_desc desc;
  libxsmm_dnn_conv_algo algo;
  libxsmm_dnn_conv_format buffer_format;
  libxsmm_dnn_conv_format filter_format;
  libxsmm_dnn_conv_fuse_op fuse_ops;
  libxsmm_dnn_conv_option options;

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

  /* internal data representation */
  libxsmm_dnn_buffer* input;
  libxsmm_dnn_buffer* output;
  libxsmm_dnn_buffer* input_relu;
  libxsmm_dnn_filter* filter;
  libxsmm_dnn_bias* bias;
  void* scratch1;
  void* scratch2;
/*#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM*/
  void* scratch3;
/*#endif*/
  void* scratch4;

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
};

typedef enum libxsmm_build_kind {
  LIBXSMM_BUILD_KIND_GEMM,
  LIBXSMM_BUILD_KIND_SSOA,
  LIBXSMM_BUILD_KIND_CFWD,
  LIBXSMM_BUILD_KIND_CBWD,
  LIBXSMM_BUILD_KIND_CUPD
} libxsmm_build_kind;

typedef union LIBXSMM_RETARGETABLE libxsmm_build_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const libxsmm_csr_soa_descriptor* ssoa;
  const libxsmm_convolution_forward_descriptor* cfwd;
  const libxsmm_convolution_backward_descriptor* cbwd;
  const libxsmm_convolution_weight_update_descriptor* cupd;
} libxsmm_build_descriptor;

typedef struct LIBXSMM_RETARGETABLE libxsmm_build_request {
  libxsmm_build_descriptor descriptor;
  libxsmm_build_kind kind;
} libxsmm_build_request;

typedef enum libxsmm_malloc_flags {
  LIBXSMM_MALLOC_FLAG_R = 1,
  LIBXSMM_MALLOC_FLAG_W = 2,
  LIBXSMM_MALLOC_FLAG_X = 4,
  LIBXSMM_MALLOC_FLAG_MMAP = 8,
  LIBXSMM_MALLOC_FLAG_RW  = LIBXSMM_MALLOC_FLAG_R | LIBXSMM_MALLOC_FLAG_W,
  LIBXSMM_MALLOC_FLAG_RWX = LIBXSMM_MALLOC_FLAG_RW | LIBXSMM_MALLOC_FLAG_X,
  /** LIBXSMM_MALLOC_FLAG_DEFAULT is an alias for setting no flag bits. */
  LIBXSMM_MALLOC_FLAG_DEFAULT = LIBXSMM_MALLOC_FLAG_RW
} libxsmm_malloc_flags;

/** Greatest common divisor. */
LIBXSMM_API size_t libxsmm_gcd(size_t a, size_t b);
/** Least common multiple. */
LIBXSMM_API size_t libxsmm_lcm(size_t a, size_t b);
/** Calculates an alignment depending on supposedly allocated size; alignment can be zero ("auto"). */
LIBXSMM_API size_t libxsmm_alignment(size_t size, size_t alignment);

/** Receive the size, the flags, or the extra attachment of the given buffer. */
LIBXSMM_API int libxsmm_malloc_info(const volatile void* memory, size_t* size, int* flags, void** extra);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXSMM_API int libxsmm_xmalloc(void** memory, size_t size, int alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
LIBXSMM_API int libxsmm_xfree(const volatile void* memory);

/** Attribute memory allocation and protect with only the necessary flags. */
LIBXSMM_API int libxsmm_malloc_attrib(void** memory, int flags,
  /** If a name is given, an executable buffer will be dumped into a file. */
  const char* name);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXSMM_REGSIZE for unmanaged code). */
LIBXSMM_API void libxsmm_build(const libxsmm_build_request* request, unsigned regindex, libxsmm_code_pointer* code);

/** Updates counters of the statistic, which is shown at program termination. */
LIBXSMM_API unsigned int libxsmm_update_mmstatistic(int flags, int m, int n, int k, unsigned int ntry, unsigned int ncol);

LIBXSMM_API int libxsmm_gemm_prefetch2uid(int prefetch);
LIBXSMM_API int libxsmm_gemm_uid2prefetch(int uid);

LIBXSMM_API size_t libxsmm_dnn_typesize(libxsmm_dnn_datatype datatype);

/** Stores the verbosity level (libxsmm_get_verbosity, libxsmm_set_verbosity). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_verbosity;
/** Target architecture (libxsmm_get_target_archid, libxsmm_set_target_archid). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_target_archid;
/** Determines the prefetch strategy, which is used in case of LIBXSMM_PREFETCH_AUTO. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_gemm_auto_prefetch;
/** Determines if (OpenMP-)tasks are preferred over thread-style parallelization. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_tasks;
/** Kind of parallel support (0: none, 1: sequential, 2: parallelized). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_mt;
/** Number of threads per core. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_nt;

#endif /*LIBXSMM_MAIN_H*/

