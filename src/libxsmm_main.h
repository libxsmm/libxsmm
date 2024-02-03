/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_MAIN_H
#define LIBXSMM_MAIN_H

#include <libxsmm.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXSMM_CAPACITY_REGISTRY) /* must be POT */
# define LIBXSMM_CAPACITY_REGISTRY 131072
#endif
#if !defined(LIBXSMM_CAPACITY_CACHE) /* must be POT */
# define LIBXSMM_CAPACITY_CACHE 16
#endif

#if !defined(LIBXSMM_PAGE_MINSIZE)
# if defined(LIBXSMM_PLATFORM_X86)
#   define LIBXSMM_PAGE_MINSIZE 4096 /* 4 KB */
# elif defined(__APPLE__)
#   define LIBXSMM_PAGE_MINSIZE 16384 /* 16 KB */
# else
#   define LIBXSMM_PAGE_MINSIZE 4096 /* 4 KB */
# endif
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
#if !defined(LIBXSMM_MALLOC_MMAP_SCRATCH) && 0
# define LIBXSMM_MALLOC_MMAP_SCRATCH
#endif
/* align if interceptor is disabled (moderated malloc) */
#if defined(LIBXSMM_MALLOC_MOD) && 0
# define LIBXSMM_MALLOC_MOD
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
    (defined(LIBXSMM_MALLOC) && (0 != LIBXSMM_MALLOC)) && \
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
    (defined(LIBXSMM_MALLOC_MOD) || (defined(LIBXSMM_MALLOC) && (0 != LIBXSMM_MALLOC)))
# define LIBXSMM_DNN_CONVOLUTION_SETUP_USE_NTS
#endif

#if defined(LIBXSMM_INTERCEPT_DYNAMIC)
# include <dlfcn.h>
# if !defined(RTLD_NEXT)
#   define LIBXSMM_RTLD_NEXT ((void*)-1l)
# else
#   define LIBXSMM_RTLD_NEXT RTLD_NEXT
# endif
#endif

#if defined(LIBXSMM_PLATFORM_AARCH64)
# if defined(_MSC_VER)
#   define LIBXSMM_ARM_ENC16(OP0, OP1, CRN, CRM, OP2) ( \
      (((OP0) & 1) << 14) | \
      (((OP1) & 7) << 11) | \
      (((CRN) & 15) << 7) | \
      (((CRM) & 15) << 3) | \
      (((OP2) & 7) << 0))
#   define ID_AA64ISAR1_EL1 LIBXSMM_ARM_ENC16(0b11, 0b000, 0b0000, 0b0110, 0b001)
#   define ID_AA64PFR0_EL1  LIBXSMM_ARM_ENC16(0b11, 0b000, 0b0000, 0b0100, 0b000)
#   define MIDR_EL1         LIBXSMM_ARM_ENC16(0b11, 0b000, 0b0000, 0b0000, 0b000)
#   define LIBXSMM_ARM_MRS(RESULT, ID) RESULT = _ReadStatusReg(ID)
# else
#   define LIBXSMM_ARM_MRS(RESULT, ID) __asm__ __volatile__( \
      "mrs %0," LIBXSMM_STRINGIFY(ID) : "=r"(RESULT))
# endif
#endif

#if defined(__powerpc64__)
# define LIBXSMM_TIMER_RDTSC(CYCLE) do { \
    CYCLE = __ppc_get_timebase(); \
  } while(0)
#elif ((defined(LIBXSMM_PLATFORM_X86) && (64 <= (LIBXSMM_BITS))) && \
      (defined(__GNUC__) || defined(LIBXSMM_INTEL_COMPILER) || defined(__PGI)))
# define LIBXSMM_TIMER_RDTSC(CYCLE) do { \
    libxsmm_timer_tickint libxsmm_timer_rdtsc_hi_; \
    __asm__ __volatile__ ("rdtsc" : "=a"(CYCLE), "=d"(libxsmm_timer_rdtsc_hi_)); \
    CYCLE |= libxsmm_timer_rdtsc_hi_ << 32; \
  } while(0)
#elif (defined(_rdtsc) || defined(_WIN32)) && defined(LIBXSMM_PLATFORM_X86)
# define LIBXSMM_TIMER_RDTSC(CYCLE) (CYCLE = __rdtsc())
#elif defined(LIBXSMM_PLATFORM_AARCH64) && 1
# if defined(ARM64_CNTVCT) /* Windows */
#   define LIBXSMM_TIMER_RDTSC(CYCLE) LIBXSMM_ARM_MRS(CYCLE, ARM64_CNTVCT)
# else
#   define LIBXSMM_TIMER_RDTSC(CYCLE) LIBXSMM_ARM_MRS(CYCLE, CNTVCT_EL0)
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

#define LIBXSMM_DESCRIPTOR_CLEAR_AUX(DST, SIZE, FLAGS) LIBXSMM_MEMSET127(DST, 0, SIZE)
#define LIBXSMM_DESCRIPTOR_CLEAR(BLOB) \
  LIBXSMM_ASSERT((LIBXSMM_DESCRIPTOR_MAXSIZE) == sizeof(*(BLOB))); \
  LIBXSMM_DESCRIPTOR_CLEAR_AUX(BLOB, LIBXSMM_DESCRIPTOR_MAXSIZE, 0)

/** Low-level/internal GEMM descriptor initialization. */
#define LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE0, DATA_TYPE1, DATA_TYPE2, FLAGS, M, N, K, LDA, LDB, LDC, PREFETCH) \
  LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K); LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(LDA, LDB, LDC); \
  LIBXSMM_DESCRIPTOR_CLEAR_AUX(&(DESCRIPTOR), sizeof(DESCRIPTOR), FLAGS); \
  (DESCRIPTOR).datatype[0] = (unsigned char)(DATA_TYPE0); (DESCRIPTOR).datatype[1] = (unsigned char)(DATA_TYPE1); \
  (DESCRIPTOR).datatype[2] = (unsigned char)(DATA_TYPE2); (DESCRIPTOR).prefetch = (unsigned char)(PREFETCH); \
  (DESCRIPTOR).flags = (unsigned int)(FLAGS); \
  (DESCRIPTOR).m   = (unsigned int)(M);   (DESCRIPTOR).n   = (unsigned int)(N);   (DESCRIPTOR).k   = (unsigned int)(K); \
  (DESCRIPTOR).lda = (unsigned int)(LDA); (DESCRIPTOR).ldb = (unsigned int)(LDB); (DESCRIPTOR).ldc = (unsigned int)(LDC)

/** Declare and construct a GEMM descriptor. */
#define LIBXSMM_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, DATA_TYPE0, DATA_TYPE1, DATA_TYPE2, FLAGS, M, N, K, LDA, LDB, LDC, PREFETCH) \
  libxsmm_gemm_descriptor DESCRIPTOR; LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE0, DATA_TYPE1, DATA_TYPE2 \
    FLAGS, M, N, K, LDA, LDB, LDC, PREFETCH)

#define LIBXSMM_REGDESC_DEFAULT
#define LIBXSMM_REGDESC(START, MODIFIER) \
  START libxsmm_gemm_descriptor MODIFIER gemm; \
  START libxsmm_meltw_descriptor MODIFIER meltw; \
  START libxsmm_meqn_descriptor MODIFIER meqn

/**
* Packed structure, which stores the argument description of GEMM routines.
* The size of the structure is padded to LIBXSMM_DESCRIPTOR_MAXSIZE.
*/
LIBXSMM_EXTERN_C LIBXSMM_PACKED(struct) libxsmm_gemm_descriptor {
  /** Extents of the matrix. */
  unsigned int m, n, k;
  /** Leading dimensions. */
  unsigned int lda, ldb, ldc;
  /** Set of flags. */
  unsigned int flags;
  /** Prefetch strategy. */
  unsigned char prefetch;
  /** Denotes the data-type. */
  unsigned char datatype[3];
  /**
   * Do not reorder elements between above and below blocks!
   */
  /** Denotes of optional eltwise data-type */
  unsigned char meltw_datatype_aux;
  /** multipurpose 64-bit field, currently used for: a) stride_a in brgemm */
  long long c1;
  /** multipurpose 64-bit field, currently used for: a) stride_b in brgemm */
  long long c2;
  /** multipurpose 8-bit field, currently used for: a) unroll hint in brgemm */
  unsigned char c3;
  /** LDx, LDy, LDz,  additional meltw LDs */
  unsigned int meltw_ldx, meltw_ldy, meltw_ldz;
  /** optional param field */
  unsigned short meltw_param;
  /** Set of flags */
  unsigned short meltw_flags;
  /** operation specifier */
  unsigned char meltw_operation;
  /* Ap, Bp, Cp */
  unsigned char eltw_ap_op;
  unsigned char eltw_bp_op;
  unsigned char eltw_cp_op;
  unsigned short eltw_ap_flags;
  unsigned short eltw_bp_flags;
  unsigned short eltw_cp_flags;
  unsigned short eltw_ap_param;
  unsigned short eltw_bp_param;
  unsigned short eltw_cp_param;
  unsigned int ldap;
  unsigned int ldbp;
  unsigned int ldcp;
  /* internal flags2 */
  unsigned char internal_flags_2;
};

/** Packed structure storing the mateltw argument description. */
LIBXSMM_EXTERN_C LIBXSMM_PACKED(struct) libxsmm_meltw_descriptor {
  /** LDx, M, and N. */
  unsigned int m, n, ldi, ldo, ldi2, ldi3;
  /** Size of data element. */
  unsigned char datatype;
  unsigned char datatype1;
  unsigned char datatype2;
  /** Set of flags */
  unsigned short flags;
  /** optional param field */
  unsigned short param;
  /** operation specifier */
  unsigned char operation;
};

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_pspgemm_csr_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
  unsigned int packed_width;
} libxsmm_pspgemm_csr_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_pspgemm_csc_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* column_ptr;
  const unsigned int* row_idx;
  const void* values;
  unsigned int packed_width;
} libxsmm_pspgemm_csc_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_pspgemm_bcsc_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  unsigned int packed_width;
  unsigned int bk;
  unsigned int bn;
} libxsmm_pspgemm_bcsc_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_pgemm_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  unsigned int packed_width;
} libxsmm_pgemm_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_pgemm_ac_rm_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  unsigned int packed_width;
} libxsmm_pgemm_ac_rm_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_pgemm_bc_rm_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  unsigned int packed_width;
} libxsmm_pgemm_bc_rm_descriptor;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_csr_reg_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxsmm_csr_reg_descriptor;

LIBXSMM_EXTERN_C typedef union libxsmm_xcopykernel {
  libxsmm_meltwfunction_unary function;
  const void *ptr_const, *ptr;
} libxsmm_xcopykernel;

LIBXSMM_EXTERN_C typedef union libxsmm_code_pointer {
  /*void (*ptr_fn)(const void*, ...);*/
  const void* ptr_const;
  void* ptr;
  uintptr_t uval;
  intptr_t ival;
  libxsmm_xmmfunction xgemm; /* GEMM: smm, dmm, wimm, or void-function */
  libxsmm_xmeltwfunction xmateltw;
  libxsmm_meqn_function xmateqn;
} libxsmm_code_pointer;

struct libxsmm_fsspmdm {
  int M, N, K, ldb, ldc, N_chunksize;
  libxsmm_gemmfunction kernel;
  libxsmm_datatype datatype;
  void* a_dense;
};

/** Packed structure storing the mateltw argument description. */
LIBXSMM_EXTERN_C LIBXSMM_PACKED(struct) libxsmm_meqn_descriptor {
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
  LIBXSMM_BUILD_KIND_PGEMM      = LIBXSMM_KERNEL_UNREGISTERED,
  LIBXSMM_BUILD_KIND_PGEMMRMAC,
  LIBXSMM_BUILD_KIND_PGEMMRMBC,
  LIBXSMM_BUILD_KIND_PSPGEMM_CSR,
  LIBXSMM_BUILD_KIND_PSPGEMM_CSC,
  LIBXSMM_BUILD_KIND_PSPGEMM_BCSC,
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
# define LIBXSMM_DESCRIPTOR_ISBIG(KIND) ((unsigned char)((KIND) >> 7))
# define LIBXSMM_DESCRIPTOR_KIND(KIND) ((unsigned char)((KIND) & 0x7F))
typedef unsigned char libxsmm_descriptor_kind;
#endif

/** All descriptor types, which are valid for code-registration. */
LIBXSMM_EXTERN_C typedef union libxsmm_descriptor {
  unsigned char data[LIBXSMM_DESCRIPTOR_MAXSIZE];
  libxsmm_descriptor_kind kind; /* kind: must be the first member after "data" entry (above) */
  LIBXSMM_REGDESC(LIBXSMM_PACKED(struct) { libxsmm_descriptor_kind /*repeated kind*/ pad; , desc; });
  LIBXSMM_PACKED(struct) { libxsmm_descriptor_kind /*repeated kind*/ pad; unsigned char size; unsigned char desc[1]; } user;
} libxsmm_descriptor;

LIBXSMM_EXTERN_C typedef struct libxsmm_build_request {
  union {
    const void *ptr_const, *ptr; /* raw content */
    LIBXSMM_REGDESC(LIBXSMM_REGDESC_DEFAULT, const*);
    const libxsmm_pspgemm_csr_descriptor* pspgemm_csr;
    const libxsmm_pspgemm_csc_descriptor* pspgemm_csc;
    const libxsmm_pspgemm_bcsc_descriptor* pspgemm_bcsc;
    const libxsmm_pgemm_descriptor* pgemm;
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

LIBXSMM_EXTERN_C typedef void* (*libxsmm_realloc_fun)(void* /*ptr*/, size_t /*size*/);

#if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)
LIBXSMM_EXTERN_C typedef struct libxsmm_malloc_fntype {
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
LIBXSMM_EXTERN_C void* __libc_memalign(size_t alignment, size_t size);
LIBXSMM_EXTERN_C void* __libc_malloc(size_t size);
#if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
LIBXSMM_EXTERN_C void* __libc_calloc(size_t num, size_t size);
#endif
#if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
LIBXSMM_EXTERN_C void* __libc_realloc(void* ptr, size_t size);
#endif
LIBXSMM_EXTERN_C void  __libc_free(void* ptr);
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
 * Attribute memory allocation and protect with only the necessary flags (revoke other flags).
 * This procedure is not suitable for executable buffers, profiler support, etc.
 */
LIBXSMM_API_INTERN int libxsmm_malloc_xattrib(void* buffer, int flags, size_t size);

/**
 * Attribute memory allocation and protect with only the necessary flags.
 * This procedure is expected to run only one time per buffer, and may
 * relocate the given memory.
 */
LIBXSMM_API_INTERN int libxsmm_malloc_attrib(void** memory, int flags,
  /** If name is given, profiler support, and code dump (verbose mode) are supported. */
  const char* name,
  /** If data_size if given, amount of memory-attribution is lowered by data_size. */
  const size_t* data_size);

/** Like libxsmm_release_scratch, but takes a lock (can be NULL). */
LIBXSMM_API_INTERN void libxsmm_xrelease_scratch(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXSMM_API int libxsmm_xmalloc(void** memory, size_t size, size_t alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
/** Release memory, which was allocated using libxsmm_[*]malloc. */
LIBXSMM_API void libxsmm_xfree(const void* memory, int check);

/** Determines the given value in double-precision (EXIT_SUCCESS if value is NULL). */
LIBXSMM_API int libxsmm_dvalue(libxsmm_datatype datatype, const void* value, double* dvalue);

/**
 * Format for instance an amount of Bytes like libxsmm_format_value(result, sizeof(result), nbytes, "KMGT", "B", 10).
 * The value returned is in requested/determined unit so that the user can decide about printing the buffer.
 */
LIBXSMM_API_INTERN size_t libxsmm_format_value(char buffer[32],
  int buffer_size, size_t nbytes, const char scale[], const char* unit, int base);

/**
 * Print the command line arguments of the current process, and get the number of written
 * characters including the prefix, the postfix, but not the terminating NULL character.
 * If zero is returned, nothing was printed (no prefix, no postfix).
 * If buffer_size is zero, buffer is assumed to be a FILE-pointer.
 */
LIBXSMM_API_INTERN int libxsmm_print_cmdline(void* buffer, size_t buffer_size, const char* prefix, const char* postfix);

/**
 * Dump data, (optionally) check attempt to dump different data into an existing file (unique),
 * or (optionally) permit overwriting an existing file.
 */
LIBXSMM_API_INTERN int libxsmm_dump(const char* title, const char* name, const void* data, size_t size, int unique, int overwrite);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXSMM_CAPACITY_REGISTRY for unmanaged code). */
LIBXSMM_API_INTERN int libxsmm_build(const libxsmm_build_request* request, unsigned int regindex, libxsmm_code_pointer* code);

/** Determines CPU-name using OS-specific instead of CPU-specific interfaces. */
LIBXSMM_API_INTERN void libxsmm_cpuid_model(char model[], size_t* model_size);

LIBXSMM_EXTERN_C typedef struct libxsmm_kernel_xinfo {
  /** Non-zero if kernel is registered. */
  unsigned int registered;
  /** Number of FLoating Point OPerationS (FLOPS). */
  unsigned int nflops;
} libxsmm_kernel_xinfo;

/** Receive information about JIT-generated code. */
LIBXSMM_API_INTERN const libxsmm_kernel_xinfo* libxsmm_get_kernel_xinfo(libxsmm_code_pointer code, const libxsmm_descriptor** desc, size_t* code_size);

/** Calculates duration in seconds from given RTC ticks. */
LIBXSMM_API double libxsmm_timer_duration_rtc(libxsmm_timer_tickint tick0, libxsmm_timer_tickint tick1);
/** Returns the current tick of platform-specific real-time clock. */
LIBXSMM_API libxsmm_timer_tickint libxsmm_timer_tick_rtc(void);
/** Returns the current tick of a (monotonic) platform-specific counter. */
LIBXSMM_API libxsmm_timer_tickint libxsmm_timer_tick_tsc(void);

LIBXSMM_API_INTERN void libxsmm_memory_init(int target_arch);
LIBXSMM_API_INTERN void libxsmm_memory_finalize(void);

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
#endif /*LIBXSMM_MAIN_H*/
