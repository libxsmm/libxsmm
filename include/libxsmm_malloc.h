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
#ifndef LIBXSMM_MALLOC_H
#define LIBXSMM_MALLOC_H

#include "libxsmm_macros.h"

/** Include <tensorflow/core/public/version.h> prior to LIBXSMM otherwise the current TensorFlow API is assumed. */
#if !defined(LIBXSMM_TF12) && (!defined(TF_VERSION_STRING) || \
  LIBXSMM_VERSION2(1, 12) <= LIBXSMM_VERSION2(TF_MAJOR_VERSION, TF_MINOR_VERSION))
# define LIBXSMM_TF12 /* TF_PATCH_VERSION does not matter */
#endif

/** Can be used with libxsmm_[get|set]_scratch_limit. */
#define LIBXSMM_SCRATCH_UNLIMITED ((size_t)LIBXSMM_UNLIMITED)
#define LIBXSMM_SCRATCH_DEFAULT 0


/** Function types accepted for memory allocation (see libxsmm_*_allocator). */
LIBXSMM_EXTERN_C typedef void* (*libxsmm_malloc_ctx)(size_t /*size*/, const void* /*context*/);
LIBXSMM_EXTERN_C typedef void* (*libxsmm_malloc_fun)(size_t /*size*/);
LIBXSMM_EXTERN_C typedef union libxsmm_malloc_function {
  libxsmm_malloc_ctx ctx_form;
  libxsmm_malloc_fun function;
} libxsmm_malloc_function;

/** Function types accepted for releasing memory (see libxsmm_*_allocator). */
LIBXSMM_EXTERN_C typedef void (*libxsmm_free_ctx)(void* /*buffer*/, const void* /*context*/);
LIBXSMM_EXTERN_C typedef void (*libxsmm_free_fun)(void* /*buffer*/);
LIBXSMM_EXTERN_C typedef union libxsmm_free_function {
  libxsmm_free_ctx ctx_form;
  libxsmm_free_fun function;
} libxsmm_free_function;

/**
 * To setup the custom default memory allocator, either a malloc_fn and a free_fn
 * are given, or two NULL-pointers designate to reset the default allocator to a
 * library-internal default. If a context is given (non-NULL), the context-based
 * form of the memory allocation is used.
 * Changing the allocator including the function for deallocation applies to
 * upcoming allocation/deallocation and works correctly for pending buffers.
 */
LIBXSMM_API int libxsmm_set_default_allocator(/* malloc_fn/free_fn must correspond */
  const void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Retrieve the default memory allocator. */
LIBXSMM_API int libxsmm_get_default_allocator(const void** context,
  libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/**
 * To setup the scratch memory allocator, a malloc_fn function and an optional free_fn
 * are given. A NULL-free acts as a "no-operation", and the deallocation is expected
 * to be controlled otherwise. If two NULL-pointers are given, the allocator is reset
 * to the currently active default memory allocator. If a context is given (non-NULL),
 * the context-based form of the memory allocation is used.
 * Changing the allocator including the function for deallocation applies to
 * upcoming allocation/deallocation and works correctly for pending buffers.
 */
LIBXSMM_API int libxsmm_set_scratch_allocator(/* malloc_fn/free_fn must correspond */
  const void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Retrieve the scratch memory allocator. */
LIBXSMM_API int libxsmm_get_scratch_allocator(const void** context,
  libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/** Allocate memory (malloc/free interface). */
LIBXSMM_API LIBXSMM_ATTRIBUTE_MALLOC void* libxsmm_malloc(size_t size);

/** Allocate aligned memory using the default allocator. */
LIBXSMM_API LIBXSMM_ATTRIBUTE_MALLOC void* libxsmm_aligned_malloc(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment);

/** Reallocate memory using the default allocator (alignment is preserved). */
LIBXSMM_API void* libxsmm_realloc(size_t size, void* ptr);

/**
 * Allocate aligned scratch memory. It is not supported
 * to query properties per libxsmm_get_malloc_info, but
 * libxsmm_get_scratch_info can used instead.
 */
LIBXSMM_API void* libxsmm_scratch_malloc(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment,
  /**
   * Identifies the call site, which is used
   * to determine the memory pool.
   */
  const void* caller);

/**
 * Binary form of libxsmm_scratch_malloc, which
 * expands the call-context automatically. This
 * macro is intentionally lower case.
 */
#define libxsmm_aligned_scratch(size, alignment) \
  libxsmm_scratch_malloc(size, alignment, \
    LIBXSMM_CALLER_ID)

/** Deallocate memory (malloc/free interface). */
LIBXSMM_API void libxsmm_free(const void* memory);

/**
 * Initialize the pool by drawing from the given storage a number of chunks of the given size.
 * If the capacity of the pool is num, the storage must be at least num x size.
 * The same num-counter must be used for pmalloc/pfree when referring to the same pool.
 */
LIBXSMM_API void libxsmm_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage);
/** Allocate from the given pool by using the original num-counter (libxsmm_pmalloc_init). */
LIBXSMM_API void* libxsmm_pmalloc(void* pool[], size_t* i);
/** Bring pointer back into the pool by using original num-counter (libxsmm_pmalloc_init). */
LIBXSMM_API void libxsmm_pfree(void* pointer, void* pool[], size_t* i);

/**
 * Release the entire scratch memory regardless
 * of whether it is still referenced or not.
 */
LIBXSMM_API void libxsmm_release_scratch(void);

/** Information about a buffer (default memory domain). */
LIBXSMM_EXTERN_C typedef struct libxsmm_malloc_info {
  /** Size of the buffer. */
  size_t size;
} libxsmm_malloc_info;

/** Retrieve information about a buffer (default memory domain). */
LIBXSMM_API int libxsmm_get_malloc_info(const void* memory, libxsmm_malloc_info* info);

/** Information about the scratch memory domain. */
LIBXSMM_EXTERN_C typedef struct libxsmm_scratch_info {
  /** Watermark memory across pools (size), unsatisfied (local), and library-internal memory. */
  size_t size, local, internal;
  /** Pending allocations (not released). */
  size_t npending;
  /** Number of allocations so far. */
  size_t nmallocs;
  /** Number of pools used. */
  unsigned int npools;
} libxsmm_scratch_info;

/** Retrieve information about the scratch memory domain. */
LIBXSMM_API int libxsmm_get_scratch_info(libxsmm_scratch_info* info);

/**
 * Limit the total size (Bytes) of the scratch memory.
 * LIBXSMM_SCRATCH_UNLIMITED removes any limit, and
 * LIBXSMM_SCRATCH_DEFAULT populates the default.
 * The related environment variable LIBXSMM_SCRATCH_LIMIT
 * allows units: <none>/b/B (Bytes), k/K, m/M, and g/G.
 */
LIBXSMM_API void libxsmm_set_scratch_limit(size_t nbytes);
/** Get the maximum size of the scratch memory domain. */
LIBXSMM_API size_t libxsmm_get_scratch_limit(void);

/**
 * Intercepts malloc/free to use scratch memory allocator.
 * (related environment variable LIBXSMM_MALLOC).
 * Optionally set the range of malloc-sizes to be intercepted.
 * The related environment variable LIBXSMM_MALLOC_LIMIT
 * allows units: <none>/b/B (Bytes), k/K, m/M, and g/G.
 */
LIBXSMM_API void libxsmm_set_malloc(int enabled, const size_t* lo, const size_t* hi);
/**
 * Determines if malloc/free are (and can be) intercepted.
 * Optionally gets the range of enabled malloc-sizes.
 */
LIBXSMM_API int libxsmm_get_malloc(size_t* lo, size_t* hi);


#if defined(__cplusplus)

/** RAII idiom to temporarily setup an allocator for the lifetime of the scope. */
template<typename kind> class libxsmm_scoped_allocator {
public:
  /** C'tor, which instantiates the new allocator (plain form). */
  libxsmm_scoped_allocator(libxsmm_malloc_fun malloc_fn, libxsmm_free_fun free_fn) {
    kind::get(m_context, m_malloc, m_free);
    kind::set(NULL/*context*/, NULL/*malloc_ctx*/, NULL/*free_ctx*/, malloc_fn, free_fn);
  }

  /** C'tor, which instantiates the new allocator (context form). */
  libxsmm_scoped_allocator(const void* context, libxsmm_malloc_ctx malloc_ctx, libxsmm_free_ctx free_ctx,
    libxsmm_malloc_fun malloc_fun = NULL, libxsmm_free_fun free_fun = NULL)
  {
    kind::get(m_context, m_malloc, m_free);
    kind::set(context, malloc_ctx, free_ctx, malloc_fun, free_fun);
  }

  /** Following the RAII idiom, the d'tor restores the previous allocator. */
  ~libxsmm_scoped_allocator() {
    kind::set(m_context,
      m_malloc.ctx_form, m_free.ctx_form,
      m_malloc.function, m_free.function);
  }

private: /* no copy/assignment */
  explicit libxsmm_scoped_allocator(const libxsmm_scoped_allocator&);
  libxsmm_scoped_allocator& operator=(const libxsmm_scoped_allocator&);

protected: /* saved/previous allocator */
  const void* m_context;
  libxsmm_malloc_function m_malloc;
  libxsmm_free_function m_free;
};

/** Allocator-kind to instantiate libxsmm_scoped_allocator<kind>. */
struct libxsmm_default_allocator {
  static void set(const void* context,
    libxsmm_malloc_ctx malloc_ctx, libxsmm_free_ctx free_ctx,
    libxsmm_malloc_fun malloc_fun, libxsmm_free_fun free_fun)
  {
    libxsmm_malloc_function malloc_fn;
    libxsmm_free_function free_fn;
    if (NULL == context) { /* use global form only when no context is given */
      malloc_fn.function = malloc_fun; free_fn.function = free_fun;
    }
    else {
      malloc_fn.ctx_form = malloc_ctx; free_fn.ctx_form = free_ctx;
    }
    libxsmm_set_default_allocator(context, malloc_fn, free_fn);
  }
  static void get(const void*& context,
    libxsmm_malloc_function& malloc_fn, libxsmm_free_function& free_fn)
  {
    libxsmm_get_default_allocator(&context, &malloc_fn, &free_fn);
  }
};

/** Allocator-kind to instantiate libxsmm_scoped_allocator<kind>. */
struct libxsmm_scratch_allocator {
  static void set(const void* context,
    libxsmm_malloc_ctx malloc_ctx, libxsmm_free_ctx free_ctx,
    libxsmm_malloc_fun malloc_fun, libxsmm_free_fun free_fun)
  {
    libxsmm_malloc_function malloc_fn;
    libxsmm_free_function free_fn;
    if (NULL != context) { /* adopt context form */
      malloc_fn.function = malloc_fun; free_fn.function = free_fun;
    }
    else { /* adopt global form */
      malloc_fn.ctx_form = malloc_ctx; free_fn.ctx_form = free_ctx;
    }
    libxsmm_set_scratch_allocator(context, malloc_fn, free_fn);
  }
  static void get(const void*& context,
    libxsmm_malloc_function& malloc_fn, libxsmm_free_function& free_fn)
  {
    libxsmm_get_scratch_allocator(&context, &malloc_fn, &free_fn);
  }
};

/** Forward-declared types/functions used to implement libxsmm_tf_allocator. */
namespace tensorflow {
  class Allocator;
#if defined(LIBXSMM_TF12)
  class DeviceBase; int DeviceNumaNode(const DeviceBase* /*device*/);
  Allocator* cpu_allocator(int /*numa_node*/);
#else
  Allocator* cpu_allocator();
#endif
}

/**
 * An object of this type adopts a memory allocator from TensorFlow.
 * All memory allocations of the requested kind within the current
 * scope (where the libxsmm_tf_allocator object lives) are subject
 * to TensorFlow's memory allocation scheme. The allocation kind
 * is usually "libxsmm_scratch_allocator"; using a second object
 * of kind "libxsmm_default_allocator" makes the default memory
 * allocation of LIBXSMM subject to TensorFlow as well.
 */
template<typename kind> class libxsmm_tf_allocator:
  public libxsmm_scoped_allocator<kind>
{
public:
  /** The TensorFlow allocator is adopted from the global CPU memory allocator. */
  explicit libxsmm_tf_allocator()
    : libxsmm_scoped_allocator<kind>(
      libxsmm_tf_allocator::malloc,
      libxsmm_tf_allocator::free)
  {}

  /** The TensorFlow allocator is adopted from the given OpKernelContext. */
  template<typename context_type>
  explicit libxsmm_tf_allocator(context_type& context)
    : libxsmm_scoped_allocator<kind>(&context,
      libxsmm_tf_allocator::template malloc_ctx<context_type>,
      libxsmm_tf_allocator::template free_ctx<context_type>,
      libxsmm_tf_allocator::malloc,
      libxsmm_tf_allocator::free)
  {}

  /** Global form of allocating memory (malloc signature). */
  static void* malloc(size_t size) {
#if defined(LIBXSMM_TF12)
    return libxsmm_tf_allocator::allocate(tensorflow::cpu_allocator(-1/*kNUMANoAffinity*/), size);
#else
    return libxsmm_tf_allocator::allocate(tensorflow::cpu_allocator(), size);
#endif
  }

  /** Global form of deallocating memory (free signature). */
  static void free(void* buffer) {
#if defined(LIBXSMM_TF12)
    libxsmm_tf_allocator::deallocate(tensorflow::cpu_allocator(-1/*kNUMANoAffinity*/), buffer);
#else
    libxsmm_tf_allocator::deallocate(tensorflow::cpu_allocator(), buffer);
#endif
  }

  /** Context based form of allocating memory. */
  template<typename context_type> static void* malloc_ctx(const void* context, size_t size) {
    typedef typename context_type::WrappedAllocator::first_type allocator_ptr;
    context_type *const tf_context = static_cast<context_type*>(context);
    allocator_ptr allocator = NULL;
    if (NULL != tf_context) {
#if !defined(LIBXSMM_TF12)
      if (NULL != tf_context->device()) {
        if (0 < tf_context->num_outputs()) {
          allocator = tf_context->device()->GetStepAllocator(
            tf_context->output_alloc_attr(0),
            tf_context->resource_manager());
        }
        else if (0 < tf_context->num_inputs()) {
          allocator = tf_context->device()->GetStepAllocator(
            tf_context->input_alloc_attr(0),
            tf_context->resource_manager());
        }
      }
#else /* include tensorflow/core/public/version.h prior to LIBXSMM otherwise the current TensorFlow API is assumed */
      const int numa_node = DeviceNumaNode(tf_context->device());
      allocator = tensorflow::cpu_allocator(numa_node);
#endif
    }
    return libxsmm_tf_allocator::allocate(allocator, size);
  }

  /** Context based form of deallocating memory. */
  template<typename context_type> static void free_ctx(const void* context, void* buffer) {
    typedef typename context_type::WrappedAllocator::first_type allocator_ptr;
    context_type *const tf_context = static_cast<context_type*>(context);
    allocator_ptr allocator = NULL;
    if (NULL != tf_context) {
#if defined(LIBXSMM_TF12)
      const int numa_node = DeviceNumaNode(tf_context->device());
      allocator = tensorflow::cpu_allocator(numa_node);
#else
      if (NULL != tf_context->device()) {
        if (0 < tf_context->num_outputs()) {
          allocator = tf_context->device()->GetStepAllocator(
            tf_context->output_alloc_attr(0),
            tf_context->resource_manager());
        }
        else if (0 < tf_context->num_inputs()) {
          allocator = tf_context->device()->GetStepAllocator(
            tf_context->input_alloc_attr(0),
            tf_context->resource_manager());
        }
      }
#endif
    }
    libxsmm_tf_allocator::deallocate(allocator, buffer);
  }

private:
  template<typename allocator_ptr> /* break interface dependency with TF */
  static void* allocate(allocator_ptr allocator, size_t size) {
    void* result;
    if (NULL != allocator) {
    /* no (useless) waste with alignment; raw result is re-aligned anyways */
      result = allocator->AllocateRaw(1/*alignment*/, size);
    }
    else {
      LIBXSMM_ASSERT_MSG(0/*false*/, "LIBXSMM ERROR: memory allocator is missing");
      result = NULL;
    }
    return result;
  }

  template<typename allocator_ptr> /* break interface dependency with TF */
  static void deallocate(allocator_ptr allocator, void* buffer) {
    LIBXSMM_ASSERT_MSG(NULL != allocator, "LIBXSMM ERROR: memory allocator is missing");
    if (NULL != allocator) allocator->DeallocateRaw(buffer);
  }
};

#endif /*defined(__cplusplus)*/

#endif /*LIBXSMM_MALLOC_H*/
