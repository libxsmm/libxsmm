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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_MALLOC_H
#define LIBXSMM_MALLOC_H

#include "libxsmm_macros.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stddef.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/** Function types accepted for memory allocation (see libxsmm_*_allocator). */
typedef LIBXSMM_RETARGETABLE void* (*libxsmm_malloc_ctx)(void* /*context*/, size_t /*size*/);
typedef LIBXSMM_RETARGETABLE void* (*libxsmm_malloc_fun)(size_t /*size*/);
typedef union LIBXSMM_RETARGETABLE libxsmm_malloc_function {
  libxsmm_malloc_ctx ctx_form;
  libxsmm_malloc_fun function;
} libxsmm_malloc_function;

/** Function types accepted for releasing memory (see libxsmm_*_allocator). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_free_ctx)(void* /*context*/, void* /*buffer*/);
typedef LIBXSMM_RETARGETABLE void (*libxsmm_free_fun)(void* /*buffer*/);
typedef union LIBXSMM_RETARGETABLE libxsmm_free_function {
  libxsmm_free_ctx ctx_form;
  libxsmm_free_fun function;
} libxsmm_free_function;

/**
 * To setup the custom default memory allocator, either a malloc_fn and a free_fn
 * are given, or two NULL-pointers designate to reset the default allocator to a
 * library-internal default. If a context is given (non-NULL ), the context-based
 * form of the memory allocation is used.
 * It is supported to change the allocator while buffers are pending.
 */
LIBXSMM_API int libxsmm_set_default_allocator(/* malloc_fn/free_fn must correspond */
  void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Retrieve the default memory allocator. */
LIBXSMM_API int libxsmm_get_default_allocator(void** context,
  libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/**
 * To setup the scratch memory allocator, a malloc_fn function and an optional free_fn
 * are given. A NULL-free acts as a "no-operation", and the deallocation is expected
 * to be controlled otherwise. If two NULL-pointers are given, the allocator is reset
 * to the currently active default memory allocator. If a context is given (non-NULL),
 * the context-based form of the memory allocation is used.
 * It is supported to change the allocator while buffers are pending.
 */
LIBXSMM_API int libxsmm_set_scratch_allocator(/* malloc_fn/free_fn must correspond */
  void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
/** Retrieve the scratch memory allocator. */
LIBXSMM_API int libxsmm_get_scratch_allocator(void** context,
  libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);

/** Allocate memory (malloc/free interface). */
LIBXSMM_API void* libxsmm_malloc(size_t size);

/** Allocate aligned default memory. */
LIBXSMM_API void* libxsmm_aligned_malloc(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment);

/**
 * Allocate aligned scratch memory. It is not supported
 * to query properties (libxsmm_get_malloc_info).
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

#define libxsmm_aligned_scratch(size, alignment) \
  libxsmm_scratch_malloc(size, alignment, LIBXSMM_CALLER)

/** Deallocate memory (malloc/free interface). */
LIBXSMM_API void libxsmm_free(const void* memory);

/**
 * Release the entire scratch memory regardless
 * of whether it is still referenced or not.
 */
LIBXSMM_API void libxsmm_release_scratch(void);

/** Information about a buffer (default memory domain). */
typedef struct LIBXSMM_RETARGETABLE libxsmm_malloc_info {
  /** Size of the buffer. */
  size_t size;
} libxsmm_malloc_info;

/** Retrieve information about a buffer (default memory domain). */
LIBXSMM_API int libxsmm_get_malloc_info(const void* memory, libxsmm_malloc_info* info);

/** Information about the scratch memory domain. */
typedef struct LIBXSMM_RETARGETABLE libxsmm_scratch_info {
  /** Total size of all scratch memory pools. */
  size_t size;
  /** Pending allocations (not released). */
  size_t npending;
  /** Number of allocations so far. */
  size_t nmallocs;
  /** Number of pools used. */
  unsigned int npools;
} libxsmm_scratch_info;

/** Retrieve information about the scratch memory domain. */
LIBXSMM_API int libxsmm_get_scratch_info(libxsmm_scratch_info* info);

/** Calculate a hash value for a given buffer. */
LIBXSMM_API unsigned int libxsmm_hash(const void* data, size_t size, unsigned int seed);


#if defined(__cplusplus)

/** RAII idiom to temporarily setup an allocator for the lifetime of the scope. */
template<typename kind> class LIBXSMM_RETARGETABLE libxsmm_scoped_allocator {
public:
  /** C'tor, which instantiates the new allocator (plain form). */
  libxsmm_scoped_allocator(libxsmm_malloc_fun malloc_fn, libxsmm_free_fun free_fn) {
    kind::get(m_context, m_malloc, m_free);
    kind::set(0/*context*/, 0/*malloc_ctx*/, 0/*free_ctx*/, malloc_fn, free_fn);
  }

  /** C'tor, which instantiates the new allocator (context form). */
  libxsmm_scoped_allocator(void* context, libxsmm_malloc_ctx malloc_ctx, libxsmm_free_ctx free_ctx,
    libxsmm_malloc_fun malloc_fun = 0, libxsmm_free_fun free_fun = 0)
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

private: /* saved/previous allocator */
  void* m_context;
  libxsmm_malloc_function m_malloc;
  libxsmm_free_function m_free;
};

/** Allocator-kind to instantiate libxsmm_scoped_allocator<kind>. */
struct LIBXSMM_RETARGETABLE libxsmm_default_allocator {
  static void set(void* context,
    libxsmm_malloc_ctx malloc_ctx, libxsmm_free_ctx free_ctx,
    libxsmm_malloc_fun malloc_fun, libxsmm_free_fun free_fun)
  {
    libxsmm_malloc_function malloc_fn;
    libxsmm_free_function free_fn;
    if (0 == context) { /* use global form only when no context is given */
      malloc_fn.function = malloc_fun; free_fn.function = free_fun;
    }
    else {
      malloc_fn.ctx_form = malloc_ctx; free_fn.ctx_form = free_ctx;
    }
    libxsmm_set_default_allocator(context, malloc_fn, free_fn);
  }
  static void get(void*& context,
    libxsmm_malloc_function& malloc_fn, libxsmm_free_function& free_fn)
  {
    libxsmm_get_default_allocator(&context, &malloc_fn, &free_fn);
  }
};

/** Allocator-kind to instantiate libxsmm_scoped_allocator<kind>. */
struct LIBXSMM_RETARGETABLE libxsmm_scratch_allocator {
  static void set(void* context,
    libxsmm_malloc_ctx malloc_ctx, libxsmm_free_ctx free_ctx,
    libxsmm_malloc_fun malloc_fun, libxsmm_free_fun free_fun)
  {
    libxsmm_malloc_function malloc_fn;
    libxsmm_free_function free_fn;
    if (0 != malloc_fun) { /* prefer/adopt global malloc/free functions */
      malloc_fn.function = malloc_fun; free_fn.function = free_fun;
      context = 0; /* global malloc/free functions */
    }
    else {
      malloc_fn.ctx_form = malloc_ctx; free_fn.ctx_form = free_ctx;
    }
    libxsmm_set_default_allocator(context, malloc_fn, free_fn);
  }
  static void get(void*& context,
    libxsmm_malloc_function& malloc_fn, libxsmm_free_function& free_fn)
  {
    libxsmm_get_scratch_allocator(&context, &malloc_fn, &free_fn);
  }
};

/** Forward-declared types/functions used to implement libxsmm_tf_allocator. */
namespace tensorflow { class Allocator; Allocator* cpu_allocator(); }

/**
 * An object of this type adopts a memory allocator from TensorFlow.
 * All memory allocations of the requested kind within the current
 * scope (where the libxsmm_tf_allocator object lives) are subject
 * to TensorFlow's memory allocation scheme. The allocation kind
 * is usually "libxsmm_scratch_allocator"; using a second object
 * of kind "libxsmm_default_allocator" makes the default memory
 * allocation of LIBXSMM subject to TensorFlow as well.
 */
template<typename kind> class LIBXSMM_RETARGETABLE libxsmm_tf_allocator:
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
      libxsmm_tf_allocator::malloc_ctx<context_type>,
      libxsmm_tf_allocator::free_ctx<context_type>,
      libxsmm_tf_allocator::malloc,
      libxsmm_tf_allocator::free)
  {}

private:
  template<typename allocator_type> /* break interface dependency with TF */
  static void* allocate(allocator_type* allocator, size_t size) {
    /* no waste with (useless) alignment; raw result is re-aligned anyways */
    return 0 != allocator ? allocator->AllocateRaw(1/*alignment*/, size) : 0;
  }

  template<typename allocator_type> /* break interface dependency with TF */
  static void deallocate(allocator_type* allocator, void* buffer) {
    /* no waste with (useless) alignment; raw result is re-aligned anyways */
    if (0 != allocator) allocator->DeallocateRaw(buffer);
  }

  static void* malloc(size_t size) {
    return allocate(tensorflow::cpu_allocator(), size);
  }

  static void free(void* buffer) {
    deallocate(tensorflow::cpu_allocator(), buffer);
  }

  template<typename context_type> static void* malloc_ctx(void* context, size_t size) {
    typedef typename context_type::WrappedAllocator::first_type allocator_ptr;
    context_type *const tf_context = static_cast<context_type*>(context);
    if (0 != tf_context && 0 != tf_context->device()) {
      allocator_ptr allocator = 0;
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
      /* no waste with (useless) alignment; raw result is re-aligned anyways */
      return 0 != allocator ? allocator->AllocateRaw(1/*alignment*/, size) : 0;
    }
  }

  template<typename context_type> static void free_ctx(void* context, void* buffer) {
    typedef typename context_type::WrappedAllocator::first_type allocator_ptr;
    context_type *const tf_context = static_cast<context_type*>(context);
    if (0 != tf_context && 0 != tf_context->device()) {
      allocator_ptr allocator = 0;
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
      if (0 != allocator) { allocator->DeallocateRaw(buffer); }
    }
  }
};

#endif /*defined(__cplusplus)*/

#endif /*LIBXSMM_MALLOC_H*/

