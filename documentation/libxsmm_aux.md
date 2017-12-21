## Service Functions

### Getting and Setting the Target Architecture

This functionality is available for the C and Fortran interface. There are [ID based](https://github.com/hfp/libxsmm/blob/master/include/libxsmm_cpuid.h#L47) (same for C and Fortran) and string based functions to query the code path (as determined by the CPUID), or to set the code path regardless of the presented CPUID features. The latter may degrade performance (if a lower set of instruction set extensions is requested), which can be still useful for studying the performance impact of different instruction set extensions.  
**NOTE**: There is no additional check performed if an unsupported instruction set extension is requested, and incompatible JIT-generated code may be executed (unknown instruction signaled).

```C
int libxsmm_get_target_archid(void);
void libxsmm_set_target_archid(int id);

const char* libxsmm_get_target_arch(void);
void libxsmm_set_target_arch(const char* arch);
```

Available code paths (IDs and corresponding strings):

* LIBXSMM_TARGET_ARCH_GENERIC: "**generic**", "none", "0"
* LIBXSMM_X86_GENERIC: "**x86**", "sse2"
* LIBXSMM_X86_SSE3: "**sse3**", "sse"
* LIBXSMM_X86_SSE4: "**wsm**", "nhm", "sse4", "sse4_2", "sse4.2"
* LIBXSMM_X86_AVX: "**snb**", "avx"
* LIBXSMM_X86_AVX2: "**hsw**", "avx2"
* LIBXSMM_X86_AVX512: "**avx3**", "avx512"
* LIBXSMM_X86_AVX512_MIC: "**knl**", "mic"
* LIBXSMM_X86_AVX512_KNM: "**knm**"
* LIBXSMM_X86_AVX512_CORE: "**skx**", "skl"

The **bold** names are returned by `libxsmm_get_target_arch` whereas `libxsmm_set_target_arch` accepts all of the above strings (similar to the environment variable LIBXSMM_TARGET).

### Getting and Setting the Verbosity

The [verbose mode](index.md#verbose-mode) (level of verbosity) can be controlled using the C or Fortran API, and there is an environment variable which corresponds to `libxsmm_set_verbosity` (LIBXSMM_VERBOSE).

```C
int libxsmm_get_verbosity(void);
void libxsmm_set_verbosity(int level);
```

### Timer Facility

Due to the performance oriented nature of LIBXSMM, timer-related functionality is available for the C and Fortran interface ([libxsmm_timer.h](https://github.com/hfp/libxsmm/blob/master/include/libxsmm_timer.h#L37) and [libxsmm.f](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm.f#L32)). The timer is used in many of the [code samples](https://github.com/hfp/libxsmm/tree/master/samples) to measure the duration of executing various code regions. The timer is based on monotonic clock tick, which uses a platform-specific resolution. The counter may rely on the time stamp counter instruction (RDTSC), but this is not necessarily counting CPU cycles due to varying CPU clock speed (Turbo Boost), different clock domains (e.g., depending on the instructions executed), and other reasons (which are out of scope in this context).

```C
unsigned long long libxsmm_timer_tick(void);
double libxsmm_timer_duration(unsigned long long tick0, unsigned long long tick1);
```

### Memory Allocation

The C interface ([libxsmm_malloc.h](https://github.com/hfp/libxsmm/blob/master/include/libxsmm_malloc.h#L37)) provides functions for aligned memory one of which allows to specify the alignment (or to request an automatically selected alignment). The automatic alignment is also available with a `malloc` compatible signature. The size of the automatic alignment depends on a heuristic, which uses the size of the requested buffer.  
**NOTE**: Only `libxsmm_free` is supported to deallocate the memory.

```C
void* libxsmm_malloc(size_t size);
void* libxsmm_aligned_malloc(size_t size, size_t alignment);
void* libxsmm_aligned_scratch(size_t size, size_t alignment);
void libxsmm_free(const volatile void* memory);
int libxsmm_get_malloc_info(const void* memory, libxsmm_malloc_info* info);
int libxsmm_get_scratch_info(libxsmm_scratch_info* info);
```

The library exposes two memory allocation domains: (1)&#160;default memory allocation, and (2)&#160;scratch memory allocation. There are similar service functions for both domains that allow to customize the allocation and deallocation function. The "context form" even supports a user-defined "object", which may represent an allocator or any other external facility. To set the allocator of the default domain is analogous to setting the allocator of the scratch memory domain (shown below).

```C
int libxsmm_set_scratch_allocator(void* context,
  libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn);
int libxsmm_get_scratch_allocator(void** context,
  libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn);
```

There are currently no claims on the properties of the default memory allocation (except when [tuning](libxsmm_tune.md#scalable_malloc) the thread scalability). In contrast, the scratch memory allocation is very effective and delivers a decent speedup over subsequent regular memory allocations. In contrast to the default allocation technique, the scratch memory establishes a watermark for repeatedly allocated and deallocated buffers. The scratch memory domain is (arbitrarily) limited to 4&#160;GB of memory, but it is possible to set a different Byte-limit (available per [libxsmm_malloc.h](https://github.com/hfp/libxsmm/blob/master/include/libxsmm_malloc.h#L37), and also per environment variable LIBXSMM_SCRATCH_LIMIT with optional "k|K", "m|M", and "g|G" units).

```C
void libxsmm_set_scratch_limit(size_t nbytes);
size_t libxsmm_get_scratch_limit(void);
```

By establishing a pool of "temporary" memory, the cost of repeated allocation and deallocation cycles is avoided when the watermark is reached. The scratch memory is scope-oriented, and supports only a limited number of pools for buffers of different life-time. The [verbose mode](index.md#verbose-mode) with a verbosity level of at least two (LIBXSMM_VERBOSE=2) shows some statistics about the populated scratch memory.

```bash
Scratch: 173 MB (mallocs=5, pools=1)
```

**NOTE**: be careful with scratch memory as it only grows during execution (in between `libxsmm_init` and `libxsmm_finalize` unless `libxsmm_release_scratch` is called). This is true even when `libxsmm_free` is (and should be) used!

### Meta Image File I/O

Loading and storing data (I/O) is normally out of LIBXSMM's scope. However, comparing results (correctness) or writing files for visual inspection is clearly desired. This is particularly useful for the DNN domain. The MHD library domain provides support for the Meta Image File format (MHD). Tools such as [ITK-SNAP](http://itksnap.org/) or [ParaView](https://www.paraview.org/) can be used to inspect, compare, and modify images (even beyond two-dimensional images).

Writing an image is per `libxsmm_mhd_write`, and loading an image is split in two stages: (1)&#160;`libxsmm_mhd_read_header`, and (2)&#160;`libxsmm_mhd_read`. The first step allows to allocate a properly sized buffer, which is then used to obtain the data per `libxsmm_mhd_read`. When reading data, an on-the-fly type conversion is supported. Further, data that is already in memory can be compared against file-data without allocating memory or reading this file into memory.

To load an image from a familiar format (JPG, PNG, etc.), one may save the raw data using for instance [IrfanView](http://www.irfanview.com/) and rely on a "header-only" MHD-file (plain text). This may look like:

```ini
NDims = 2
DimSize = 202 134
ElementType = MET_UCHAR
ElementNumberOfChannels = 1
ElementDataFile = mhd_image.raw
```

In the above case, a single channel (gray-scale) 202x134-image is described with pixel data stored separately (`mhd_image.raw`). Multi-channel images are expected to interleave the pixel data. The pixel type is per `libxsmm_mhd_elemtype` ([libxsmm_mhd.h](https://github.com/hfp/libxsmm/blob/master/include/libxsmm_mhd.h#L38)).

### Thread Synchronization

LIBXSMM comes with a number of light-weight abstraction layers (macro and API-based), which are distinct from the internal API (include files in [src](https://github.com/hfp/libxsmm/tree/master/src) directory) and that are exposed for general use (and hence part of the [include](https://github.com/hfp/libxsmm/tree/master/include) directory).

The synchronization layer is mainly based on macros: LIBXSMM_LOCK_\* provide spin-locks, mutexes, and reader-writer locks (LIBXSMM_LOCK_SPINLOCK, LIBXSMM_LOCK_MUTEX, and LIBXSMM_LOCK_RWLOCK respectively). Usually the spin-lock is also named LIBXSMM_LOCK_DEFAULT. The implementation is intentionally based on OS-native primitives unless LIBXSMM is reconfigured (per LIBXSMM_LOCK_SYSTEM), or built using `make OMP=1` (using OpenMP inside of the library is not recommended). The life-cycle of a lock looks like:

```C
/* attribute variable and lock variable */
LIBXSMM_LOCK_ATTR_TYPE(LIBXSMM_LOCK_DEFAULT) attr;
LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_DEFAULT) lock;
/* attribute initialization */
LIBXSMM_LOCK_ATTR_INIT(LIBXSMM_LOCK_DEFAULT, &attr);
/* lock initialization per initialized attribute */
LIBXSMM_LOCK_INIT(LIBXSMM_LOCK_DEFAULT, &lock, &attr);
/* the attribute can be destroyed */
LIBXSMM_LOCK_ATTR_DESTROY(LIBXSMM_LOCK_DEFAULT, &attr);
/* lock destruction (usage: see below/next code block) */
LIBXSMM_LOCK_DESTROY(LIBXSMM_LOCK_DEFAULT, &lock);
```

Once the lock is initialized (or an array of locks), it can be exclusively locked or try-locked, and released at the end of the locked section (LIBXSMM_LOCK_ACQUIRE, LIBXSMM_LOCK_TRYLOCK, and LIBXSMM_LOCK_RELEASE respectively):

```C
LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_DEFAULT, &lock);
/* locked code section */
LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, &lock);
```

If the lock-kind is LIBXSMM_LOCK_RWLOCK, non-exclusive a.k.a. shared locking allows to permit multiple readers (LIBXSMM_LOCK_ACQREAD, LIBXSMM_LOCK_TRYREAD, and LIBXSMM_LOCK_RELREAD) if the lock is not acquired exclusively (see above). An attempt to only read-lock anything else but an RW-lock is an exclusive lock (see above).

```C
if (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK) ==
    LIBXSMM_LOCK_TRYREAD(LIBXSMM_LOCK_RWLOCK, &rwlock))
{ /* locked code section */
  LIBXSMM_LOCK_RELREAD(LIBXSMM_LOCK_RWLOCK, &rwlock);
}
```

Locking different sections for read (LIBXSMM_LOCK_ACQREAD, LIBXSMM_LOCK_RELREAD) and write (LIBXSMM_LOCK_ACQUIRE, LIBXSMM_LOCK_RELEASE) may look like:

```C
LIBXSMM_LOCK_ACQREAD(LIBXSMM_LOCK_RWLOCK, &rwlock);
/* locked code section: only reads are performed */
LIBXSMM_LOCK_RELREAD(LIBXSMM_LOCK_RWLOCK, &rwlock);

LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_RWLOCK, &rwlock);
/* locked code section: exclusive write (no R/W) */
LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_RWLOCK, &rwlock);
```

Depending on the platform or when using OpenMP to implement the low-level synchronization primitives, the LIBXSMM_LOCK_RWLOCK may be not implemented (OSX) or not available (OMP). In any case, LIBXSMM also implements own lock primitives which are available per API (libxsmm_mutex_\*, and libxsmm_rwlock_\*). This experimental implementation can be used independently of the LIBXSMM_LOCK_\* macros. Future versions of the library eventually map the macros to LIBXSMM's own low-level primitives.

