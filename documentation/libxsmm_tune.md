## Customization

### Intercepted Allocations<a name="scalable_malloc"></a>

To improve thread-scalability and to avoid frequent memory allocation/deallocation, the [scratch memory allocator](libxsmm_aux.md#memory-allocation) can be leveraged by intercepting existing malloc/free calls. This facility is built into LIBXSMM's main library, but disabled at compile-time (by default); build with `make MALLOC=1` to permanently enable, or build with `make MALLOC=-1` to even require an environment variable `LIBXSMM_MALLOC=1` or an API-call (`libxsmm_set_malloc`). Both runtime settings allow an optional lower and/or an upper bound to select malloc-calls based on the size of the allocation. For the environment option, an extra variable is introduced, e.g., use `LIBXSMM_MALLOC=1 LIBXSMM_MALLOC_LIMIT=4m:1g`.

```C
void libxsmm_set_malloc(int enabled, const size_t* lo, const size_t* hi);
int libxsmm_get_malloc(size_t* lo, size_t* hi);
```

Querying the status may return zero even if there was an attempt to enable this facility (limitation/experimental implementation). Please note, the regular [Scratch Memory API](libxsmm_aux.md#memory-allocation) (e.g., `libxsmm_[get|set]_scratch_limit`) and the related environment variables can apply as well (`LIBXSMM_SCRATCH_LIMIT`, `LIBXSMM_SCRATCH_POOLS`, `LIBXSMM_SCRATCH_SCALE`). If intercepted memory allocations are enabled, the scratch limit is adjusted by default to allow unlimited growth of the scratch domain. Further, an increased verbosity level can help to gain some insight (`LIBXSMM_VERBOSE=3`).

Intercepting malloc/free is supported by linking LIBXSMM's static or shared main library. The latter of which can be used to intercept calls of an existing and unchanged binary (LD_PRELOAD mechanism). To statically link with LIBXSMM and to intercept existing malloc/free calls, the following changes to the application's link stage are recommended:

```bash
gcc [...] -Wl,--export-dynamic \
  -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc \
  -Wl,--wrap=memalign,--wrap=free \
  /path/to/libxsmm.a
```

The main library causes a BLAS-dependency which may be already fulfilled for the application in question. However, if this is not the case (unresolved symbols), `libxsmmnoblas.a` must be linked in addition. Depending on the dependencies of the application, the link order may also need to be adjusted. Other i.e. a GNU-compatible compiler (as shown above), can induce additional requirements (compiler runtime libraries).

**Note**: The Intel Compiler may need "libirc", i.e., `-lirc` in front of `libxsmm.a`. Linking LIBXSMM's static library may require above mentioned linker flags (`--wrap`) in particular when using Intel Fortran (IFORT) as a linker driver unless `CALL libxsmm_init()` is issued (or at least one symbol of LIBXSMM's main library is referenced; check with `nm application | grep libxsmm`). Linking the static library by using the GNU compiler does not strictly need special flags when linking the application.

Linking the shared library form of LIBXSMM (`make STATIC=0`) has similar requirements with respect to the application but does not require `-Wl,--wrap` although `-Wl,--export-dynamic` is necessary if the application is statically linked (beside of LIBXSMM linked in a shared fashion). The LD_PRELOAD based mechanism does not need any changes to the link step of an application. However, `libxsmmnoblas` may be required if the application does not already link against BLAS.

```bash
LD_PRELOAD="libxsmm.so libxsmmnoblas.so"
LD_LIBRARY_PATH=/path/to/libxsmm/lib:${LD_LIBRARY_PATH}
LIBXSMM_MALLOC=1
```

**Note**: If the application already uses BLAS, of course `libxsmmnoblas` must not be used!

The following code can be compiled and linked with `gfortran example.f -o example`:

```fortran
      PROGRAM allocate_test
        DOUBLE PRECISION, ALLOCATABLE :: a(:), b(:), c(:)
        INTEGER :: i, repeat = 100000
        DOUBLE PRECISION :: t0, t1, d

        ALLOCATE(b(16*1024))
        ALLOCATE(c(16*1024))
        CALL CPU_TIME(t0)
        DO i = 1, repeat
          ALLOCATE(a(16*1024*1024))
          DEALLOCATE(a)
        END DO
        CALL CPU_TIME(t1)
        DEALLOCATE(b)
        DEALLOCATE(c)
        d = t1 - t0

        WRITE(*, "(A,F10.1,A)") "duration:", (1D3 * d), " ms"
      END PROGRAM
```

Running with `LIBXSMM_VERBOSE=3 LIBXSMM_MALLOC=1 LD_PRELOAD=... LD_LIBRARY_PATH=... ./example` displays: `Scratch: 132 MB (mallocs=1, pools=1)` which shows the innermost allocation/deallocation was served by the scratch memory allocator.

### Static Specialization

By default, LIBXSMM uses the [JIT backend](index.md#jit-backend) which is automatically building optimized code (JIT=1). Matrix multiplication kernels can be also statically specialized at compile-time of the library (M, N, and K values). This mechanism also extends the interface of the library because function prototypes are included into both the C and FORTRAN interface.

```bash
make M="2 4" N="1" K="$(echo $(seq 2 5))"
```

The above example is generating the following set of (M,N,K) triplets:

```bash
(2,1,2), (2,1,3), (2,1,4), (2,1,5),
(4,1,2), (4,1,3), (4,1,4), (4,1,5)
```

The index sets are in a loop-nest relationship (M(N(K))) when generating the indexes. Moreover, an empty index set resolves to the next non-empty outer index set of the loop nest (including to wrap around from the M to K set). An empty index set does not participate in the loop-nest relationship. Here is an example of generating multiplication routines which are "squares" with respect to M and N (N inherits the current value of the "M loop"):

```bash
make M="$(echo $(seq 2 5))" K="$(echo $(seq 2 5))"
```

An even more flexible specialization is possible by using the MNK variable when building the library. It takes a list of indexes which are eventually grouped (using commas):

```bash
make MNK="2 3, 23"
```

Each group of the above indexes is combined into all possible triplets generating the following set of (M,N,K) values:

```bash
(2,2,2), (2,2,3), (2,3,2), (2,3,3),
(3,2,2), (3,2,3), (3,3,2), (3,3,3), (23,23,23)
```

Of course, both mechanisms (M/N/K and MNK based) can be combined by using the same command line (make). Static optimization and JIT can also be combined (no need to turn off the JIT backend).

### User-Data Dispatch

It can be desired to dispatch user-defined data, i.e., to query a value based on a key. This functionality can be used to, e.g., dispatch multiple kernels in one step if a code location relies on multiple kernels. This way, one can pay the cost of dispatch one time per task rather than according to the number of JIT-kernels used by this task. This functionality is detailed in the section about [Service Functions](libxsmm_aux.md#user-data-dispatch).

### Targeted Compilation<a name="tuning"></a>

Specifying a code path is not necessary if the JIT backend is not disabled. However, disabling JIT compilation, statically generating a collection of kernels, and targeting a specific instruction set extension for the entire library looks like:

```bash
make JIT=0 AVX=3 MNK="1 2 3 4 5"
```

The above example builds a library which cannot be deployed to anything else but the <span>Intel&#160;Knights&#160;Landing processor family&#160;("KNL")</span> or future <span>Intel&#160;Xeon</span> processors supporting foundational <span>Intel&#160;AVX&#8209;512</span> instructions (<span>AVX&#8209;512F</span>). The latter might be even more adjusted by supplying MIC=1 (along with AVX=3), however this does not matter since critical code is in inline assembly (and not affected). Similarly, SSE=0 (or JIT=0 without SSE or AVX build flag) employs an "arch-native" approach whereas AVX=1, AVX=2 (with FMA), and AVX=3 are specifically selecting the kind of <span>Intel&#160;AVX</span> code. Moreover, controlling the target flags manually or adjusting the code optimizations is also possible. The following example is GCC-specific and corresponds to OPT=3, AVX=3, and MIC=1:

```bash
make OPT=3 TARGET="-mavx512f -mavx512cd -mavx512er -mavx512pf"
```

An extended interface can be generated which allows to perform software prefetches. Prefetching data might be helpful when processing batches of matrix multiplications where the next operands are farther away or otherwise unpredictable in their memory location. The prefetch strategy can be specified similar as shown in the section [Generator Driver](libxsmm_be.md#generator-driver), i.e., by either using the number of the shown enumeration, or by exactly using the name of the prefetch strategy. The only exception is PREFETCH=1 which is automatically selecting a strategy per an internal table (navigated by CPUID flags). The following example is requesting the "AL2jpst" strategy:

```bash
make PREFETCH=8
```

The prefetch interface is extending the signature of all kernels by three arguments (pa, pb, and pc). These additional arguments are specifying the locations of the operands of the next multiplication (the next a, b, and c matrices). Providing unnecessary arguments in case of the three-argument kernels is not big a problem (beside of some additional call-overhead), however running a 3-argument kernel with more than three arguments and thereby picking up garbage data is misleading or disabling the hardware prefetcher (due to software prefetches). In this case, a misleading prefetch location is given plus an eventual page fault due to an out-of-bounds (garbage-)location.

Further, a generated configuration ([template](https://github.com/libxsmm/libxsmm/blob/main/include/libxsmm_config.h)) of the library encodes the parameters for which the library was built for (static information). This helps optimizing client code related to the library's functionality. For example, the LIBXSMM_MAX_\* and LIBXSMM_AVG_\* information can be used with the LIBXSMM_PRAGMA_LOOP_COUNT macro to hint loop trip counts when handling matrices related to the problem domain of LIBXSMM.

### Auto-dispatch

The function `libxsmm_?mmdispatch` helps amortizing the cost of the dispatch when multiple calls with the same M, N, and K are needed. The automatic code dispatch is orchestrating two levels:

1. Specialized routine (implemented in assembly code),
2. BLAS library call (fallback).

Both levels are accessible directly, which allows to customize the code dispatch. The fallback level may be supplied by the <span>Intel&#160;Math&#160;Kernel&#160;Library&#160;(Intel&#160;MKL)&#160;11.2</span> DIRECT CALL feature.

Further, a preprocessor symbol denotes the largest problem-size (*M* x *N* x *K*) that belongs to the first level, and therefore determines if a matrix multiplication falls back to BLAS. The problem-size threshold can be configured by using for example:

```bash
make THRESHOLD=$((60 * 60 * 60))
```

The maximum of the given threshold and the largest requested specialization refines the value of the threshold. Please note that explicitly JIT'ting and executing a kernel is possible and independent of the threshold. If a problem-size is below the threshold, dispatching the code requires to figure out whether a specialized routine exists or not.

For statically generated code, the precision can be selected:

```bash
make PRECISION=2
```

The default preference is to generate and register both single and double-precision code (PRECISION=0). Specifying <span>PRECISION=1&#124;2</span> is generating and registering single-precision or double-precision code respectively.

The automatic dispatch is highly convenient because existing GEMM calls can serve specialized kernels (even in a binary compatible fashion), however there is (and always will be) an overhead associated with looking up the code-registry and checking whether the code determined by the GEMM call is already JIT'ted or not. This lookup has been optimized with various techniques such as specialized CPU instructions to calculate CRC32 checksums, to avoid costly synchronization (needed for thread-safety) until it is ultimately known that the requested kernel is not yet JIT'ted, and by implementing a small thread-local cache of recently dispatched kernels. The latter of which can be adjusted in size (only power-of-two sizes) but also disabled:

```bash
make CACHE=0
```

Please note that measuring the relative cost of automatically dispatching a requested kernel depends on the kernel size (obviously smaller matrices are multiplied faster on an absolute basis), however smaller matrix multiplications are bottlenecked by memory bandwidth rather than arithmetic intensity. The latter implies the highest relative overhead when (artificially) benchmarking the very same multiplication out of the CPU-cache.

