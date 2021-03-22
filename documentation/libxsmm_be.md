## Backend

### Code Generator (JIT)

There can be situations in which it is up-front not clear which problem-sizes will be needed when running an application. To leverage LIBXSMM's high-performance kernels, the library implements a JIT (Just-In-Time) code generation backend which generates the requested kernels on the fly (in-memory). This is accomplished by emitting the corresponding byte-code directly into an executable buffer. The actual JIT code is generated per the CPUID flags, and therefore does not rely on the code path selected when building the library. In the current implementation, some limitations apply to the JIT backend specifically:

1. To stay agnostic to any threading model used, Pthread mutexes are guarding the updates of the JIT'ted code cache (link line with `-lpthread` is required); building with OMP=1 employs an OpenMP critical section as an alternative locking mechanism.
2. There is limited support for the Windows calling convention (only kernels without prefetch signature).

The JIT backend can also be disabled at build time (`make JIT=0`) as well as at runtime (`LIBXSMM_TARGET=0`, or anything prior to <span>Intel&#160;AVX</span>). The latter is an environment variable which allows to set a code path independent of the CPUID (<span>LIBXSMM_TARGET=0&#124;1&#124;sse&#124;snb&#124;hsw&#124;knl&#124;knm&#124;skx&#124;clx&#124;cpx&#124;spr</span>). Please note that LIBXSMM_TARGET cannot enable the JIT backend if it was disabled at build time (JIT=0).

One can use the afore mentioned THRESHOLD parameter to control the matrix sizes for which the JIT compilation will be automatically performed. However, explicitly requested kernels (by calling `libxsmm_?mmdispatch`) fall not under a threshold for the problem-size. In any case, JIT code generation can be used for accompanying statically generated code.

### Generator Driver

In rare situations, it might be useful to directly incorporate generated C code (with inline assembly regions). This is accomplished by invoking a driver program (with certain command line arguments).

**Note**: The stand-alone generator-driver is considered legacy (deprecated). Associated functionality may be removed and future instruction set extensions may not be addressed with printed assembly code. The cost of dispatching JIT-code for every code region of an application, and for every visit of such region, can be amortized in several ways and without dispensing JIT-generated code. Dispatching [multiple kernels at once](libxsmm_aux.md#user-data-dispatch) or (most effectively) tabulating JIT'ted function pointers manually, can elleviate or remove first-time code generation and (more important) the cost of subsequently dispatching kernels (when code was already JIT-generated).

The generator driver program is usually built as part of LIBXSMM's build process, but also available as a separate build target:

```bash
make generator
bin/libxsmm_gemm_generator
```

The code generator driver program accepts the following arguments:

1. Select: dense, dense_asm, sparse, sparse_csr, or sparse_csr_reg
2. Filename of a file to append to
3. Routine name to be created
4. M parameter
5. N parameter
6. K parameter
7. LDA (0 indicates A is sparse if 1st arg. is "sparse*")
8. LDB (0 indicates B is sparse if 1st arg. is "sparse*")
9. LDC parameter
10. Alpha (1)
11. Beta: (0 or 1)
12. Alignment override for A (1 auto, 0 unalignment)
13. Alignment override for C (1 auto, 0 unalignment)
14. Architecture (noarch, wsm, snb, hsw, knl, knm, skx, clx, cpx)
15. Prefetch strategy, see below (only nopf or pfsigonly for "sparse*")
16. SP (single-precision), DP (double-recision), or I16 (only "dense*")
17. CSC file in Matrix market format (only if 1st arg. is "sparse*").

<a name="prefetch-strategy"></a>The prefetch strategy can be:

1. "nopf": data is not prefetched, just three arguments: A, B, and C
2. "pfsigonly": no prefetches, kernel signature: A, B, C, A', B', and C'
3. "BL2viaC": uses accesses to C to prefetch B'
4. "AL2": uses accesses to A to prefetch A
5. "curAL2": prefetches current A ahead in the kernel
6. "AL2_BL2viaC": combines AL2 and BL2viaC
7. "curAL2_BL2viaC": combines curAL2 and BL2viaC

Here are some examples of invoking the driver program:

```bash
bin/libxsmm_gemm_generator dense foo.c foo 16 16 16 32 32 32 1 1 1 1 hsw nopf DP
bin/libxsmm_gemm_generator dense_asm foo.c foo 16 16 16 32 32 32 1 1 1 1 knl AL2_BL2viaC DP
bin/libxsmm_gemm_generator sparse foo.c foo 16 16 16 32 0 32 1 1 1 1 hsw nopf DP bar.csc
```

Please note, there are additional examples given in samples/generator and samples/seissol.

### Development Concepts

The low-level code generator is hosted by a single translation unit ([src/generator_x86_instructions.c](https://github.com/hfp/libxsmm/blob/master/src/generator_x86_instructions.h)). The code generator emits instructions as enumerated in [src/generator_common.h](https://github.com/hfp/libxsmm/blob/master/src/generator_common.h). A kernel then is a buffered stream of instructions in either binary/encoded or textual form. The latter is leveraged by stand-alone generator drivers that can print <span>C&#160;functions</span> with an assembly section (inline). A [generator driver](#generator-driver) may exists for some of LIBXSMM's function domains. Please note that emitting the textual form is not needed to inspect the emitted code since the binary encoded form can be easily disassembled ([objdump](index.md#objdump)).

The binary encoded form is directly suitable for execution by casting the code-buffer into a function-pointer of the corresponding signature. It is advised to rely on LIBXSMM's internal memory allocation routines to acquire an executable buffer (see libxsmm_malloc_flags, libxsmm_xmalloc, and libxsmm_malloc_attrib in [src/libxsmm_main.h](https://github.com/hfp/libxsmm/blob/master/src/libxsmm_main.h)). This ensures correct behavior in security-hardened environments. As a bonus, [profiler support](libxsmm_prof.md) for the emitted code is enabled transparently.

To debug the JIT'ted code, GNU GDB can be used to disassemble a given memory address (`disas address,+length`). Having the code disassembled side-by-side (while debugging) helps to look ahead and to have some orientation. For the latter, [objdump](index.md#objdump) can be used to acquire the source code (assembly) along with hexadecimal line numbers (length). The offset position (for GDB's disas) directly corresponds to objectdump's line numbers.

The kernel development is much like assembly programming, except that an API is used to emit instructions. For further reference, some existing source code for building kernels can be inspected (e.g., matcopy). This may help to capture the concept of mapping registers (basically a table to avoid hard-coding register names).

