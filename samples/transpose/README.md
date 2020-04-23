# Matrix Transpose (TCOPY)

## Overview

This code sample aims to benchmark the performance of matrix transposes. The C/C++ and [FORTRAN sample code](https://github.com/hfp/libxsmm/blob/master/samples/transpose/transpose.f) differ slightly with the C/C++ code sample offering a richer set of command line options as well as build settings available inside of the [translation unit](https://github.com/hfp/libxsmm/blob/master/samples/transpose/transpose.c).

The available command line options of the sample code may be reviewed by looking into the source code. Generally, the idea is to support the following:

> transpose  [&lt;kind&gt;  [&lt;m&gt;  [&lt;n&gt;  [&lt;ldi&gt;  [&lt;ldo&gt;]]]]]  
transposef                [&lt;m&gt;  [&lt;n&gt;  [&lt;ldi&gt;  [&lt;ldo&gt;]]]]

Above, `m` and `n` specify the matrix shape, and `ldi` the leading dimension of the matrix. The argument `ldo` allows to specify an output dimension, which may differ from `ldi`. The transpose kind shall be either out-of-place (`o`) or in-place (`i`).

Running the C sample code may look like:

```bash
$ ./transpose.sh o 20000
m=20000 n=20000 ldi=20000 ldo=20000 size=3052MB (double, out-of-place)
        bandwidth: 18.8 GB/s
        duration: 159 ms
```

Instead of executing a wrapper script, one may affinitize the multi-threaded execution manually (OpenMP runtime). In case of an executable built using the Intel Compiler this may look like:

```bash
LIBXSMM_VERBOSE=2 KMP_AFFINITY=balanced,granularity=fine,1 \
./transpose o 20000
m=20000 n=20000 ldi=20000 ldo=20000 size=3052MB (double, out-of-place)
        bandwidth: 21.1 GB/s
        duration: 141 ms

Registry: 20 MB (gemm=0 mcopy=0 tcopy=1)
```

In the above case one can see from the verbose output (`LIBXSMM_VERBOSE=2`) that one kernel (tcopy) served transposing the entire matrix. To avoid duplicating JIT-kernels under contention (code registry), one may also consider `LIBXSMM_TRYLOCK=1`, which is available per API-call as well.

## OpenTuner

To tune the tile sizes ("block sizes") internal to LIBXSMM's transpose routine, the [OpenTuner](http://opentuner.org/) extensible framework for program autotuning can be used. In case of issues during the tuning phase ("no value has been set for this column"), please install the latest 1.2.x revision of SQLAlchemy (`pip install sqlalchemy==1.2.19`). A tuning script (`transpose_opentuner.py`) is provided, which accepts a range of matrix sizes as command line arguments.

> transpose_opentuner.py &lt;begin&gt; &lt;end&gt; [*nexperiments-per-epoch*] [*tile-size-m*] [*tile-size-n*]

To start a tuning experiment for a new set of arguments, it is highly recommended to start from scratch. Otherwise the population of previously generated tuning results is fetched from a database and used to tune an eventually unrelated range of matrix shapes. To get reliable timings, the total time for all experiments per epoch is minimized (hence a different number of experiments per epoch also asks for an own database). Optionally, the initial block size can be seeded (`tile-size-m` and `tile-size-n`).

```bash
rm -rf opentuner.db
```

The script tunes matrices with randomized shape according to the specified range. The leading dimension is chosen tightly for the experiments. The optimizer not only maximizes the performance but also minimizes the value of *M&#160;\*&#160;N* (which also helps to prune duplicated results due to an additional preference).

```bash
rm -rf opentuner.db
./transpose_opentuner.py --no-dups 1 1024 1000

rm -rf opentuner.db
./transpose_opentuner.py --no-dups 1024 2048 100

rm -rf opentuner.db
./transpose_opentuner.py --no-dups 2048 3072 20

rm -rf opentuner.db
./transpose_opentuner.py --no-dups 3072 4096 20

rm -rf opentuner.db
./transpose_opentuner.py --no-dups 4096 5120 16

rm -rf opentuner.db
./transpose_opentuner.py --no-dups 5120 6144 12

rm -rf opentuner.db
./transpose_opentuner.py --no-dups 6144 7168 8

rm -rf opentuner.db
./transpose_opentuner.py --no-dups 7168 8192 6
```

The tuning script uses the environment variables `LIBXSMM_TCOPY_M` and `LIBXSMM_TCOPY_N`, which are internal to LIBXSMM. These variables are used to adjust certain thresholds in `libxsmm_otrans` or to request a specific tiling-scheme inside of the `libxsmm_otrans_omp` routine.

