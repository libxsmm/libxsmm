# XGEMM: Tiled GEMM Routines

## Overview

This sample code calls the `libxsmm_?gemm_omp` routines provided by the LIBXSMM extension library (`libxsmmext`). These routines are meant for big(ger) xGEMM routines, and thereby provide an OpenMP-based parallelization.

The driver program (`xgemm.c`) currently accepts all typical GEMM arguments (except for the transposition specifier): `m`, `n`, `k`, `lda`, `ldb`, `ldc`, `alpha`, and `beta`. All arguments are optional (or will inherit defaults from previously specified arguments). Matrix transposition as part of the `libxsmm_?gemm_omp` routines will become available in an upcoming release of LIBXSMM. Please also note that unsupported Alpha or Beta values will cause a fall back to the related BLAS routine. The single-precision matrix multiplications require to change the `ITYPE` in `xgemm.c`.

```bash
./xgemm.sh 2000
```

## OpenTuner

To tune the tile sizes ("block sizes") internal to LIBXSMM, the [OpenTuner](http://opentuner.org/) extensible framework for program autotuning can be used. In case of issues during the tuning phase ("no value has been set for this column"), please install the latest 1.2.x revision of SQLAlchemy (`pip install sqlalchemy==1.2.19`). A tuning script (`xgemm_opentuner.py`) is provided, which optionally accepts a list of grouped parameters as command line arguments. The syntax of the arguments is per LIBXSMM's `MNK` build-option, and expands to "triplets" specifying the matrix shapes. For instance, four matrix multiplications of square-matrices can be benchmarked and tuned using the following command.

```bash
./xgemm_opentuner.py 1024,1280,1536,1792
```

To start a tuning experiment for a new set of arguments, it is highly recommended to start from scratch. Otherwise the population of previously generated tuning results is fetched from a database and used to tune an unrelated range of matrix shapes. Optionally, the initial block size can be seeded (`tile-size-m`, `tile-size-n`, and `tile-size-k`).

```bash
rm -rf opentuner.db
```

The script tunes the geometric mean of the performance for each of the requested triplets. However, the optimizer not only maximizes the performance but also minimizes the value of *M&#160;\*&#160;N&#160;\*&#160;K* (which also helps to prune duplicated results due to an additional preference). As a limitation of the current implementation, the multiplication kernels are not accompanied by copy-kernels (and not accompanied by transpose kernels). This negatively impacts performance on power-of-two matrix shapes (POT) due to trashing the LLC. However, it has been found, that tuning for POT shapes likely achieves superior performance when compared to tuning for non-POT shapes of the same range.

```bash
rm -rf opentuner.db
./xgemm_opentuner.py --no-dups 192,256,320,512,768

rm -rf opentuner.db
./xgemm_opentuner.py --no-dups 1024,1280,1536,1792

rm -rf opentuner.db
./xgemm_opentuner.py --no-dups 2048,2304,2560,2816

rm -rf opentuner.db
./xgemm_opentuner.py --no-dups 3072,3328,3584,3840

rm -rf opentuner.db
./xgemm_opentuner.py --no-dups 4096,4416,4736

rm -rf opentuner.db
./xgemm_opentuner.py --no-dups 5120,5440,5760

rm -rf opentuner.db
./xgemm_opentuner.py --no-dups 6144,6464,6784

rm -rf opentuner.db
./xgemm_opentuner.py --no-dups 7168,7488,7808
```

Above, the series of matrix multiplications from 192-8K is separately tuned in eight ranges. The tuning script uses the environment variables `LIBXSMM_TGEMM_M`, `LIBXSMM_TGEMM_N`, and `LIBXSMM_TGEMM_K` which are internal to LIBXSMM. These variables are used to request a specific tiling-scheme within LIBXSMM's `libxsmm_?gemm_omp` routines.

