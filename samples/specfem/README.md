# SPECFEM Sample

This sample contains a dummy example from a spectral-element stiffness kernel taken from [SPECFEM3D_GLOBE](https://github.com/geodynamics/specfem3d_globe).

It is based on a 4th-order, spectral-element stiffness kernel for simulations of elastic wave propagation through the Earth. Matrix sizes used are (25,5), (5,25) and (5,5) determined by different cut-planes through a three dimensional (5,5,5)-element with a total of 125 GLL points.


## Usage Step-by-Step

This example needs the LIBXSMM library to be built with static kernels, using MNK="5 25" (for matrix size (5,25), (25,5) and (5,5)).

1. In LIBXSMM root directory, compile the library with:

  - general default compilation:
```
make MNK="5 25" ALPHA=1 BETA=0
```

  additional compilation examples are:

  - compilation using only single precision version & aggressive optimization:
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3
```

  - for Sandy Bridge CPUs:
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3 AVX=1
```

  - for Haswell CPUs:
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3 AVX=2
```

  - for Knights Corner (KNC) (and thereby creating a Sandy Bridge version):
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3 AVX=1 \
OFFLOAD=1 KNC=1
```

  - installing libraries into a sub-directory workstation/:
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3 AVX=1 \
OFFLOAD=1 KNC=1 \
PREFIX=workstation/ install-minimal
```

2. Compile this example code by typing:

  - for default CPU host:
```
cd sample/specfem
make
```

  - for Knights Corner (KNC):
```
cd sample/specfem
make KNC=1
```

  - additionally, adding some specific Fortran compiler flags, for example:
```
cd sample/specfem
make FCFLAGS="-O3 -fopenmp" [...]
```

Note that steps 1 & 2 could be shortened:

  - by specifying a "specfem" make target in the LIBXSMM root directory:
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3 AVX=1 specfem
```

  - for Knights Corner, this would need two steps:
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3 AVX=1 OFFLOAD=1 KNC=1
make OPT=3 specfem_mic
```

Run the performance test:

  - for default CPU host:
```
./specfem.sh
```

  - for Knights Corner (KNC):
```
./specfem.sh -mic
```

## Results

Using Intel Compiler suite: icpc 15.0.2, icc 15.0.2, and ifort 15.0.2

### Sandy Bridge - Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz

- library compilation by (root directory):
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3 AVX=1
```

- single threaded example run:
```
cd sample/specfem
make; OMP_NUM_THREADS=1 ./specfem.sh
```

  Output:
```
===============================================================
average over           15 repetitions
 timing with Deville loops    =   0.1269
 timing with unrolled loops   =   0.1737 / speedup =   -36.87 %
 timing with LIBXSMM dispatch =   0.1697 / speedup =   -33.77 %
 timing with LIBXSMM prefetch =   0.1611 / speedup =   -26.98 %
 timing with LIBXSMM static   =   0.1392 / speedup =    -9.70 %
===============================================================
```


### Haswell - Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz

- library compilation by (root directory):
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3 AVX=2
```

- single threaded example run:
```
cd sample/specfem
make; OMP_NUM_THREADS=1 ./specfem.sh
```

  Output:
```
===============================================================
average over           15 repetitions
 timing with Deville loops    =   0.1028
 timing with unrolled loops   =   0.1385 / speedup =   -34.73 %
 timing with LIBXSMM dispatch =   0.1408 / speedup =   -37.02 %
 timing with LIBXSMM prefetch =   0.1327 / speedup =   -29.07 %
 timing with LIBXSMM static   =   0.1151 / speedup =   -11.93 %
===============================================================
```

- multi-threaded example run:
```
cd sample/specfem
make OPT=3; OMP_NUM_THREADS=24 ./specfem.sh
```

  Output:
```
OpenMP information:
  number of threads =           24

[...]

===============================================================
average over           15 repetitions
 timing with Deville loops    =   0.0064
 timing with unrolled loops   =   0.0349 / speedup =  -446.71 %
 timing with LIBXSMM dispatch =   0.0082 / speedup =   -28.34 %
 timing with LIBXSMM prefetch =   0.0076 / speedup =   -19.59 %
 timing with LIBXSMM static   =   0.0068 / speedup =    -5.78 %
===============================================================
```


### Knights Corner - Intel Xeon Phi B1PRQ-5110P/5120D

- library compilation by (root directory):
```
make MNK="5 25" ALPHA=1 BETA=0 PRECISION=1 OPT=3 OFFLOAD=1 KNC=1
```

- multi-threaded example run:
```
cd sample/specfem
make FCFLAGS="-O3 -fopenmp -warn" OPT=3 KNC=1; ./specfem.sh -mic
```

  Output:
```
OpenMP information:
  number of threads =          236

[...]

===============================================================
average over           15 repetitions
 timing with Deville loops    =   0.0164
 timing with unrolled loops   =   0.6982 / speedup = -4162.10 %
 timing with LIBXSMM dispatch =   0.0170 / speedup =    -3.89 %
 timing with LIBXSMM static   =   0.0149 / speedup =     9.22 %
===============================================================
```
