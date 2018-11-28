### reproducer for EDGE

#### build

1. Use `Makefile` directly. eg:

```
CXX=icpc make EDGE_ORDER=5 EDGE_PRECISION=32 EDGE_CFR=16
```
This example should build the reproducers `./bin/local_5_32_16` and `./bin/neigh_5_32_16` for the 5th order with single precision and fused runs.

2. Use the build script `./build_reproducer.sh`. eg:

```
EDGE_CXX=icpc EDGE_ARCH=skx ./build_reproducer.sh
```
By default, it should build the reproducers for all combinations of `$(order)_$(precision)_$(cfr)` where cfr>1 (always fused runs).
This can also be changed by setting the variable `EDGE_CONFIGS`. eg:
```
export EDGE_CONFIGS="5_32_16 6_64_8"
EDGE_CXX=icpc EDGE_ARCH=skx ./build_script
```

Currently, only Intel C++ compiler is verified to build the reproducer.

#### run reproducers

Run the reproducer directly. It reads in two arguments. The first is #Steps (the number of time steps/repetations), and the second is #Element (the number of total elements).
```
./local_5_32_16 10 100
```

Run the reproducer on KNL with MCDRAM and multi-threading.
```
export OMP_NUM_THREADS=2
export KMP_AFFINITY=granularity=fine,compact,1,2
numactl -m 1 ./local_5_32_16 10 100
```
The input matrix files' directory and names are hard-coded in the reproducer. The directory of input matrix files is `$(reproducer_dir)/../mats`.

*The correctness of reproducers is under verification.*
*QFMA fill-in is not added into the reproducers currently.*
