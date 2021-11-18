# SVE in LIBXSMM

This example will provide tests and benchmarks for running LIBSMM with the ARM AARCH64 SVE extension. Currently, only the processor A64FX implements this extension.

You can compile and run it in the following way:

```bash
cd /path/to/libxsmm
make
or
make BLAS=0 LIBXSMM_NO_BLAS=1 STATIC=0 -j 48

cd /path/to/libxsmm/samples/sve
make compile

make run
```


