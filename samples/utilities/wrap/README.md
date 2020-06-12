# Wrapped DGEMM

This code sample is calling DGEMM and there is no dependency on the LIBXSMM API as it only relies on LAPACK/BLAS interface. Two variants are linked when building the source code: (1) code which is dynamically linked against LAPACK/BLAS, (2) code which is linked using `--wrap=`*symbol* as possible when using a GNU&#160;GCC compatible tool chain. For more information, see the [Call Wrapper](https://libxsmm.readthedocs.io/libxsmm_mm/#call-wrapper) section of the reference documentation.

The same (source-)code will execute in three flavors when running `dgemm-test.sh`: (1) code variant which is dynamically linked against the originally supplied LAPACK/BLAS library, (2) code variant which is linked using the wrapper mechanism of the GNU&#160;GCC tool chain, and (3) the first code but using the LD_PRELOAD mechanism (available under Linux).

**Command Line Interface (CLI)**

* Optionally takes the number of repeated DGEMM calls
* Shows the performance of the workload (wall time)

