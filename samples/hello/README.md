# Hello LIBXSMM

This example is focused on a specific functionality but may be considered as "Hello LIBXSMM". Copy and paste the example code and build it either manually and as described in our [main documentation](https://libxsmm.readthedocs.io/#hello-libxsmm) (see underneath the source code), or use GNU Make:

```bash
cd /path/to/libxsmm
make

cd /path/to/libxsmm/samples/hello
make

./hello
```

Alternatively, one can use the Bazel build system. To further simplify, [Bazelisk](https://github.com/bazelbuild/bazelisk) is used to boot-strap [Bazel](https://bazel.build/):

```bash
cd /path/to/libxsmm/samples/hello
bazelisk build //...

./bazel-bin/hello
```

The [C/C++ code](https://github.com/libxsmm/libxsmm/blob/main/samples/hello/hello.cpp) given here uses LIBXSMM in header-only form (`#include <libxsmm_source.h>`), which is in contrast to the code shown in the [main documentation](https://libxsmm.readthedocs.io/#hello-libxsmm). The [Fortran code](https://github.com/libxsmm/libxsmm/blob/main/samples/hello/hello.f) (`hello.f`) can be manually compiled like `gfortran -I/path/to/libxsmm/include hello.f -L/path/to/libxsmm/lib -libxsmmf -lxsmm -lxsmmnoblas -o hello` or as part of the above described invocation of GNU Make.

