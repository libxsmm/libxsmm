# Hello LIBXSMM

This example is focused on a specific functionality but may be considered as "Hello LIBXSMM". Build the example either manually and as described in our [main documentation](https://libxsmm.readthedocs.io/#hello-libxsmm) (see underneath the source code), or use GNU Make:

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

