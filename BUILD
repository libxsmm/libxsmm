# LIBXSMM's minimal starting point for building with Bazel
# which relies on header-only LIBXSMM with "zero-config".
# https://libxsmm.readthedocs.io/libxsmm_samples/#hello-libxsmm
#
# Please note, this BUILD file does not build the LIBXSMM library
# but uses the header-only form. See below reference for details:
# https://libxsmm.readthedocs.io/#zero-config
#
cc_library(
    name = "xsmm",
    includes = ["include"],
    hdrs = glob([
        "include/*.h",
        "src/*.h",
        "src/*.c",
        "src/template/*.tpl.c",
    ]),
    # Remove "__BLAS=0" (defines) and link (linkopts)
    # with BLAS library if fallback to GEMM is needed
    # or desired.
    defines = ["__BLAS=0"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
