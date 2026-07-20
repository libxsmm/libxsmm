# CMake-native test support mirroring the GNU Makefile test entries.
# Keep the build independent from the GNU Makefile path: CTest may reuse the
# existing shell runners, but all executables are built by CMake.

find_package(OpenMP COMPONENTS C)

option(LIBXSMM_CTEST_WITH_BLAS_REFERENCE
  "Build selected sample tests with an external BLAS reference path" OFF)
if(LIBXSMM_CTEST_WITH_BLAS_REFERENCE)
  find_package(BLAS REQUIRED)
endif()

set(LIBXSMM_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(LIBXSMM_BINARY_ROOT "${PROJECT_BINARY_DIR}")
set(LIBXSMM_SAMPLE_ROOT "${LIBXSMM_BINARY_ROOT}/samples")
set(LIBXSMM_SCRIPT_ROOT "${LIBXSMM_BINARY_ROOT}/scripts")

file(MAKE_DIRECTORY "${LIBXSMM_TEST_DIR}")
file(MAKE_DIRECTORY "${LIBXSMM_SAMPLE_ROOT}")
file(MAKE_DIRECTORY "${LIBXSMM_SCRIPT_ROOT}")

file(COPY "${PROJECT_SOURCE_DIR}/tests/"
  DESTINATION "${LIBXSMM_TEST_DIR}"
  FILES_MATCHING
    PATTERN "*.sh"
    PATTERN "*.c"
    PATTERN "*.h"
    PATTERN "README.md")

file(COPY "${PROJECT_SOURCE_DIR}/scripts/"
  DESTINATION "${LIBXSMM_SCRIPT_ROOT}")

foreach(sample_dir IN ITEMS
    utilities/dispatch
    utilities/memcmp
    equation
    eltwise
    xgemm
    xgemm_norm_packed
    xgemm_sparse_Ainregs)
  file(COPY "${PROJECT_SOURCE_DIR}/samples/${sample_dir}/"
    DESTINATION "${LIBXSMM_SAMPLE_ROOT}/${sample_dir}")
endforeach()

function(libxsmm_set_runtime_output target rel_dir output_name)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${LIBXSMM_BINARY_ROOT}/${rel_dir}"
    RUNTIME_OUTPUT_NAME "${output_name}")
endfunction()

function(libxsmm_link_common target)
  target_link_libraries(${target} PRIVATE libxsmm::libxsmm)
  if(OpenMP_C_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
  endif()
endfunction()

function(libxsmm_link_source_only target)
  target_include_directories(${target} PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${XSMM_GENERATED_INCLUDE_DIR}")
  target_compile_definitions(${target} PRIVATE
    LIBXSMM_DEFAULT_CONFIG
    LIBXSMM_BLAS_CONST
    __BLAS=0)
  target_link_libraries(${target} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
  if(XSMM_LIBM)
    target_link_libraries(${target} PRIVATE m)
  endif()
  if(XSMM_LIBRT)
    target_link_libraries(${target} PRIVATE rt)
  endif()
  if(OpenMP_C_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
  endif()
endfunction()

function(libxsmm_add_test_program name)
  set(target "libxsmm_test_${name}")
  if(name STREQUAL "headeronly")
    add_executable(${target}
      "${PROJECT_SOURCE_DIR}/tests/headeronly.c"
      "${PROJECT_SOURCE_DIR}/tests/headeronly_aux.c")
  else()
    add_executable(${target} "${PROJECT_SOURCE_DIR}/tests/${name}.c")
  endif()
  target_include_directories(${target} PRIVATE
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/include"
    "${XSMM_GENERATED_INCLUDE_DIR}")
  target_compile_definitions(${target} PRIVATE LIBXSMM_BLAS_CONST)

  if(name IN_LIST LIBXSMM_HEADER_ONLY_TESTS)
    libxsmm_link_source_only(${target})
  else()
    libxsmm_link_common(${target})
  endif()
  libxsmm_set_runtime_output(${target} "tests" "${name}")
endfunction()

function(libxsmm_add_c_sample target rel_dir output_name source)
  add_executable(${target} "${PROJECT_SOURCE_DIR}/${source}")
  target_include_directories(${target} PRIVATE
    "${PROJECT_SOURCE_DIR}/${rel_dir}"
    "${PROJECT_SOURCE_DIR}/include"
    "${XSMM_GENERATED_INCLUDE_DIR}")
  libxsmm_link_common(${target})
  libxsmm_set_runtime_output(${target} "${rel_dir}" "${output_name}")
endfunction()

function(libxsmm_add_fsspmdm_sample target rel_dir output_name source)
  libxsmm_add_c_sample(${target} "${rel_dir}" "${output_name}" "${source}")
  if(LIBXSMM_CTEST_WITH_BLAS_REFERENCE)
    target_compile_definitions(${target} PRIVATE __USE_BLAS=1)
    target_link_libraries(${target} PRIVATE ${BLAS_LIBRARIES})
  else()
    target_compile_definitions(${target} PRIVATE __USE_BLAS=0)
  endif()
endfunction()

set(LIBXSMM_HEADER_ONLY_TESTS
  atomics
  gemmflags
  hash
  headeronly
  matdiff
  memory)

set(LIBXSMM_TEST_ENTRIES
  atomics.c
  gemmflags.c
  hash.c
  headeronly.c
  malloc.c
  matdiff.c
  math.c
  memory.c
  registry.c
  rng.c
  threadsafety.c
  timer.c
  vla.c
  dispatch.sh
  eltwise.sh
  equation.sh)
list(APPEND LIBXSMM_TEST_ENTRIES
  fsspmdm.sh
  memcmp.sh
  packed.sh
  smm.sh)

foreach(test_src IN ITEMS
    atomics
    gemmflags
    hash
    headeronly
    malloc
    matdiff
    math
    memory
    registry
    rng
    threadsafety
    timer
    vla)
  libxsmm_add_test_program(${test_src})
endforeach()

libxsmm_add_c_sample(libxsmm_sample_dispatch
  "samples/utilities/dispatch" "dispatch"
  "samples/utilities/dispatch/dispatch.c")
libxsmm_add_c_sample(libxsmm_sample_memcmp
  "samples/utilities/memcmp" "memcmp"
  "samples/utilities/memcmp/memcmp.c")

foreach(sample IN ITEMS
    equation_simple
    equation_gather_reduce
    equation_relu
    equation_simple_layernorm
    equation_matmul
    equation_softmax
    equation_layernorm
    equation_splitSGD
    equation_bf16_x3_split_f32
    equation_gather_dot
    equation_gather_bcstmul_add)
  libxsmm_add_c_sample("libxsmm_sample_${sample}"
    "samples/equation" "${sample}"
    "samples/equation/${sample}.c")
endforeach()

foreach(sample IN ITEMS
    eltwise_unary_reduce
    eltwise_unary_gather_scatter
    eltwise_binary_simple
    eltwise_ternary_simple
    eltwise_unary_dropout
    eltwise_unary_transform
    eltwise_unary_simple
    eltwise_unary_relu
    eltwise_unary_quantization)
  libxsmm_add_c_sample("libxsmm_sample_${sample}"
    "samples/eltwise" "${sample}"
    "samples/eltwise/${sample}.c")
endforeach()

foreach(sample IN ITEMS
    gemm_kernel
    gemm_kernel_fused
    gemm_kernel_parallel)
  libxsmm_add_c_sample("libxsmm_sample_${sample}"
    "samples/xgemm" "${sample}"
    "samples/xgemm/${sample}.c")
endforeach()

foreach(sample IN ITEMS asparse_packed_csr bsparse_packed_csr bsparse_packed_csc dense_packedacrm dense_packedbcrm)
  libxsmm_add_c_sample("libxsmm_sample_${sample}_f64"
    "samples/xgemm_norm_packed" "${sample}_f64"
    "samples/xgemm_norm_packed/${sample}.c")
  libxsmm_add_c_sample("libxsmm_sample_${sample}_f32"
    "samples/xgemm_norm_packed" "${sample}_f32"
    "samples/xgemm_norm_packed/${sample}.c")
  target_compile_definitions("libxsmm_sample_${sample}_f32" PRIVATE __EDGE_EXECUTE_F32__)
endforeach()

libxsmm_add_c_sample(libxsmm_sample_gimmik
  "samples/xgemm_sparse_Ainregs" "gimmik"
  "samples/xgemm_sparse_Ainregs/gimmik.c")
libxsmm_add_fsspmdm_sample(libxsmm_sample_pyfr_driver_asp_reg
  "samples/xgemm_sparse_Ainregs" "pyfr_driver_asp_reg"
  "samples/xgemm_sparse_Ainregs/pyfr_driver_asp_reg.c")

add_custom_target(libxsmm_generate_eltwise_test_scripts ALL
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_unary_simple_test_scripts.sh
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_unary_transform_test_scripts.sh
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_unary_reduce_test_scripts.sh
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_unary_relu_test_scripts.sh
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_unary_dropout_test_scripts.sh
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_unary_quant_test_scripts.sh
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_unary_gather_scatter_test_scripts.sh
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_binary_test_scripts.sh
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_ternary_test_scripts.sh
  WORKING_DIRECTORY "${LIBXSMM_SAMPLE_ROOT}/eltwise/kernel_test"
  COMMENT "Generating LIBXSMM eltwise test scripts")

add_custom_target(libxsmm_generate_xgemm_test_scripts ALL
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_gemm_test_scripts.sh
  COMMAND "${CMAKE_COMMAND}" -E env bash ./generate_spmm_test_scripts.sh
  WORKING_DIRECTORY "${LIBXSMM_SAMPLE_ROOT}/xgemm/kernel_test"
  COMMENT "Generating LIBXSMM xgemm test scripts")

foreach(target IN ITEMS
    libxsmm_sample_eltwise_unary_reduce
    libxsmm_sample_eltwise_unary_gather_scatter
    libxsmm_sample_eltwise_binary_simple
    libxsmm_sample_eltwise_ternary_simple
    libxsmm_sample_eltwise_unary_dropout
    libxsmm_sample_eltwise_unary_transform
    libxsmm_sample_eltwise_unary_simple
    libxsmm_sample_eltwise_unary_relu
    libxsmm_sample_eltwise_unary_quantization)
  add_dependencies(${target} libxsmm_generate_eltwise_test_scripts)
endforeach()

foreach(target IN ITEMS
    libxsmm_sample_gemm_kernel
    libxsmm_sample_gemm_kernel_fused
    libxsmm_sample_gemm_kernel_parallel)
  add_dependencies(${target} libxsmm_generate_xgemm_test_scripts)
endforeach()

function(libxsmm_require_test_files test_name)
  set(files ${ARGN})
  if(TEST "libxsmm-${test_name}")
    set_tests_properties("libxsmm-${test_name}" PROPERTIES
      REQUIRED_FILES "${files}")
  endif()
endfunction()

foreach(test_entry IN LISTS LIBXSMM_TEST_ENTRIES)
  get_filename_component(test_name "${test_entry}" NAME_WE)
  add_test(NAME "libxsmm-${test_name}"
    COMMAND "${LIBXSMM_TEST_DIR}/test.sh" "${test_entry}")
  set_tests_properties("libxsmm-${test_name}" PROPERTIES
    WORKING_DIRECTORY "${LIBXSMM_TEST_DIR}"
    LABELS libxsmm)
endforeach()

# Avoid false positives from shell runners that may otherwise skip missing files.
libxsmm_require_test_files(dispatch
  "${LIBXSMM_SAMPLE_ROOT}/utilities/dispatch/dispatch")
libxsmm_require_test_files(memcmp
  "${LIBXSMM_SAMPLE_ROOT}/utilities/memcmp/memcmp")
libxsmm_require_test_files(equation
  "${LIBXSMM_SAMPLE_ROOT}/equation/equation_simple"
  "${LIBXSMM_SAMPLE_ROOT}/equation/equation_matmul"
  "${LIBXSMM_SAMPLE_ROOT}/equation/equation_test/equation_simple.sh")
libxsmm_require_test_files(eltwise
  "${LIBXSMM_SAMPLE_ROOT}/eltwise/eltwise_unary_simple"
  "${LIBXSMM_SAMPLE_ROOT}/eltwise/eltwise_binary_simple")
libxsmm_require_test_files(smm
  "${LIBXSMM_SAMPLE_ROOT}/xgemm/gemm_kernel"
  "${LIBXSMM_SAMPLE_ROOT}/xgemm/gemm_kernel_fused")
libxsmm_require_test_files(packed
  "${LIBXSMM_SAMPLE_ROOT}/xgemm_norm_packed/dense_packedacrm_f32"
  "${LIBXSMM_SAMPLE_ROOT}/xgemm_norm_packed/dense_packedacrm_f64"
  "${LIBXSMM_SAMPLE_ROOT}/xgemm_norm_packed/dense_packedbcrm_f32"
  "${LIBXSMM_SAMPLE_ROOT}/xgemm_norm_packed/dense_packedbcrm_f64")
libxsmm_require_test_files(fsspmdm
  "${LIBXSMM_SAMPLE_ROOT}/xgemm_sparse_Ainregs/pyfr_driver_asp_reg")
