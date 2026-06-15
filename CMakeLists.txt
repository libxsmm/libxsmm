cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(LIBXSMM_VERSION "1.17")

project(libxsmm
  VERSION ${LIBXSMM_VERSION}
  DESCRIPTION "Library for specialized dense and sparse matrix operations and deep learning primitives"
  LANGUAGES C CXX)

message(WARNING
  "CMake is not the reference build system for LIBXSMM yet; "
  "GNU Make remains the tested and preferred build path.")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckLanguage)

option(BUILD_SHARED_LIBS "Build libxsmm as a shared library" OFF)
option(BUILD_TESTING "Build unittests" OFF)
option(LIBXSMM_FORTRAN "Build Fortran module interface (auto-detected if not set)" OFF)

set(LIBXSMM_INSTALL_PKGCONFIGDIR "${CMAKE_INSTALL_LIBDIR}/pkgconfig" CACHE STRING
  "Installation directory for pkg-config files")

# Platform
set(LINUX False)
set(MACOS False)
set(WINDOWS False)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(LINUX True)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(WINDOWS True)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(MACOS True)
endif()

check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
  set(LIBXSMM_FORTRAN ON CACHE BOOL "" FORCE)
elseif(LIBXSMM_FORTRAN)
  message(FATAL_ERROR "LIBXSMM_FORTRAN requested but no Fortran compiler found")
endif()

if(LIBXSMM_FORTRAN)
  enable_language(Fortran)
endif()

# display config
message(STATUS "******** Build Summary ********")
message(STATUS "General:")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "Options:")
message(STATUS "  BUILD_SHARED_LIBS     : ${BUILD_SHARED_LIBS}")
message(STATUS "  LIBXSMM_FORTRAN       : ${LIBXSMM_FORTRAN}")

# Setup current directory as project root directory.
set(XSMM_ROOT_DIR ${PROJECT_SOURCE_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${XSMM_ROOT_DIR}/include/libxsmm/*.c)
list(REMOVE_ITEM XSMM_SRCS
  ${XSMM_ROOT_DIR}/include/libxsmm/libxsmm_generator_gemm_driver.c
  ${XSMM_ROOT_DIR}/include/libxsmm/libxsmm_binaryexport_generator.c)

if(NOT XSMM_SRCS)
  file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${XSMM_ROOT_DIR}/src/*.c)
  list(REMOVE_ITEM XSMM_SRCS
    ${XSMM_ROOT_DIR}/src/libxsmm_generator_gemm_driver.c
    ${XSMM_ROOT_DIR}/src/libxsmm_binaryexport_generator.c)
endif()

set(XSMM_INCLUDE_DIRS ${XSMM_ROOT_DIR}/include)

# Generate libxsmm_version.h
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(XSMM_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${XSMM_GENERATED_INCLUDE_DIR})

execute_process(
  COMMAND ${Python3_EXECUTABLE}
          ${XSMM_ROOT_DIR}/scripts/libxsmm_config.py
          ${XSMM_ROOT_DIR}/src/template/libxsmm_version.h
  OUTPUT_FILE ${XSMM_GENERATED_INCLUDE_DIR}/libxsmm_version.h
  RESULT_VARIABLE LIBXSMM_VERSION_HEADER_RC
  WORKING_DIRECTORY ${XSMM_ROOT_DIR})

if(NOT LIBXSMM_VERSION_HEADER_RC EQUAL 0)
  message(FATAL_ERROR "Failed to generate libxsmm_version.h")
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${XSMM_ROOT_DIR}/version.txt
  ${XSMM_ROOT_DIR}/scripts/libxsmm_config.py
  ${XSMM_ROOT_DIR}/scripts/libxsmm_utilities.py
  ${XSMM_ROOT_DIR}/src/template/libxsmm_version.h)

if(LIBXSMM_FORTRAN)
  set(XSMM_GENERATED_FORTRAN_DIR ${CMAKE_CURRENT_BINARY_DIR}/fortran)
  file(MAKE_DIRECTORY ${XSMM_GENERATED_FORTRAN_DIR})

  set(LIBXSMM_PRECISION 0 CACHE STRING "LIBXSMM interface precision selection")
  set(LIBXSMM_FORTRAN_INTERFACE_LEVEL 2 CACHE STRING "LIBXSMM Fortran interface level")
  set(LIBXSMM_PREFETCH_TYPE -1 CACHE STRING "LIBXSMM prefetch type")
  set(LIBXSMM_ILP64 0 CACHE STRING "Use ILP64 integers in generated LIBXSMM interfaces")
  set(LIBXSMM_CACHELINE 64 CACHE STRING "LIBXSMM cacheline alignment")
  set(LIBXSMM_SYNC 1 CACHE STRING "Enable LIBXSMM synchronization support")
  set(LIBXSMM_JIT 1 CACHE STRING "Enable LIBXSMM JIT support")
  set(LIBXSMM_FLAGS 0 CACHE STRING "LIBXSMM default GEMM flags")
  set(LIBXSMM_ALPHA 1 CACHE STRING "LIBXSMM default alpha value")
  set(LIBXSMM_BETA 1 CACHE STRING "LIBXSMM default beta value")
  set(LIBXSMM_MALLOC 0 CACHE STRING "LIBXSMM malloc mode")

  math(EXPR LIBXSMM_FORTRAN_INTERFACE_ARG
    "${LIBXSMM_PRECISION} + (${LIBXSMM_FORTRAN_INTERFACE_LEVEL} * 4)")

  set(XSMM_GENERATED_FORTRAN_INTERFACE
    ${XSMM_GENERATED_FORTRAN_DIR}/libxsmm_interface.f)
  set(XSMM_GENERATED_FORTRAN_SOURCE
    ${XSMM_GENERATED_FORTRAN_DIR}/libxsmm.f)

  execute_process(
    COMMAND ${Python3_EXECUTABLE}
            ${XSMM_ROOT_DIR}/scripts/libxsmm_interface.py
            ${XSMM_ROOT_DIR}/src/template/libxsmm.f
            ${LIBXSMM_FORTRAN_INTERFACE_ARG}
            ${LIBXSMM_PREFETCH_TYPE}
    OUTPUT_FILE ${XSMM_GENERATED_FORTRAN_INTERFACE}
    RESULT_VARIABLE LIBXSMM_FORTRAN_INTERFACE_RC
    WORKING_DIRECTORY ${XSMM_ROOT_DIR})

  if(NOT LIBXSMM_FORTRAN_INTERFACE_RC EQUAL 0)
    message(FATAL_ERROR "Failed to generate libxsmm Fortran interface")
  endif()

  execute_process(
    COMMAND ${Python3_EXECUTABLE}
            ${XSMM_ROOT_DIR}/scripts/libxsmm_config.py
            ${XSMM_GENERATED_FORTRAN_INTERFACE}
            ${LIBXSMM_ILP64}
            ${LIBXSMM_CACHELINE}
            ${LIBXSMM_PRECISION}
            ${LIBXSMM_PREFETCH_TYPE}
            ${LIBXSMM_SYNC}
            ${LIBXSMM_JIT}
            ${LIBXSMM_FLAGS}
            ${LIBXSMM_ALPHA}
            ${LIBXSMM_BETA}
            ${LIBXSMM_MALLOC}
    OUTPUT_FILE ${XSMM_GENERATED_FORTRAN_SOURCE}
    RESULT_VARIABLE LIBXSMM_FORTRAN_SOURCE_RC
    WORKING_DIRECTORY ${XSMM_ROOT_DIR})

  if(NOT LIBXSMM_FORTRAN_SOURCE_RC EQUAL 0)
    message(FATAL_ERROR "Failed to configure libxsmm Fortran interface")
  endif()

  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${XSMM_ROOT_DIR}/src/template/libxsmm.f)
endif()

if(BUILD_SHARED_LIBS)
  add_library(xsmm SHARED ${XSMM_SRCS})
else()
  add_library(xsmm STATIC ${XSMM_SRCS})
endif()
add_library(libxsmm::libxsmm ALIAS xsmm)

set_target_properties(xsmm PROPERTIES
  EXPORT_NAME libxsmm
  OUTPUT_NAME xsmm
  POSITION_INDEPENDENT_CODE ON)
if(BUILD_SHARED_LIBS)
  set_target_properties(xsmm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

target_include_directories(xsmm
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${XSMM_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(xsmm
  PUBLIC
    LIBXSMM_DEFAULT_CONFIG
  PRIVATE
    $<$<BOOL:${LINUX}>:LIBXSMM_BUILD=2>
    $<$<NOT:$<BOOL:${LINUX}>>:LIBXSMM_BUILD=1>)

# Keep the core library independent from classic BLAS wrappers/fallbacks.
target_compile_definitions(xsmm PRIVATE __BLAS=0)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(xsmm PRIVATE -U_DEBUG)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(xsmm PUBLIC Threads::Threads)
target_link_libraries(xsmm PUBLIC ${CMAKE_DL_LIBS})

include(CheckLibraryExists)
check_library_exists(m sqrt "" XSMM_LIBM)
if(XSMM_LIBM)
  target_link_libraries(xsmm PUBLIC m)
endif()
check_library_exists(rt sched_yield "" XSMM_LIBRT)
if(XSMM_LIBRT)
  target_link_libraries(xsmm PUBLIC rt)
endif()

set(LIBXSMM_INSTALL_TARGETS xsmm)

if(LIBXSMM_FORTRAN)
  if(BUILD_SHARED_LIBS)
    add_library(xsmmf SHARED ${XSMM_GENERATED_FORTRAN_SOURCE})
  else()
    add_library(xsmmf STATIC ${XSMM_GENERATED_FORTRAN_SOURCE})
  endif()
  add_library(libxsmm::libxsmmf ALIAS xsmmf)

  set(XSMM_Fortran_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/mod)
  set_target_properties(xsmmf PROPERTIES
    EXPORT_NAME libxsmmf
    OUTPUT_NAME xsmmf
    POSITION_INDEPENDENT_CODE ON
    Fortran_MODULE_DIRECTORY ${XSMM_Fortran_MODULE_DIR})
  if(BUILD_SHARED_LIBS)
    set_target_properties(xsmmf PROPERTIES
      VERSION ${PROJECT_VERSION}
      SOVERSION ${PROJECT_VERSION_MAJOR})
  endif()

  target_link_libraries(xsmmf PUBLIC libxsmm::libxsmm)
  target_include_directories(xsmmf
    PUBLIC
      $<BUILD_INTERFACE:${XSMM_Fortran_MODULE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  list(APPEND LIBXSMM_INSTALL_TARGETS xsmmf)
endif()

foreach(generator IN ITEMS gemm binaryexport)
  if(generator STREQUAL "gemm")
    set(generator_source ${XSMM_ROOT_DIR}/src/libxsmm_generator_gemm_driver.c)
    set(generator_target libxsmm_gemm_generator)
  else()
    set(generator_source ${XSMM_ROOT_DIR}/src/libxsmm_binaryexport_generator.c)
    set(generator_target libxsmm_binaryexport_generator)
  endif()

  if(EXISTS "${generator_source}")
    add_executable(${generator_target} ${generator_source})
    target_include_directories(${generator_target} PRIVATE
      ${XSMM_INCLUDE_DIRS}
      ${XSMM_GENERATED_INCLUDE_DIR})
    target_link_libraries(${generator_target} PRIVATE libxsmm::libxsmm)
    install(TARGETS ${generator_target}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

install(TARGETS ${LIBXSMM_INSTALL_TARGETS}
  EXPORT libxsmmTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${XSMM_ROOT_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.f")
install(FILES ${XSMM_GENERATED_INCLUDE_DIR}/libxsmm_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(LIBXSMM_FORTRAN)
  install(FILES ${XSMM_GENERATED_FORTRAN_SOURCE}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    RENAME libxsmm.f)
  install(DIRECTORY ${XSMM_Fortran_MODULE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.mod")
endif()
install(EXPORT libxsmmTargets
  NAMESPACE libxsmm::
  FILE libxsmmTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxsmm)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxsmmConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libxsmmConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxsmm)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/libxsmmConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libxsmmConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/libxsmmConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libxsmm)

set(LIBXSMM_PC_PRIVATE_LIBS "")
if(CMAKE_THREAD_LIBS_INIT)
  list(APPEND LIBXSMM_PC_PRIVATE_LIBS "${CMAKE_THREAD_LIBS_INIT}")
endif()
if(CMAKE_DL_LIBS)
  list(APPEND LIBXSMM_PC_PRIVATE_LIBS "-l${CMAKE_DL_LIBS}")
endif()
if(XSMM_LIBM)
  list(APPEND LIBXSMM_PC_PRIVATE_LIBS "-lm")
endif()
if(XSMM_LIBRT)
  list(APPEND LIBXSMM_PC_PRIVATE_LIBS "-lrt")
endif()
list(JOIN LIBXSMM_PC_PRIVATE_LIBS " " LIBXSMM_PC_PRIVATE_LIBS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxsmm.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libxsmm.pc
  @ONLY)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libxsmm.pc
  DESTINATION ${LIBXSMM_INSTALL_PKGCONFIGDIR})

if(LIBXSMM_FORTRAN)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxsmmf.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/libxsmmf.pc
    @ONLY)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/libxsmmf.pc
    DESTINATION ${LIBXSMM_INSTALL_PKGCONFIGDIR})
endif()
