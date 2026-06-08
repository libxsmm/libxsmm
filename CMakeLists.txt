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

option(XSMM_STATIC "Build libxsmm as a static library" ON)
option(XSMM_BLAS "Build libxsmm with external BLAS fallback support" OFF)
set(LIBXSMM_INSTALL_PKGCONFIGDIR "${CMAKE_INSTALL_LIBDIR}/pkgconfig" CACHE STRING
  "Installation directory for pkg-config files")

# display config
message(STATUS "******** Build Summary ********")
message(STATUS "General:")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "Options:")
message(STATUS "  XSMM_STATIC           : ${XSMM_STATIC}")
message(STATUS "  XSMM_BLAS             : ${XSMM_BLAS}")

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

# Setup current directory as project root directory.
set(XSMM_ROOT_DIR ${PROJECT_SOURCE_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${XSMM_ROOT_DIR}/include/libxsmm/*.c)
list(REMOVE_ITEM XSMM_SRCS ${XSMM_ROOT_DIR}/include/libxsmm/libxsmm_generator_gemm_driver.c)

if(NOT XSMM_SRCS)
  file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${XSMM_ROOT_DIR}/src/*.c)
  list(REMOVE_ITEM XSMM_SRCS ${XSMM_ROOT_DIR}/src/libxsmm_generator_gemm_driver.c)
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

if(XSMM_STATIC)
  add_library(xsmm STATIC ${XSMM_SRCS})
else()
  add_library(xsmm SHARED ${XSMM_SRCS})
endif()
add_library(libxsmm::libxsmm ALIAS xsmm)

set_target_properties(xsmm PROPERTIES
  EXPORT_NAME libxsmm
  OUTPUT_NAME xsmm
  POSITION_INDEPENDENT_CODE ON)
if(NOT XSMM_STATIC)
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

if(NOT XSMM_BLAS)
  target_compile_definitions(xsmm PRIVATE __BLAS=0)
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(xsmm PUBLIC ${BLAS_LIBRARIES})
endif()

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

install(TARGETS xsmm
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
if(XSMM_BLAS)
  list(APPEND LIBXSMM_PC_PRIVATE_LIBS "${BLAS_LIBRARIES}")
endif()
list(JOIN LIBXSMM_PC_PRIVATE_LIBS " " LIBXSMM_PC_PRIVATE_LIBS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/libxsmm.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libxsmm.pc
  @ONLY)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libxsmm.pc
  DESTINATION ${LIBXSMM_INSTALL_PKGCONFIGDIR})
