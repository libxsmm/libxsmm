cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(LIBXSMM_VERSION "1.17")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt")
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" LIBXSMM_VERSION_TXT LIMIT_COUNT 1)
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)(\\.([0-9]+))?" LIBXSMM_VERSION_MATCH "${LIBXSMM_VERSION_TXT}")
  if(LIBXSMM_VERSION_MATCH)
    set(LIBXSMM_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(LIBXSMM_VERSION_MINOR "${CMAKE_MATCH_2}")
    if(CMAKE_MATCH_4)
      set(LIBXSMM_VERSION_UPDATE "${CMAKE_MATCH_4}")
    else()
      set(LIBXSMM_VERSION_UPDATE "0")
    endif()
    set(LIBXSMM_VERSION "${LIBXSMM_VERSION_MAJOR}.${LIBXSMM_VERSION_MINOR}.${LIBXSMM_VERSION_UPDATE}")
  endif()
endif()

project(libxsmm VERSION ${LIBXSMM_VERSION} LANGUAGES C CXX)

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

if(XSMM_STATIC)
  add_library(xsmm STATIC ${XSMM_SRCS})
else()
  add_library(xsmm SHARED ${XSMM_SRCS})
endif()
add_library(libxsmm::libxsmm ALIAS xsmm)

set_target_properties(xsmm PROPERTIES
  EXPORT_NAME libxsmm
  OUTPUT_NAME xsmm)
if(NOT XSMM_STATIC)
  set_target_properties(xsmm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

target_include_directories(xsmm
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
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

# The source-tree libxsmm_source.h uses ../src for header-only usage.  The
# install-tree layout used by make install places the implementation files under
# include/libxsmm instead, so generate the same relocatable header for CMake.
file(READ "${XSMM_ROOT_DIR}/include/libxsmm_source.h" LIBXSMM_SOURCE_HEADER)
string(REPLACE "\"../src/" "\"libxsmm/" LIBXSMM_SOURCE_HEADER "${LIBXSMM_SOURCE_HEADER}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/libxsmm_source.h" "${LIBXSMM_SOURCE_HEADER}")

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
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libxsmm_source.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${XSMM_ROOT_DIR}/src/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libxsmm
  FILES_MATCHING
    PATTERN "*.c"
    PATTERN "*.h")

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
