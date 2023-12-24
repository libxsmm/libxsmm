cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(libxsmm C CXX)

message(WARNING
    "CMake is the not preferred tool to build libxsmm for the time being. It's only for some target usages and \"make\" is the tool of choice."
)

option(XSMM_STATIC      "static build" ON)

# display config
message(STATUS "******** Build Summary ********")
message(STATUS "General:")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "Options:")
message(STATUS "  XSMM_STATIC           : ${XSMM_STATIC}")

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

set(CMAKE_INSTALL_MESSAGE NEVER)
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

target_include_directories(xsmm PUBLIC ${XSMM_INCLUDE_DIRS})

# https://github.com/libxsmm/libxsmm/tree/main#link-instructions
# Please pass BLAS config by "-D__BLAS=0"
# target_compile_definitions(xsmm PRIVATE __BLAS=0)

add_definitions(-DLIBXSMM_DEFAULT_CONFIG -U_DEBUG)

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
