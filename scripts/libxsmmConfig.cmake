include(CMakeFindDependencyMacro)
include(FindPackageHandleStandardArgs)

get_filename_component(_libxsmm_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(LIBXSMM_VERSION "@VERSION@")

set(_libxsmm_suffixes_save "${CMAKE_FIND_LIBRARY_SUFFIXES}")
if(DEFINED BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
endif()
find_library(LIBXSMM_LIBRARY NAMES xsmm HINTS "${_libxsmm_prefix}/lib" NO_DEFAULT_PATH)
if(NOT LIBXSMM_LIBRARY)
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${_libxsmm_suffixes_save}")
  find_library(LIBXSMM_LIBRARY NAMES xsmm HINTS "${_libxsmm_prefix}/lib" NO_DEFAULT_PATH)
endif()
set(CMAKE_FIND_LIBRARY_SUFFIXES "${_libxsmm_suffixes_save}")
unset(_libxsmm_suffixes_save)

find_path(LIBXSMM_INCLUDE_DIR NAMES libxsmm.h
  HINTS "${_libxsmm_prefix}/include" "${_libxsmm_prefix}" NO_DEFAULT_PATH)

if(LIBXSMM_LIBRARY AND LIBXSMM_INCLUDE_DIR)
  find_dependency(Threads)

  if(NOT TARGET libxsmm::libxsmm)
    add_library(libxsmm::libxsmm UNKNOWN IMPORTED)
    set_target_properties(libxsmm::libxsmm PROPERTIES
      IMPORTED_LOCATION "${LIBXSMM_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBXSMM_INCLUDE_DIR}")

    set(_libxsmm_interface_libraries "")
    if(TARGET Threads::Threads)
      list(APPEND _libxsmm_interface_libraries Threads::Threads)
    endif()
    if(CMAKE_DL_LIBS)
      list(APPEND _libxsmm_interface_libraries "${CMAKE_DL_LIBS}")
    endif()
    if(UNIX AND NOT APPLE)
      list(APPEND _libxsmm_interface_libraries m rt)
    elseif(UNIX)
      list(APPEND _libxsmm_interface_libraries m)
    endif()
    if(_libxsmm_interface_libraries)
      set_property(TARGET libxsmm::libxsmm PROPERTY
        INTERFACE_LINK_LIBRARIES "${_libxsmm_interface_libraries}")
    endif()
    unset(_libxsmm_interface_libraries)
  else()
    set_property(TARGET libxsmm::libxsmm PROPERTY
      IMPORTED_LOCATION "${LIBXSMM_LIBRARY}")
  endif()
endif()

find_package_handle_standard_args(libxsmm
  REQUIRED_VARS LIBXSMM_LIBRARY LIBXSMM_INCLUDE_DIR
  VERSION_VAR LIBXSMM_VERSION)

unset(_libxsmm_prefix)
