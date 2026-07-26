include(FindPackageHandleStandardArgs)

get_filename_component(_libxsmm_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(LIBXSMM_VERSION "@VERSION@")
set(_libxsmm_threads "@THREADS@")

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
  HINTS "${_libxsmm_prefix}/include/libxsmm"
        "${_libxsmm_prefix}/include"
        "${_libxsmm_prefix}"
  NO_DEFAULT_PATH)

find_path(LIBXSMM_FORTRAN_MODULE_DIR NAMES libxsmm.mod LIBXSMM.mod
  HINTS "${_libxsmm_prefix}/include/libxsmm"
        "${_libxsmm_prefix}/include"
        "${_libxsmm_prefix}"
  NO_DEFAULT_PATH)

set(_libxsmm_suffixes_save "${CMAKE_FIND_LIBRARY_SUFFIXES}")
if(DEFINED BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
endif()
find_library(LIBXSMM_FORTRAN_LIBRARY NAMES xsmmf
  HINTS "${_libxsmm_prefix}/lib" NO_DEFAULT_PATH)
if(NOT LIBXSMM_FORTRAN_LIBRARY)
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${_libxsmm_suffixes_save}")
  find_library(LIBXSMM_FORTRAN_LIBRARY NAMES xsmmf
    HINTS "${_libxsmm_prefix}/lib" NO_DEFAULT_PATH)
endif()
set(CMAKE_FIND_LIBRARY_SUFFIXES "${_libxsmm_suffixes_save}")
unset(_libxsmm_suffixes_save)

if(LIBXSMM_LIBRARY AND LIBXSMM_INCLUDE_DIR)
  if(NOT TARGET libxsmm::libxsmm)
    add_library(libxsmm::libxsmm UNKNOWN IMPORTED)
    set_target_properties(libxsmm::libxsmm PROPERTIES
      IMPORTED_LOCATION "${LIBXSMM_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBXSMM_INCLUDE_DIR}")

    set(_libxsmm_interface_libraries "")
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

    # Do not call FindThreads here: it requires C or CXX, and therefore
    # prevents a Fortran-only project from discovering LIBXSMM. Preserve the
    # thread usage requirements from the GNU Make build directly instead.
    if(_libxsmm_threads AND UNIX AND NOT APPLE)
      set_property(TARGET libxsmm::libxsmm APPEND PROPERTY
        INTERFACE_COMPILE_OPTIONS "-pthread")
      set_property(TARGET libxsmm::libxsmm APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "-pthread")
    endif()
  else()
    set_property(TARGET libxsmm::libxsmm PROPERTY
      IMPORTED_LOCATION "${LIBXSMM_LIBRARY}")
  endif()
endif()

set(libxsmm_Fortran_FOUND FALSE)
if(LIBXSMM_FORTRAN_LIBRARY AND LIBXSMM_FORTRAN_MODULE_DIR
    AND TARGET libxsmm::libxsmm)
  if(NOT TARGET libxsmm::libxsmmf)
    add_library(libxsmm::libxsmmf UNKNOWN IMPORTED)
    set_target_properties(libxsmm::libxsmmf PROPERTIES
      IMPORTED_LOCATION "${LIBXSMM_FORTRAN_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBXSMM_FORTRAN_MODULE_DIR}"
      INTERFACE_LINK_LIBRARIES "libxsmm::libxsmm")
  else()
    set_property(TARGET libxsmm::libxsmmf PROPERTY
      IMPORTED_LOCATION "${LIBXSMM_FORTRAN_LIBRARY}")
  endif()
  set(libxsmm_Fortran_FOUND TRUE)
endif()

find_package_handle_standard_args(libxsmm
  REQUIRED_VARS LIBXSMM_LIBRARY LIBXSMM_INCLUDE_DIR
  VERSION_VAR LIBXSMM_VERSION
  HANDLE_COMPONENTS)

unset(_libxsmm_prefix)
unset(_libxsmm_threads)
unset(LIBXSMM_FORTRAN_LIBRARY)
unset(LIBXSMM_FORTRAN_MODULE_DIR)
