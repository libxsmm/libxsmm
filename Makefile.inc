#############################################################################
# Copyright (c) 2015-2018, Intel Corporation                                #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions        #
# are met:                                                                  #
# 1. Redistributions of source code must retain the above copyright         #
#    notice, this list of conditions and the following disclaimer.          #
# 2. Redistributions in binary form must reproduce the above copyright      #
#    notice, this list of conditions and the following disclaimer in the    #
#    documentation and/or other materials provided with the distribution.   #
# 3. Neither the name of the copyright holder nor the names of its          #
#    contributors may be used to endorse or promote products derived        #
#    from this software without specific prior written permission.          #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#############################################################################
# Hans Pabst (Intel Corp.)
#############################################################################

MAKEINC = $(abspath $(dir $(filter %Makefile.inc,$(MAKEFILE_LIST))))

ifeq (d,$(filter d,$(MAKEFLAGS)))
  SHELL = bash -xv
endif

ifeq (Windows_NT,$(OS))
  UNAME ?= Windows_NT
  ENV ?= $(NULL)
  # Cygwin/MinGW based
  DLIBEXT ?= dll
  SLIBEXT ?= lib
endif

# Command line utilities
#PKGCFG ?= $(shell which pkg-config 2>/dev/null)
CP ?= $(shell which cp 2>/dev/null)
MV ?= $(shell which mv 2>/dev/null)
UNAME ?= $(shell uname)
MAKE ?= make
ENV ?= env

# Python interpreter per PYTHON=/path/to/python
PYTHON3 ?= $(shell which python3 2>/dev/null)
PYTHON ?= $(shell which python 2>/dev/null)

ifeq (,$(PYTHON))
ifneq (,$(PYTHON3))
  SHELL := env PATH=$(MAKEINC):$(PATH) $(SHELL)
  PYTHON = $(shell ln -s $(PYTHON3) $(MAKEINC)/python 2>/dev/null && which python 2>/dev/null)
else
  $(info ERROR: No Python interpreter found!)
endif
endif

ifneq (Darwin,$(UNAME))
  ifneq (,$(strip $(CP)))
    CP += -u
  endif
  DLIBEXT ?= so
  SLIBEXT ?= a
else
  DLIBEXT ?= dylib
  SLIBEXT ?= a
endif

# Regular expression to match "main" (good-enough pattern)
CMAIN = "main\s*(.*)"
FMAIN = "^\s*PROGRAM\s\s*\w\w*\(\s\s*\!.*\)*$$"

# Debugging and symbols (e.g., when profiling)
SYM ?= 0
DBG ?= 0

# Optimization level
ifeq (0,$(DBG))
  OPT ?= 2
else
  OPT ?= 0
endif

# Optimization flag derived from OPT flag
ifeq (0,$(shell echo "$(OPT)" | grep -q "^[0-9]\+$$"; echo "$$?"))
  OPTFLAG = -O$(OPT)
else
  OPTFLAG = -$(OPT)
endif

# Instrumentation level (trace)
ifeq (,$(strip $(INSTRUMENT)))
  INSTRUMENT = 0
endif
TRACE ?= 0

# Number of repeated calls (tests),
# or used to scale the problem size
TESTSIZE ?= 1

# PYMOD=1: enable Python module development
PYMOD ?= 0

# Static or shared binary
STATIC ?= 0

# PIC: PIC or pic
PIC ?= pic

# Intrinsics support level
INTRINSICS ?= 2
ifneq (0,$(INTRINSICS))
  ifeq (1,$(INTRINSICS))
    DFLAGS += -DLIBXSMM_INTRINSICS_STATIC
  endif
else # disabled
  DFLAGS += -DLIBXSMM_INTRINSICS_NONE
endif

OFFLOAD ?= 0
ifneq (0,$(OFFLOAD))
  MPSS ?= 1
  KNC ?= 1
else
  MPSS ?= 0
  KNC ?= 0
endif

DEPDIR ?= $(ROOTDIR)
ifeq (0,$(KNC))
  LIBNAME ?= $(DEPDIR)/lib/libxsmm
else ifneq (3,$(AVX))
  ifeq (0,$(OFFLOAD))
    LIBNAME ?= $(DEPDIR)/lib/mic/libxsmm
  else
    LIBNAME ?= $(DEPDIR)/lib/libxsmm
  endif
else
  LIBNAME ?= $(DEPDIR)/lib/libxsmm
endif

# Internal utilities
MKTEMP = $(DEPDIR)/.mktmp.sh
FLOCK = $(DEPDIR)/.flock.sh

# THREADS refers to foundational TRT (and not necessarily Posix Threads)
THREADS ?= 1

# Threading runtime
ifeq (0,$(THREADS))
  override OMP = 0
endif
OMP ?= 0

# Code conformance (beyond -Wall)
PEDANTIC ?= 0

# Embed InterProcedural Optimization information into libraries
IPO ?= 0

# ILP64=0 (LP64 with 32-bit integers), and ILP64=0 (64-bit integers)
ILP64 ?= 0

# TBB enabled (1) or disabled (0)
# availability depends on TBBROOT
TBB ?= 0
# TBB runtime compatible with oldest supported GCC
TBB_OLDRTL ?= 0

# C Compiler is used for link stage
ifneq (Darwin,$(UNAME))
  XGROUP_BEGIN = -Wl,--start-group
  XGROUP_END = -Wl,--end-group
  XLIB_BEGIN = -Wl,--as-needed
  XLIB_END = -Wl,--no-as-needed
  XRPATH = -Wl,--rpath=$(NULL)
else
  XRPATH = -Xlinker -rpath -Xlinker $(NULL)
endif

# Enable absolute library paths
ABSLIBS ?= 0

abslibrpath = $(strip $(if $(wildcard $1),$(XRPATH)$(dir $(abspath $1)),$(NULL)))
absliblpath = $(strip $(if $(filter .$(DLIBEXT),$(suffix $1)), \
  $(if $1,-L$(dir $(abspath $1)),-L$(dir $1)),$(NULL)))
abslibfile = $(strip $(if $(filter .$(DLIBEXT),$(suffix $1)), \
  $(if $(patsubst lib%,%,$(basename $(notdir $1))), \
    -l$(patsubst lib%,%,$(basename $(notdir $1))),$(NULL)),$1))

ifneq (0,$(ABSLIBS))
  abslibpath = $(strip $(call abslibrpath,$1) $(call absliblpath,$1))
else
  abslibpath = $(call absliblpath,$1)
endif

BRACE_OPEN := (
BRACE_CLOSE := )

libpath = $(strip $(if $1, \
  $(if $(shell $1 $2 -l$3 2>&1 | grep "\-l$3"),$(NULL), \
    $(subst //,/,$(abspath $(shell $1 $2 -l$3 -Wl,--verbose 2>/dev/null | grep "$3" | tail -n1 | grep -o "[^ ()]*" | grep "\/")))), \
  $(NULL)))
abslib = $(strip $(call abslibpath,$1) $(call abslibfile,$1))

# Embedd soname into shared library
SONAMELNK ?= 2
ifneq (Darwin,$(UNAME))
  ifeq (0,$(shell ln -fs this-file-does-not-exist .ln 2>/dev/null && echo "$$?" && rm .ln 2>/dev/null))
    solink = -o $1.$2.$3.$4 -Wl,-soname=$(strip $(notdir $1).$5)
    ifneq (0,$(SONAMELNK))
      solink += $(shell cd $(dir $1) && ln -fs $(notdir $1.$2.$3.$4) $(notdir $1.$5))
    endif
    ifneq (0,$(shell echo "$$((1<$(SONAMELNK) || 0>$(SONAMELNK)))"))
      solink += $(shell cd $(dir $1) && ln -fs $(notdir $1.$5) $(notdir $1))
    endif
  else # MinGW
    solink = -o $1 -Wl,-soname=$(strip $(notdir $1).$5)
  endif
else # osx
  solink = -o $(basename $1).$2$(suffix $1) \
           -install_name $(notdir $(basename $1).$2$(suffix $1)) \
           -current_version $2.$3.$4 -compatibility_version $5
  ifneq (0,$(SONAMELNK))
    solink += $(shell cd $(dir $1) && ln -fs $(notdir $(basename $1).$2$(suffix $1)) $(notdir $1))
  endif
endif

# Pickup OpenMP library name if passed as OMP=libname|name
ifneq (0,$(shell echo "$(OMP)" | grep -q "^-*[0-9]\+$$"; echo "$$?")) # NaN
  OMPRT = $(patsubst lib%,%,$(OMP))
endif

ifneq (0,$(PYMOD))
  PYVERSION_STRING = $(shell $(PYTHON) --version 2>&1 | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
  PYVERSION = $(shell echo "$(PYVERSION_STRING)" | cut -d"." -f1,2)
  PYROOT = $(abspath $(dir $(shell which $(PYTHON) 2>/dev/null))/..)
  PYINC = $(wildcard $(PYROOT)/include/python$(PYVERSION)/Python.h)
  ifneq (,$(PYINC))
    LDFLAGS += -lpython$(PYVERSION)
    IFLAGS += -I$(dir $(PYINC))
    DFLAGS += -D__PYTHON
    # Avoid (unresolved) BLAS (alternative: BLAS=1|2)
    ifeq (,$(filter environment% override command%,$(origin BLAS)))
      override BLAS = 0
    endif
    # Enable shared library (req. for Python module)
    override STATIC = 0
  endif
endif

# Explicitly disables BLAS by user's intervention
# Makefile defines what happens (perhaps nothing)
NOBLAS ?= 0
DNOBLAS = -D__BLAS=0
ifneq (,$(filter environment% override command%,$(origin BLAS)))
ifeq (0,$(BLAS))
  BLAS_FLAGS += $(DNOBLAS)
  NOBLAS = 1
endif
endif

# Secondary static
ifneq (file,$(origin STATIC)) # prefer user override/preference (in any case)
  DEPSTATIC ?= $(STATIC)
else ifneq (,$(wildcard $(LIBNAME).$(SLIBEXT))) # prefer static (library exists)
  DEPSTATIC ?= 1
else ifneq (0,$(STATIC))
  DEPSTATIC ?= $(STATIC)
else
  DEPSTATIC ?= 0
endif

# Library extension
ifneq (0,$(DEPSTATIC))
  LIBEXT ?= $(SLIBEXT)
else
  LIBEXT ?= $(DLIBEXT)
endif

# Separate control on how to link against the BLAS library
BLAS_STATIC ?= $(DEPSTATIC)
ifneq (0,$(BLAS_STATIC))
  BLASLIBEXT ?= $(SLIBEXT)
else # shared (DLL)
  BLASLIBEXT ?= $(IMPEXT)
endif

# BLAS is not used by default
BLAS ?= 0

LIBDEP = $(LIBNAME).$(IMPEXT)
MAINLIB = $(call abslib,$(LIBDEP))

FORTDEP = $(LIBNAME)f.$(IMPEXT)
FORTLIB = $(call abslib,$(FORTDEP))

EXTDEP = $(LIBNAME)ext.$(IMPEXT)
EXTLIB = $(XLIB_BEGIN) $(call abslib,$(EXTDEP)) $(XLIB_END)
ifeq (0,$(BLAS))
  # provides libxsmmnoblas to satisfy BLAS symbols
  NOBLASLIB = $(XLIB_BEGIN) $(call abslib,$(LIBNAME)noblas.$(IMPEXT)) $(XLIB_END)
else
  NOBLASLIB = $(NULL)
endif
XBLASLIB ?= $(NOBLASLIB)

COMPILER_VERSION_FLAG ?= --version
CC_VERSION_FLAG ?= $(COMPILER_VERSION_FLAG)
FC_VERSION_FLAG ?= $(COMPILER_VERSION_FLAG)
CXX_VERSION_FLAG ?= $(CC_VERSION_FLAG)

# Automatically pickup the environment (make -e not needed),
# or pickup the Intel Compiler (if available).
ifneq (,$(notdir $(shell which icpc 2>/dev/null)))
  CXX = icpc
endif
ifneq (,$(notdir $(shell which icc 2>/dev/null)))
  CC = icc
endif
ifneq (,$(notdir $(shell which ifort 2>/dev/null)))
  FC = ifort
endif

# check if the Intel Development Tools are available
INTEL ?= $(shell echo "$$((2==$(words $(filter icpc icc, \
  $(shell $(CXX) $(CXX_VERSION_FLAG) 2>/dev/null | head -n1 | cut -d' ' -f1) \
  $(shell $(CC) $(CC_VERSION_FLAG) 2>/dev/null | head -n1 | cut -d' ' -f1)))))")

ifneq (0,$(INTEL))
  ifeq (0,$(OFFLOAD))
    CXXFLAGS += -qno-offload
    CFLAGS += -qno-offload
    ifneq (,$(strip $(FC)))
    ifeq (ifort,$(shell $(FC) $(FC_VERSION_FLAG) 2>/dev/null | head -n1 | cut -d" " -f1))
      FCFLAGS += -qno-offload
    endif
    endif
  endif
else
  ifeq (,$(notdir $(shell which $(CXX) 2>/dev/null)))
    CXX = g++
  else ifneq (0,$(shell $(CXX) $(CXX_VERSION_FLAG) >/dev/null 2>/dev/null; echo "$$?"))
    CXX = g++
  endif
  ifeq (,$(notdir $(shell which $(CC) 2>/dev/null)))
    CC = gcc
  else ifneq (0,$(shell $(CC) $(CC_VERSION_FLAG) >/dev/null 2>/dev/null; echo "$$?"))
    CC = gcc
  endif
  ifneq (0,$(FORTRAN))
  ifneq (,$(strip $(FC)))
  ifneq (0,$(shell $(FC) $(FC_VERSION_FLAG) >/dev/null 2>/dev/null; echo "$$?"))
    ifneq (,$(notdir $(shell which gfortran 2>/dev/null)))
      FC = gfortran
    else
      FC = $(NULL)
    endif
  endif
  endif
  endif
endif
ifeq (0,$(FORTRAN))
  FC = $(NULL)
endif
ifeq (,$(strip $(FC)))
  DFLAGS += -DLIBXSMM_NOFORTRAN
endif

CXX_VERSION_STRING = $(shell $(CXX) $(CXX_VERSION_FLAG) 2>/dev/null | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
CXX_VERSION_MAJOR = $(shell echo "$(CXX_VERSION_STRING)" | cut -d"." -f1)
CXX_VERSION_MINOR = $(shell echo "$(CXX_VERSION_STRING)" | cut -d"." -f2)
CXX_VERSION_PATCH = $(shell echo "$(CXX_VERSION_STRING)" | cut -d"." -f3)
ifeq (3,$(words $(CXX_VERSION_MAJOR) $(CXX_VERSION_MINOR) $(CXX_VERSION_PATCH)))
  CXX_VERSION = $(shell echo "$$(($(CXX_VERSION_MAJOR)*10000+$(CXX_VERSION_MINOR)*100+$(CXX_VERSION_PATCH)))")
else ifeq (2,$(words $(CXX_VERSION_MAJOR) $(CXX_VERSION_MINOR)))
  CXX_VERSION = $(shell echo "$$(($(CXX_VERSION_MAJOR)*10000+$(CXX_VERSION_MINOR)*100))")
  CXX_VERSION_PATCH = 0
else
  CXX_VERSION_STRING = $(NULL)
  CXX_VERSION = 0
endif

CC_VERSION_STRING = $(shell $(CC) $(CC_VERSION_FLAG) 2>/dev/null | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
CC_VERSION_MAJOR = $(shell echo "$(CC_VERSION_STRING)" | cut -d"." -f1)
CC_VERSION_MINOR = $(shell echo "$(CC_VERSION_STRING)" | cut -d"." -f2)
CC_VERSION_PATCH = $(shell echo "$(CC_VERSION_STRING)" | cut -d"." -f3)
ifeq (3,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR) $(CC_VERSION_PATCH)))
  CC_VERSION = $(shell echo "$$(($(CC_VERSION_MAJOR)*10000+$(CC_VERSION_MINOR)*100+$(CC_VERSION_PATCH)))")
else ifeq (2,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR)))
  CC_VERSION = $(shell echo "$$(($(CC_VERSION_MAJOR)*10000+$(CC_VERSION_MINOR)*100))")
  CC_VERSION_PATCH = 0
else
  CC_VERSION_STRING = $(NULL)
  CC_VERSION = 0
endif

ifneq (,$(strip $(FC)))
  FC_VERSION_STRING := $(shell $(FC) $(FC_VERSION_FLAG) 2>/dev/null | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
  FC_VERSION_MAJOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f1)
  FC_VERSION_MINOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f2)
  FC_VERSION_PATCH = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f3)
endif
ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
  FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100+$(FC_VERSION_PATCH)))")
else ifeq (2,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR)))
  FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100))")
  FC_VERSION_PATCH = 0
else
  FC_VERSION_STRING = $(NULL)
  FC_VERSION = 0
endif

ifneq (,$(strip $(FC)))
ifeq (GNU,$(shell $(FC) $(FC_VERSION_FLAG) 2>/dev/null | head -n1 | cut -d" " -f1))
  ifneq (0,$(shell echo "$$((40500>$(FC_VERSION)))"))
  ifneq (gfortran,$(notdir $(FC)))
    FC = gfortran
    FC_VERSION_STRING := $(shell $(FC) $(FC_VERSION_FLAG) 2>/dev/null | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
    FC_VERSION_MAJOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f1)
    FC_VERSION_MINOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f2)
    FC_VERSION_PATCH = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f3)
    ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
      FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100+$(FC_VERSION_PATCH)))")
    else ifeq (2,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR)))
      FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100))")
      FC_VERSION_PATCH = 0
    else
      FC_VERSION = 0
    endif
    ifneq (0,$(shell echo "$$((40500>$(FC_VERSION)))"))
      # keep FC_VERSION_STRING available for message about outdated compiler
      override FC = $(NULL)
      FC_VERSION = 0
      FORTRAN = 0
    endif
  else
    # keep FC_VERSION_STRING available for message about outdated compiler
    override FC = $(NULL)
    FC_VERSION = 0
    FORTRAN = 0
  endif
  endif
else ifneq (0,$(INTEL))
  ifneq (0,$(shell echo "$$((130000>$(FC_VERSION) && 0<$(FC_VERSION)))"))
    override FC = $(NULL)
    FC_VERSION = 0
    FORTRAN = 0
  endif
endif
endif

# compiler names
CXX_NAME = $(basename $(shell $(CXX) $(CXX_VERSION_FLAG) 2>/dev/null | head -n1 \
           | sed -e "s/^\([^0-9][^0-9]*\) ..*/\1/" -e "s/ version//" -e "s/ $(BRACE_OPEN)..*//" \
           | rev | cut -d" " -f1 | rev))
ifeq (,$(strip $(CXX_NAME)))
  CXX_NAME = $(basename $(notdir $(CXX)))
endif
CC_NAME = $(basename $(shell $(CC) $(CC_VERSION_FLAG) 2>/dev/null | head -n1 \
          | sed -e "s/^\([^0-9][^0-9]*\) ..*/\1/" -e "s/ version//" -e "s/ $(BRACE_OPEN)..*//" \
          | rev | cut -d" " -f1 | rev))
ifeq (,$(strip $(CC_NAME)))
  CC_NAME = $(basename $(notdir $(CC)))
endif
ifneq (,$(strip $(FC)))
  FC_NAME = $(firstword $(notdir $(FC)))
endif
ifeq (,$(strip $(FC_NAME)))
  FC_NAME = $(basename $(notdir $(FC)))
endif

# native GCC?
GCC ?= 0
ifeq (0,$(shell $(CC) $(CC_VERSION_FLAG) 2>/dev/null | grep -q "Free Software Foundation"; echo "$$?"))
  GCC = 1
else ifeq (GCC,$(shell $(CC) $(CC_VERSION_FLAG) 2>/dev/null | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
  GCC = 1
else ifneq (pgcc,$(CC_NAME))
  ifneq (,$(findstring pgcc,$(CC_NAME)))
    GCC = 1
  endif
endif
ifeq (1,$(GCC))
  ifeq (0,$(shell $(CXX) $(CXX_VERSION_FLAG) 2>/dev/null | grep -q "Free Software Foundation"; echo "$$?"))
    GCC = 2
  else ifeq (GCC,$(shell $(CXX) $(CXX_VERSION_FLAG) 2>/dev/null | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
    GCC = 2
  else ifneq (,$(findstring g++,$(CXX_NAME)))
    GCC = 2
  endif
  ifeq (2,$(GCC))
  ifneq (,$(strip $(FC)))
    ifeq (GCC,$(shell $(FC) $(FC_VERSION_FLAG) 2>/dev/null | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
      GCC = 3
    else ifeq (0,$(shell $(FC) $(FC_VERSION_FLAG) 2>/dev/null | grep -q "Free Software Foundation"; echo "$$?"))
      GCC = 3
    else ifneq (pgfortran,$(CC_NAME))
      ifneq (,$(findstring gfortran,$(FC_NAME)))
        GCC = 3
      endif
    endif
  endif
  endif
endif

FORCE_CXX ?= 0
ifneq (0,$(FORCE_CXX))
  CC = $(CXX)
endif

ifeq (Windows_NT,$(UNAME))
ifeq (MINGW64,$(MSYSTEM))
  MINGW = 64
else ifeq (MINGW32,$(MSYSTEM))
  MINGW = 32
else ifeq (0,$(shell $(CC) -dM -E - < /dev/null 2>/dev/null | grep -q "__MINGW64__"; echo "$$?"))
  MINGW = 64
else ifeq (0,$(shell $(CC) -dM -E - < /dev/null 2>/dev/null | grep -q "__MINGW32__"; echo "$$?"))
  MINGW = 32
endif
endif
MINGW ?= 0

#ifneq (0,$(MINGW))
#  IMPEXT ?= $(SLIBEXT)
#endif
IMPEXT ?= $(LIBEXT)

# Too many ICEs with Clang hence we want to know if we are on Clang
ifeq (0,$(GCC)) # not GCC
ifeq (0,$(INTEL)) # not Intel
ifeq (0,$(shell $(CC) -dM -E - < /dev/null 2>/dev/null | grep -q "__clang__"; echo "$$?"))
  CLANG = 1
endif
endif
endif
CLANG ?= 0

# Make GCC version number available even when not using GCC
ifneq (0,$(shell echo "$$((1<=$(GCC)))"))
  GCC_VERSION_STRING = $(CXX_VERSION_STRING)
  GCC_VERSION_MAJOR = $(CXX_VERSION_MAJOR)
  GCC_VERSION_MINOR = $(CXX_VERSION_MINOR)
  GCC_VERSION_PATCH = $(CXX_VERSION_PATCH)
else ifeq (0,$(CLANG))
  GCCBIN = $(notdir $(shell which gcc 2>/dev/null))
  ifneq (,$(strip $(GCCBIN)))
    GCC_VERSION_STRING = $(shell $(GCCBIN) $(CXX_VERSION_FLAG) 2>/dev/null | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
    GCC_VERSION_MAJOR = $(shell echo "$(GCC_VERSION_STRING)" | cut -d"." -f1)
    GCC_VERSION_MINOR = $(shell echo "$(GCC_VERSION_STRING)" | cut -d"." -f2)
    GCC_VERSION_PATCH = $(shell echo "$(GCC_VERSION_STRING)" | cut -d"." -f3)
  endif
endif
ifeq (3,$(words $(GCC_VERSION_MAJOR) $(GCC_VERSION_MINOR) $(GCC_VERSION_PATCH)))
  GCC_VERSION = $(shell echo "$$(($(GCC_VERSION_MAJOR)*10000+$(GCC_VERSION_MINOR)*100+$(GCC_VERSION_PATCH)))")
else ifeq (2,$(words $(GCC_VERSION_MAJOR) $(GCC_VERSION_MINOR)))
  GCC_VERSION = $(shell echo "$$(($(GCC_VERSION_MAJOR)*10000+$(GCC_VERSION_MINOR)*100))")
  GCC_VERSION_PATCH = 0
else
  GCC_VERSION_STRING = $(NULL)
  GCC_VERSION = 0
endif

# Select code path (if not selected otherwise)
CPUFLAGS = $(strip $(shell if [ -e /proc/cpuinfo ]; then \
    grep -m1 flags /proc/cpuinfo | cut -d: -f2-; \
  elif [ "Darwin" = "$(UNAME)" ]; then \
    sysctl -a machdep.cpu.features \
      machdep.cpu.extfeatures \
      machdep.cpu.leaf7_features \
    | cut -d: -f2- | tr "\n" " " | tr [:upper:] [:lower:]; \
  fi))
SSE ?= 0
ifeq (0,$(SSE)) # discover AVX
  ifeq (1,$(words $(filter avx512f,$(CPUFLAGS))))
    AVX ?= 3
    SSE = 0
    ifeq (2,$(words $(filter avx512pf avx512er,$(CPUFLAGS)))) # KNL
      MIC ?= 1
    else # SKX
      MIC ?= 0
    endif
  else ifeq (1,$(words $(filter avx avx1.0,$(CPUFLAGS))))
    ifeq (1,$(words $(filter fma,$(CPUFLAGS))))
      AVX ?= 2
    else
      AVX ?= 1
    endif
    SSE = 0
  endif
else ifeq (1,$(SSE)) # discover SSE
  ifeq (1,$(words $(filter sse4_2 sse4.2,$(CPUFLAGS))))
    SSE = 4
  else ifneq (0,$(words $(filter sse3 ssse3,$(CPUFLAGS))))
    SSE = 3
  else #ifeq (,$(SSE))
    SSE = 0
  endif
else ifneq (0,$(KNC))
  MPSS = 1
endif
AVX ?= 0

ifneq (0,$(INTEL))
  SUITE = Intel Compiler
else ifneq (0,$(GCC))
  SUITE = GNU Compiler Collection
else
  SUITE = Compiler
  ifeq (0,$(CLANG))
    COMPATIBLE ?= 1
  endif
  ifneq (0,$(COMPATIBLE))
    ifeq (Cray,$(shell $(CC) -V 2>&1 | head -n1 | cut -d' ' -f1))
      LDFLAGS += -hsystem_alloc
      CRAY ?= 1
    else
      PGI ?= $(shell $(CC) $(CC_VERSION_FLAG) 2>/dev/null | if grep -q "PGI"; then echo "1"; else echo "0"; fi)
    endif
  endif
endif
CRAY ?= 0
PGI ?= 0

# linker setup
LD = $(CC)
XLD = $(CXX)

ifeq (0,$(DEPSTATIC))
  LIB_LD := $(LD) -shared
  LIB_XLD := $(XLD) -shared
else
  LIB_LD := $(LD)
  LIB_XLD := $(XLD)
endif

ifeq (3,$(GCC)) # try to avoid some more references to Fortran runtime libraries
  LIBGFORTRAN = $(call libpath,$(FC) $(SLDFLAGS),$(NULL),gfortran)
  ifneq (,$(strip $(LIBGFORTRAN)))
    FLDFLAGS += $(call abslibpath,$(LIBGFORTRAN)) $(XLIB_BEGIN) $(call abslibfile,$(LIBGFORTRAN)) $(XLIB_END)
    LIB_FLD = $(LIB_LD)
    FLD = $(LD)
  endif
endif

ifeq (,$(strip $(FLD)))
ifneq (,$(strip $(FC)))
  FLD = $(FC)
  ifeq (0,$(DEPSTATIC))
    LIB_FLD := $(FLD) -shared
  else
    LIB_FLD := $(FLD)
  endif
else # fallback
  LIB_FLD := $(LIB_LD)
  FLD := $(LD)
endif
endif

ifneq (0,$(CRAY))
ifeq (0,$(DEPSTATIC))
  XLD := $(XLD) -dynamic
  FLD := $(FLD) -dynamic
  LD := $(LD) -dynamic
endif
endif

# Avoid more sophisticated flags of the GCC tool chain,
# and improve compatibility with compilers supposed to be
# compatible with the GCC tool chain
COMPATIBLE ?= 0

ifneq (0,$(INTEL))
  AR ?= xiar
  CXXLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
  FCLDFLAGS += $(XLIB_BEGIN) -lc -lstdc++ $(XLIB_END)
  CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
  CXXFLAGS += -Wall -diag-disable 1879,3415,3948,10006,10010,10411,13003
  CFLAGS += -Wall -diag-disable 1879,3415,3948,10006,10010,10411,13003
  LDFLAGS += -diag-disable 1879,3415,10006,10010,10411
  FCFLAGS += -diag-disable 10006,10010,10411,13003
  ifneq (0,$(THREADS))
    FCMTFLAGS += -threads
  endif
  CPEDANTIC += -Wcheck -diag-disable 177,1419,2547,10382
  FPEDANTIC += -warn all,notruncated_source -diag-disable 7025,7373,10237,10342,10382
  CWARNEXTRA = -Wremarks
  ifeq (1,$(PEDANTIC))
    FSTD = -std03
    CSTD = -std=c99
    CXXFLAGS += -Wcheck
    CFLAGS += $(CSTD) -Wcheck
    FMFLAGS += $(FSTD) $(FPEDANTIC) -diag-disable 10010
    FCFLAGS += -free
  else ifneq (0,$(PEDANTIC))
    FSTD = -std03
    CSTD = -std=c89
    CXXFLAGS += $(CPEDANTIC)
    CFLAGS += $(CSTD) $(CPEDANTIC) $(CWARNEXTRA)
    FCFLAGS += $(FSTD) $(FPEDANTIC)
    FMFLAGS += -fixed
  else
    CSTD = -std=c89
    FCFLAGS += -free
  endif
  # flag specifying output directory must be last
  FMFLAGS += -module
  CXXFLAGS += $(OPTFLAG)
  CFLAGS += $(OPTFLAG)
  FCFLAGS += $(OPTFLAG)
  ifeq (0,$(DBG))
    # consider more accurate -fp-model (C/C++: precise, Fortran: source)
    CXXFLAGS += -fno-alias -ansi-alias -qoverride_limits #-fp-model fast=2
    CFLAGS += -fno-alias -ansi-alias -qoverride_limits #-fp-model fast=2
    FCFLAGS += -qoverride_limits #-fp-model fast=2
    ifneq (0,$(shell echo "$$((130000<=$(FC_VERSION)))"))
      FCFLAGS += -align array64byte
    endif
    ifneq (0,$(IPO))
      CXXFLAGS += -ipo
      CFLAGS += -ipo
      FCFLAGS += -ipo
    endif
    ifneq (0,$(INSTRUMENT))
      SYM = $(INSTRUMENT)
    else ifeq (0,$(shell echo "$$((0>$(SYM)))"))
      DFLAGS += -DNDEBUG
    else
      DFLAGS += -D_DEBUG
    endif
  else # debugging enabled
    ifneq (0,$(shell echo "$$((1<$(DBG) || 0>$(DBG)))"))
      DFLAGS += -D_DEBUG
    endif
    FCFLAGS += -check
    SYM = $(DBG)
  endif
  ifneq (0,$(INSTRUMENT))
    CXXFLAGS += -finstrument-functions
    CFLAGS += -finstrument-functions
    FCFLAGS += -finstrument-functions
    DFLAGS += -D__TRACE=$(INSTRUMENT)
  endif
  ifneq (0,$(shell echo "$$((3>$(DBG)))"))
    ifeq (0,$(COMPATIBLE))
    ifneq (,$(filter environment% override command%,$(origin COMPATIBLE)))
      ifeq (3,$(AVX))
        ifeq (0,$(MIC))
          ifneq (file,$(origin MIC))
            CTARGET = -xCORE-AVX512
          else
            CTARGET = -xCOMMON-AVX512
          endif
        else
          CTARGET = -xMIC-AVX512
        endif
      else ifeq (2,$(AVX))
        CTARGET = -xCORE-AVX2
      else ifeq (1,$(AVX))
        CTARGET = -xAVX
      else ifneq (0,$(SSE))
        ifeq (1,$(SSE)) # default
          CTARGET = -xSSE4.2
        else ifeq (3,$(SSE))
          ifneq (Darwin,$(UNAME))
            CTARGET = -xSSE3
          else # no systems with less than SSE4.2
            CTARGET = -xSSE4.2
          endif
        else ifeq (4,$(SSE))
          CTARGET = -xSSE4.2
        else
          CTARGET = -xSSE$(SSE)
        endif
      else ifneq (0,$(AVX))
        CTARGET = -xHost
      endif
    endif
    endif
    ifeq (3,$(AVX))
      ifneq (,$(MIC)) # specified
        ifneq (0,$(MIC)) # MIC
          CTARGET = -xMIC-AVX512
        else # SKX
          CTARGET = -xCORE-AVX512
        endif
      else
        CTARGET = -xCOMMON-AVX512
      endif
    else ifeq (2,$(AVX))
      CTARGET = -march=core-avx2
    endif
  endif
  ifneq (0,$(SYM))
    ifeq (1,$(SYM))
      CXXFLAGS := -g $(CXXFLAGS)
      CFLAGS := -g $(CFLAGS)
      FCFLAGS := -g -traceback $(FCFLAGS)
    else
      CXXFLAGS := -g3 -debug inline-debug-info $(CXXFLAGS)
      CFLAGS := -g3 -debug inline-debug-info $(CFLAGS)
      FCFLAGS := -g -traceback $(FCFLAGS)
    endif
  endif
  OMPFLAG = -fopenmp
  ifeq (,$(strip $(OMPRT)))
    OMPRT = iomp5
  endif
  ifneq (0,$(OMP))
    CXXFLAGS += $(OMPFLAG)
    CFLAGS += $(OMPFLAG)
    FCFLAGS += $(OMPFLAG)
    LDFLAGS += $(OMPFLAG)
  endif
  ifneq (0,$(SIMD))
  ifneq (0,$(shell echo "$$((150000<=$(CXX_VERSION)))"))
    DFLAGS += -DLIBXSMM_OPENMP_SIMD
    CXXFLAGS += -qopenmp-simd
    CFLAGS += -qopenmp-simd
    FCFLAGS += -qopenmp-simd
  endif
  endif
  ifeq (1,$(STATIC))
    SLDFLAGS += -no-intel-extensions -static-intel -static-libstdc++
    ifneq (Darwin,$(UNAME))
      SLDFLAGS += -static-libgcc
    endif
    DFLAGS += -D__STATIC=1
  else ifneq (0,$(STATIC))
    DFLAGS += -D__STATIC=$(STATIC)
    SLDFLAGS += -static
  endif
  ifneq (,$(strip $(R8)))
  ifneq (0,$(R8))
    FCFLAGS += -autodouble
  endif
  endif
  # workaround for certain bits introduced by GCC 7.0
  ifneq (0,$(shell echo "$$(((180000<=$(CC_VERSION) && 180001>$(CC_VERSION)) || (170006>$(CC_VERSION) && 0!=$(CC_VERSION))))"))
    CFLAGS += -D_Float128=__float128
  endif
else # GCC assumed
  FCLDFLAGS += $(XLIB_BEGIN) -lc -lstdc++ $(XLIB_END)
  CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
  CXXLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
  LDFLAGS += $(XLIB_BEGIN) -lm $(XLIB_END)
  ifeq (0,$(COMPATIBLE))
    ifneq (0,$(shell echo "$$((40700<=$(GCC_VERSION)))"))
      CXXFLAGS += -std=c++11
    endif
    CXXFLAGS += -Wall -Wno-unused-function #-Wno-attributes
    CFLAGS += -Wall -Wno-unused-function #-Wno-attributes
    FSTD = -std=f2003
    CPEDANTIC += -pedantic -Wextra -Wno-variadic-macros
    FPEDANTIC += -pedantic -Wextra -Wunused-variable \
                 -Wcharacter-truncation -Wline-truncation \
                 -Wconversion -Wintrinsics-std \
                 -Wimplicit-interface -Wimplicit-procedure
    ifneq (0,$(shell echo "$$((40200<=$(FC_VERSION)))"))
      CPEDANTIC += -Wno-overlength-strings
    else ifneq (0,$(CLANG))
      CPEDANTIC += -Wno-overlength-strings
    endif
    ifneq (0,$(shell echo "$$((40500<=$(FC_VERSION)))"))
      CPEDANTIC += -Wshadow
    endif
    ifneq (0,$(shell echo "$$((50000<=$(FC_VERSION)))"))
      FWARNEXTRA = -Wuse-without-only -Wc-binding-type \
                   -Wrealloc-lhs -Wrealloc-lhs-all \
                   -Wreal-q-constant -Wconversion-extra
    endif
    ifeq (0,$(MINGW))
      CPEDANTIC += -Wformat=2
    else # MinGW
      CXXFLAGS += -fno-asynchronous-unwind-tables
      FCFLAGS += -fno-asynchronous-unwind-tables
      CFLAGS += -fno-asynchronous-unwind-tables
      ifneq (0,$(PEDANTIC))
      ifneq (1,$(PEDANTIC))
        CFLAGS += -Wno-format
      endif
      endif
    endif
    FPEDANTIC += $(FWARNEXTRA)
    ifeq (1,$(PEDANTIC))
      CSTD = -std=c99
      CXXFLAGS += $(CPEDANTIC) -Wno-long-long
      #CXXFLAGS += -Wno-missing-field-initializers
      CFLAGS += $(CSTD) $(CPEDANTIC)
      FCFLAGS += -ffree-form
      FMFLAGS += $(FSTD) -pedantic -Wunused-variable $(FWARNEXTRA)
    else ifneq (0,$(PEDANTIC))
      ifneq (Darwin,$(UNAME))
        CPEDANTIC += -Wno-long-long
        CSTD = -std=c89
      else ifneq (0,$(shell echo "$$((1<=$(GCC)))"))
        CPEDANTIC += -Wno-long-long
        CSTD = -std=c89
      else # Clang may run into ICEs under OSX
        CSTD = -std=c99
      endif
      CXXFLAGS += $(CPEDANTIC) -Wno-long-long #-Wzero-as-null-pointer-constant
      #CXXFLAGS += -Wno-missing-field-initializers
      CFLAGS += $(CSTD) $(CPEDANTIC)
      FCFLAGS += $(FSTD) $(FPEDANTIC)
    else ifeq (0,$(COMPATIBLE))
      CPEDANTIC += -Wno-long-long #-Wno-missing-field-initializers
      FCFLAGS += -ffree-form
      CSTD = -std=c89
    endif
    # flag specifying output directory must be last
    FMFLAGS += -J
  else # fallback
    FMFLAGS += -I
  endif
  CXXFLAGS += $(OPTFLAG)
  CFLAGS += $(OPTFLAG)
  FCFLAGS += $(OPTFLAG)
  ifeq (0,$(DBG))
    ifeq (0,$(COMPATIBLE))
      CXXFLAGS += -funroll-loops
      CFLAGS += -funroll-loops
      FCFLAGS += -funroll-loops
    endif
    ifneq (0,$(IPO))
      CXXFLAGS += -flto -ffat-lto-objects
      CFLAGS += -flto -ffat-lto-objects
      FCFLAGS += -flto -ffat-lto-objects
      FLDFLAGS += -fno-lto
      LDFLAGS += -flto
    endif
    ifneq (0,$(INSTRUMENT))
      SYM = $(INSTRUMENT)
    else ifeq (0,$(shell echo "$$((0>$(SYM)))"))
      DFLAGS += -DNDEBUG
    else
      DFLAGS += -D_DEBUG
    endif
  else
    ifneq (0,$(shell echo "$$((1<$(DBG) || 0>$(DBG)))"))
      DFLAGS += -D_DEBUG
    endif
    SYM = $(DBG)
  endif
  ifneq (0,$(INSTRUMENT))
    CXXFLAGS += -finstrument-functions
    CFLAGS += -finstrument-functions
    FCFLAGS += -finstrument-functions
    # e.g. clang does not need/understand below flag
    ifneq (0,$(shell echo "$$((40300<=$(GCC_VERSION)))"))
      CFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
      ifneq (0,$(shell echo "$$((2<=$(GCC)))"))
        CXXFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
        ifneq (0,$(shell echo "$$((3<=$(GCC)))"))
          FCFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
        endif
      endif
    endif
    DFLAGS += -D__TRACE=$(INSTRUMENT)
  endif
  ifeq (Windows_NT,$(UNAME))
    LDFLAGS += $(XLIB_BEGIN) -ldbghelp $(XLIB_END)
  else ifeq (FreeBSD,$(UNAME))
    LDFLAGS += $(XLIB_BEGIN) -lexecinfo $(XLIB_END)
  endif
  ifneq (0,$(SYM))
    ifeq (1,$(SYM))
      CXXFLAGS := -g $(CXXFLAGS)
      CFLAGS := -g $(CFLAGS)
      FCFLAGS := -g $(FCFLAGS)
    else ifeq (2,$(SYM))
      CXXFLAGS := -g $(CXXFLAGS) -fsanitize=thread -fno-omit-frame-pointer
      CFLAGS := -g $(CFLAGS) -fsanitize=thread -fno-omit-frame-pointer
      FCFLAGS := -g $(FCFLAGS) -fsanitize=thread -fno-omit-frame-pointer
      LDFLAGS := -g $(LDFLAGS) -fsanitize=thread -ltsan -fno-omit-frame-pointer
      #ELDFLAGS := -pie
    else
      CXXFLAGS := -g3 $(CXXFLAGS)
      CFLAGS := -g3 $(CFLAGS)
      FCFLAGS := -g $(FCFLAGS)
    endif
  endif
  ifeq (0,$(COMPATIBLE))
    ifneq (0,$(shell echo "$$((1<=$(GCC)))"))
      OMPFLAG = -fopenmp
    endif
  endif
  ifeq (,$(OMPFLAG))
  ifneq (,$(filter environment% override command%,$(origin OMP)))
    OMPFLAG = -fopenmp
  endif
  endif
  # account for missing TLS/OMP
  ifeq (,$(OMPFLAG))
    THREADS ?= 0
    OMP = 0
  endif
  ifeq (,$(strip $(OMPRT)))
    OMPRT = gomp
  endif
  ifneq (0,$(OMP))
  ifneq (,$(OMPFLAG))
    # clang: OMP=libomp
    ifneq (0,$(shell echo "$(OMP)" | grep -q "^-*[0-9]\+$$"; echo "$$?")) # NaN
      CXXFLAGS += $(OMPFLAG)=$(OMP)
      CFLAGS += $(OMPFLAG)=$(OMP)
      # omit passing special OMP into Fortran compiler
      FCFLAGS += $(OMPFLAG)
      CLDFLAGS += $(OMPFLAG)=$(OMP)
    else ifneq (0,$(shell $(LD) $(SLDFLAGS) -l$(OMPRT) 2>&1 | grep -q "\-l$(OMPRT)"; echo "$$?"))
      CXXFLAGS += $(OMPFLAG)
      CFLAGS += $(OMPFLAG)
      FCFLAGS += $(OMPFLAG)
      LDFLAGS += $(OMPFLAG)
    endif
  endif
  endif
  ifeq (1,$(STATIC))
    SLDFLAGS += -Bstatic -static-libstdc++
    ifneq (Darwin,$(UNAME))
      SLDFLAGS += -static-libgcc
    endif
    DFLAGS += -D__STATIC=1
  else ifneq (0,$(STATIC))
    DFLAGS += -D__STATIC=$(STATIC)
    ifeq (0,$(shell $(LD) -static -ldummydoesnotexist 2>&1 | grep -q "\-ldummydoesnotexist"; echo "$$?"))
      SLDFLAGS += -static
    endif
  endif
  ifneq (,$(strip $(R8)))
  ifneq (0,$(R8))
    FCFLAGS += -fdefault-real-8 -fdefault-double-8
  endif
  endif
endif

ifeq (,$(CTARGET))
ifneq (0,$(shell echo "$$((3>$(DBG)))"))
  # prevents Clang BE error (CRC32 and others)
  ifneq (Darwin,$(UNAME))
  ifeq (3,$(GCC))
  ifeq (0,$(DEPSTATIC))
    CTARGET = -msse3
  else ifneq (0,$(SHARED))
    ifneq (,$(SHARED))
      CTARGET = -msse3
    endif
  endif
  endif
  endif
  ifeq (,$(CTARGET))
  ifneq (0,$(shell echo "$$((2<=$(AVX)))"))
    ifeq (3,$(AVX))
    ifneq (0,$(shell echo "$$(((0!=$(CLANG) && 40000<=$(CC_VERSION)) || 50000<=$(GCC_VERSION) || 0==$(CC_VERSION)))"))
    ifeq (0,$(shell $(CC) -E -mavx512f -mavx512cd /dev/null 2>/dev/null >/dev/null; echo "$$?"))
      CTARGET = -mavx512f -mavx512cd
      # fix for Clang
      CTARGET += -mfma
      ifneq (,$(MIC)) # specified
        ifneq (0,$(MIC)) # MIC
          ifeq (0,$(shell $(CC) -E -mavx512pf -mavx512er /dev/null 2>/dev/null >/dev/null; echo "$$?"))
            CTARGET += -mavx512pf -mavx512er
          endif
        else # SKX
          ifeq (0,$(shell $(CC) -E -mavx512dq -mavx512bw -mavx512vl /dev/null 2>/dev/null >/dev/null; echo "$$?"))
            CTARGET += -mavx512dq -mavx512bw -mavx512vl
            #CTARGET += -mavx512ifma -mavx512vbmi
          endif
        endif
      endif
    endif
    endif
    endif
    ifeq (,$(CTARGET)) # fall-back to AVX2
      ifneq (0,$(PGI))
        #DFLAGS += -D__AVX2__ -D__FMA__ -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__SSSE3__ -D__SSE3__
        CTARGET = -tp=haswell
      endif
      CTARGET ?= -march=core-avx2
    endif
  else ifeq (1,$(AVX))
    ifneq (0,$(PGI))
      #DFLAGS += -D__AVX__ -D__SSE4_2__ -D__SSE4_1__ -D__SSSE3__ -D__SSE3__
      CTARGET = -tp=sandybridge
    endif
    CTARGET ?= -mavx
  else ifneq (0,$(SSE))
    ifeq (1,$(SSE)) # default
      ifneq (0,$(PGI))
        #DFLAGS += -D__SSE4_2__ -D__SSE4_1__ -D__SSSE3__ -D__SSE3__
        CTARGET = -tp=nehalem
      endif
      CTARGET ?= -msse4.2
    else ifeq (2,$(SSE))
      # implicitly present (64-bit ABI)
    else
      ifeq (3,$(SSE))
        ifneq (0,$(PGI))
          #DFLAGS += -D__SSSE3__ -D__SSE3__
          CTARGET = -tp=penryn
        endif
        ifneq (Darwin,$(UNAME))
          CTARGET ?= -msse3
        else # prevents Clang BE error (CRC32 and others)
          CTARGET ?= -msse4.2
        endif
      else ifeq (4,$(SSE))
        ifneq (0,$(PGI))
          #DFLAGS += -D__SSE4_2__ -D__SSE4_1__ -D__SSSE3__ -D__SSE3__
          CTARGET = -tp=nehalem
        endif
        CTARGET ?= -msse4.2
      else # perhaps better to use TARGET flag directly
        CTARGET = -msse$(SSE)
      endif
    endif
  else ifneq (0,$(AVX))
    CTARGET = -march=native
  endif
  endif
endif
endif

# take user's TARGET into account
ifneq (undefined,$(origin TARGET))
  CTARGET = $(TARGET)
endif

# check target flags by feeding an ad-hoc C program into the compiler
ifneq (0,$(shell CINFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX-c.c); \
  echo "int main(void) { return 0; }" > $${CINFILE}; \
  CTEST=$$($(CC) $(CTARGET) -c $${CINFILE} 2>&1); \
  if [ "" = "$${CTEST}" ]; then echo "$$?"; else echo "1"; fi; \
  rm -f /tmp/.libxsmm_??????-c.c .libxsmm_??????-c.o* 2>/dev/null))
# revoke target flags
  CTARGET = $(NULL)
endif

ifneq (,$(strip $(FC))) # check target flags by feeding an ad-hoc Fortran program into the compiler
ifeq (0,$(shell FINFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX-f.f); \
  echo "      PROGRAM test" > $${FINFILE}; \
  echo "      END PROGRAM" >> $${FINFILE}; \
  FTEST=$$($(FC) $(CTARGET) -c $${FINFILE} 2>&1); \
  if [ "" = "$${FTEST}" ]; then echo "$$?"; else echo "1"; fi; \
  rm -f /tmp/.libxsmm_??????-f.f .libxsmm_??????-f.o* 2>/dev/null))
# inherit CTARGET flags
  FTARGET = $(CTARGET)
endif
endif

ifeq (0,$(COMPATIBLE))
  ifneq (0,$(OPT))
  ifeq (0,$(INTEL))
    ifneq (0,$(SIMD))
      ifneq (0,$(GCC))
        ifneq (,$(CTARGET))
        ifneq (0,$(shell echo "$$((40900<=$(CC_VERSION)))"))
          DFLAGS += -DLIBXSMM_OPENMP_SIMD
          CFLAGS += -fopenmp-simd
          ifneq (1,$(GCC))
          ifneq (0,$(shell echo "$$((40900<=$(CXX_VERSION)))"))
            CXXFLAGS += -fopenmp-simd
            ifneq (,$(FTARGET))
            ifneq (2,$(GCC))
            ifneq (0,$(shell echo "$$((40900<=$(FC_VERSION)))"))
              FCFLAGS += -fopenmp-simd
            endif
            endif
            endif
          endif
          endif
        endif
        endif
      else ifneq (0,$(CLANG))
        ifneq (,$(strip $(SIMD))) # explicit
        ifneq (Darwin,$(UNAME))
        ifneq (,$(CTARGET))
        ifneq (0,$(shell echo "$$((60000<=$(CC_VERSION)))"))
        ifneq (0,$(shell echo "$$((60000<=$(CXX_VERSION)))"))
          DFLAGS += -DLIBXSMM_OPENMP_SIMD
          CXXFLAGS += -fopenmp-simd
          CFLAGS += -fopenmp-simd
          ifneq (,$(FTARGET))
          ifneq (,$(strip $(FC)))
          ifneq (0,$(shell echo "$$((40900<=$(FC_VERSION)))"))
            FCFLAGS += -fopenmp-simd
          endif
          endif
          endif
        endif
        endif
        endif
        endif
        endif
      endif
    endif # SIMD
    ifneq (,$(CTARGET))
      CXXFLAGS += -ftree-vectorize
      CFLAGS += -ftree-vectorize
    endif
    ifneq (,$(CTARGET))
      FCFLAGS += -ftree-vectorize
    endif
  endif
  endif
  ifneq (Darwin,$(UNAME))
    ifneq (0,$(HARDEN)) # not defined: enabled
      ifneq (,$(strip $(HARDEN))) # explicit
        ifneq (0,$(shell echo "$$((40900<=$(FC_VERSION)))"))
          CXXFLAGS += -fstack-protector-strong
          CFLAGS += -fstack-protector-strong
        else
          CXXFLAGS += -fstack-protector
          CFLAGS += -fstack-protector
        endif
        #DFLAGS += -D_FORTIFY_SOURCE=2
      #else
        #DFLAGS += -D_FORTIFY_SOURCE=1
      endif
    else
      CXXFLAGS += -fno-stack-protector
      CFLAGS += -fno-stack-protector
    endif
    ifneq (Windows_NT,$(UNAME))
      # Linux distributions may apply similar hardening
      LDFLAGS += -Wl,-z,relro,-z,now
      ifneq (0,$(DEPSTATIC))
        ifneq (0,$(HARDEN))
          ifneq (,$(strip $(HARDEN))) # explicit
            LDFLAGS += -Wl,--export-dynamic
          else ifneq (0,$(SYM))
            LDFLAGS += -Wl,--export-dynamic
          endif
        else ifneq (0,$(SYM))
          LDFLAGS += -Wl,--export-dynamic
        endif
      endif
    endif
    # flag -fvisibility=hidden is causing crashes (assertions during validation)
    CXXFLAGS += -fdata-sections -ffunction-sections #-fvisibility=hidden -fvisibility-inlines-hidden
    CFLAGS += -fdata-sections -ffunction-sections #-fvisibility=hidden
    FCFLAGS += -fdata-sections -ffunction-sections #-fvisibility=hidden
    # --gc-sections: relies on section-flags present at compile-stage
    LDFLAGS += -Wl,--gc-sections
  endif
  ifeq (0,$(EXP))
    CXXFLAGS += -fno-exceptions
  endif
endif
ifneq (Windows_NT,$(UNAME))
ifneq (,$(strip $(PIC)))
  PICFLAG = -f$(PIC)
endif
endif

ifneq (0,$(THREADS))
  ifeq (0,$(COMPATIBLE))
    CXXFLAGS += -pthread
    CFLAGS += -pthread
    ifneq (0,$(shell echo "$$((0==$(INTEL) || 0!=$(DEPSTATIC)))"))
      FLDFLAGS += -lpthread
      ifneq (Windows_NT,$(UNAME))
        ifneq (Darwin,$(UNAME))
          CLDFLAGS += -pthread
        else ifeq (0,$(CLANG))
          CLDFLAGS += -pthread
        endif
      endif
    endif
  endif
else
  DFLAGS += -DLIBXSMM_NO_SYNC
endif
ifeq (0,$(VLA))
  DFLAGS += -DLIBXSMM_NO_VLA
endif

# Information which can be displayed by the actual Makefile
ifneq (,$(strip $(FC)))
  GINFO = $(SUITE): $(strip $(CC_NAME) $(CC_VERSION_STRING)), $(strip $(CXX_NAME) $(CXX_VERSION_STRING)), and $(strip $(FC_NAME) $(FC_VERSION_STRING))
  FINFO = Fortran target: $(if $(FTARGET),$(FTARGET),<compiler default>)
else
  GINFO = $(SUITE): $(strip $(CC_NAME) $(CC_VERSION_STRING)), and $(strip $(CXX_NAME) $(CXX_VERSION_STRING))
  FINFO = Fortran: <none>
endif
CINFO = C / C++ target: $(if $(CTARGET),$(CTARGET),<compiler default>)

ifneq (0,$(shell $(LD) $(SLDFLAGS) -lrt 2>&1 | grep -q "\-lrt"; echo "$$?"))
  LIBRT = -lrt
endif
ifneq (0,$(shell $(LD) $(SLDFLAGS) -ldl 2>&1 | grep -q "\-ldl"; echo "$$?"))
  LIBDL = -ldl
endif
ifneq (,$(strip $(LIBRT) $(LIBDL)))
  LDFLAGS += $(XLIB_BEGIN) $(LIBRT) $(LIBDL) $(XLIB_END)
endif
ifeq (Darwin,$(UNAME))
  # avoid Homebrew based GCC AS; apply the flag only to the non-GCC components
  ifneq (0,$(shell echo "$$((1<=$(GCC)))"))
    LDFLAGS += -Wa,-q
    CFLAGS += -Wa,-q
    ifneq (0,$(shell echo "$$((2<=$(GCC)))"))
      CXXFLAGS += -Wa,-q
    endif
  endif
  FLDFLAGS += -Wa,-q
  FCFLAGS += -Wa,-q
endif

ifneq (0,$(shell $(LD) $(SLDFLAGS) -l$(OMPRT) 2>&1 | grep -q "\-l$(OMPRT)"; echo "$$?"))
  OMPLIBFILE = $(call libpath,$(LD) $(SLDFLAGS),$(NULL),$(OMPRT))
  ifneq (,$(strip $(OMPLIBFILE)))
    OMPLIB = $(call abslib,$(OMPLIBFILE))
  else
    OMPLIB = -l$(OMPRT)
  endif
else
  OMPLIB = $(NULL)
endif

ifeq (0,$(OMP))
  EXTLIB += $(OMPLIB)
endif

ifeq (Darwin,$(UNAME))
ifneq (,$(OMPLIBFILE))
  ifeq (0,$(ABSLIBS))
    LDFLAGS += $(call abslibrpath,$(OMPLIBFILE))
  endif
  LDFLAGS += $(call abslib,$(OMPLIBFILE))
endif
endif

ifneq (0,$(TBB))
ifneq (,$(TBBROOT))
  ifneq (Windows_NT,$(UNAME))
    TBBLIB_DIR = $(TBBROOT)/lib/intel64
    TBBLIB_DIRGCC = gcc$(GCC_VERSION_MAJOR).$(GCC_VERSION_MINOR)
    TBBLIB_MALLOC = $(wildcard $(TBBLIB_DIR)/$(TBBLIB_DIRGCC)/libtbbmalloc.so)
    ifeq (,$(TBBLIB_MALLOC))
      ifneq (0,$(TBB_OLDRTL))
        TBBLIB_DIRGCC = $(shell ls -1 "$(TBBLIB_DIR)" | tr "\n" " " | cut -d" " -f1)
      else
        TBBLIB_DIRGCC = $(shell ls -1 "$(TBBLIB_DIR)" | tr "\n" " " | rev | cut -d" " -f2 | rev)
      endif
      TBBLIB_MALLOC = $(wildcard $(TBBLIB_DIR)/$(TBBLIB_DIRGCC)/libtbbmalloc.so)
    endif
    ifneq (,$(TBBLIB_MALLOC))
      IFLAGS += -I$(TBBROOT)/include
      DFLAGS += -D__TBB
      LDFLAGS += $(XLIB_BEGIN) $(call abslib,$(TBBLIB_MALLOC)) $(XLIB_END)
    endif
  else # TODO: Windows support
  endif
endif
endif

MAKE_ILP64 = 0
ifneq (,$(strip $(ILP64)))
ifneq (0,$(ILP64))
  MAKE_ILP64 = $(ILP64)
endif
endif
ifneq (0,$(MAKE_ILP64))
  BLAS_BITS = 64
  MKL_BITS = ilp64
else
  MKL_BITS = lp64
endif

ifneq (,$(strip $(MKLROOT)))
  MKL ?= $(BLAS)
  ifneq (,$(strip $(FC)))
    ifneq (0,$(shell echo "$$((0==$(GCC) || 40600<=$(GCC_VERSION)))"))
      MKL_DIRECT ?= 0
    else
      MKL_DIRECT = 0
    endif
  else
    MKL_DIRECT = 0
  endif
else
  MKL = 0
endif
ifneq (0,$(MKL))
  BLAS_FLAGS += -D__CBLAS
endif
ifeq (1,$(MKL_DIRECT))
  ifeq (1,$(MKL))
    BLAS_FLAGS += -DMKL_DIRECT_CALL_SEQ
  else ifneq (0,$(MKL))
    BLAS_FLAGS += -DMKL_DIRECT_CALL
  endif
  ifneq (0,$(GCC))
    CXXFLAGS += -Wno-unused-value
	CFLAGS += -Wno-unused-value
  endif
endif
ifneq (Darwin,$(UNAME))
  MKL_ARCH = intel64
endif

ifeq (1,$(MKL)) # sequential
  BLAS_FLAGS += -D__BLAS=1 -D__MKL=1
  BLAS_IFLAGS += -I$(MKLROOT)/include
  ifeq (0,$(BLAS_STATIC)) # shared
    BLAS_LDFLAGS += -L$(MKLROOT)/lib/$(MKL_ARCH) -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_sequential
    ifneq (0,$(OFFLOAD))
      BLAS_LDFLAGS += -qoffload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_sequential"
    endif
  else # static
    BLAS_LDFLAGS += $(XGROUP_BEGIN) \
      $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
      $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_core.$(SLIBEXT) \
      $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_sequential.$(SLIBEXT) \
    $(XGROUP_END)
    ifneq (0,$(OFFLOAD))
      BLAS_LDFLAGS += -qoffload-option,mic,ld,"--start-group \
        $(MKLROOT)/lib/mic/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
        $(MKLROOT)/lib/mic/libmkl_core.$(SLIBEXT) \
        $(MKLROOT)/lib/mic/libmkl_sequential.$(SLIBEXT) \
      --end-group"
    endif
  endif
  ifneq (0,$(THREADS))
    BLAS_LDFLAGS += -lm $(LIBDL)
  endif
  BLAS_LDFLAGS += -lm $(LIBDL)
  ifneq (0,$(OFFLOAD))
    BLAS_LDFLAGS += -qoffload-option,mic,ld,"-lm $(LIBDL)"
  endif
else ifneq (0,$(MKL)) # multi-threaded
  BLAS_FLAGS += -D__BLAS=$(MKL) -D__MKL=$(MKL)
  BLAS_IFLAGS += -I$(MKLROOT)/include
  ifeq (0,$(BLAS_STATIC)) # shared
    ifneq (0,$(INTEL))
      BLAS_LDFLAGS += -L$(MKLROOT)/lib/$(MKL_ARCH) -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_intel_thread
      ifneq (0,$(OFFLOAD))
        BLAS_LDFLAGS += -qoffload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_intel_thread"
      endif
    else # assuming GNU toolchain
      BLAS_LDFLAGS += -L$(MKLROOT)/lib/$(MKL_ARCH) -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_gnu_thread
    endif
  else # static
    ifneq (0,$(INTEL))
      BLAS_LDFLAGS += $(XGROUP_BEGIN) \
        $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
        $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_core.$(SLIBEXT) \
        $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_intel_thread.$(SLIBEXT) \
      $(XGROUP_END)
      ifneq (0,$(OFFLOAD))
        BLAS_LDFLAGS += -qoffload-option,mic,ld,"--start-group \
          $(MKLROOT)/lib/mic/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
          $(MKLROOT)/lib/mic/libmkl_core.$(SLIBEXT) \
          $(MKLROOT)/lib/mic/libmkl_intel_thread.$(SLIBEXT) \
        --end-group"
      endif
    else # assuming GNU toolchain
      BLAS_LDFLAGS += $(XGROUP_BEGIN) \
        $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
        $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_core.$(SLIBEXT) \
        $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_gnu_thread.$(SLIBEXT) \
      $(XGROUP_END) $(LIBDL)
    endif
  endif
  ifeq (0,$(OMP))
    BLAS_LDFLAGS += $(OMPLIB)
  endif
  ifneq (0,$(THREADS))
    BLAS_LDFLAGS += -lpthread
  endif
  BLAS_LDFLAGS += -lm $(LIBDL)
  ifneq (0,$(INTEL))
  ifneq (0,$(OFFLOAD))
    BLAS_LDFLAGS += -qoffload-option,mic,ld,"-lm $(LIBDL)"
    ifeq (0,$(OMP))
      BLAS_LDFLAGS += -qoffload-option,mic,ld,"$(OMPLIB)"
    endif
  endif
  endif
else ifneq (0,$(BLAS)) # generic
  ifeq (,$(strip $(BLASLIB)))
    ifneq (1,$(BLAS))
      ifneq (0,$(OMP))
        BLAS_THREADS = o
      else
        BLAS_THREADS = p
      endif
    endif
    ifneq (,$(BLASDIR))
      BLASROOT = $(BLASDIR)/..
    else
      BLASROOT = /usr
    endif
    ifeq (0,$(BLAS_STATIC)) # shared
      BLASLIBFILE = $(call libpath,$(LD) $(SLDFLAGS),$(BLASDIR),openblas$(BLAS_THREADS)$(BLAS_BITS))
      ifeq (,$(BLASLIBFILE)) # newer distributions symlink a non-decorated library to threaded OpenBLAS
        BLASLIBFILE = $(call libpath,$(LD) $(SLDFLAGS),$(BLASDIR),openblas$(BLAS_BITS))
      endif
    else # static
      BLASLIBFILE = $(call libpath,$(LD) -static,$(BLASDIR),openblas$(BLAS_THREADS)$(BLAS_BITS))
      ifeq (,$(BLASLIBFILE)) # newer distributions symlink a non-decorated library to threaded OpenBLAS
        BLASLIBFILE = $(call libpath,$(LD) -static,$(BLASDIR),openblas$(BLAS_BITS))
      endif
    endif
    # most people expect to pickup OpenBLAS (if available) even when libblas/liblapack are available as well; use OPENBLAS=0 to avoid this
    OPENBLAS := $(if $(BLASLIBFILE),1,0)

    ifneq (0,$(OPENBLAS)) # OpenBLAS
      # OpenBLAS also carries the CBLAS bits
      BLAS_FLAGS += -D__BLAS=$(BLAS) -D__CBLAS -D__OPENBLAS
      ifneq (,$(wildcard $(BLASROOT)/include/openblas/f77blas.h))
        BLAS_FLAGS += -D__OPENBLAS77 -I$(BLASROOT)/include/openblas
      else ifneq (,$(wildcard $(BLASROOT)/include/x86_64-linux-gnu/f77blas.h))
        BLAS_FLAGS += -D__OPENBLAS77 -I$(BLASROOT)/include/x86_64-linux-gnu
      endif
    else # BLAS (reference)
      BLAS_FLAGS += -D__BLAS
      ifeq (Windows_NT,$(UNAME)) # no particular action about static linkage (use DLL)
        BLASLIBFILE = $(call libpath,$(LD) $(SLDFLAGS),$(BLASDIR),blas$(BLAS_BITS).dll)
      else ifeq (0,$(BLAS_STATIC)) # potentially shared
        BLASLIBFILE = $(call libpath,$(LD) $(SLDFLAGS),$(BLASDIR),blas$(BLAS_BITS))
      else # static
        BLASLIBFILE = $(call libpath,$(LD) -static,$(BLASDIR),blas$(BLAS_BITS))
      endif
    endif
    ifneq (,$(BLASLIBFILE))
      ifeq (0,$(BLAS_STATIC)) # shared
        BLAS_LDFLAGS += $(call abslib,$(BLASLIBFILE))
      else # static
        BLAS_LDFLAGS += $(BLASLIBFILE)
      endif
    else # fall-back
      BLAS_LDFLAGS += -lblas
    endif
    ifeq (0,$(OPENBLAS)) # BLAS (reference)
      ifneq (0,$(DEPSTATIC))
      ifneq (,$(LIBGFORTRAN))
        BLAS_CLDFLAGS += $(XLIB_BEGIN) $(call abslibfile,$(LIBGFORTRAN)) -lm $(XLIB_END)
      endif
      endif
      QUADMATH = $(shell $(LD) $(SLDFLAGS) -lquadmath 2>&1 | grep -q "\-lquadmath"; echo "$$?")
      ifneq (0,$(QUADMATH))
        BLAS_CLDFLAGS += $(XLIB_BEGIN) -lquadmath -lm $(XLIB_END)
      endif
    endif
  else # BLAS library is specified via BLASLIB
    BLAS_FLAGS += -D__BLAS=$(BLAS)
    ifneq (./,$(BLASDIR)$(dir $(BLASLIB)))
      ifeq (./,$(dir $(BLASLIB)))
        BLAS_LDFLAGS += $(call abslib,$(BLASDIR)/$(if $(suffix $(BLASLIB)),$(BLASLIB),lib$(BLASLIB).$(BLASLIBEXT)))
      else
        BLAS_LDFLAGS += $(call abslib,$(if $(suffix $(BLASLIB)),$(BLASLIB),$(BLASLIB).$(BLASLIBEXT)))
      endif
    else # fallback
      BLAS_LDFLAGS += -l$(BLASLIB)
    endif
  endif
endif

# no-BLAS flags: cleanup and extra flags
NOBLAS_CXXFLAGS := $(strip $(PICFLAG) $(CXXFLAGS) $(ECXXFLAGS) $(EFLAGS))
NOBLAS_FCFLAGS := $(strip $(PICFLAG) $(FCFLAGS) $(EFCFLAGS) $(EFLAGS))
NOBLAS_CFLAGS := $(strip $(PICFLAG) $(CFLAGS) $(ECFLAGS) $(EFLAGS))
NOBLAS_LDFLAGS := $(strip $(PICFLAG) $(LDFLAGS) $(ELDFLAGS))
NOBLAS_CLDFLAGS := $(strip $(CLDFLAGS))
NOBLAS_FLAGS := $(strip $(DFLAGS))
NOBLAS_IFLAGS := $(strip $(IFLAGS))

# regular flags: cleanup
CXXFLAGS := $(strip $(NOBLAS_CXXFLAGS) $(BLAS_CXXFLAGS))
FCFLAGS := $(strip $(NOBLAS_FCFLAGS) $(BLAS_FCFLAGS))
CFLAGS := $(strip $(NOBLAS_CFLAGS) $(BLAS_CFLAGS))
LDFLAGS := $(strip $(NOBLAS_LDFLAGS) $(BLAS_LDFLAGS))
CLDFLAGS := $(strip $(NOBLAS_CLDFLAGS) $(BLAS_CLDFLAGS))
DFLAGS := $(strip $(NOBLAS_FLAGS) $(BLAS_FLAGS))
IFLAGS := $(strip $(NOBLAS_IFLAGS) $(BLAS_IFLAGS))
# cleanup eventually duplicated flags
LDFLAGS := $(subst $(NULL) $(XLIB_END) $(XLIB_BEGIN),$(NULL),$(LDFLAGS))
CLDFLAGS := $(subst $(NULL) $(XLIB_END) $(XLIB_BEGIN),$(NULL),$(CLDFLAGS))
# cleanup duplicated slashes
LDFLAGS := $(subst //,/,$(LDFLAGS))
CLDFLAGS := $(subst //,/,$(CLDFLAGS))
IFLAGS  := $(subst //,/,$(IFLAGS))

.PRECIOUS: $(BLDDIR)/%-cpp.o $(BLDDIR)/%-c.o $(BLDDIR)/%-f.o \
           $(BLDDIR)/%-f90.o $(BLDDIR)/%-f90.o $(BLDDIR)/%-f77.o \
           %/.make

.SUFFIXES:

# derives the extension of a filename
extname = $(subst .,,$(suffix $(1)))
# derives the name of an object files from a given source file
objname = $(foreach ARG, $(1),$(addprefix $(BLDDIR)/, $(patsubst %$(suffix $(ARG)),%-$(call extname,$(ARG)).o,$(notdir $(ARG)))))

STATE := $(foreach V,$(sort $(.VARIABLES)), $(if \
  $(filter-out environment% default automatic,$(origin $V)), \
  $(if $(filter-out .SHELLSTATUS .DEFAULT_GOAL MAKEFILE_LIST MAKEOVERRIDES MAKEFLAGS SHELL \
    $(EXCLUDE_STATE) CPUFLAGS TEST TESTSIZE BRACE_OPEN BRACE_CLOSE FLOCK MKTEMP \
    abslibfile abslibpath absliblpath abslibrpath \
    libpath abslib solink extname objname,$V), \
    $V=$($V)?)))
.state: $(shell echo '$(STATE)' | $(DEPDIR)/.state.sh)

%/.make:
	@mkdir -p $(basename $@)
	@touch $@

.make:
	@touch $@

