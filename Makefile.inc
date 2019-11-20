#############################################################################
# Copyright (c) 2015-2019, Intel Corporation                                #
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

MAKE_VERSION_MAJOR = $(shell echo "$(MAKE_VERSION)" | cut -d"." -f1)
MAKE_VERSION_MINOR = $(shell echo "$(MAKE_VERSION)" | cut -d"." -f2)
MAKE_VERSION_PATCH = $(shell echo "$(MAKE_VERSION)" | cut -d"." -f3)
ifeq (,$(MAKE_VERSION_MAJOR))
MAKE_VERSION_MAJOR = 0
endif
ifeq (,$(MAKE_VERSION_MINOR))
MAKE_VERSION_MINOR = 0
endif
ifeq (,$(MAKE_VERSION_PATCH))
MAKE_VERSION_PATCH = 0
endif
MAKE_VERSION_INT = $(shell echo "$$(($(MAKE_VERSION_MAJOR)*10000+$(MAKE_VERSION_MINOR)*100+$(MAKE_VERSION_PATCH)))")

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explicitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(strip $(MAKE_PARALLEL)))
ifneq (0,$(shell echo "$$((38200>$(MAKE_VERSION_INT)))"))
.NOTPARALLEL:
else ifneq (0,$(shell echo "$$((40000<=$(MAKE_VERSION_INT)))"))
MAKEFLAGS += -O
endif
else ifneq (0,$(shell echo "$$((40000<=$(MAKE_VERSION_INT)))"))
MAKEFLAGS += -O
endif

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

which = $(shell which $1 2>/dev/null)
which ?= $(shell command -v $1)

# Command line utilities
#PKGCFG ?= $(call which,pkg-config)
CP ?= $(call which,cp)
MV ?= $(call which,mv)
UNAME ?= $(shell uname)
MAKE ?= make
ENV ?= env

# Python interpreter per PYTHON=/path/to/python
PYTHON3 ?= $(call which,python3)
PYTHON ?= $(call which,python)

ifeq (,$(PYTHON))
ifneq (,$(PYTHON3))
  SHELL := $(ENV) PATH=$(MAKEINC):$(PATH) $(SHELL)
  PYTHON = $(shell ln -s $(PYTHON3) $(MAKEINC)/python 2>/dev/null; command -v python)
endif
endif

ifneq (Darwin,$(UNAME))
  ifneq (,$(strip $(CP)))
  ifneq (FreeBSD,$(UNAME))
    CP += -u
  endif
  endif
  DLIBEXT ?= so
  SLIBEXT ?= a
else
  DLIBEXT ?= dylib
  SLIBEXT ?= a
endif

# Regular expression to match "main" (good-enough pattern)
CMAIN = "main[[:space:]]*(.*)"
FMAIN = "^[[:space:]]*PROGRAM[[:space:]][[:space:]]*\w\w*\([[:space:]][[:space:]]*\!.*\)*$$"

# Debugging and symbols (e.g., when profiling)
SYM ?= 0
DBG ?= 0

# Instrumentation level (trace)
ifeq (,$(strip $(INSTRUMENT)))
  INSTRUMENT = 0
endif
TRACE ?= 0

ifeq (0,$(DBG))
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
  SYM = $(DBG)
endif

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

# Avoid more sophisticated flags of the GCC tool chain,
# and improve compatibility with compilers supposed to be
# compatible with the GCC tool chain
COMPATIBLE ?= 0

# Control visibility of symbols
# 0: hidden unless explicitly marked visible
# 1: default visibility
VISIBILITY ?= 0

# Number of repeated calls (tests),
# or used to scale the problem size
TESTSIZE ?= 1

# PYMOD=1: enable Python module development
PYMOD ?= 0

# Static or shared binary
STATIC ?= 0

# PIC: PIC or pic
PIC ?= pic

PLATFORM ?= 0
ifneq (0,$(PLATFORM))
  DFLAGS += -DLIBXSMM_PLATFORM_FORCE
endif

# Intrinsics support level
INTRINSICS ?= 2
ifneq (0,$(INTRINSICS))
  ifeq (1,$(INTRINSICS))
    ifeq (0,$(INTEL))
      DFLAGS += -DLIBXSMM_INTRINSICS_STATIC
    else ifneq (,$(CTARGET))
      DFLAGS += -DLIBXSMM_INTRINSICS_STATIC
    endif
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

# Warning about unused functions
UNUSED ?= 0

# Embed InterProcedural Optimization information into libraries
IPO ?= 0
FAT ?= 0

# ILP64=0 (LP64 with 32-bit integers), and ILP64=0 (64-bit integers)
ILP64 ?= 0

# TBB enabled (1) or disabled (0)
# availability depends on TBBROOT
TBB ?= 0
# TBB runtime compatible with oldest supported GCC
TBB_OLDRTL ?= 0

# Enable absolute library paths
ABSLIBS ?= 0

REVERSION := "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/"
BRACE_OPEN := (
BRACE_CLOSE := )

libpath = $(strip $(subst / $(NULL), $(NULL),$(if $1, \
  $(if $(shell $1 $2 -l$3 2>&1 | grep "\-l$3"),$(NULL), \
    $(subst //,/,$(abspath $(shell $1 $2 -l$3 -Wl,--verbose 2>&1 \
    | grep "lib$3" | sed "s/[^./]\(\.*\/..*[^[:space:]$(BRACE_CLOSE)]\).*/:\1/" \
    | rev | cut -d":" -f1 | rev | sed -n "s/[[:space:]]*\(\.*\/..*\)/\1/p" \
    | cut -d" " -f1 | tail -n1)))), \
  $(NULL))))
abslib = $(strip $(subst / $(NULL), $(NULL),$(call abslibpath,$1) $(call abslibfile,$1)))

# Embedd soname into shared library
SONAMELNK ?= 2

# Pickup OpenMP library name if passed as OMP=libname|name
ifneq (0,$(shell echo "$(OMP)" | grep -q "^-*[0-9]\+$$"; echo "$$?")) # NaN
  OMPRT = $(patsubst lib%,%,$(OMP))
endif

ifneq (0,$(PYMOD))
ifneq (,$(PYTHON))
  PYVERSION_STRING = $(shell $(PYTHON) --version 2>&1 | head -n1 | sed $(REVERSION))
  PYVERSION = $(shell echo "$(PYVERSION_STRING)" | cut -d"." -f1,2)
  PYROOT = $(abspath $(dir $(call which,$(PYTHON)))/..)
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

LNKSOFT ?= 1
ifeq (0,$(STATIC))
  ifeq (Windows_NT,$(UNAME))
    LNKSOFT = 0
  else ifeq (Darwin,$(UNAME))
    LNKSOFT = 0
  endif
endif

# BLAS is not used by default
ifneq (0,$(LNKSOFT))
  BLAS ?= 0
else
  BLAS ?= 2
endif

# Automatically pickup the environment (make -e is not required),
# or pickup the Intel Compiler (if available).
GNU ?= 0
ifeq (0,$(GNU))
  ifeq (,$(shell echo "$${CXX}"))
    ifneq (,$(notdir $(call which,icpx)))
      CXX = icpx
    else ifneq (,$(notdir $(call which,icpc)))
      CXX = icpc
    endif
  endif
  ifeq (,$(shell echo "$${CC}"))
    ifneq (,$(notdir $(call which,icx)))
      CC = icx
    else ifneq (,$(notdir $(call which,icc)))
      CC = icc
    endif
  endif
  ifeq (__INTEL_COMPILER,$(shell echo "__INTEL_COMPILER" | $(CC) $(ECFLAGS) $(EFLAGS) -E -P - 2>/dev/null))
    ICX = 1
  endif
endif
ICX ?= 0

# adopt extra flags from C if not set individually
ECXXFLAGS ?= $(ECFLAGS)

COMPILER_VERSION_FLAG ?= $(ECXXFLAGS) $(EFLAGS) --version 2>/dev/null
CC_VERSION_FLAG ?= $(COMPILER_VERSION_FLAG)
FC_VERSION_FLAG ?= $(COMPILER_VERSION_FLAG)
CXX_VERSION_FLAG ?= $(CC_VERSION_FLAG)

# check if the Intel Development Tools are available
INTEL ?= $(shell echo "$$((2==$(words $(filter icpc% icpx% icc% icx%, \
  $(shell $(CXX) $(CXX_VERSION_FLAG) 2>/dev/null | grep -m1 . | cut -d' ' -f1) \
  $(shell $(CC)  $(CC_VERSION_FLAG)  2>/dev/null | grep -m1 . | cut -d' ' -f1)))))")

ifneq (0,$(INTEL))
  ifeq (,$(shell echo "$${FC}"))
    ifneq (,$(notdir $(call which,ifort)))
      FC = ifort
    else ifneq (,$(notdir $(call which,ifx)))
      FC = ifx
    endif
  endif
else
  ifeq (,$(call which,$(CXX)))
    CXX = g++
  else ifneq (0,$(shell $(CXX) $(CXX_VERSION_FLAG) >/dev/null 2>/dev/null; echo "$$?"))
    CXX = g++
  else ifneq (gcc,$(notdir $(call which,$(CC))))
    ifeq (g++,$(notdir $(call which,$(CXX))))
      CC = gcc
    endif
  endif
  ifeq (,$(call which,$(CC)))
    CC = gcc
  else ifneq (0,$(shell $(CC) $(CC_VERSION_FLAG) >/dev/null 2>/dev/null; echo "$$?"))
    CC = gcc
  endif
  ifeq (Cray,$(shell $(CC) -V 2>&1 | head -n1 | cut -d' ' -f1))
    COMPILER_VERSION_FLAG ?= -V 2>&1
  endif
endif

CXX_VERSION_STRING = $(shell $(CXX) $(CXX_VERSION_FLAG) | grep -m1 . | sed $(REVERSION))
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

CC_VERSION_STRING = $(shell $(CC) $(CC_VERSION_FLAG) | grep -m1 . | sed $(REVERSION))
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

# disable Fortran per user-request
ifeq (0,$(FORTRAN))
  override FC = $(NULL)
endif

# fixup FC-default given by MAKE
ifneq (,$(strip $(FC)))
ifeq (0,$(INTEL))
  FIXFC ?= 0
  ifeq (,$(call which,$(FC)))
    FIXFC = 1
  else ifneq (0,$(shell $(FC) $(FC_VERSION_FLAG) >/dev/null 2>/dev/null; echo "$$?"))
    FIXFC = 1
  else ifneq (gfortran,$(notdir $(call which,$(FC))))
    ifeq (g++,$(notdir $(call which,$(CXX))))
      FIXFC = 1
    endif
  endif
  ifneq (0,$(FIXFC))
    ifneq (,$(notdir $(call which,gfortran)))
      MKL_FCRTL = gf
      GFC = gfortran
      FC = $(GFC)
    else
      FC = $(NULL)
    endif
  endif
endif
endif

ifneq (,$(strip $(FC)))
  FC_VERSION_STRING := $(shell $(FC) $(FC_VERSION_FLAG) | grep -m1 . | sed $(REVERSION))
  FC_VERSION_MAJOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f1)
  FC_VERSION_MINOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f2)
  FC_VERSION_PATCH = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f3)
  ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
    FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100+$(FC_VERSION_PATCH)))")
  else ifeq (2,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR)))
    FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100))")
    FC_VERSION_PATCH = 0
  else
    FC_VERSION_STRING = $(NULL)
    FC_VERSION = 0
  endif
  ifeq (GNU,$(shell $(FC) $(FC_VERSION_FLAG) | grep -m1 . | cut -d" " -f1))
    ifneq (0,$(shell echo "$$((40500>$(FC_VERSION)))"))
      ifneq (gfortran,$(notdir $(FC)))
        FC = gfortran
        FC_VERSION_STRING := $(shell $(FC) $(FC_VERSION_FLAG) | grep -m1 . | sed $(REVERSION))
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
          override FC = $(NULL)
        endif
      else
        override FC = $(NULL)
      endif
    endif
  else ifneq (0,$(INTEL))
    ifneq (0,$(shell echo "$$((130000>$(FC_VERSION) && 0<$(FC_VERSION)))"))
      override FC = $(NULL)
    endif
  endif
endif

ifeq (,$(strip $(FC)))
  ifeq (0,$(FORTRAN))
    FC_VERSION_STRING = $(NULL)
  else # keep FC_VERSION_STRING for message about outdated compiler
    FORTRAN = 0
  endif
endif

# compiler names
ifeq (0,$(PLATFORM))
  CXX_NAME = $(basename $(shell $(CXX) $(CXX_VERSION_FLAG) | grep -m1 . \
           | sed -e "s/^\([^0-9][^0-9]*\) ..*/\1/" -e "s/[[:space:]][[:space:]]*[Vv]ersion//" \
                 -e "s/[[:space:]][[:space:]]*$(BRACE_OPEN)..*//" -e "s/[[:space:]][[:space:]]*[[:punct:]]//" \
                 -e "s/[[:space:]][[:space:]]*[0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*//" \
           | rev | cut -d" " -f1 | rev))
endif
ifeq (,$(strip $(CXX_NAME)))
  CXX_NAME = $(basename $(notdir $(CXX)))
endif
ifeq (0,$(PLATFORM))
  CC_NAME = $(basename $(shell $(CC) $(CC_VERSION_FLAG) | grep -m1 . \
           | sed -e "s/^\([^0-9][^0-9]*\) ..*/\1/" -e "s/[[:space:]][[:space:]]*[Vv]ersion//" \
                 -e "s/[[:space:]][[:space:]]*$(BRACE_OPEN)..*//" -e "s/[[:space:]][[:space:]]*[[:punct:]]//" \
                 -e "s/[[:space:]][[:space:]]*[0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*//" \
          | rev | cut -d" " -f1 | rev))
endif
ifeq (,$(strip $(CC_NAME)))
  CC_NAME = $(basename $(notdir $(CC)))
endif
ifneq (,$(strip $(FC)))
  FC_NAME = $(firstword $(notdir $(FC)))
endif
ifeq (,$(strip $(FC_NAME)))
  FC_NAME = $(basename $(notdir $(FC)))
endif

ifneq (0,$(FORTRAN))
  ifeq (,$(strip $(GFC)))
    ifneq (,$(strip $(FC)))
      ifeq (GCC,$(shell $(FC) $(FC_VERSION_FLAG) | grep -m1 . | sed "s/.* (\(..*\)) .*/\1/"))
        GFC = $(FC)
      else ifeq (0,$(shell $(FC) $(FC_VERSION_FLAG) | grep -q "Free Software Foundation"; echo "$$?"))
        GFC = $(FC)
      else ifneq (pgfortran,$(CC_NAME))
        ifneq (,$(findstring gfortran,$(FC_NAME)))
          GFC = $(FC)
        endif
      endif
    else
      FORTRAN = 0
    endif
  endif
else
  FC = $(NULL)
endif
ifeq (,$(strip $(FC)))
  DFLAGS += -DLIBXSMM_NOFORTRAN
endif

# native GCC?
GCC ?= 0
ifeq (0,$(shell $(CC) $(CC_VERSION_FLAG) | grep -q "Free Software Foundation"; echo "$$?"))
  GCC = 1
else ifeq (GCC,$(shell $(CC) $(CC_VERSION_FLAG) | grep -m1 . | sed "s/.* (\(..*\)) .*/\1/"))
  GCC = 1
else ifneq (pgcc,$(CC_NAME))
  ifneq (,$(findstring pgcc,$(CC_NAME)))
    GCC = 1
  endif
endif
ifeq (1,$(GCC))
  ifeq (0,$(shell $(CXX) $(CXX_VERSION_FLAG) | grep -q "Free Software Foundation"; echo "$$?"))
    GCC = 2
  else ifeq (GCC,$(shell $(CXX) $(CXX_VERSION_FLAG) | grep -m1 . | sed "s/.* (\(..*\)) .*/\1/"))
    GCC = 2
  else ifneq (,$(findstring g++,$(CXX_NAME)))
    GCC = 2
  endif
  ifeq (2,$(GCC))
  ifneq (,$(strip $(GFC)))
    MKL_FCRTL = gf
    GCC = 3
  endif
  endif
endif
# Fortran runtime library
MKL_FCRTL ?= intel

# adopt fully equipped archiver
CCAR = $(call which,$(CC)-ar)
ifneq (,$(CCAR))
  ifeq (default,$(origin AR))
    AR = $(CC)-ar
  else
    AR ?= $(CC)-ar
  endif
endif

ifneq (,$(FORCE_CXX))
ifneq (0,$(FORCE_CXX))
  override CC = $(CXX)
  ifeq (0,$(shell $(CC) -E -x c++ /dev/null 2>/dev/null >/dev/null; echo "$$?"))
    override CC += -x c++
  endif
endif
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

# Library extension
ifneq (0,$(DEPSTATIC))
  LIBEXT ?= $(SLIBEXT)
else
  LIBEXT ?= $(DLIBEXT)
endif

# Import-library
ifeq (0,$(MINGW))
  ILIBEXT ?= $(DLIBEXT)
else # MinGW
  ILIBEXT ?= a
endif

# Separate control on how to link against the BLAS library
BLAS_STATIC ?= $(DEPSTATIC)

# PGI: ignore env. variable (same name)
override PGI = 0
ifeq (0,$(GCC)) # not GCC
ifeq (0,$(INTEL)) # not Intel
ifeq (0,$(shell $(CC) -dM -E - < /dev/null 2>/dev/null | grep -q "__clang__"; echo "$$?"))
  CLANG = 1
else
  override PGI = $(shell $(CC) $(CC_VERSION_FLAG) | if grep -q "PGI"; then echo "1"; else echo "0"; fi)
endif
endif
endif
CLANG ?= 0

# Make GCC version number available even when not using GCC
ifneq (0,$(GCC))
  GCC_VERSION_STRING = $(CXX_VERSION_STRING)
  GCC_VERSION_MAJOR = $(CXX_VERSION_MAJOR)
  GCC_VERSION_MINOR = $(CXX_VERSION_MINOR)
  GCC_VERSION_PATCH = $(CXX_VERSION_PATCH)
else ifeq (0,$(CLANG))
  GCCBIN = $(notdir $(call which,gcc))
  ifneq (,$(strip $(GCCBIN)))
    GCC_VERSION_STRING = $(shell $(GCCBIN) $(CXX_VERSION_FLAG) | grep -m1 . | sed $(REVERSION))
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
SSE ?= 1
ifeq (0,$(SSE)) # discover AVX
  ifeq (1,$(words $(filter avx512f,$(CPUFLAGS))))
    AVX ?= 3
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
  endif
else ifeq (1,$(SSE)) # discover SSE
  ifeq (1,$(words $(filter sse4_2 sse4.2,$(CPUFLAGS))))
    SSE = 4
  else ifneq (0,$(words $(filter sse3 ssse3,$(CPUFLAGS))))
    SSE = 3
  else ifneq (,$(CPUFLAGS))
    SSE = 0
  endif
else ifneq (0,$(KNC))
  MPSS = 1
endif
AVX ?= 0

ifneq (0,$(INTEL))
  SUITE = Intel Compiler
  MKL_OMPRTL = intel
else ifneq (0,$(GCC))
  SUITE = GNU Compiler Collection
  MKL_OMPRTL = gnu
else ifneq (0,$(PGI))
  SUITE = $(if $(filter-out 0,$(PGI)),PGI $(NULL),$(NULL))Compiler
  MKL_OMPRTL = pgi
else
  ifeq (0,$(CLANG))
    COMPATIBLE = 1
  endif
  ifneq (0,$(COMPATIBLE))
  ifeq (Cray,$(shell $(CC) -V 2>&1 | head -n1 | cut -d' ' -f1))
    SUITE = Cray Compiler
    LDFLAGS += -hsystem_alloc
    CRAY ?= 1
    # prevent codegen issues
    #SYM = 1
  endif
  endif
endif
SUITE ?= Compiler
MKL_OMPRTL ?= gnu
CRAY ?= 0
PGI ?= 0

# linker setup
LD = $(CC)
XLD = $(CXX)

ifeq (0,$(DEPSTATIC))
  LIB_LD := $(LD) -shared $(PICFLAG)
  LIB_XLD := $(XLD) -shared $(PICFLAG)
else
  LIB_LD := $(LD)
  LIB_XLD := $(XLD)
endif

ifeq (,$(strip $(FLD)))
ifneq (,$(strip $(FC)))
  FLD = $(FC)
  ifeq (0,$(DEPSTATIC))
    LIB_FLD := $(FLD) -shared $(PICFLAG)
  else
    LIB_FLD := $(FLD)
  endif
else # fallback
  LIB_FLD := $(LIB_LD)
  FLD := $(LD)
endif
endif

FREEFORM ?= 1
ifeq (,$(strip $(GFC)))
  ifneq (0,$(INTEL))
    ifneq (,$(strip $(LIB_FLD)))
      LIB_FLD += -nofor-main
    endif
    ifneq (0,$(FREEFORM))
      FFORM_FLAG = -free
    endif
  endif
endif
ifneq (0,$(FREEFORM))
ifeq (0,$(PGI))
  FFORM_FLAG ?= -ffree-form
endif
endif

ifneq (0,$(DEPSTATIC))
  ifeq (0,$(COMPATIBLE))
  ifneq (Darwin,$(UNAME))
  ifneq (Windows_NT,$(UNAME))
    ifneq (0,$(HARDEN))
      ifneq (,$(strip $(HARDEN))) # explicit
        DYNAMIC = 1
      else ifneq (0,$(SYM))
        DYNAMIC = 1
      endif
    else ifneq (0,$(SYM))
      DYNAMIC = 1
    endif
  endif
  endif
  endif
else
  DYNAMIC = 1
endif
DYNAMIC ?= 0

# CCE: resolve linker issue
ifneq (0,$(DYNAMIC))
ifeq (0,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c); \
  echo "int main(void) { return 0; }" > $${INFILE}; \
  RESULT=$$($(CC) -dynamic $${INFILE} -o $${INFILE}.exe 2>&1); \
  if [ "" = "$${RESULT}" ]; then echo "$$?"; else echo "1"; fi; \
  rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
    XLD := $(XLD) -dynamic
    FLD := $(FLD) -dynamic
    LD := $(LD) -dynamic
endif
endif

# Compiler is used for link stage
ifneq (Darwin,$(UNAME))
  ifneq (ld,$(notdir $(LD)))
    XLNKOPT = -Wl,
  endif
  linkopt = $(XLNKOPT)$(if $2,$1=$2,$1)
  abslibrpath = $(strip $(subst / $(NULL), $(NULL),$(if \
    $(filter $(ILIBEXT),$(SLIBEXT))$(filter-out .$(SLIBEXT),$(suffix $(firstword $1))), \
    $(call linkopt,--rpath,$(dir $(abspath $(firstword $1)))),$(NULL))))
  XGROUP_BEGIN = $(XLNKOPT)--start-group
  XGROUP_END = $(XLNKOPT)--end-group
  ifneq (0,$(ASNEEDED))
    XLIB_BEGIN = $(XLNKOPT)--as-needed
    XLIB_END = $(XLNKOPT)--no-as-needed
  endif
else # OSX
  ifneq (ld,$(notdir $(LD)))
    XLNKOPT = -Xlinker
  endif
  linkopt = $(XLNKOPT) $1 $(XLNKOPT) $2
  abslibrpath = $(strip $(subst / $(NULL), $(NULL),$(if \
    $(filter $(ILIBEXT),$(SLIBEXT))$(filter-out .$(SLIBEXT),$(suffix $(firstword $1))), \
    $(call linkopt,-rpath,$(dir $(abspath $(firstword $1)))),$(NULL))))
endif

absliblpath = $(strip $(subst / $(NULL), $(NULL), \
  $(if $(filter .$(ILIBEXT),$(suffix $(firstword $1))), \
  $(if $(firstword $1),-L$(dir $(abspath $(firstword $1))),-L$(dir $(firstword $1))), \
  $(if $(filter .$(SLIBEXT),$(suffix $(firstword $1))),$(NULL),-L$(firstword $1)))))

ifneq (Windows_NT1,$(OS)$(DEPSTATIC))
  abslibfile = $(strip $(if $(filter .$(ILIBEXT),$(suffix $(firstword $1))), \
    $(if $(patsubst lib%,%,$(basename $(notdir $(firstword $1)))), \
       -l$(patsubst lib%,%,$(basename $(notdir $(firstword $1)))),$(NULL)),$(abspath $(firstword $1))))
else # Cygwin/MinGW (static)
  abslibfile = $(strip $(if $(filter .$(ILIBEXT),$(suffix $(firstword $1))), \
    $(if $(patsubst lib%,%,$(basename $(notdir $(firstword $1)))), \
       -l$(basename $(notdir $(firstword $1))),$(NULL)),$(abspath $(firstword $1))))
endif

LIBDEP = $(LIBNAME).$(LIBEXT)
MAINLIB = $(call abslib,$(LIBDEP))

FORTDEP = $(LIBNAME)f.$(LIBEXT)
FORTLIB = $(call abslib,$(FORTDEP))

EXTDEP = $(LIBNAME)ext.$(LIBEXT)
EXTLIB = $(XLIB_BEGIN) $(call abslib,$(EXTDEP)) $(XLIB_END)

# provides libxsmmnoblas to satisfy BLAS symbols
NOBLASDEP = $(LIBNAME)noblas.$(LIBEXT)
NOBLASLIB = $(XLIB_BEGIN) $(call abslib,$(NOBLASDEP)) $(XLIB_END)

ifeq (0,$(BLAS))
  MAINLIB += $(NOBLASLIB)
  #EXTLIB += $(NOBLASLIB)
  LIBDEP += $(NOBLASDEP)
endif

ifneq (Darwin,$(UNAME))
  ifeq (0,$(shell ln -fs this-file-does-not-exist .ln 2>/dev/null && echo "$$?" && rm .ln 2>/dev/null))
    solink = -o $1.$2.$3.$4 $(XLNKOPT)-soname=$(strip $(notdir $1).$5)
    ifneq (0,$(SONAMELNK))
      solink += $(shell cd $(dir $1) && ln -fs $(notdir $1.$2.$3.$4) $(notdir $1.$5))
    endif
    ifneq (0,$(shell echo "$$((1<$(SONAMELNK) || 0>$(SONAMELNK)))"))
      solink += $(shell cd $(dir $1) && ln -fs $(notdir $1.$5) $(notdir $1))
    endif
  else # MinGW
    solink = -o $1 $(XLNKOPT)-soname=$(strip $(notdir $1).$5)
  endif
else # osx
  solink = -o $(basename $1).$2$(suffix $1) \
           -install_name $(notdir $(basename $1).$2$(suffix $1)) \
           -current_version $2.$3.$4 -compatibility_version $5
  ifneq (0,$(SONAMELNK))
    solink += $(shell cd $(dir $1) && ln -fs $(notdir $(basename $1).$2$(suffix $1)) $(notdir $1))
  endif
endif

ifneq (0,$(INTEL))
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
else
  ifeq (1,$(STATIC))
    ifeq (0,$(PGI))
      SLDFLAGS += -Bstatic
      ifeq (0,$(CLANG))
        SLDFLAGS += -static-libstdc++
        ifneq (Darwin,$(UNAME))
          SLDFLAGS += -static-libgcc
        endif
      endif
    endif
    DFLAGS += -D__STATIC=1
  else ifneq (0,$(STATIC))
    DFLAGS += -D__STATIC=$(STATIC)
    ifeq (0,$(shell $(LD) -static -ldummydoesnotexist 2>&1 | grep -q "\-ldummydoesnotexist"; echo "$$?"))
      SLDFLAGS += -static
    endif
  endif
endif

LIBGFORTRAN = $(call libpath,$(FC),$(SLDFLAGS),gfortran)
ifneq (,$(strip $(LIBGFORTRAN)))
ifeq (3,$(GCC)) # try to avoid some more references to Fortran runtime libraries
  FLDFLAGS += $(call abslibpath,$(LIBGFORTRAN)) $(XLIB_BEGIN) $(call abslibfile,$(LIBGFORTRAN)) $(XLIB_END)
  LIB_FLD = $(LIB_LD)
  #FLD = $(LD)
endif
endif

ifneq (0,$(INTEL))
  AR ?= xiar
  CXXLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
  FCLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
  ifneq (0,$(FORCE_CXX))
    FCLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
    CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
  endif
  ifneq (0,$(shell echo "$$((200000<=$(CXX_VERSION)))"))
    CXXFLAGS += -std=c++14
  else ifneq (0,$(COMPATIBLE))
    ifneq (0,$(shell echo "$$((170000<=$(CXX_VERSION)))"))
      CXXFLAGS += -std=c++14
    else ifneq (0,$(shell echo "$$((140000<=$(CXX_VERSION)))"))
      CXXFLAGS += -std=c++11
    endif
  endif
  CXXFLAGS += -Wall -diag-disable 1879,3415,3948,10006,10010,10411,13003
  CFLAGS += -Wall -diag-disable 1879,3415,3948,10006,10010,10411,13003
  ifneq (0,$(UNUSED))
    CXXFLAGS += -Wno-unused-function
    CFLAGS += -Wno-unused-function
  endif
  ifeq (,$(strip $(GFC)))
    ifneq (ld,$(notdir $(LD)))
      LDFLAGS += -diag-disable 1879,3415,10006,10010,10411
    endif
    FCFLAGS += -diag-disable 10006,10010,10411,13003
    ifneq (0,$(THREADS))
      FCMTFLAGS += -threads
    endif
    FPEDANTIC += -warn all,notruncated_source -diag-disable 7025,7373,10237,10342,10382
  endif
  ifeq (0,$(ICX))
    CPEDANTIC += -Wcheck
  endif
  CPEDANTIC += -diag-disable 177,1419,1572,2547,10382
  CWARNEXTRA = -Wremarks
  ifeq (1,$(PEDANTIC))
    CSTD = -std=c99
    CFLAGS += $(CSTD)
    ifeq (,$(strip $(GFC)))
      FSTD = -std03
      FMFLAGS += $(FSTD) $(FPEDANTIC) -diag-disable 10010
      FCFLAGS += $(FFORM_FLAG)
    endif
    ifeq (0,$(ICX))
      CXXFLAGS += -Wcheck
      CFLAGS += -Wcheck
    endif
  else ifneq (0,$(PEDANTIC))
    CSTD = -std=c89
    CXXFLAGS += $(CPEDANTIC)
    CFLAGS += $(CSTD) $(CPEDANTIC)
    ifneq (990000,$(CC_VERSION))
      CFLAGS += $(CWARNEXTRA)
    else ifneq (2,$(PEDANTIC))
      CFLAGS += $(CWARNEXTRA)
    endif
    ifeq (,$(strip $(GFC)))
      FSTD = -std03
      FCFLAGS += $(FSTD) $(FPEDANTIC)
      FMFLAGS += -fixed
    endif
  else
    CSTD = -std=c89
    ifeq (,$(strip $(GFC)))
      FCFLAGS += $(FFORM_FLAG)
    endif
  endif
  ifeq (,$(strip $(GFC)))
    # flag specifying output directory must be last
    FMFLAGS += -module
  else
    FMFLAGS += -J
  endif
  CXXFLAGS += $(OPTFLAG)
  CFLAGS += $(OPTFLAG)
  FCFLAGS += $(OPTFLAG)
  ifeq (0,$(OFFLOAD))
  ifeq (0,$(ICX))
    ifeq (,$(strip $(GFC)))
      FCFLAGS += -qno-offload
    endif
    CXXFLAGS += -qno-offload
    CFLAGS += -qno-offload
  endif
  endif
  ifeq (0,$(DBG))
    # consider more accurate -fp-model (C/C++: precise, Fortran: source)
    ifeq (0,$(ICX))
      #CXXFLAGS += -qoverride_limits #-fp-model fast=2
      #CFLAGS += -qoverride_limits #-fp-model fast=2
      CXXFLAGS += -fno-alias
      CFLAGS += -fno-alias
    endif
    CXXFLAGS += -ansi-alias
    CFLAGS += -ansi-alias
    ifeq (,$(strip $(GFC)))
    ifneq (,$(strip $(FC)))
      #FCFLAGS += -qoverride_limits #-fp-model fast=2
      ifneq (0,$(shell echo "$$((130000<=$(FC_VERSION)))"))
        FCFLAGS += -align array64byte
      endif
      ifneq (0,$(IPO))
        FCFLAGS += -ipo
      endif
    endif
    endif
    ifneq (0,$(IPO))
      CXXFLAGS += -ipo
      CFLAGS += -ipo
    endif
  else ifeq (,$(strip $(GFC))) # debugging enabled
    ifeq (0,$(ICX))
      FCFLAGS += -check
    endif
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
        ifeq (,$(MIC))
          CTARGET = -xCOMMON-AVX512
        else ifneq (0,$(MIC))
          CTARGET = -xMIC-AVX512
        else
          CTARGET = -xCORE-AVX512
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
      ifeq (,$(MIC))
        CTARGET = -xCOMMON-AVX512
      else ifneq (0,$(MIC))
        CTARGET = -xMIC-AVX512
      else
        CTARGET = -xCORE-AVX512
      endif
    else ifeq (2,$(AVX))
      CTARGET = -march=core-avx2
    endif
  endif
  ifneq (0,$(SYM))
    ifeq (1,$(SYM))
      CXXFLAGS := -g $(CXXFLAGS)
      CFLAGS := -g $(CFLAGS)
    else
      CXXFLAGS := -g3 -debug inline-debug-info $(CXXFLAGS)
      CFLAGS := -g3 -debug inline-debug-info $(CFLAGS)
    endif
    ifeq (,$(strip $(GFC)))
      FCFLAGS := -g -traceback $(FCFLAGS)
    endif
  endif
  OMPFLAG_FORCE = -fopenmp
  ifeq (,$(strip $(OMPRT)))
    OMPRT = iomp5
  endif
  ifneq (0,$(OMP))
    CXXFLAGS += $(OMPFLAG_FORCE)
    FCFLAGS += $(OMPFLAG_FORCE)
    CFLAGS += $(OMPFLAG_FORCE)
    ifneq (,$(strip $(GFC)))
      LDFLAGS += $(XLIB_BEGIN) -l$(OMPRT) $(XLIB_END)
    else
      LDFLAGS += $(OMPFLAG_FORCE)
    endif
  endif
  ifneq (0,$(SIMD))
  ifneq (0,$(shell echo "$$((150000<=$(CXX_VERSION)))"))
    DFLAGS += -DLIBXSMM_OPENMP_SIMD
    CXXFLAGS += -qopenmp-simd
    CFLAGS += -qopenmp-simd
    ifeq (,$(strip $(GFC)))
      FCFLAGS += -qopenmp-simd
    endif
    SIMD ?= 1
  endif
  endif
  ifeq (,$(strip $(GFC)))
  ifneq (,$(strip $(R8)))
  ifneq (0,$(R8))
    FCFLAGS += -autodouble
  endif
  endif
  endif
  # workaround for certain bits introduced by GCC 7.0
  ifneq (0,$(shell echo "$$(((180000<=$(CC_VERSION) && 180001>$(CC_VERSION)) || (170006>$(CC_VERSION) && 0!=$(CC_VERSION))))"))
    CFLAGS += -D_Float128=__float128
  endif
else # GCC assumed
  FCLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
  CXXLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
  LDFLAGS += $(XLIB_BEGIN) -lm $(XLIB_END)
  ifneq (0,$(FORCE_CXX))
    FCLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
    CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
  endif
  ifeq (,$(filter-out 0,$(COMPATIBLE) $(PGI)))
    ifneq (0,$(shell echo "$$((50000<=$(GCC_VERSION)))"))
      CXXFLAGS += -std=c++14
    else ifneq (0,$(shell echo "$$((40700<=$(GCC_VERSION)))"))
      CXXFLAGS += -std=c++11
    endif
    CXXFLAGS += -Wall
    CFLAGS += -Wall
    FSTD = -std=f2003
    CPEDANTIC += -pedantic -Wextra -Wno-variadic-macros
    FPEDANTIC += -pedantic -Wextra -Wunused-variable \
                 -Wcharacter-truncation -Wline-truncation \
                 -Wconversion -Wintrinsics-std \
                 -Wimplicit-interface -Wimplicit-procedure
    ifneq (0,$(shell echo "$$((40200<=$(CC_VERSION)))"))
      CPEDANTIC += -Wno-overlength-strings
    else ifneq (0,$(CLANG))
      CPEDANTIC += -Wno-overlength-strings
    endif
    ifneq (0,$(shell echo "$$((40500<=$(CC_VERSION)))"))
      CPEDANTIC += -Wshadow
    endif
    ifneq (,$(strip $(FC)))
    ifneq (0,$(shell echo "$$((50000<=$(FC_VERSION)))"))
      FWARNEXTRA = -Wuse-without-only -Wc-binding-type \
                   -Wrealloc-lhs -Wrealloc-lhs-all \
                   -Wreal-q-constant -Wconversion-extra
    endif
    endif
    ifneq (0,$(UNUSED))
      CXXFLAGS += -Wno-unused-function #-Wno-attributes
      CFLAGS += -Wno-unused-function #-Wno-attributes
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
      FCFLAGS += $(FFORM_FLAG)
      FMFLAGS += $(FSTD) -pedantic -Wunused-variable $(FWARNEXTRA)
    else ifneq (0,$(PEDANTIC))
      ifneq (Darwin,$(UNAME))
        CPEDANTIC += -Wno-long-long
        CSTD = -std=c89
      else ifneq (0,$(GCC))
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
      FCFLAGS += $(FFORM_FLAG)
      CSTD = -std=c89
    endif
    # flag specifying output directory must be last
    ifeq (0,$(PGI))
      FMFLAGS += -J
    endif
  else # fallback
    FMFLAGS += -I
  endif
  CXXFLAGS += $(OPTFLAG)
  CFLAGS += $(OPTFLAG)
  FCFLAGS += $(OPTFLAG)
  ifeq (0,$(DBG))
    ifeq (,$(filter-out 0,$(COMPATIBLE) $(PGI)))
      CXXFLAGS += -funroll-loops
      CFLAGS += -funroll-loops
      FCFLAGS += -funroll-loops
    endif
    ifneq (0,$(IPO))
      CXXFLAGS += -flto
      CFLAGS += -flto
      FCFLAGS += -flto
      #FLDFLAGS += -fno-lto
      LDFLAGS += $(XLNKOPT)-flto
      ifneq (0,$(FAT))
        CXXFLAGS += -ffat-lto-objects
        CFLAGS += -ffat-lto-objects
        FCFLAGS += -ffat-lto-objects
      endif
    endif
  endif
  ifneq (0,$(INSTRUMENT))
    DFLAGS += -D__TRACE=$(INSTRUMENT)
    ifeq (0,$(PGI))
      CXXFLAGS += -finstrument-functions
      CFLAGS += -finstrument-functions
      FCFLAGS += -finstrument-functions
      # e.g. clang does not need/understand below flag
      ifneq (0,$(shell echo "$$((40300<=$(GCC_VERSION)))"))
        CFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
        ifneq (,$(filter 2 3,$(GCC)))
          CXXFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
          ifeq (3,$(GCC))
            FCFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
          endif
        endif
      endif
    endif
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
      ifneq (,$(filter 2 3,$(GCC)))
        CXXFLAGS := -g3 $(CXXFLAGS)
        CFLAGS := -g3 $(CFLAGS)
      else
        CXXFLAGS := -g $(CXXFLAGS)
        CFLAGS := -g $(CFLAGS)
      endif
      ifeq (3,$(GCC))
        FCFLAGS := -g3 $(FCFLAGS)
      else
        FCFLAGS := -g $(FCFLAGS)
      endif
    endif
  endif
  ifeq (0,$(COMPATIBLE))
    ifneq (0,$(PGI))
      OMPFLAG_FORCE = -mp
      OMPRT = omp
    else ifneq (0,$(GCC))
      OMPFLAG_FORCE = -fopenmp
    else ifneq (Darwin,$(UNAME))
      ifneq (0,$(shell echo "$$((0!=$(CLANG) && 30900<=$(CC_VERSION)))"))
        OMPFLAG_FORCE = -fopenmp
        OMPRT = omp
      endif
    endif
  endif
  ifeq (,$(OMPFLAG_FORCE))
    ifneq (,$(filter environment% override command%,$(origin OMP)))
      OMPFLAG_FORCE = -fopenmp
    endif
  else ifeq (FreeBSD,$(UNAME))
    # avoid include path at begin of compile line
    OMPFLAG_FORCE := -I/usr/local/include $(OMPFLAG_FORCE)
  endif
  # account for missing TLS/OMP
  ifeq (,$(OMPFLAG_FORCE))
    THREADS ?= 0
    OMP = 0
  endif
  ifeq (,$(strip $(OMPRT))) # fallback
    OMPRT = gomp
  endif
  OMPLIBFILE = $(call libpath,$(LD),$(SLDFLAGS) $(OMPFLAG_FORCE) -L/usr/local/lib,$(OMPRT))
  ifneq (0,$(OMP))
  ifneq (,$(OMPFLAG_FORCE))
  ifneq (,$(OMPLIBFILE))
    # clang: OMP=libomp
    ifneq (0,$(shell echo "$(OMP)" | grep -q "^-*[0-9]\+$$"; echo "$$?")) # NaN
      ifneq (0,$(CLANG))
        CXXLDFLAGS += $(call absliblpath,$(OMPLIBFILE))
        CLDFLAGS += $(call absliblpath,$(OMPLIBFILE))
      endif
      CXXLDFLAGS += $(OMPFLAG_FORCE)=$(OMP)
      CXXFLAGS += $(OMPFLAG_FORCE)=$(OMP)
      CLDFLAGS += $(OMPFLAG_FORCE)=$(OMP)
      CFLAGS += $(OMPFLAG_FORCE)=$(OMP)
    else ifneq (Darwin,$(UNAME))
      ifneq (0,$(CLANG))
        CXXLDFLAGS += $(call absliblpath,$(OMPLIBFILE))
        CLDFLAGS += $(call absliblpath,$(OMPLIBFILE))
      endif
      CXXLDFLAGS += $(OMPFLAG_FORCE)
      CLDFLAGS += $(OMPFLAG_FORCE)
      CXXFLAGS += $(OMPFLAG_FORCE)
      CFLAGS += $(OMPFLAG_FORCE)
    endif
  endif
  endif
  endif
  ifneq (0,$(OMP))
  ifneq (,$(OMPFLAG_FORCE))
    ifeq (3,$(GCC)) # pure GNU toolchain
      FLDFLAGS += $(OMPFLAG_FORCE)
      FCFLAGS += $(OMPFLAG_FORCE)
    else # mixed toolchain
      LDTEST = $(call libpath,$(FLD),$(SLDFLAGS) $(OMPFLAG_FORCE) -L/usr/local/lib,$(OMPRT))
      ifneq (,$(LDTEST))
        FLDFLAGS += $(OMPFLAG_FORCE)
        FCFLAGS += $(OMPFLAG_FORCE)
      else
        LDTEST = $(call libpath,$(FLD),$(SLDFLAGS) $(OMPFLAG_FORCE) -L/usr/local/lib,gomp)
        FLDFLAGS += $(call absliblpath,$(OMPLIBFILE))
        ifneq (,$(LDTEST))
          FLDFLAGS += $(OMPFLAG_FORCE)
          FCFLAGS += $(OMPFLAG_FORCE)
        endif
      endif
    endif
  endif
  endif
  ifneq (,$(strip $(R8)))
  ifneq (0,$(R8))
    FCFLAGS += -fdefault-real-8 -fdefault-double-8
  endif
  endif
endif

ifeq (undefined,$(origin TARGET))
  ifneq (,$(CTARGET))
  ifneq (0,$(shell $(CC) -E $(CTARGET) /dev/null 2>/dev/null >/dev/null; echo "$$?"))
    undefine CTARGET
  endif
  endif
  MAINTAINER ?= 0
  ifeq (,$(CTARGET))
  ifneq (0,$(shell echo "$$((3>$(DBG)))"))
    # detect maintainer build and limit to SSE3
    ifeq (,$(filter Windows_NT Darwin,$(UNAME)))
    ifeq (,$(SPACK_ENV_PATH)) # not under Spack
    ifeq (3,$(GCC)) # pure GNU toolchain
    ifeq (0,$(DEPSTATIC))
      MAINTAINER = 1
    else ifneq (0,$(SHARED))
      ifneq (,$(SHARED))
        MAINTAINER = 1
      endif
    endif
    endif
    endif
    endif
    ifeq (,$(CTARGET))
    ifeq (3,$(AVX))
    ifneq (0,$(shell echo "$$(((0!=$(GCC) && 60000<=$(GCC_VERSION)) || \
          (0!=$(CLANG) && (40000<=$(CC_VERSION) || 0==$(CC_VERSION))) || \
          (0!=$(PGI) && 190000<=$(CC_VERSION))))"))
    ifeq (0,$(shell $(CC) -E -mfma -mavx512f -mavx512cd /dev/null 2>/dev/null >/dev/null; echo "$$?"))
      CTARGET = -mfma -mavx512f -mavx512cd
      ifneq (0,$(lastword $(sort 0 $(MIC)))) # MIC
        ifeq (0,$(shell $(CC) -E -mavx512pf -mavx512er /dev/null 2>/dev/null >/dev/null; echo "$$?"))
          CTARGET += -mavx512pf -mavx512er
        endif
      else ifeq (0,$(shell $(CC) -E -mavx512dq -mavx512bw -mavx512vl /dev/null 2>/dev/null >/dev/null; echo "$$?"))
        ifneq (Darwin,$(UNAME))
          CTARGET += -mavx512dq -mavx512bw -mavx512vl
          #CTARGET += -mavx512ifma -mavx512vbmi
        else ifneq (,$(filter 1 2 3,$(INTEL) $(GCC)))
          CTARGET += -mavx512dq -mavx512bw -mavx512vl
          #CTARGET += -mavx512ifma -mavx512vbmi
        else ifneq (0,$(shell echo "$$((0!=$(CLANG) && 80100<=$(CC_VERSION)))"))
          CTARGET += -mavx512dq -mavx512bw -mavx512vl
        endif
      endif
    endif
    endif
    endif
    endif
    ifeq (,$(CTARGET)) # fallback to AVX2
    ifneq (,$(filter 2 3,$(AVX)))
      ifneq (0,$(shell echo "$$((0!=$(INTEL) || 0!=$(CLANG) || (40800<=$(GCC_VERSION) && 0==$(PGI)) || 0==$(CC_VERSION)))"))
        CTARGET ?= -march=core-avx2
      else
        CTARGET ?= -mavx2 -mfma
      endif
    endif
    endif
    ifeq (,$(CTARGET)) # fallback to AVX
    ifneq (,$(filter 1 2 3,$(AVX)))
      ifneq (0,$(shell echo "$$((0!=$(INTEL) || 0!=$(CLANG) || 40400<=$(GCC_VERSION) || 0==$(CC_VERSION)))"))
        CTARGET ?= -mavx
      endif
    endif
    endif
    ifneq (,$(filter 1 2 3 4,$(SSE) $(AVX)))
      ifeq (,$(CTARGET)) # SSE-4.2
        ifneq (,$(filter 40 41 42 43,$(SSE)$(AVX)))
          ifneq (0,$(GCC))
            ifneq (0,$(shell echo "$$((40300<=$(CC_VERSION)))"))
              ifeq (0,$(MAINTAINER))
                CTARGET = -msse4.2
              else
                TARGET = -msse4.2
                CTARGET = -msse3
              endif
            else
              CTARGET = -msse3
            endif
          endif
          CTARGET ?= -msse4.2
        else ifneq (,$(filter-out 0 1,$(SSE))) # better to use TARGET flag directly
          CTARGET = -msse$(SSE)
        endif
      endif
      ifeq (,$(CTARGET)) # SSE3
      ifneq (,$(filter 30 31 32 33,$(SSE)$(AVX)))
        ifneq (Darwin,$(UNAME))
          CTARGET ?= -msse3
        else # prevents Clang BE error (CRC32 and others)
          CTARGET ?= -msse4.2
        endif
      endif
      endif
      # SSE=2 is implicitly present (64-bit ABI)
      ifeq (,$(CTARGET)) # default SSE
      ifneq (,$(filter 01 02 03 10 11 12 13,$(SSE)$(AVX)))
        ifneq (0,$(GCC))
          ifneq (0,$(shell echo "$$((40300<=$(CC_VERSION)))"))
            ifeq (0,$(MAINTAINER))
              CTARGET = -msse4.2
            else
              TARGET = -msse4.2
              CTARGET = -msse3
            endif
          else
            CTARGET = -msse3
          endif
        endif
        CTARGET ?= -msse4.2
      endif
      endif
    else ifneq (0,$(AVX))
      CTARGET = -march=native
    endif
  endif # CTARGET
  endif # DBG
else # take user's TARGET into account
  CTARGET := $(TARGET)
endif

ifeq (0,$(shell $(CC) -E $(CTARGET) /dev/null 2>/dev/null >/dev/null; echo "$$?"))
  # check target flags by feeding an ad-hoc C program into the compiler
  ifneq (0,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c); \
    echo "int main(void) { return 0; }" > $${INFILE}; \
    RESULT=$$($(CC) $(CTARGET) -c $${INFILE} -o $${INFILE}.o 2>&1); \
    if [ "" = "$${RESULT}" ]; then echo "$$?"; else echo "1"; fi; \
    rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
      CTARGET = $(NULL)
  endif
  ifneq (,$(strip $(FC))) # check target flags by feeding an ad-hoc Fortran program into the compiler
  # INFILE cannot start with a leading dot when used with certain compiler
  ifeq (0,$(shell INFILE=$$($(MKTEMP) /tmp/_libxsmm_XXXXXX.f); \
    echo "      PROGRAM test" > $${INFILE}; \
    echo "      END PROGRAM" >> $${INFILE}; \
    RESULT=$$($(FC) $(CTARGET) -c $${INFILE} -o $${INFILE}.o 2>&1); \
    if [ "" = "$${RESULT}" ]; then echo "$$?"; else echo "1"; fi; \
    rm -f /tmp/$$(basename $${INFILE} .f).* _libxsmm_??????.* 2>/dev/null))
      # inherit CTARGET flags
      FTARGET = $(CTARGET)
  endif
  endif
else # revoke target flags
  CTARGET = $(NULL)
  FTARGET = $(NULL)
endif
# avoid unnecessary state
ifeq ($(CTARGET),$(FTARGET))
  TARGET ?= $(CTARGET)
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
              SIMD ?= 1
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
        ifneq (0,$(shell echo "$$((60000<=$(CC_VERSION) && 60000<=$(CXX_VERSION)))"))
          DFLAGS += -DLIBXSMM_OPENMP_SIMD
          CXXFLAGS += -fopenmp-simd
          CFLAGS += -fopenmp-simd
          ifneq (,$(FTARGET))
          ifneq (,$(strip $(FC)))
          ifneq (0,$(shell echo "$$((40900<=$(FC_VERSION)))"))
            FCFLAGS += -fopenmp-simd
            SIMD ?= 1
          endif
          endif
          endif
        endif
        endif
        endif
        endif
      endif
    endif # SIMD
    ifeq (0,$(PGI))
      ifneq (,$(CTARGET))
        CXXFLAGS += -ftree-vectorize
        CFLAGS += -ftree-vectorize
      endif
      ifneq (,$(CTARGET))
        FCFLAGS += -ftree-vectorize
      endif
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
    ifneq (ld,$(notdir $(LD)))
    ifneq (Windows_NT,$(UNAME))
      ifneq (0,$(DEPSTATIC))
        ifneq (0,$(HARDEN))
          ifneq (,$(strip $(HARDEN))) # explicit
            LDFLAGS := $(XLNKOPT)--export-dynamic $(LDFLAGS)
          else ifneq (0,$(SYM))
            LDFLAGS := $(XLNKOPT)--export-dynamic $(LDFLAGS)
          endif
        else ifneq (0,$(SYM))
          LDFLAGS := $(XLNKOPT)--export-dynamic $(LDFLAGS)
        endif
      endif
      # Linux distributions may apply similar hardening
      LDFLAGS := $(XLNKOPT)-z,relro,-z,now $(LDFLAGS)
    endif
    endif
    ifeq (0,$(PGI))
      CXXFLAGS += -fdata-sections -ffunction-sections
      CFLAGS += -fdata-sections -ffunction-sections
      ifeq (0,$(INTEL))
        FCFLAGS += -fdata-sections -ffunction-sections
      else ifneq (,$(strip $(GFC)))
        FCFLAGS += -fdata-sections -ffunction-sections
      endif
      # --gc-sections: relies on section-flags present at compile-stage
      LDFLAGS := $(XLNKOPT)--gc-sections $(LDFLAGS)
      ifeq (0,$(VISIBILITY)) # -fvisibility=hidden may cause crashes
        CXXFLAGS += -fvisibility=hidden -fvisibility-inlines-hidden
        #FCFLAGS += -fvisibility=hidden
        CFLAGS += -fvisibility=hidden
      endif
    endif
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
  ifeq (,$(filter-out 0,$(COMPATIBLE) $(PGI)))
    CFLAGS += -pthread
    CXXFLAGS += -pthread
    FLDFLAGS += $(XLIB_BEGIN) -lpthread $(XLIB_END)
    ifneq (Windows_NT,$(UNAME))
      ifneq (0,$(INTEL))
        ifeq (0,$(OFFLOAD))
          CLDFLAGS += $(XLIB_BEGIN) -lpthread $(XLIB_END)
        else
          CLDFLAGS += -pthread
        endif
      else ifneq (Darwin,$(UNAME))
        CLDFLAGS += -pthread
      else ifeq (0,$(CLANG))
        CLDFLAGS += -pthread
      endif
    endif
  else ifneq (0,$(PGI))
    LIBATOMIC ?= 1
  endif
endif

LIBATOMIC ?= 0
ifneq (0,$(THREADS))
ifneq (0,$(LIBATOMIC))
  DFLAGS += -DLIBXSMM_LIBATOMIC
  LDFLAGS += -latomic
endif
endif

ifeq (0,$(VLA))
  DFLAGS += -DLIBXSMM_NO_VLA
else ifneq (,$(VLA))
  DFLAGS += -DLIBXSMM_VLA
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
  ifneq (0,$(GCC))
    LDFLAGS += -Wa,-q
    CFLAGS += -Wa,-q
    ifneq (,$(filter 2 3,$(GCC)))
      CXXFLAGS += -Wa,-q
    endif
  endif
  FLDFLAGS += -Wa,-q
  FCFLAGS += -Wa,-q
endif

ifeq (0,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c); \
  echo "\#include <features.h>" > $${INFILE}; \
  echo "\#if !defined(__GNU_LIBRARY__) && !defined(__GLIBC__)" >> $${INFILE}; \
  echo "0" >> $${INFILE}; \
  echo "\#endif" >> $${INFILE}; \
  RESULT=$$($(CC) -c $${INFILE} -o $${INFILE}.o 2>&1); \
  if [ "" = "$${RESULT}" ]; then echo "$$?"; else echo "1"; fi; \
  rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
    GLIBC = 1
endif
GLIBC ?= 0

OMPLIBFILE ?= $(call libpath,$(LD),$(SLDFLAGS) $(OMPFLAG_FORCE) -L/usr/local/lib,$(OMPRT))
ifneq (,$(strip $(OMPLIBFILE)))
  OMPLIB = $(call abslib,$(OMPLIBFILE))
endif
ifneq (,$(OMPFLAG_FORCE))
ifeq (0,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c); \
  echo "\#if defined(_OPENMP)" > $${INFILE}; \
  echo "\# include <omp.h>" >> $${INFILE}; \
  echo "int main() { return 0; }" >> $${INFILE}; \
  echo "\#endif" >> $${INFILE}; \
  RESULT=$$($(CC) $(OMPFLAG_FORCE) $${INFILE} -o $${INFILE}.x 2>&1); \
  if [ "" = "$${RESULT}" ]; then echo "$$?"; else echo "1"; fi; \
  rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
    OMPFLAG = $(OMPFLAG_FORCE)
endif
endif
ifeq (0,$(OMP))
  EXTLIB += $(OMPLIB)
endif

ifneq (0,$(TBB))
ifneq (,$(TBBROOT))
  ifneq (Windows_NT,$(UNAME))
    TBBLIB_DIR = $(TBBROOT)/lib/intel64
    TBBLIB_DIRGCC = gcc$(GCC_VERSION_MAJOR).$(GCC_VERSION_MINOR)
    TBBLIB_MALLOC = $(wildcard $(TBBLIB_DIR)/$(TBBLIB_DIRGCC)/libtbbmalloc.$(ILIBEXT))
    ifeq (,$(TBBLIB_MALLOC))
      ifneq (0,$(TBB_OLDRTL))
        TBBLIB_DIRGCC = $(shell ls -1 "$(TBBLIB_DIR)" | tr "\n" " " | cut -d" " -f1)
      else
        TBBLIB_DIRGCC = $(shell ls -1 "$(TBBLIB_DIR)" | tr "\n" " " | rev | cut -d" " -f2 | rev)
      endif
      TBBLIB_MALLOC = $(wildcard $(TBBLIB_DIR)/$(TBBLIB_DIRGCC)/libtbbmalloc.$(ILIBEXT))
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

ifneq (0,$(BLAS))
  ifneq (Darwin,$(UNAME))
    MKL_PLATFORM = linux
  else # OSX
    MKL_PLATFORM = mac
  endif
endif

# enable MKL if available
ifeq (,$(strip $(MKLROOT)))
  MKL_INCFILE = $(strip $(lastword $(sort $(wildcard /opt/intel/compilers_and_libraries_*/$(MKL_PLATFORM)/mkl/include/mkl.h))))
  ifneq (,$(MKL_INCFILE))
    MKLROOT = $(abspath $(dir $(MKL_INCFILE))/..)
  endif
endif
ifeq (,$(strip $(MKLROOT)))
  MKL_INCFILE = $(strip $(wildcard /usr/include/mkl/mkl.h))
  ifneq (,$(MKL_INCFILE))
    MKLROOT = $(abspath $(dir $(MKL_INCFILE))/../..)
  endif
endif
ifneq (,$(MKL_INCFILE))
  #BLAS_STATIC = 0
  ABSLIBS = 1
endif

ifneq (0,$(BLAS_STATIC))
  BLASLIBEXT ?= $(SLIBEXT)
else # shared (DLL)
  BLASLIBEXT ?= $(ILIBEXT)
endif

ifneq (0,$(ABSLIBS))
#ifeq (0,$(DEPSTATIC))
  abslibpath = $(strip $(call abslibrpath,$1) $(call absliblpath,$1))
#endif
endif
abslibpath ?= $(call absliblpath,$1)

ifeq (Darwin,$(UNAME))
ifneq (,$(OMPLIBFILE))
  ifeq (0,$(ABSLIBS))
    LDFLAGS += $(call abslibrpath,$(OMPLIBFILE))
  endif
  LDFLAGS += $(call abslib,$(OMPLIBFILE))
endif
endif

ifneq (,$(strip $(MKLROOT)))
  ifeq (0,$(PGI))
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
    MKL ?= 0
  endif
else # disable
  MKL = 0
endif
ifneq (0,$(MKL))
  ifeq (1,$(MKL_DIRECT))
    ifeq (1,$(MKL))
      BLAS_FLAGS += -DMKL_DIRECT_CALL_SEQ
    else
      BLAS_FLAGS += -DMKL_DIRECT_CALL
    endif
    ifneq (0,$(GCC))
      CXXFLAGS += -Wno-unused-value
      CFLAGS += -Wno-unused-value
    endif
  else ifneq (0,$(MKL_DIRECT))
    ifeq (1,$(MKL))
      BLAS_FLAGS += -DMKL_DIRECT_CALL_SEQ_JIT
    else
      BLAS_FLAGS += -DMKL_DIRECT_CALL_JIT
    endif
    ifneq (0,$(GCC))
      CXXFLAGS += -Wno-unused-value
      CFLAGS += -Wno-unused-value
    endif
  endif
  BLAS_FLAGS += -D__CBLAS
endif
ifneq (Darwin,$(UNAME))
  ifneq (,$(wildcard $(MKLROOT)/lib/x86_64-linux-gnu/libmkl_rt.*))
    MKL_ARCH = x86_64-linux-gnu
  else
    MKL_ARCH = intel64
  endif
endif

ifeq (1,$(MKL)) # sequential
  BLAS_FLAGS += -D__BLAS=1 -D__MKL=1
  MKL_THREADS = sequential
  ifneq (0,$(INTEL))
  ifneq (0,$(OFFLOAD))
    BLAS_LDFLAGS += -qoffload-option,mic,ld,"-lm $(LIBDL)"
  endif
  endif
else ifneq (0,$(MKL)) # multi-threaded
  BLAS_FLAGS += -D__BLAS=$(MKL) -D__MKL=$(MKL)
  MKL_THREADS = $(MKL_OMPRTL)_thread
  ifeq (0,$(OMP))
    ifneq (gnu,$(MKL_OMPRTL))
      BLAS_LDFLAGS += $(OMPLIB)
    else
      BLAS_LDFLAGS += -lgomp
    endif
  else ifeq (gnu,$(OMPLIB)$(MKL_OMPRTL))
    BLAS_LDFLAGS += -lgomp
  endif
  ifneq (0,$(THREADS))
    BLAS_LDFLAGS += $(XLIB_BEGIN) -lpthread $(XLIB_END)
  endif
  ifneq (0,$(INTEL))
  ifneq (0,$(OFFLOAD))
    BLAS_LDFLAGS += -qoffload-option,mic,ld,"-lm $(LIBDL)"
    ifeq (0,$(OMP))
      BLAS_LDFLAGS += -qoffload-option,mic,ld,"$(OMPLIB)"
    endif
  endif
  endif
endif
ifneq (0,$(MKL))
  ifeq (,$(MKL_INCFILE))
    BLAS_IFLAGS += -I$(MKLROOT)/include
  else
    BLAS_IFLAGS += -I$(dir $(MKL_INCFILE))
  endif
  BLAS_LDFLAGS += $(XLIB_BEGIN) -lm $(LIBDL) $(XLIB_END)
  ifeq (0,$(BLAS_STATIC)) # shared
    BLAS_LDFLAGS := $(call abslibpath,$(MKLROOT)/lib/$(MKL_ARCH)) \
      $(call abslibfile,libmkl_$(MKL_FCRTL)_$(MKL_BITS).$(ILIBEXT)) \
      $(call abslibfile,libmkl_core.$(ILIBEXT)) \
      $(call abslibfile,libmkl_$(MKL_THREADS).$(ILIBEXT)) \
      $(BLAS_LDFLAGS)
    ifneq (0,$(INTEL))
    ifneq (0,$(OFFLOAD))
      BLAS_LDFLAGS += -qoffload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_$(MKL_FCRTL)_$(MKL_BITS) -lmkl_core -lmkl_$(MKL_THREADS)"
    endif
    endif
  else # static
    BLAS_LDFLAGS := $(XGROUP_BEGIN) \
      $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_$(MKL_FCRTL)_$(MKL_BITS).$(SLIBEXT) \
      $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_core.$(SLIBEXT) \
      $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_$(MKL_THREADS).$(SLIBEXT) \
    $(XGROUP_END) $(BLAS_LDFLAGS)
    ifneq (0,$(INTEL))
    ifneq (0,$(OFFLOAD))
      BLAS_LDFLAGS += -qoffload-option,mic,ld,"--start-group \
        $(MKLROOT)/lib/mic/libmkl_$(MKL_FCRTL)_$(MKL_BITS).$(SLIBEXT) \
        $(MKLROOT)/lib/mic/libmkl_core.$(SLIBEXT) \
        $(MKLROOT)/lib/mic/libmkl_$(MKL_THREADS).$(SLIBEXT) \
      --end-group"
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
    BLASDIR ?= $(wildcard /usr/local/lib /usr/lib)
    BLASROOT = $(wildcard $(patsubst %,%/..,$(BLASDIR)))
    ifeq (0,$(BLAS_STATIC)) # shared
      LDTEST = $(filter-out -static,$(SLDFLAGS)) $(patsubst %,-L%,$(BLASDIR))
    else # static
      LDTEST = $(SLDFLAGS) $(patsubst %,-L%,$(BLASDIR))
    endif
    BLASLIBFILE = $(call libpath,$(LD),$(LDTEST),openblas$(BLAS_THREADS)$(BLAS_BITS))
    ifeq (,$(BLASLIBFILE)) # newer distributions symlink a non-decorated library to threaded OpenBLAS
      BLASLIBFILE = $(call libpath,$(LD),$(LDTEST),openblas$(BLAS_BITS))
    endif
    # most people expect to pickup OpenBLAS (if available) even when libblas/liblapack are available; use OPENBLAS=0 to avoid this
    ifeq (0,$(PGI))
      OPENBLAS := $(if $(BLASLIBFILE),1,0)
    else # PGI
      OPENBLAS := 0
    endif
    ifneq (0,$(OPENBLAS)) # OpenBLAS
      # OpenBLAS also carries the CBLAS bits
      BLAS_FLAGS += -D__BLAS=$(BLAS) -D__CBLAS -D__OPENBLAS
      ifneq (,$(wildcard $(patsubst %,%/include/openblas/f77blas.h,$(BLASROOT))))
        BLAS_FLAGS += -D__OPENBLAS77 -I$(dir $(wildcard $(patsubst %,%/include/openblas/f77blas.h,$(BLASROOT))))
      else ifneq (,$(wildcard $(patsubst %,%/include/x86_64-linux-gnu/f77blas.h,$(BLASROOT))))
        BLAS_FLAGS += -D__OPENBLAS77 -I$(dir $(wildcard $(patsubst %,%/include/x86_64-linux-gnu/f77blas.h,$(BLASROOT))))
      else ifneq (,$(wildcard $(patsubst %,%/include/f77blas.h,$(BLASROOT))))
        BLAS_FLAGS += -D__OPENBLAS77 -I$(dir $(wildcard $(patsubst %,%/include/f77blas.h,$(BLASROOT))))
      endif
    else # BLAS (reference)
      BLAS_FLAGS += -D__BLAS
      ifeq (Windows_NT,$(UNAME)) # no particular action about static linkage (use DLL)
        BLASLIBFILE = $(call libpath,$(LD),$(LDTEST),blas$(BLAS_BITS).dll)
      else
        BLASLIBFILE = $(call libpath,$(LD),$(LDTEST),blas$(BLAS_BITS))
      endif
    endif
    ifneq (,$(BLASLIBFILE))
      ifeq (0,$(BLAS_STATIC)) # shared
        BLAS_LDFLAGS += $(call abslib,$(BLASLIBFILE))
      else # static
        BLAS_LDFLAGS += $(BLASLIBFILE)
      endif
    else # fallback
      BLAS_LDFLAGS += -lblas
    endif
    ifeq (0,$(OPENBLAS)) # BLAS (reference)
      ifneq (0,$(DEPSTATIC))
        ifneq (,$(LIBGFORTRAN))
          LDTEST += $(XLIB_BEGIN) $(call abslibfile,$(LIBGFORTRAN)) -lm $(XLIB_END)
        else ifeq (3,$(GCC))
          LDTEST += $(XLIB_BEGIN) -lgfortran -lm $(XLIB_END)
        endif
      endif
      ifeq (0,$(PGI))
        QUADMATH = $(shell $(LD) $(SLDFLAGS) -lquadmath 2>&1 | grep -q "\-lquadmath"; echo "$$?")
        ifneq (0,$(QUADMATH))
          LDTEST += $(XLIB_BEGIN) -lquadmath -lm $(XLIB_END)
        endif
        ifneq (3,$(GCC))
          BLAS_LDFLAGS += $(LDTEST)
        else
          BLAS_CLDFLAGS += $(LDTEST)
        endif
      endif
    endif
    ifeq (0,$(OMP)) # Fortran compiler cannot link OpenMP runtime
      ifeq (,$(call libpath,$(FLD),$(SLDFLAGS) $(OMPFLAG_FORCE) -L/usr/local/lib,$(OMPRT)))
        BLAS_CLDFLAGS += $(XLIB_BEGIN) $(call abslibfile,$(OMPLIBFILE)) $(XLIB_END)
      else # common link BLAS/OMP-link flags (all compilers)
        BLAS_LDFLAGS += $(XLIB_BEGIN) $(call abslibfile,$(OMPLIBFILE)) $(XLIB_END)
      endif
    endif
  else # BLAS library is specified via BLASLIB
    BLAS_FLAGS += -D__BLAS=$(BLAS)
    ifneq (,$(findstring openblas,$(notdir $(BLASLIB))))
      BLAS_FLAGS += -D__CBLAS -D__OPENBLAS
    endif
    ifneq (./,$(firstword $(BLASDIR))$(dir $(BLASLIB)))
      ifeq (./,$(dir $(BLASLIB)))
        BLAS_LDFLAGS += $(call abslib,$(firstword $(BLASDIR))/$(if $(suffix $(BLASLIB)),$(BLASLIB),lib$(BLASLIB).$(BLASLIBEXT)))
      else
        BLAS_LDFLAGS += $(call abslib,$(if $(suffix $(BLASLIB)),$(BLASLIB),$(BLASLIB).$(BLASLIBEXT)))
      endif
    else # fallback
      BLAS_LDFLAGS += -l$(BLASLIB)
    endif
  endif
endif

cleanup = $(foreach flag,$(subst / $(NULL), $(NULL),$(subst //,/,$1)),$(flag))
# cleanup eventually duplicated flags and slashes
cleanld = $(strip $(shell echo "$1" | sed \
  -e "s/[[:space:]]$(XLIB_BEGIN)[[:space:]]$(XLIB_END)/ /g" \
  -e "s/[[:space:]]$(XLIB_END)[[:space:]]$(XLIB_BEGIN)/ /g" \
  -e "s/[[:space:]][[:space:]][[:space:]]*/ /g" \
  -e "s/\/[[:space:]]/ /g" \
  -e "s/\/\//\//g"))

# no-BLAS flags: cleanup and extra flags
NOBLAS_CXXFLAGS := $(call cleanup,$(PICFLAG) $(CXXFLAGS) $(ECXXFLAGS) $(EFLAGS))
NOBLAS_FCFLAGS := $(call cleanup,$(PICFLAG) $(FCFLAGS) $(EFCFLAGS) $(EFLAGS))
NOBLAS_CFLAGS := $(call cleanup,$(PICFLAG) $(CFLAGS) $(ECFLAGS) $(EFLAGS))
NOBLAS_FLAGS := $(call cleanup,$(DFLAGS))
NOBLAS_IFLAGS := $(call cleanup,$(IFLAGS))
NOBLAS_LDFLAGS := $(call cleanld,$(LDFLAGS) $(ELDFLAGS))
NOBLAS_CLDFLAGS := $(call cleanld,$(CLDFLAGS))

# regular flags: cleanup
CXXFLAGS := $(call cleanup,$(NOBLAS_CXXFLAGS) $(BLAS_CXXFLAGS))
FCFLAGS := $(call cleanup,$(NOBLAS_FCFLAGS) $(BLAS_FCFLAGS))
CFLAGS := $(call cleanup,$(NOBLAS_CFLAGS) $(BLAS_CFLAGS))
DFLAGS := $(call cleanup,$(NOBLAS_FLAGS) $(BLAS_FLAGS))
IFLAGS := $(call cleanup,$(NOBLAS_IFLAGS) $(BLAS_IFLAGS))
CLDFLAGS := $(call cleanld,$(NOBLAS_CLDFLAGS) $(BLAS_CLDFLAGS))
LDFLAGS := $(call cleanld,$(NOBLAS_LDFLAGS) $(BLAS_LDFLAGS))
CXXLDFLAGS := $(call cleanld,$(CXXLDFLAGS))
FLDFLAGS := $(call cleanld,$(FLDFLAGS))

.PRECIOUS: $(BLDDIR)/%-cpp.o $(BLDDIR)/%-c.o $(BLDDIR)/%-f.o \
           $(BLDDIR)/%-f90.o $(BLDDIR)/%-f90.o $(BLDDIR)/%-f77.o \
           %/.make

.SUFFIXES:

# applyif(A1,A2,A3,A4) evaluates to A4 if the basenames (A1, A2) match and if A1 is non-zero (precondition)
applyif = $(if $(filter 0,$1),$(NULL),$(if $(filter $2,$(basename $(notdir $3))),$(if $1,$4,$(NULL)),$(NULL)))
# derives the extension of a filename
extname = $(subst .,,$(suffix $(1)))
# derives the name of an object files for a given source file
objname = $(foreach ARG, $(1),$(addprefix $(BLDDIR)/, $(patsubst %$(suffix $(ARG)),%-$(call extname,$(ARG)).o,$(notdir $(ARG)))))

STATE := $(foreach V,$(sort $(.VARIABLES)),$(if \
  $(filter-out environment% default automatic,$(origin $V)), \
  $(if $(filter-out $(EXCLUDE_STATE) EXCLUDE_STATE .% _% MAKE_% HAVE_% LIB_% MAKEFILE_LIST MAKEOVERRIDES MAKEFLAGS SHELL BLASLIBEXT BLDDIR REVERSION \
    CC_VERSION CC_VERSION_FLAG CC_VERSION_STRING CINFO COMPILER_VERSION_FLAG CP CXX_VERSION CXX_VERSION_FLAG CXX_VERSION_STRING DEPDIR DFLAGS FINFO \
    ENV EXTDEP FSTD FC_VERSION FC_VERSION_FLAG FC_VERSION_STRING FORTDEP GCC_VERSION GCC_VERSION_STRING GINFO TESTSIZE DLIBEXT DYNAMIC FLOCK LIBEXT \
    ILIBEXT LIBNAME LICFILE MAKEINC MKL_INCFILE MV ROOTDIR SLIBEXT XLNKOPT BRACE_OPEN BRACE_CLOSE MKTEMP OMPLIBFILE SHARED GNU CCAR CLDFLAGS SUITE \
    FPEDANTIC FWARNEXTRA XGROUP_BEGIN XGROUP_END SLDFLAGS PICFLAG PIC OPTFLAG OPT OMPFLAG_FORCE FORCE_CXX MINGW NOBLAS% MAINLIB LIBRT PGI \
    ICX GFC CLANG INTEL CRAY GCC GCCBIN LIBGFORTRAN LIBDEP FORTLIB FMFLAGS FIXFC FCLDFLAGS EXTLIB PYTHON% PYMOD CWARNEXTRA CSTD DNOBLAS LDTEST \
    PEDANTIC LIBDL MAINTAINER CMAIN FMAIN CPEDANTIC FREEFORM OPSYS FFORM_FLAG UNUSED PRESTATE AVX SSE \
    which cleanup cleanld linkopt abslibfile abslibpath absliblpath abslibrpath libpath abslib solink applyif extname objname \
    $(if $(filter $(BLAS_STATIC),$(DEPSTATIC)),BLAS_STATIC,$(NULL)) $(if $(filter-out $(FTARGET),$(CTARGET)),TARGET,CTARGET FTARGET),$V), \
    $(if $($V),$V=$(subst $(USER),$$USER,$(subst $(HOME),$$HOME,$(filter-out $(XLNKOPT)--export-dynamic -dynamic,$($V))))?))))

PRESTATE := $(shell echo '$(STATE)' | $(DEPDIR)/.state.sh)
.state: $(firstword $(PRESTATE))
ifneq (0,$(STATIC))
ifneq (STATIC2,$(word 2,$(PRESTATE))$(words $(PRESTATE)))
	@rm -f $(OUTDIR)/lib*.$(DLIBEXT) $(OUTDIR)/lib*.$(DLIBEXT).*
endif
endif

%/.make:
	@mkdir -p $(basename $@)
	@touch $@

.make:
	@touch $@

