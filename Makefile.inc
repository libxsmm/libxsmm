###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

MAKE_VERSION_MAJOR = $(shell echo "$(MAKE_VERSION)" | cut -d. -f1)
MAKE_VERSION_MINOR = $(shell echo "$(MAKE_VERSION)" | cut -d. -f2)
MAKE_VERSION_PATCH = $(shell echo "$(MAKE_VERSION)" | cut -d. -f3)
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

#MAKEINC = $(abspath $(dir $(filter %Makefile.inc,$(MAKEFILE_LIST))))
MAKEINC = $(dir $(filter %Makefile.inc,$(MAKEFILE_LIST)))

ifeq (d,$(filter d,$(MAKEFLAGS)))
  #SHELL = bash -xv
  SHELL += -xv
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
OPTFLAG ?= -O$(patsubst O%,%,$(OPT))

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

# Additional library search paths
LIBFIND ?= /usr/local/lib

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

# Embedd soname into shared library
SONAMELNK ?= 2

# utilities to handle paths with spaces
SPACES ?= 1

REVERSION0 := [0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*
REVERSION1 := s/..* \($(REVERSION0)\)[ \S]*.*/\1/
REVERSION2 := s/..* \([0-9]\{5\}[0-9]*\)[ \S]*.*/\1/
CHAR_OPEN := (
CHAR_CLOSE := )
CHAR_HASH := \#

ifneq (0,$(SPACES))
  unquote = $(shell echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}" | sed -e 's/^[" ][" ]*//' -e 's/[" ][" ]*$$//')
  quote = $(strip $(if $(filter 0 1,$(words $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})), \
                                            $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}, \
                            "$(call unquote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})"))
  qapath = $(call quote,$(shell export "VAR=$(call unquote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})" && \
    if [ ! -d "$${VAR}" ]; then cd "$$(dirname "$${VAR}" 2>/dev/null)" 2>/dev/null && echo "$$(pwd -P)/$$(basename "$${VAR}" 2>/dev/null)"; \
    else cd "$${VAR}" 2>/dev/null && pwd -P; fi))
  qndir = $(call quote,$(shell export "VAR=$(call unquote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})" && \
    if [ ! -d "$${VAR}" ]; then basename "$${VAR}" 2>/dev/null; fi))
  qdir = $(call quote,$(shell dirname "$(call unquote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})x" 2>/dev/null)/)
else
  quote =  $(strip                $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})
  qapath = $(abspath $(call quote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
  qndir =  $(notdir  $(call quote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
  qdir =   $(dir     $(call quote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
endif
qsuffix = $(suffix   $(strip      $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
qbname =  $(basename $(call quote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
qname =   $(basename $(call quote,$(shell echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}" | sed "s/\.$(REVERSION0)//")))

# absolute directory paths
ABSDIR := $(call qapath,$(ROOTDIR))
HOMEDIR := $(call qapath,$(HOME))
HEREDIR := $(call qapath,.)

# Pickup OpenMP library name if passed as OMP=libname|name
ifneq (0,$(shell echo "$(OMP)" | grep -q "^-*[0-9]\+$$"; echo "$$?")) # NaN
  OMPRT = $(patsubst lib%,%,$(OMP))
endif

uniqadd = $(filter-out $($2),$($1)) $(if $3,$3,$($2))
uniqmov = $(if $(filter $($2),$($1)),$(call uniqadd,$1,$2,$3),$($1))
ctest = $(if $(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c) && \
  echo "int main(void) { return 0; }" > $${INFILE} && \
  $1 $2 $${INFILE} -o $${INFILE}.x 2>/dev/null >/dev/null && echo "OK"; \
  rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null),$2)
ldlib = $(if $1,$(call ctest,$1 $2,-l$3))

ifneq (0,$(PYMOD))
  ifneq (,$(PYTHON))
    PYVERSION_STRING = $(shell $(PYTHON) --version 2>&1 | head -n1 | sed -n "$(REVERSION1)p")
    PYVERSION = $(shell echo "$(PYVERSION_STRING)" | cut -d. -f1,2)
    PYROOT = $(call qapath,$(call qdir,$(call which,$(PYTHON)))/..)
    PYINC = $(wildcard $(PYROOT)/include/python$(PYVERSION)/Python.h)
    PYLIB = $(call ldlib,$(LD),$(SLDFLAGS),python$(PYVERSION))
    ifneq (,$(PYINC))
    ifneq (,$(PYLIB))
      LDFLAGS += $(PYLIB)
      IFLAGS += -I$(call qdir,$(PYINC))
      DFLAGS += -D__PYTHON
      # Avoid (unresolved) BLAS (alternative: BLAS=1|2)
      ifeq (,$(filter environment% override command%,$(origin BLAS))$(BLAS))
        NOBLAS = 1
      endif
    endif
    endif
  endif
  # Enable shared library (req. for Python module)
  override STATIC = 0
endif

# Explicitly disable BLAS by user's intervention
# Makefile defines what happens (perhaps nothing)
NOBLAS ?= 0
DNOBLAS = -D__BLAS=0
ifneq (,$(filter environment% override command%,$(origin BLAS)))
ifeq (0,$(BLAS))
  NOBLAS = 1
endif
endif
ifneq (0,$(NOBLAS))
  BLAS_FLAGS += $(DNOBLAS)
endif

# Secondary static
ifneq (file,$(origin STATIC)) # prefer user override/preference (in any case)
  DEPSTATIC ?= $(STATIC)
else ifneq (,$(wildcard $(LIBNAME).$(SLIBEXT)*)) # prefer static (library exists)
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
    ifneq (,$(filter icpc icpc1,$(call qndir,$(call which,icpc))$(INTEL)))
      CXX = icpc
    else ifneq (,$(filter icpx icpx2,$(call qndir,$(call which,icpx))$(INTEL)))
      CXX = icpx
    endif
  endif
  ifeq (,$(shell echo "$${CC}"))
    ifneq (,$(filter icc icc1,$(call qndir,$(call which,icc))$(INTEL)))
      CC = icc
    else ifneq (,$(filter icx icx2,$(call qndir,$(call which,icx))$(INTEL)))
      CC = icx
    endif
  endif
  ifeq (__INTEL_COMPILER,$(shell echo "__INTEL_COMPILER" \
                         | $(CC) $(ECFLAGS) $(EFLAGS) -E -P - 2>/dev/null \
                         | sed "/^[[:space:]]*$$/d" 2>/dev/null))
    ICX = $(INTEL)
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
  $(shell $(CXX) $(CXX_VERSION_FLAG) 2>/dev/null | head -n1 | cut -d' ' -f1) \
  $(shell $(CC)  $(CC_VERSION_FLAG)  2>/dev/null | head -n1 | cut -d' ' -f1)))))")

ifneq (0,$(INTEL))
  ifeq (,$(shell echo "$${FC}"))
    ifneq (,$(call qndir,$(call which,ifort)))
      FC = ifort
    else ifneq (,$(call qndir,$(call which,ifx)))
      FC = ifx
    endif
  endif
else
  ifeq (,$(call which,$(CXX)))
    CXX = g++
  else ifneq (0,$(shell $(CXX) $(CXX_VERSION_FLAG) >/dev/null 2>/dev/null; echo "$$?"))
    CXX = g++
  else ifneq (gcc,$(call qndir,$(call which,$(CC))))
    ifeq (g++,$(call qndir,$(call which,$(CXX))))
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

CXX_VERSION = $(shell $(CXX) $(CXX_VERSION_FLAG) | head -n1 | sed -n "$(REVERSION1)p")
ifeq (,$(CXX_VERSION))
  CXX_VERSION = $(shell $(CXX) $(CXX_VERSION_FLAG) | head -n1 | sed -n "$(REVERSION2)p")
endif
CXX_VERSION_MAJOR = $(shell echo "$(CXX_VERSION)" | cut -d. -f1)
CXX_VERSION_MINOR = $(shell echo "$(CXX_VERSION)" | cut -d. -f2)
CXX_VERSION_PATCH = $(shell echo "$(CXX_VERSION)" | cut -d. -f3)
ifeq (3,$(words $(CXX_VERSION_MAJOR) $(CXX_VERSION_MINOR) $(CXX_VERSION_PATCH)))
  CXX_VERSION_NUM = $(shell echo "$$(($(CXX_VERSION_MAJOR)*10000+$(CXX_VERSION_MINOR)*100+$(CXX_VERSION_PATCH)))")
else ifeq (2,$(words $(CXX_VERSION_MAJOR) $(CXX_VERSION_MINOR)))
  CXX_VERSION_NUM = $(shell echo "$$(($(CXX_VERSION_MAJOR)*10000+$(CXX_VERSION_MINOR)*100))")
  CXX_VERSION_PATCH = 0
else
  CXX_VERSION = $(NULL)
  CXX_VERSION_NUM = 0
endif

CC_VERSION = $(shell $(CC) $(CC_VERSION_FLAG) | head -n1 | sed -n "$(REVERSION1)p")
ifeq (,$(CC_VERSION))
  CC_VERSION = $(shell $(CC) $(CC_VERSION_FLAG) | head -n1 | sed -n "$(REVERSION2)p")
endif
CC_VERSION_MAJOR = $(shell echo "$(CC_VERSION)" | cut -d. -f1)
CC_VERSION_MINOR = $(shell echo "$(CC_VERSION)" | cut -d. -f2)
CC_VERSION_PATCH = $(shell echo "$(CC_VERSION)" | cut -d. -f3)
ifeq (3,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR) $(CC_VERSION_PATCH)))
  CC_VERSION_NUM = $(shell echo "$$(($(CC_VERSION_MAJOR)*10000+$(CC_VERSION_MINOR)*100+$(CC_VERSION_PATCH)))")
else ifeq (2,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR)))
  CC_VERSION_NUM = $(shell echo "$$(($(CC_VERSION_MAJOR)*10000+$(CC_VERSION_MINOR)*100))")
  CC_VERSION_PATCH = 0
else
  CC_VERSION = $(NULL)
  CC_VERSION_NUM = 0
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
  else ifneq (gfortran,$(call qndir,$(call which,$(FC))))
    ifeq (g++,$(call qndir,$(call which,$(CXX))))
      FIXFC = 1
    endif
  endif
  ifneq (0,$(FIXFC))
    ifneq (,$(call qndir,$(call which,gfortran)))
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
  FC_VERSION := $(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | sed -n "$(REVERSION1)p")
  ifeq (,$(FC_VERSION))
    FC_VERSION := $(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | sed -n "$(REVERSION2)p")
  endif
  FC_VERSION_MAJOR = $(shell echo "$(FC_VERSION)" | cut -d. -f1)
  FC_VERSION_MINOR = $(shell echo "$(FC_VERSION)" | cut -d. -f2)
  FC_VERSION_PATCH = $(shell echo "$(FC_VERSION)" | cut -d. -f3)
  ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
    FC_VERSION_NUM = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100+$(FC_VERSION_PATCH)))")
  else ifeq (2,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR)))
    FC_VERSION_NUM = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100))")
    FC_VERSION_PATCH = 0
  else
    FC_VERSION = $(NULL)
    FC_VERSION_NUM = 0
  endif
  ifeq (GNU,$(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | cut -d" " -f1))
    ifneq (0,$(shell echo "$$((40500>$(FC_VERSION_NUM)))"))
      ifneq (gfortran,$(call qndir,$(FC)))
        FC = gfortran
        FC_VERSION := $(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | sed -n "$(REVERSION1)p")
        ifeq (,$(FC_VERSION))
          FC_VERSION := $(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | sed -n "$(REVERSION2)p")
        endif
        FC_VERSION_MAJOR = $(shell echo "$(FC_VERSION)" | cut -d. -f1)
        FC_VERSION_MINOR = $(shell echo "$(FC_VERSION)" | cut -d. -f2)
        FC_VERSION_PATCH = $(shell echo "$(FC_VERSION)" | cut -d. -f3)
        ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
          FC_VERSION_NUM = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100+$(FC_VERSION_PATCH)))")
        else ifeq (2,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR)))
          FC_VERSION_NUM = $(shell echo "$$(($(FC_VERSION_MAJOR)*10000+$(FC_VERSION_MINOR)*100))")
          FC_VERSION_PATCH = 0
        else
          FC_VERSION_NUM = 0
        endif
        ifneq (0,$(shell echo "$$((40500>$(FC_VERSION_NUM)))"))
          override FC = $(NULL)
        endif
      else
        override FC = $(NULL)
      endif
    endif
  else ifneq (0,$(INTEL))
    ifneq (0,$(shell echo "$$((130000>$(FC_VERSION_NUM) && 0<$(FC_VERSION_NUM)))"))
      override FC = $(NULL)
    endif
  endif
endif

ifeq (,$(strip $(FC)))
  ifeq (0,$(FORTRAN))
    FC_VERSION = $(NULL)
  else # keep FC_VERSION for message about outdated compiler
    FORTRAN = 0
  endif
endif

# compiler names
ifeq (0,$(PLATFORM))
  CXX_NAME = $(call qbname,$(shell $(CXX) $(CXX_VERSION_FLAG) | head -n1 \
           | sed -e "s/^\([^0-9][^0-9]*\) ..*/\1/" -e "s/[[:space:]][[:space:]]*[Vv]ersion//" \
                 -e "s/[[:space:]][[:space:]]*$(CHAR_OPEN)..*//" -e "s/[[:space:]][[:space:]]*[[:punct:]]//" \
                 -e "s/[[:space:]][[:space:]]*$(REVERSION0)//" \
           | rev | cut -d" " -f1 | rev))
endif
ifeq (,$(strip $(CXX_NAME)))
  CXX_NAME = $(call qname,$(call qndir,$(CXX)))
endif
ifeq (0,$(PLATFORM))
  CC_NAME = $(call qbname,$(shell $(CC) $(CC_VERSION_FLAG) | head -n1 \
           | sed -e "s/^\([^0-9][^0-9]*\) ..*/\1/" -e "s/[[:space:]][[:space:]]*[Vv]ersion//" \
                 -e "s/[[:space:]][[:space:]]*$(CHAR_OPEN)..*//" -e "s/[[:space:]][[:space:]]*[[:punct:]]//" \
                 -e "s/[[:space:]][[:space:]]*$(REVERSION0)//" \
          | rev | cut -d" " -f1 | rev))
endif
ifeq (,$(strip $(CC_NAME)))
  CC_NAME = $(call qname,$(call qndir,$(CC)))
endif
ifneq (,$(strip $(FC)))
  FC_NAME = $(firstword $(call qndir,$(FC)))
endif
ifeq (,$(strip $(FC_NAME)))
  FC_NAME = $(call qname,$(call qndir,$(FC)))
endif

ifneq (0,$(FORTRAN))
  ifeq (,$(strip $(GFC)))
    ifneq (,$(strip $(FC)))
      ifeq (GCC,$(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
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
else ifeq (GCC,$(shell $(CC) $(CC_VERSION_FLAG) | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
  GCC = 1
else ifneq (pgcc,$(CC_NAME))
  ifneq (,$(findstring pgcc,$(CC_NAME)))
    GCC = 1
  endif
endif
ifeq (1,$(GCC))
  ifeq (0,$(shell $(CXX) $(CXX_VERSION_FLAG) | grep -q "Free Software Foundation"; echo "$$?"))
    GCC = 2
  else ifeq (GCC,$(shell $(CXX) $(CXX_VERSION_FLAG) | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
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
  override CC := $(CXX) $(call ctest,$(CC),-xc++)
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
  GCC_VERSION = $(CXX_VERSION)
  GCC_VERSION_MAJOR = $(CXX_VERSION_MAJOR)
  GCC_VERSION_MINOR = $(CXX_VERSION_MINOR)
  GCC_VERSION_PATCH = $(CXX_VERSION_PATCH)
else ifeq (0,$(CLANG))
  GCCBIN = $(call qndir,$(call which,gcc))
  ifneq (,$(strip $(GCCBIN)))
    GCC_VERSION = $(shell $(GCCBIN) $(CXX_VERSION_FLAG) | head -n1 | sed -n "$(REVERSION1)p")
    GCC_VERSION_MAJOR = $(shell echo "$(GCC_VERSION)" | cut -d. -f1)
    GCC_VERSION_MINOR = $(shell echo "$(GCC_VERSION)" | cut -d. -f2)
    GCC_VERSION_PATCH = $(shell echo "$(GCC_VERSION)" | cut -d. -f3)
  endif
endif
ifeq (3,$(words $(GCC_VERSION_MAJOR) $(GCC_VERSION_MINOR) $(GCC_VERSION_PATCH)))
  GCC_VERSION_NUM = $(shell echo "$$(($(GCC_VERSION_MAJOR)*10000+$(GCC_VERSION_MINOR)*100+$(GCC_VERSION_PATCH)))")
else ifeq (2,$(words $(GCC_VERSION_MAJOR) $(GCC_VERSION_MINOR)))
  GCC_VERSION_NUM = $(shell echo "$$(($(GCC_VERSION_MAJOR)*10000+$(GCC_VERSION_MINOR)*100))")
  GCC_VERSION_PATCH = 0
else
  GCC_VERSION = $(NULL)
  GCC_VERSION_NUM = 0
endif

MAINTAINER ?= 0
# detect maintainer build and limit to SSE3
ifeq (,$(filter Windows_NT Darwin,$(UNAME)))
ifeq (,$(SPACK_ENV_PATH)) # not under Spack
ifeq (3,$(GCC)) # pure GNU toolchain
ifneq (0,$(shell echo "$$((60000<=$(GCC_VERSION_NUM)))"))
  MAINTAINER = 1
endif
endif
endif
endif

# Select code path (if not selected otherwise)
CPUFLAGS = $(strip $(shell if [ -e /proc/cpuinfo ]; then \
    grep -m1 flags /proc/cpuinfo | cut -d: -f2-; \
  elif [ "Darwin" = "$(UNAME)" ]; then \
    sysctl -a machdep.cpu.features \
      machdep.cpu.extfeatures \
      machdep.cpu.leaf7_features \
    | cut -d: -f2- | tr -s "\n" " " \
    | tr [:upper:] [:lower:]; \
  fi))
SSE ?= 1
ifeq (0,$(SSE)) # discover AVX
  ifeq (1,$(words $(filter avx512f,$(CPUFLAGS))))
    ifeq (2,$(words $(filter avx512pf avx512er,$(CPUFLAGS)))) # KNL
      AVX ?= 3
      MIC ?= 1
    else
      AVX ?= 2
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
MIC ?= 0

ifneq (0,$(INTEL))
  SUITE = Intel Compiler
  MKL_OMPRTL = intel
else ifneq (0,$(GCC))
  SUITE = GNU Compiler Collection
  MKL_OMPRTL = gnu
else ifneq (0,$(PGI))
  SUITE = $(if $(filter-out 0,$(PGI)),PGI $(NULL))Compiler
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

# prepend compiler-local library directory
LIBFIND := $(call qapath,$(call qdir,$(call which,$(LD)))/../lib) $(LIBFIND)

ifneq (Windows_NT,$(UNAME))
ifneq (,$(strip $(PIC)))
  ifneq (,$(call ctest,$(CC),-f$(PIC)))
    PIC := PIC
  endif
  PICFLAG = -f$(PIC)
endif
endif

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

ifneq (,$(COMMON))
ifneq (0,$(COMMON))
  CFLAGS += $(call ctest,$(CC),-fcommon)
else
  CFLAGS += $(call ctest,$(CC),-fno-common)
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
ifneq (,$(call ctest,$(LD),-dynamic))
  EXCLUDE_VALUE += -dynamic
  XLD := $(XLD) -dynamic
  LD := $(LD) -dynamic
  ifeq (0,$(INTEL))
    FLD := $(FLD) -dynamic
  else ifneq (,$(strip $(GFC)))
    FLD := $(FLD) -dynamic
  endif
endif
endif

# enable MKL (if available)
ifeq (,$(strip $(MKLROOT)))
  BLAS_INCFILE = $(strip $(lastword $(sort $(wildcard /opt/intel/compilers_and_libraries_*/$(MKL_PLATFORM)/mkl/include/mkl.h))))
  ifneq (,$(BLAS_INCFILE))
    MKLROOT = $(call qapath,$(call qdir,$(BLAS_INCFILE))/..)
  endif
endif
ifeq (,$(strip $(MKLROOT)))
  BLAS_INCFILE = $(strip $(wildcard /usr/include/mkl/mkl.h))
  ifneq (,$(BLAS_INCFILE))
    MKLROOT = $(call qapath,$(call qdir,$(BLAS_INCFILE))/../..)
  endif
endif
ifneq (,$(BLAS_INCFILE))
  #BLAS_STATIC = 0
  ABSLIBS = 1
endif

# Compiler is used for link stage
ifneq (Darwin,$(UNAME))
  ifneq (ld,$(call qndir,$(LD)))
    XLNKOPT = -Wl,
  endif
  XLNKVERBOSE = --verbose
  linkopt = $(if $1,$(XLNKOPT)$(if $2,$1=$(call quote,$2),$1))
  abslibrpath = $(strip $(if \
    $(filter $(ILIBEXT),$(SLIBEXT))$(if $(filter %.$(SLIBEXT),$1),$(NULL),$1), \
    $(call linkopt,--rpath,$(call qdir,$(call qapath,$1)))))
  XGROUP_BEGIN = $(call linkopt,--start-group)
  XGROUP_END = $(call linkopt,--end-group)
  ifneq (0,$(ASNEEDED))
    XLIB_BEGIN = $(call linkopt,--as-needed)
    XLIB_END = $(call linkopt,--no-as-needed)
  endif
else # OSX
  ifneq (ld,$(call qndir,$(LD)))
    XLNKOPT = -Xlinker
  endif
  #XLNKVERBOSE = -t
  linkopt = $(if $1,$(XLNKOPT) $(if $2,$1 $(XLNKOPT) $(call quote,$2),$1))
  abslibrpath = $(strip $(if \
    $(filter $(ILIBEXT),$(SLIBEXT))$(if $(filter %.$(SLIBEXT),$1),$(NULL),$1), \
    $(call linkopt,-rpath,$(call qdir,$(call qapath,$1)))))
endif

absliblpath = $(strip $(if $1, \
  $(if $(findstring .$(ILIBEXT),$1),-L$(call qdir,$(call qapath,$1)))))

ifneq (0,$(ABSLIBS))
  abslibpath = $(strip $(call abslibrpath,$1) $(call absliblpath,$1))
endif
abslibpath ?= $(call absliblpath,$1)

libpath = $(strip $(if $1,$(shell $1 $2 -l$3 $(call linkopt,$(XLNKVERBOSE)) 2>&1 \
    | grep "lib$3" | tr " " "\n" | sed -n "/\//p" | sed "s/[$(CHAR_OPEN)]\(..*\)[$(CHAR_CLOSE)]/\1/" \
    | xargs -I {} sh -c "if [ -f '{}' ]; then echo '{}'; else ls -1 {}.* 2>/dev/null; fi" | tail -n1)))

ifneq (Windows_NT1,$(OS)$(DEPSTATIC))
  abslibfile = $(strip $(if $(findstring .$(ILIBEXT),$1), \
    $(if $(patsubst lib%,%,$(call qname,$(call qndir,$1))), \
       -l$(patsubst lib%,%,$(call qname,$(call qndir,$1)))), \
    $(if $(filter $(call qndir,$1),$(call qapath,$1)),$(call qapath,$1),$1)))
else # Cygwin/MinGW (static)
  abslibfile = $(strip $(if $(findstring .$(ILIBEXT),$1), \
    $(if $(call qname,$(call qndir,$1)), \
       -l$(call qname,$(call qndir,$1))), \
    $(if $(filter $(call qndir,$1),$(call qapath,$1)),$(call qapath,$1),$1)))
endif
abslib = $(strip $(call abslibpath,$1) $(call abslibfile,$1))

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
  MAINLIB := $(MAINLIB) $(NOBLASLIB)
  #EXTLIB := $(EXTLIB) $(NOBLASLIB)
  LIBDEP := $(LIBDEP) $(NOBLASDEP)
endif

ifneq (Darwin,$(UNAME))
  ifeq (0,$(shell ln -fs this-file-does-not-exist .ln 2>/dev/null && echo "$$?" && rm .ln 2>/dev/null))
    solink = -o "$1.$2.$3.$4" $(call linkopt,-soname,$(strip $(call qndir,$1).$5))
    ifneq (0,$(SONAMELNK))
      solink += $(shell cd $(call qdir,$1) 2>/dev/null && \
        ln -fs $(call qndir,$1.$2.$3.$4) $(call qndir,$1.$5))
    endif
    ifneq (0,$(shell echo "$$((1<$(SONAMELNK) || 0>$(SONAMELNK)))"))
      solink += $(shell cd $(call qdir,$1) 2>/dev/null && \
        ln -fs $(call qndir,$1.$5) $(call qndir,$1))
    endif
  else # MinGW
    solink = -o $(call quote,$1) $(call linkopt,-soname,$(strip $(call qndir,$1).$5))
  endif
else # macOS
  solink = -o $(call qbname,$1).$2$(call qsuffix,$1) \
           -install_name $(call qndir,$(call qbname,$1).$2$(call qsuffix,$1)) \
           -current_version $2.$3.$4 -compatibility_version $5
  ifneq (0,$(SONAMELNK))
    solink += $(shell cd $(call qdir,$1) 2>/dev/null && \
      ln -fs $(call qndir,$(call qbname,$1).$2$(call qsuffix,$1)) $(call qndir,$1))
  endif
endif

ifneq (0,$(INTEL))
  ifeq (1,$(STATIC))
    ifeq (0,$(ICX))
      SLDFLAGS += -no-intel-extensions -static-intel -static-libstdc++
    endif
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

LIBATOMIC ?= 0
ifneq (0,$(THREADS))
ifneq (0,$(LIBATOMIC))
ifneq (,$(call ldlib,$(LD),$(SLDFLAGS),atomic))
  LDFLAGS += $(call ldlib,$(LD),$(SLDFLAGS),atomic)
  DFLAGS += -DLIBXSMM_LIBATOMIC
endif
endif
endif

LIBGFORTRAN = $(call qapath,$(call libpath,$(FLD),$(SLDFLAGS),gfortran))
LIBPTHREAD = $(call ldlib,$(LD),$(SLDFLAGS),pthread)
LIBCPP = $(call ldlib,$(LD),$(SLDFLAGS),stdc++)
LIBRT = $(call ldlib,$(LD),$(SLDFLAGS),rt)
LIBDL = $(call ldlib,$(LD),$(SLDFLAGS),dl)
ifneq (0,$(INTEL))
  LIBM = $(call ldlib,$(LD),$(SLDFLAGS),imf)
endif
LIBM ?= $(call ldlib,$(LD),$(SLDFLAGS),m)

# (default) runtime library dependencies
ifneq (0,$(FORCE_CXX)) # incl. undefined
  FCLDFLAGS += $(XLIB_BEGIN) $(call ldlib,$(FLD),$(SLDFLAGS),stdc++) $(XLIB_END)
  CLDFLAGS += $(XLIB_BEGIN) $(LIBCPP) $(XLIB_END)
endif
FCLDFLAGS += $(XLIB_BEGIN) $(call ldlib,$(FLD),$(SLDFLAGS),c) $(XLIB_END)
CXXLDFLAGS += $(XLIB_BEGIN) $(call ldlib,$(LD),$(SLDFLAGS),c) $(XLIB_END)

ifneq (0,$(INTEL))
  AR ?= xiar
  ifneq (0,$(SYM))
    ifeq (1,$(SYM))
      CXXFLAGS += -g
      CFLAGS += -g
    else
      CXXFLAGS += -g3 -debug inline-debug-info
      CFLAGS += -g3 -debug inline-debug-info
    endif
    ifeq (,$(strip $(GFC)))
      FCFLAGS += -g -traceback
    endif
  endif
  ifneq (0,$(shell echo "$$((170000<=$(CXX_VERSION_NUM)))"))
    CXXFLAGS += -std=c++14
  else ifneq (0,$(shell echo "$$((140000<=$(CXX_VERSION_NUM)))"))
    CXXFLAGS += -std=c++11
  endif
  CXXFLAGS += -Wall -diag-disable 1879,3415,3948,10006,10010,10411,13003
  CFLAGS += -Wall -diag-disable 1879,3415,3948,10006,10010,10411,13003
  ifneq (0,$(UNUSED))
    CXXFLAGS += -Wno-unused-function
    CFLAGS += -Wno-unused-function
  endif
  ifeq (,$(strip $(GFC)))
    ifneq (ld,$(call qndir,$(LD)))
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
  else ifneq (1,$(ICX))
    CPEDANTIC += -diag-disable 593,1599,2415
  endif
  CPEDANTIC += -diag-disable 177,1419,1572,2547,10382
  CWARNEXTRA = -Wremarks
  ifeq (1,$(PEDANTIC))
    ifeq (,$(filter-out 0,$(FORCE_CXX)))
      CSTD = -std=c99
    endif
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
    ifneq (,$(filter 0 1,$(ICX)))
    ifeq (,$(filter-out 0,$(FORCE_CXX)))
      CSTD = -std=c89
    endif
    endif
    CXXFLAGS += $(CPEDANTIC)
    CFLAGS += $(CSTD) $(CPEDANTIC)
    ifneq (990000,$(CC_VERSION_NUM))
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
    ifneq (,$(filter 0 1,$(ICX)))
      ifeq (,$(filter-out 0,$(FORCE_CXX)))
        CSTD = -std=c89
      endif
    else ifneq (0,$(ICX))
      CXXFLAGS += -Wno-pass-failed
      CFLAGS += -Wno-pass-failed
    endif
    ifeq (,$(strip $(GFC)))
      FCFLAGS += $(FFORM_FLAG)
    endif
  endif
  ifeq (0,$(OFFLOAD))
  ifeq (0,$(ICX))
    ifneq (0,$(shell echo "$$((150000<=$(CC_VERSION_NUM)))"))
      NO_OFFLOAD_FLAG = -qno-offload
    else
      NO_OFFLOAD_FLAG = -no-offload
    endif
  endif
  endif
  CXXFLAGS += $(OPTFLAG) $(NO_OFFLOAD_FLAG)
  CFLAGS += $(OPTFLAG) $(NO_OFFLOAD_FLAG)
  ifeq (,$(strip $(GFC)))
    FCFLAGS += $(OPTFLAG) $(NO_OFFLOAD_FLAG)
    # flag specifying output directory must be last
    FMFLAGS += -module
  else
    FCFLAGS += $(OPTFLAG)
    FMFLAGS += -J
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
      ifneq (0,$(shell echo "$$((130000<=$(FC_VERSION_NUM)))"))
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
  OMPFLAG_FORCE = -fopenmp
  ifeq (,$(strip $(OMPRT)))
    OMPRT = iomp5
  endif
  ifneq (0,$(OMP))
    CXXFLAGS += $(OMPFLAG_FORCE)
    FCFLAGS += $(OMPFLAG_FORCE)
    CFLAGS += $(OMPFLAG_FORCE)
    ifneq (,$(strip $(GFC)))
      LDFLAGS += $(XLIB_BEGIN) $(call ldlib,$(LD),$(SLDFLAGS),$(OMPRT)) $(XLIB_END)
    else
      LDFLAGS += $(OMPFLAG_FORCE)
    endif
  endif
  ifneq (0,$(SIMD))
  ifneq (0,$(shell echo "$$((150000<=$(CXX_VERSION_NUM)))"))
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
  ifneq (0,$(shell echo "$$(((180000<=$(CC_VERSION_NUM) && 180001>$(CC_VERSION_NUM)) || (170006>$(CC_VERSION_NUM) && 0!=$(CC_VERSION_NUM))))"))
    CFLAGS += -D_Float128=__float128
  endif
else # GCC assumed
  ifneq (0,$(SYM))
    ifeq (1,$(SYM))
      CXXFLAGS += -g
      CFLAGS += -g
      FCFLAGS += -g
    else ifeq (2,$(SYM))
      CXXFLAGS += -g -fsanitize=thread -fno-omit-frame-pointer
      CFLAGS += -g -fsanitize=thread -fno-omit-frame-pointer
      FCFLAGS += -g -fsanitize=thread -fno-omit-frame-pointer
      LDFLAGS += -g -fsanitize=thread $(call ldlib,$(LD),$(SLDFLAGS),tsan) -fno-omit-frame-pointer
    else
      ifneq (,$(filter 2 3,$(GCC)))
        CXXFLAGS += -g3
        CFLAGS += -g3
      else
        CXXFLAGS += -g
        CFLAGS += -g
      endif
      ifeq (3,$(GCC))
        FCFLAGS += -g3
      else
        FCFLAGS += -g
      endif
    endif
  endif
  ifeq (0,$(COMPATIBLE))
    ifneq (0,$(shell echo "$$((50000<=$(GCC_VERSION_NUM)))"))
      CXXFLAGS += -std=c++14
    else ifneq (0,$(shell echo "$$((40700<=$(GCC_VERSION_NUM)))"))
      CXXFLAGS += -std=c++11
    else ifneq (0,$(CLANG))
      ifneq (0,$(shell echo "$$((40000<=$(CXX_VERSION_NUM)))"))
        CXXFLAGS += -std=c++14
      else
        CXXFLAGS += -std=c++11
      endif
    else ifneq (,$(filter-out 2 3,$(GCC)))
      CXXFLAGS += -std=c++11
    endif
  endif
  ifeq (,$(filter-out 0,$(COMPATIBLE) $(PGI)))
    CXXFLAGS += -Wall
    CFLAGS += -Wall
    FSTD = -std=f2003
    CPEDANTIC += -pedantic -Wextra -Wno-variadic-macros
    FPEDANTIC += -pedantic -Wextra -Wunused-variable \
                 -Wcharacter-truncation -Wline-truncation \
                 -Wconversion -Wintrinsics-std \
                 -Wimplicit-interface -Wimplicit-procedure
    ifneq (0,$(shell echo "$$((40200<=$(CC_VERSION_NUM)))"))
      CPEDANTIC += -Wno-overlength-strings
    else ifneq (0,$(CLANG))
      CPEDANTIC += -Wno-overlength-strings
    endif
    ifneq (0,$(shell echo "$$((40500<=$(CC_VERSION_NUM)))"))
      CPEDANTIC += -Wshadow
    endif
    ifneq (,$(strip $(FC)))
    ifneq (0,$(shell echo "$$((50000<=$(FC_VERSION_NUM)))"))
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
      ifneq (,$(filter 0 1,$(PEDANTIC)))
        CFLAGS += -Wno-format
      endif
    endif
    FPEDANTIC += $(FWARNEXTRA)
    ifeq (1,$(PEDANTIC))
      ifeq (,$(filter-out 0,$(FORCE_CXX)))
        CSTD = -std=c99
      endif
      CXXFLAGS += $(CPEDANTIC) -Wno-long-long
      #CXXFLAGS += -Wno-missing-field-initializers
      CFLAGS += $(CSTD) $(CPEDANTIC)
      FCFLAGS += $(FFORM_FLAG)
      FMFLAGS += $(FSTD) -pedantic -Wunused-variable $(FWARNEXTRA)
    else ifneq (0,$(PEDANTIC))
      ifneq (Darwin,$(UNAME))
        ifeq (,$(filter-out 0,$(FORCE_CXX)))
          CSTD = -std=c89
        endif
        CPEDANTIC += -Wno-long-long
        CXXFLAGS += $(CPEDANTIC)
      else ifneq (0,$(GCC))
        ifeq (,$(filter-out 0,$(FORCE_CXX)))
          CSTD = -std=c89
        endif
        CPEDANTIC += -Wno-long-long
        CXXFLAGS += $(CPEDANTIC)
      else # Clang may run into ICEs under macOS
        ifeq (,$(filter-out 0,$(FORCE_CXX)))
          CSTD = -std=c99
        endif
        CXXFLAGS += $(CPEDANTIC) -Wno-long-long
      endif
      #CXXFLAGS += -Wno-missing-field-initializers #-Wzero-as-null-pointer-constant
      CFLAGS += $(CSTD) $(CPEDANTIC)
      FCFLAGS += $(FSTD) $(FPEDANTIC)
    else ifeq (0,$(COMPATIBLE))
      CPEDANTIC += -Wno-long-long #-Wno-missing-field-initializers
      FCFLAGS += $(FFORM_FLAG)
      ifeq (,$(filter-out 0,$(FORCE_CXX)))
        CSTD = -std=c89
      endif
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
    ifneq (0,$(IPO))
      CXXFLAGS += -flto
      CFLAGS += -flto
      FCFLAGS += -flto
      #FLDFLAGS += -fno-lto
      LDFLAGS += $(call linkopt,-flto)
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
      ifneq (0,$(shell echo "$$((40300<=$(GCC_VERSION_NUM)))"))
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
    LDFLAGS += $(XLIB_BEGIN) $(call ldlib,$(LD),$(SLDFLAGS),dbghelp) $(XLIB_END)
  else ifeq (FreeBSD,$(UNAME))
    LDFLAGS += $(XLIB_BEGIN) $(call ldlib,$(LD),$(SLDFLAGS),execinfo) $(XLIB_END)
  endif
  ifeq (0,$(COMPATIBLE))
    ifneq (0,$(PGI))
      OMPFLAG_FORCE = -mp
      OMPRT = omp
    else ifneq (0,$(GCC))
      OMPFLAG_FORCE = -fopenmp
    else ifneq (Darwin,$(UNAME))
      ifneq (0,$(shell echo "$$((0!=$(CLANG) && 30900<=$(CC_VERSION_NUM)))"))
        OMPFLAG_FORCE = -fopenmp
        OMPRT = omp
      endif
    else # Darwin
      OMPFLAG_FORCE = -Xpreprocessor -fopenmp
      OMPRT = omp
    endif
  endif
  ifeq (,$(OMPFLAG_FORCE))
    ifneq (,$(filter environment% override command%,$(origin OMP)))
      OMPFLAG_FORCE = -fopenmp
    endif
  else ifeq (FreeBSD,$(UNAME))
    # avoid include path at begin of compile line
    OMPFLAG_FORCE += -I/usr/local/include
  endif
  # account for missing TLS/OMP
  ifeq (,$(OMPFLAG_FORCE))
    THREADS ?= 0
    OMP = 0
  endif
  ifeq (,$(strip $(OMPRT))) # fallback
    OMPRT = gomp
  endif
  OMPLIBFILE = $(call libpath,$(LD),$(SLDFLAGS) $(OMPFLAG_FORCE) $(addprefix -L,$(LIBFIND)),$(OMPRT))
  ifeq (,$(OMPLIBFILE))
  ifneq (,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c) && \
      printf "$(CHAR_HASH)include <omp.h>\nint main(void) { return omp_get_max_threads(); }\n" > $${INFILE} && \
      $(LD) $(SLDFLAGS) $(OMPFLAG_FORCE) $(addprefix -L,$(LIBFIND)) -l$(OMPRT) $${INFILE} -o $${INFILE}.x 2>/dev/null >/dev/null && echo "OK"; \
      rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
    LDFLAGS += $(addprefix -L,$(LIBFIND))
    OMPLIBFILE = -l$(OMPRT)
  endif
  endif
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
      else
        ifneq (Darwin,$(UNAME))
          ifneq (0,$(CLANG))
            CXXLDFLAGS += $(call absliblpath,$(OMPLIBFILE))
            CLDFLAGS += $(call absliblpath,$(OMPLIBFILE))
          endif
          CXXLDFLAGS += $(OMPFLAG_FORCE)
          CLDFLAGS += $(OMPFLAG_FORCE)
        endif
        CXXFLAGS += $(OMPFLAG_FORCE)
        CFLAGS += $(OMPFLAG_FORCE)
      endif
    endif
    ifeq (3,$(GCC)) # pure GNU toolchain
      FLDFLAGS += -fopenmp
      FCFLAGS += -fopenmp
    else # mixed toolchain
      OMPTEST = $(call ldlib,$(FLD),$(SLDFLAGS),$(OMPRT))
      ifneq (,$(OMPTEST))
        FLDFLAGS += $(OMPTEST)
      else
        OMPTEST = $(call ldlib,$(FLD),$(SLDFLAGS) $(addprefix -L,$(LIBFIND)),$(OMPRT))
        ifneq (,$(OMPTEST))
          FLDFLAGS += $(addprefix -L,$(LIBFIND)) $(OMPTEST)
        else # last try
          FLDFLAGS += $(call ldlib,$(FLD),$(SLDFLAGS),gomp)
        endif
      endif
      OMPTEST = $(call ctest,$(FC),-fopenmp)
      ifneq (,$(OMPTEST))
        FCFLAGS += $(OMPTEST)
      else
        FCFLAGS += $(call ctest,$(FC),$(OMPFLAG_FORCE))
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
  ifeq (,$(call ctest,$(CC),$(CTARGET)))
    undefine CTARGET
  endif
  endif
  ifeq (,$(CTARGET))
  ifneq (0,$(shell echo "$$((3>$(DBG)))"))
    ifeq (3,$(AVX))
    ifneq (0,$(shell echo "$$(((0!=$(GCC) && 60000<=$(GCC_VERSION_NUM)) || \
          (0!=$(CLANG) && (40000<=$(CC_VERSION_NUM) || 0==$(CC_VERSION_NUM))) || \
          (0!=$(PGI) && 190000<=$(CC_VERSION_NUM))))"))
    ifneq (,$(call ctest,$(CC),-mfma -mavx512f -mavx512cd))
      CTARGET = -mfma -mavx512f -mavx512cd
      ifneq (0,$(lastword $(sort 0 $(MIC)))) # MIC
        ifneq (,$(call ctest,$(CC),-mavx512pf -mavx512er))
          CTARGET += -mavx512pf -mavx512er
        endif
      else ifneq (,$(call ctest,$(CC),-mavx512dq -mavx512bw -mavx512vl))
        ifneq (Darwin,$(UNAME))
          CTARGET += -mavx512dq -mavx512bw -mavx512vl
          #CTARGET += -mavx512ifma -mavx512vbmi
        else ifneq (,$(filter 1 2 3,$(INTEL) $(GCC)))
          CTARGET += -mavx512dq -mavx512bw -mavx512vl
          #CTARGET += -mavx512ifma -mavx512vbmi
        else ifneq (0,$(shell echo "$$((0!=$(CLANG) && 80100<=$(CC_VERSION_NUM)))"))
          CTARGET += -mavx512dq -mavx512bw -mavx512vl
        endif
      endif
    endif
    endif
    endif
    ifeq (,$(CTARGET)) # fallback to AVX2
    ifneq (,$(filter 2 3,$(AVX)))
      ifneq (0,$(shell echo "$$((0!=$(INTEL) || 0!=$(CLANG) || (40800<=$(GCC_VERSION_NUM) && 0==$(PGI)) || 0==$(CC_VERSION_NUM)))"))
        CTARGET ?= -march=core-avx2
      else
        CTARGET ?= -mavx2 -mfma
      endif
    endif
    endif
    ifeq (,$(CTARGET)) # fallback to AVX
    ifneq (,$(filter 1 2 3,$(AVX)))
      ifneq (0,$(shell echo "$$((0!=$(INTEL) || 0!=$(CLANG) || 40400<=$(GCC_VERSION_NUM) || 0==$(CC_VERSION_NUM)))"))
        CTARGET ?= -mavx
      endif
    endif
    endif
    ifneq (,$(filter 1 2 3 4,$(SSE) $(AVX)))
      ifeq (,$(CTARGET)) # SSE-4.2 or default-SSE
        ifneq (,$(filter 01 02 03 10 11 12 13 40 41 42 43,$(SSE)$(AVX)))
          ifneq (0,$(GCC))
            ifneq (0,$(shell echo "$$((40300<=$(CC_VERSION_NUM)))"))
              ifeq (0,$(MAINTAINER))
                CTARGET = -msse4.2
              else # maintainer build
                ifeq (0,$(DEPSTATIC))
                  TARGET = -msse4.2
                  CTARGET = -msse3
                else ifneq (0,$(SHARED))
                  ifneq (,$(SHARED))
                    TARGET = -msse4.2
                    CTARGET = -msse3
                  else
                    CTARGET = -msse4.2
                  endif
                else
                  CTARGET = -msse4.2
                endif
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
      # stop here as SSE=2 is implicitly present (64-bit ABI)
    else ifneq (0,$(AVX))
      CTARGET = -march=native
    endif
  endif # CTARGET
  endif # DBG
else # take user's TARGET into account
  CTARGET := $(TARGET)
endif

ifneq (,$(call ctest,$(CC),$(CTARGET)))
  # check target flags by feeding an ad-hoc C program into the compiler
  ifneq (0,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c) && \
      echo "int main(void) { return 0; }" > $${INFILE} && \
      RESULT=$$($(CC) $(CTARGET) -c $${INFILE} -o $${INFILE}.o 2>&1); \
      if [ "" = "$${RESULT}" ]; then echo "$$?"; else echo "1"; fi; \
      rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
    CTARGET = $(NULL)
  endif
  ifneq (,$(strip $(FC))) # check target flags by feeding an ad-hoc Fortran program into the compiler
  # INFILE cannot start with a leading dot when used with certain compiler
  ifeq (0,$(shell INFILE=$$($(MKTEMP) /tmp/_libxsmm_XXXXXX.f) && \
      printf "      PROGRAM test\n      END PROGRAM\n" > $${INFILE} && \
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

ifeq (0,$(SYM))
  LDFLAGS += $(call ctest,$(LD),-s)
endif

ifeq (0,$(COMPATIBLE))
  ifneq (0,$(OPT))
  ifeq (0,$(INTEL))
    ifneq (0,$(SIMD))
      ifneq (0,$(GCC))
        ifneq (,$(CTARGET))
        ifneq (0,$(shell echo "$$((40900<=$(CC_VERSION_NUM)))"))
          DFLAGS += -DLIBXSMM_OPENMP_SIMD
          CFLAGS += -fopenmp-simd
          ifneq (1,$(GCC))
          ifneq (0,$(shell echo "$$((40900<=$(CXX_VERSION_NUM)))"))
            CXXFLAGS += -fopenmp-simd
            ifneq (,$(FTARGET))
            ifneq (2,$(GCC))
            ifneq (0,$(shell echo "$$((40900<=$(FC_VERSION_NUM)))"))
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
        ifneq (0,$(shell echo "$$((60000<=$(CC_VERSION_NUM) && 60000<=$(CXX_VERSION_NUM)))"))
          DFLAGS += -DLIBXSMM_OPENMP_SIMD
          CXXFLAGS += -fopenmp-simd
          CFLAGS += -fopenmp-simd
          ifneq (,$(FTARGET))
          ifneq (,$(strip $(FC)))
          ifneq (0,$(shell echo "$$((40900<=$(FC_VERSION_NUM)))"))
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
  endif
  endif
  ifneq (,$(filter 1 2,$(patsubst O%,%,$(OPT))))
  ifneq (,$(CTARGET))
    ifeq (,$(filter-out 0,$(PGI)))
      CXXFLAGS += -funroll-loops
      CFLAGS += -funroll-loops
      FCFLAGS += -funroll-loops
    endif
    ifeq (,$(filter-out 0,$(PGI) $(INTEL)))
      CXXFLAGS += -ftree-vectorize
      CFLAGS += -ftree-vectorize
      FCFLAGS += -ftree-vectorize
    endif
  endif
  endif
  ifneq (Darwin,$(UNAME))
    ifneq (0,$(HARDEN)) # not defined: enabled
      ifneq (,$(strip $(HARDEN))) # explicit
        ifneq (0,$(shell echo "$$((40900<=$(FC_VERSION_NUM)))"))
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
    ifneq (ld,$(call qndir,$(LD)))
    ifneq (Windows_NT,$(UNAME))
      ifneq (0,$(DEPSTATIC))
        EXCLUDE_VALUE += $(call linkopt,--export-dynamic)
        ifneq (0,$(HARDEN))
          ifneq (,$(strip $(HARDEN))) # explicit
            LDFLAGS := $(call linkopt,--export-dynamic) $(LDFLAGS)
          else ifneq (0,$(SYM))
            LDFLAGS := $(call linkopt,--export-dynamic) $(LDFLAGS)
          endif
        else ifneq (0,$(SYM))
          LDFLAGS := $(call linkopt,--export-dynamic) $(LDFLAGS)
        endif
      endif
      # Linux distributions may apply similar hardening
      LDFLAGS := $(XLNKOPT)-z,relro,-z,now $(LDFLAGS)
    endif
    endif
    ifeq (0,$(PGI))
    ifneq (0,$(OPT))
      CXXFLAGS += -fdata-sections -ffunction-sections
      CFLAGS += -fdata-sections -ffunction-sections
      ifeq (0,$(INTEL))
        FCFLAGS += -fdata-sections -ffunction-sections
      else ifneq (,$(strip $(GFC)))
        FCFLAGS += -fdata-sections -ffunction-sections
      endif
      # --gc-sections: relies on section-flags present at compile-stage
      LDFLAGS := $(call linkopt,--gc-sections) $(LDFLAGS)
      ifeq (0,$(VISIBILITY)) # -fvisibility=hidden may cause crashes
        CXXFLAGS += -fvisibility=hidden -fvisibility-inlines-hidden
        #FCFLAGS += -fvisibility=hidden
        CFLAGS += -fvisibility=hidden
      endif
    endif
    endif
  endif
  ifeq (0,$(EXP))
    CXXFLAGS += -fno-exceptions
  endif
endif

ifeq (0,$(VLA))
  DFLAGS += -DLIBXSMM_NO_VLA
else ifneq (,$(VLA))
  DFLAGS += -DLIBXSMM_VLA
endif

# Information which can be displayed by the actual Makefile
ifneq (,$(strip $(FC)))
  GINFO = $(SUITE): $(strip $(CC_NAME) $(CC_VERSION)), $(strip $(CXX_NAME) $(CXX_VERSION)), and $(strip $(FC_NAME) $(FC_VERSION))
  FINFO = Fortran target: $(if $(FTARGET),$(FTARGET),<compiler default>)
else
  GINFO = $(SUITE): $(strip $(CC_NAME) $(CC_VERSION)), and $(strip $(CXX_NAME) $(CXX_VERSION))
  FINFO = Fortran: <none>
endif
CINFO = C / C++ target: $(if $(CTARGET),$(CTARGET),<compiler default>)

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

ifeq (,$(filter-out 0,$(COMPATIBLE) $(PGI)))
  ifneq (,$(filter-out 0,$(THREADS) $(BLAS)))
    CFLAGS += -pthread
    CXXFLAGS += -pthread
  endif
  FLDFLAGS += $(XLIB_BEGIN) $(call ldlib,$(FLD),$(SLDFLAGS),pthread) $(XLIB_END)
  ifneq (Windows_NT,$(UNAME))
    ifneq (0,$(INTEL))
      ifeq (0,$(OFFLOAD))
        CLDFLAGS += $(XLIB_BEGIN) $(LIBPTHREAD) $(XLIB_END)
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

ifeq (0,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c) && \
    printf "$(CHAR_HASH)include <features.h>\n$(CHAR_HASH)if !defined(__GNU_LIBRARY__) && !defined(__GLIBC__)\n0\n$(CHAR_HASH)endif\n" > $${INFILE} && \
    RESULT=$$($(CC) -c $${INFILE} -o $${INFILE}.o 2>&1); if [ "" = "$${RESULT}" ]; then echo "$$?"; else echo "1"; fi; \
    rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
  GLIBC = 1
endif
GLIBC ?= 0

OMPLIBFILE ?= $(call libpath,$(LD),$(SLDFLAGS) $(OMPFLAG_FORCE) $(addprefix -L,$(LIBFIND)),$(OMPRT))
ifneq (,$(strip $(OMPLIBFILE)))
  OMPLIB = $(call abslib,$(OMPLIBFILE))
endif

ifneq (,$(OMPFLAG_FORCE))
ifeq (0,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c) && \
    printf "$(CHAR_HASH)if defined(_OPENMP)\n$(CHAR_HASH) include <omp.h>\nint main() { return 0; }\n$(CHAR_HASH)endif\n" > $${INFILE} && \
    RESULT=$$($(CC) $(OMPFLAG_FORCE) $${INFILE} -o $${INFILE}.x 2>&1); if [ "" = "$${RESULT}" ]; then echo "$$?"; else echo "1"; fi; \
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
        TBB_LIBDIRGCC = $(shell ls -1 $(call quote,$(TBB_LIBDIR)) | head -n1)
      else
        TBB_LIBDIRGCC = $(shell ls -1 $(call quote,$(TBB_LIBDIR)) | tail -n1)
      endif
      TBB_LIBMALLOC = $(wildcard $(TBB_LIBDIR)/$(TBB_LIBDIRGCC)/libtbbmalloc.$(ILIBEXT))
    endif
    ifneq (,$(TBB_LIBMALLOC))
      IFLAGS += -I$(call quote,$(TBBROOT)/include)
      DFLAGS += -D__TBB
      LDFLAGS += $(XLIB_BEGIN) $(call abslib,$(TBB_LIBMALLOC)) $(XLIB_END)
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
  else # macOS
    MKL_PLATFORM = mac
  endif
endif

ifneq (0,$(BLAS_STATIC))
  BLASLIBEXT ?= $(SLIBEXT)
else # shared (DLL)
  BLASLIBEXT ?= $(ILIBEXT)
endif

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
      ifneq (0,$(shell echo "$$((0==$(GCC) || 40600<=$(GCC_VERSION_NUM)))"))
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
    BLAS_LDFLAGS += -qoffload-option,mic,ld,"$(LIBM) $(LIBDL)"
  endif
  endif
else ifneq (0,$(MKL)) # multi-threaded
  BLAS_FLAGS += -D__BLAS=$(MKL) -D__MKL=$(MKL)
  MKL_THREADS = $(MKL_OMPRTL)_thread
  ifeq (0,$(OMP))
    ifneq (gnu,$(MKL_OMPRTL))
      BLAS_LDFLAGS += $(OMPLIB)
    else
      BLAS_LDFLAGS += $(call ldlib,$(LD),$(SLDFLAGS),gomp)
    endif
  else ifeq (gnu,$(OMPLIB)$(MKL_OMPRTL))
    BLAS_LDFLAGS += $(call ldlib,$(LD),$(SLDFLAGS),gomp)
  endif
  ifneq (0,$(THREADS))
    BLAS_LDFLAGS += $(XLIB_BEGIN) $(LIBPTHREAD) $(XLIB_END)
  endif
  ifneq (0,$(INTEL))
  ifneq (0,$(OFFLOAD))
    BLAS_LDFLAGS += -qoffload-option,mic,ld,"$(LIBM) $(LIBDL)"
    ifeq (0,$(OMP))
      BLAS_LDFLAGS += -qoffload-option,mic,ld,"$(OMPLIB)"
    endif
  endif
  endif
endif
ifneq (0,$(MKL))
  ifeq (,$(BLAS_INCFILE))
    BLAS_IFLAGS += -I$(call quote,$(MKLROOT)/include)
  else
    BLAS_IFLAGS += -I$(call qdir,$(BLAS_INCFILE))
  endif
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
    BLASDIR ?= $(wildcard $(LIBFIND) /usr/lib /usr/lib64)
    BLASROOT = $(wildcard $(patsubst %,%/..,$(BLASDIR)))
    ifeq (0,$(BLAS_STATIC)) # shared
      BLASTEST = $(filter-out -static,$(SLDFLAGS)) $(patsubst %,-L%,$(BLASDIR))
    else # static
      BLASTEST = $(SLDFLAGS) $(patsubst %,-L%,$(BLASDIR))
    endif
    BLASLIBFILE = $(call libpath,$(LD),$(BLASTEST),openblas$(BLAS_THREADS)$(BLAS_BITS))
    ifeq (,$(BLASLIBFILE)) # newer distributions symlink non-decorated library to threaded OpenBLAS
      BLASLIBFILE = $(call libpath,$(LD),$(BLASTEST),openblas$(BLAS_BITS))
    endif
    # most people expect to pickup OpenBLAS (if available) even when libblas/liblapack are available; use OPENBLAS=0 to avoid this
    ifeq (0,$(PGI))
      OPENBLAS := $(if $(BLASLIBFILE),1,0)
    #else ifeq (Darwin,$(UNAME)) # Apple
      #OPENBLAS := 0
      #BLASREF = 0
    else # PGI
      OPENBLAS := 0
    endif
    ifneq (0,$(OPENBLAS)) # OpenBLAS
      # OpenBLAS also carries the CBLAS bits
      BLAS_FLAGS += -D__BLAS=$(BLAS) -D__CBLAS -D__OPENBLAS
      ifneq (,$(wildcard $(patsubst %,%/include/openblas/f77blas.h,$(BLASROOT))))
        BLAS_FLAGS += -D__OPENBLAS77 -I$(call qdir,$(firstword $(wildcard $(patsubst %,%/include/openblas/f77blas.h,$(BLASROOT)))))
      else ifneq (,$(wildcard $(patsubst %,%/include/x86_64-linux-gnu/f77blas.h,$(BLASROOT))))
        BLAS_FLAGS += -D__OPENBLAS77 -I$(call qdir,$(firstword $(wildcard $(patsubst %,%/include/x86_64-linux-gnu/f77blas.h,$(BLASROOT)))))
      else ifneq (,$(wildcard $(patsubst %,%/include/f77blas.h,$(BLASROOT))))
        BLAS_FLAGS += -D__OPENBLAS77 -I$(call qdir,$(firstword $(wildcard $(patsubst %,%/include/f77blas.h,$(BLASROOT)))))
      endif
    else ifneq (0,$(filter 0,$(BLASREF))) # BLAS (reference)
      BLAS_FLAGS += -D__BLAS
      ifeq (Windows_NT,$(UNAME)) # no particular action about static linkage (use DLL)
        BLASLIBFILE = $(call libpath,$(LD),$(BLASTEST),blas$(BLAS_BITS).dll)
      else
        BLASLIBFILE = $(call libpath,$(LD),$(BLASTEST),blas$(BLAS_BITS))
      endif
    endif
    ifneq (,$(BLASLIBFILE))
      ifeq (0,$(BLAS_STATIC)) # shared
        BLAS_LDFLAGS += $(call abslib,$(BLASLIBFILE))
      else # static
        BLAS_LDFLAGS += $(BLASLIBFILE)
      endif
      BLASREF ?= 1
    else ifneq (,$(call ctest,$(LD),-framework Accelerate)) # macOS
      BLAS_INCFILE = $(shell $(LD) $(call linkopt,-v) 2>&1 | grep -i frameworks 2>/dev/null | xargs -I {} find {} -name cblas.h 2>/dev/null | head -n1)
      ifneq (,$(BLAS_INCFILE))
        BLAS_FLAGS += -I$(call qapath,$(call qdir,$(BLAS_INCFILE)))
      endif
      #EXCLUDE_VALUE += -framework Accelerate
      BLAS_LDFLAGS += -framework Accelerate
      BLAS_FLAGS += -D__CBLAS
    else # fallback
      BLAS_LDFLAGS += $(call ldlib,$(LD),$(SLDFLAGS),blas)
      BLASREF ?= 1
    endif
    BLASREF ?= 0
    ifneq (0,$(BLASREF)) # BLAS (reference)
      ifeq (0,$(PGI))
        ifneq (0,$(DEPSTATIC))
          ifneq (,$(LIBGFORTRAN))
            BLASTEST += $(XLIB_BEGIN) $(LIBGFORTRAN) $(LIBM) $(XLIB_END)
          else ifeq (3,$(GCC))
            BLASTEST += $(XLIB_BEGIN) $(call ldlib,$(LD),$(SLDFLAGS),gfortran) $(LIBM) $(XLIB_END)
          endif
        endif
        QUADMATH = $(call ldlib,$(LD),$(SLDFLAGS),quadmath)
        ifneq (,$(QUADMATH))
          BLASTEST += $(XLIB_BEGIN) $(QUADMATH) $(call ldlib,$(LD),$(SLDFLAGS),m) $(XLIB_END)
        endif
        ifneq (3,$(GCC))
          BLAS_LDFLAGS += $(BLASTEST)
        else
          BLAS_CLDFLAGS += $(BLASTEST)
        endif
      else # PGI
        BLAS_FLAGS += -D__CBLAS
      endif
    endif
    ifneq (,$(filter-out 0,$(OPENBLAS) $(BLASREF))) # OpenBLAS or RefBLAS
    ifeq (0,$(OMP)) # Fortran compiler cannot link OpenMP runtime
      ifeq (,$(call libpath,$(FLD),$(SLDFLAGS) $(OMPFLAG_FORCE) $(addprefix -L,$(LIBFIND)),$(OMPRT)))
        BLAS_CLDFLAGS += $(XLIB_BEGIN) $(call abslibfile,$(OMPLIBFILE)) $(XLIB_END)
      else # common link BLAS/OMP-link flags (all compilers)
        BLAS_LDFLAGS += $(XLIB_BEGIN) $(call abslibfile,$(OMPLIBFILE)) $(XLIB_END)
      endif
    endif
    endif
  else # BLAS library is specified via BLASLIB
    BLAS_FLAGS += -D__BLAS=$(BLAS)
    ifneq (,$(findstring openblas,$(call qndir,$(BLASLIB))))
      BLAS_FLAGS += -D__CBLAS -D__OPENBLAS
    endif
    ifneq (./,$(firstword $(BLASDIR))$(call qdir,$(BLASLIB)))
      ifeq (./,$(call qdir,$(BLASLIB)))
        BLAS_LDFLAGS += $(call abslib,$(firstword $(BLASDIR))/$(if $(call qsuffix,$(BLASLIB)),$(BLASLIB),lib$(BLASLIB).$(BLASLIBEXT)))
      else
        BLAS_LDFLAGS += $(call abslib,$(if $(call qsuffix,$(BLASLIB)),$(BLASLIB),$(BLASLIB).$(BLASLIBEXT)))
      endif
    else # fallback
      BLAS_LDFLAGS += $(call ldlib,$(LD),$(SLDFLAGS),$(BLASLIB))
    endif
  endif
endif

cleanup = $(foreach flag,$(subst //,/,$1),$(flag))
# cleanup eventually duplicated flags and slashes
cleanld = $(strip $(shell echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}" | sed \
  -e "s/[[:space:]]$(XLIB_BEGIN)[[:space:]]$(XLIB_END)/ /g" \
  -e "s/[[:space:]]$(XLIB_END)[[:space:]]$(XLIB_BEGIN)/ /g" \
  -e "s/\/\//\//g"))

# common runtime libraries
LDFLAGS := $(call uniqmov,LDFLAGS,LIBPTHREAD,$(XLIB_BEGIN) $(LIBPTHREAD) $(XLIB_END))
LDFLAGS := $(call uniqmov,LDFLAGS,LIBCPP,$(XLIB_BEGIN) $(LIBCPP) $(XLIB_END))
ifeq (0,$(INTEL))
  LDFLAGS := $(call uniqadd,LDFLAGS,LIBM,$(XLIB_BEGIN) $(LIBM) $(XLIB_END))
endif
ifneq (0,$(MKL))
  LDFLAGS := $(call uniqadd,LDFLAGS,LIBM,$(XLIB_BEGIN) $(LIBM) $(XLIB_END))
endif
LDFLAGS := $(call uniqadd,LDFLAGS,LIBRT,$(XLIB_BEGIN) $(LIBRT) $(XLIB_END))
LDFLAGS := $(call uniqadd,LDFLAGS,LIBDL,$(XLIB_BEGIN) $(LIBDL) $(XLIB_END))

# no-BLAS flags: cleanup and extra flags
NOBLAS_CXXFLAGS := $(call cleanup,$(PICFLAG) $(CXXFLAGS) $(ECXXFLAGS) $(EFLAGS))
NOBLAS_FCFLAGS := $(call cleanup,$(PICFLAG) $(FCFLAGS) $(EFCFLAGS) $(EFLAGS))
NOBLAS_CFLAGS := $(call cleanup,$(PICFLAG) $(CFLAGS) $(ECFLAGS) $(EFLAGS))
NOBLAS_FLAGS := $(call cleanup,$(DFLAGS))
NOBLAS_LDFLAGS := $(call cleanld,$(LDFLAGS) $(ELDFLAGS))
NOBLAS_CLDFLAGS := $(call cleanld,$(CLDFLAGS))
# merged ("=" rather than ":=")
NOBLAS_IFLAGS = $(call cleanup,$(IFLAGS))

# regular flags: cleanup
CXXFLAGS := $(call cleanup,$(BLAS_CXXFLAGS) $(NOBLAS_CXXFLAGS))
FCFLAGS := $(call cleanup,$(BLAS_FCFLAGS) $(NOBLAS_FCFLAGS))
CFLAGS := $(call cleanup,$(BLAS_CFLAGS) $(NOBLAS_CFLAGS))
DFLAGS := $(call cleanup,$(BLAS_FLAGS) $(NOBLAS_FLAGS))
IFLAGS := $(call cleanup,$(BLAS_IFLAGS) $(NOBLAS_IFLAGS))
CLDFLAGS := $(call cleanld,$(BLAS_CLDFLAGS) $(NOBLAS_CLDFLAGS))
LDFLAGS := $(call cleanld,$(BLAS_LDFLAGS) $(NOBLAS_LDFLAGS))
CXXLDFLAGS := $(call cleanld,$(CXXLDFLAGS))
FLDFLAGS := $(call cleanld,$(FLDFLAGS))

.PRECIOUS: $(BLDDIR)/%-cpp.o $(BLDDIR)/%-c.o $(BLDDIR)/%-f.o \
           $(BLDDIR)/%-f90.o $(BLDDIR)/%-f90.o $(BLDDIR)/%-f77.o \
           %/.make

.SUFFIXES:

# applyif(A1,A2,A3,A4) evaluates to A4 if the basenames (A1, A2) match and if A1 is non-zero (precondition)
applyif = $(if $(filter 0,$1),$(NULL),$(if $(filter $2,$(call qname,$(call qndir,$3))),$(if $1,$4)))
# derives the extension of a filename
extname = $(subst .,,$(call qsuffix,$(1)))
# derives the name of an object files for a given source file
objname = $(foreach ARG,$(1),$(addprefix $(BLDDIR)/,$(patsubst %$(call qsuffix,$(ARG)),%-$(call extname,$(ARG)).o,$(call qndir,$(ARG)))))

STATE := $(foreach V,$(sort $(.VARIABLES)),$(if \
  $(filter-out environment% default automatic,$(origin $V)), \
  $(if $(filter-out $(EXCLUDE_STATE) EXCLUDE_% .% _% MAKE_% HAVE_% LIB_% %STATE MAKEFILE_LIST MAKEOVERRIDES MAKEFLAGS SHELL BLDDIR REVERSION% AVX \
    COMPILER_VERSION_FLAG CC_VERSION_% CXX_VERSION_% FC_VERSION_% GCC_VERSION_% CINFO CP DEPDIR DFLAGS FINFO SSE ENV EXTDEP FORTDEP FSTD GINFO \
    DLIBEXT DYNAMIC FLOCK LIBEXT ICX ILIBEXT LIBNAME LICFILE MAKEINC MV ROOTDIR SLIBEXT XLNK% CHAR_OPEN CHAR_CLOSE CHAR_HASH MKTEMP OMPLIBFILE \
    CPEDANTIC FREEFORM FFORM_FLAG FORTLIB FIXFC INCDIR OUTDIR SRCDIR XLIB_% GFC CLANG INTEL CRAY GCC GCCBIN FMFLAGS FCLDFLAGS EXTLIB PYTHON% \
    GNU CCAR CLDFLAGS LIBGFORTRAN PGI FPEDANTIC FWARNEXTRA XGROUP_BEGIN XGROUP_END SLDFLAGS PICFLAG PIC OPTFLAG OPT OMPFLAG_FORCE FORCE_CXX \
    NOBLAS% DNOBLAS MAINLIB FLDFLAGS OPSYS LIBDEP LIBFIND LIBDL LIBRT LIBM LIBCPP BLAS_FLAGS BLAS_LDFLAGS BLAS_THREADS SUITE PEDANTIC CMAIN \
    LIBPTHREAD CSTD CWARNEXTRA OMPTEST QUADMATH BLASDIR BLASLIBEXT BLASTEST NO_OFFLOAD_FLAG FYPPEXE INCLUDEMAKE LIBDIR OBJDIR ONEVERSION \
    HEREDIR HOMEDIR UNUSED TESTSIZE SHARED PYMOD MINGW FMAIN VERSION \
    abslib applyif uniqadd uniqmov ldlib which cleanup cleanld linkopt abslibfile abslibpath absliblpath abslibrpath libpath solink \
    ctest extname objname unquote quote qapath qndir qdir qsuffix qbname qname \
    BLAS_INCFILE $(if $(filter 0,$(MKL)),MKL_%) $(if $(filter 0,$(TBB)),TBB_%) \
    $(if $(filter $(BLAS_STATIC),$(DEPSTATIC)),BLAS_STATIC) $(if $(filter-out $(FTARGET),$(CTARGET)),TARGET,CTARGET FTARGET),$V), \
    $(if $($V),$V=$(subst $(USER),$$USER,$(subst $(HOME),$$HOME,$(subst $(HOMEDIR),$$HOME,$(filter-out $(EXCLUDE_VALUE),$($V)))))?))))

DIRSTATE ?= .
PRESTATE := $(shell echo '$(STATE)' | $(DEPDIR)/.state.sh $(DIRSTATE))
$(DIRSTATE)/.state: $(firstword $(PRESTATE))

%/.make:
	@mkdir -p $(call qbname,$@)
	@touch $@

.make:
	@touch $@

.DEFAULT_GOAL := $(NULL)

