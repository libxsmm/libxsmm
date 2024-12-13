###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

MAKE_VERSION_MAJOR := $(shell echo "$(MAKE_VERSION)" | cut -d. -f1)
MAKE_VERSION_MINOR := $(shell echo "$(MAKE_VERSION)" | cut -d. -f2)
MAKE_VERSION_PATCH := $(shell echo "$(MAKE_VERSION)" | cut -d. -f3)
ifeq (,$(MAKE_VERSION_MAJOR))
MAKE_VERSION_MAJOR := 0
endif
ifeq (,$(MAKE_VERSION_MINOR))
MAKE_VERSION_MINOR := 0
endif
ifeq (,$(MAKE_VERSION_PATCH))
MAKE_VERSION_PATCH := 0
endif
# calculate version number given by up to three components
version = $(shell echo "$$(( \
  $(if $1,$(word 1,$1)*10000,0) + \
  $(if $2,$2*100,$(if $(word 2,$1),$(word 2,$1)*100,0)) + \
  $(if $3,$3,$(if $(word 3,$1),$(word 3,$1),0))))")
MAKE_VERSION_INT := $(call version,$(MAKE_VERSION_MAJOR),$(MAKE_VERSION_MINOR),$(MAKE_VERSION_PATCH))

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explicitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(strip $(MAKE_PARALLEL)))
# Force parallel build for old GNU Make
#ifneq (0,$(shell echo "$$((38200>$(MAKE_VERSION_INT)))"))
#.NOTPARALLEL:
#else
ifneq (0,$(shell echo "$$((40201<$(MAKE_VERSION_INT)))"))
MAKEFLAGS += -O
endif
else ifneq (0,$(shell echo "$$((40201<$(MAKE_VERSION_INT)))"))
MAKEFLAGS += -O
endif

#MAKEINC := $(abspath $(dir $(filter %Makefile.inc,$(MAKEFILE_LIST))))
MAKEINC := $(patsubst %/,%,$(dir $(filter %Makefile.inc,$(MAKEFILE_LIST))))

COMMAND := $(shell which command 2>/dev/null)
ifneq (,$(COMMAND))
  which = $(shell $(COMMAND) -v $1)
else
  which = $(shell which $(firstword $1) 2>/dev/null)
endif

# ensure a number or instead return zero (by default) or 2nd argument (if given)
qnum = $(shell echo "$1" | grep "^-*[0-9][0-9]*$$" 2>/dev/null || echo "$(if $2,$2,0)")

ifeq (,$(HOSTNAME))
  HOSTNAME := $(shell hostname 2>/dev/null)
endif
ifeq (Windows_NT,$(OS))
  MNAME ?= $(if $(filter AMD64,$(PROCESSOR_ARCHITECTURE)),x86_64,$(PROCESSOR_ARCHITECTURE))
  UNAME ?= Windows_NT
  # Cygwin/MinGW based
  DLIBEXT ?= dll
  SLIBEXT ?= lib
else
  MNAME ?= $(shell uname -m 2>/dev/null)
  UNAME ?= $(shell uname 2>/dev/null)
  ifneq (Darwin,$(UNAME))
    ENVBIN ?= $(call which,env)
  endif
endif

# Command line utilities
#PKGCFG ?= $(call which,pkg-config)
ifneq (,$(call which,stdbuf))
  FLUSH ?= stdbuf -o0 -e0
endif
MKDIR ?= $(call which,mkdir)
CP ?= $(call which,cp)
MV ?= $(call which,mv)
MAKE ?= make

SED := $(call which,gsed)
# GNU sed is desired (macOS)
ifeq (,$(SED))
  SED := $(call which,sed)
endif

# Python interpreter per PYTHON=/path/to/python
PYTHON3 := $(call which,python3)
# Python3 by default
ifneq (,$(PYTHON3))
  PYTHON := $(PYTHON3)
else ifneq (,$(call which,python))
  ifneq (Windows_NT,$(UNAME))
    PYTHON3 := $(shell ln -s $(call which,python) $(MAKEINC)/python3 2>/dev/null)
    PYTHON := $(MAKEINC)/python3
  else
    PYTHON := python
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
CMAIN := main[[:space:]]*(.*)
FMAIN := ^[[:space:]]*PROGRAM[[:space:]][[:space:]]*\w\w*\([[:space:]][[:space:]]*\!.*\)*$$
# Regular expression to mazch variable name (Make-key)
VNAME := ^ *[^0-9[:punct:]][A-Z0-9_][A-Z0-9_]*

# Debugging and symbols (e.g., when profiling)
SYM ?= 0
DBG ?= 0

# Instrumentation level (trace)
ifeq (,$(strip $(INSTRUMENT)))
  INSTRUMENT := 0
endif
TRACE ?= 0

ifeq (0,$(DBG))
  ifneq (0,$(INSTRUMENT))
    SYM := $(INSTRUMENT)
  endif
  ifneq (,$(filter 0 1,$(SYM)))
    DFLAGS += -DNDEBUG
  endif
else # debugging enabled
  ifneq (0,$(shell echo "$$((1<$(DBG)||0>$(DBG)))"))
    DFLAGS += -D_DEBUG
  endif
  SYM := $(DBG)
endif

# Optimization level
ifeq (0,$(DBG))
  OPT ?= 2
else
  OPT ?= 0
endif

# Optimization flag derived from OPT flag
OPT := $(patsubst O%,%,$(OPT))
OPTFLAGS ?= -O$(OPT)
OPTFLAG1 ?= -O$(shell echo "$$((1<$(OPT)?1:$(OPT)))")

# Kind of Clang/GCC based analysis (thread, address, ...)
SANITIZE ?= $(NULL)

# VMAX=0 (default): Prefer default vector size
# VMAX=1: Prefer widest vector size (VLEN=MAX)
VMAX ?= 0

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

# Delay for user-interaction (sleep)
WAIT ?= 12

# Prefer static or shared library/dependencies
STATIC ?= 1

# PIC: PIC or pic
PIC ?= pic

# Add object files to archive (not "?=")
#MAKE_AR ?= rm -f $(basename $@).$(SLIBEXT); $(AR) -rs
MAKE_AR ?= $(AR) -rs

PLATFORM ?= 0
ifneq (0,$(call qnum,$(PLATFORM))) # NaN
  DFLAGS += -DLIBXSMM_PLATFORM_FORCE
endif

DEPDIR ?= $(ROOTDIR)
LIBNAME ?= $(DEPDIR)/lib/libxsmm

# Additional library search paths
LIBFIND ?= /usr/local/lib

# Utilities
MKTEMP := $(DEPDIR)/.mktmp.sh
FLOCK := $(DEPDIR)/.flock.sh

# Special characters
CHAR_OPEN := (
CHAR_CLOSE := )
CHAR_HASH := \#

# make -d triggers Shell's verbose output
ifeq (d,$(filter d,$(MAKEFLAGS)))
  SHELL := $(SHELL) -xv
endif

ifneq (,$(UMASK))
  UMASK_CURRENT := $(shell umask)
  ifneq (0,$(shell echo "$$(($(UMASK)<$(UMASK_CURRENT)))"))
    $(info WARNING: umask $(UMASK_CURRENT) does not match requirement ($(UMASK)))
  endif
  SHELL := $(shell printf "$(CHAR_HASH)!$(SHELL)\numask $(UMASK)\n$(SHELL) \"\$$$\@\"\n" \
          >$(DEPDIR)/.shell.sh; chmod +x $(DEPDIR)/.shell.sh 2>/dev/null; echo "$(DEPDIR)/.shell.sh")
endif

# THREADS refers to foundational TRT (and not necessarily Posix Threads)
THREADS ?= 1

# Threading runtime
ifeq (0,$(THREADS))
  override OMP := 0
endif
OMP ?= 0

# Code conformance (beyond -Wall)
PEDANTIC ?= 0

# Static code analysis (linking is not intended)
ANALYZE ?= 0

# Warning about unused functions
UNUSED ?= 0

# Embed InterProcedural Optimization information into libraries
IPO ?= 0
FAT ?= 0

# ILP64=0 (LP64 with 32-bit integers), and ILP64=0 (64-bit integers)
ILP64 ?= 0

# TBB Malloc enabled (1) or disabled (0)
# availability depends on TBBROOT
TBB_MALLOC ?= 0
# TBB runtime compatible with oldest supported GCC
TBB_OLDRTL ?= 0

# Enable absolute library paths
ABSLIBS ?= 1

# Embedd soname into shared library
SONAMELNK ?= 2

# utilities to handle paths with spaces
SPACES ?= 0

REVERSION0 := [0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*
REVERSION1 := s/..* \($(REVERSION0)\)[ \S]*.*/\1/
REVERSION2 := s/..* \([0-9]\{5\}[0-9]*\)[ \S]*.*/\1/

ifneq (0,$(SPACES))
  unquote = $(shell echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}" | $(SED) -e 's/^[" ][" ]*//' -e 's/[" ][" ]*$$//')
  quote = $(strip $(if $(filter 0 1,$(words $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})), \
                                            $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}, \
                            "$(call unquote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})"))
  qapath = $(call quote,$(shell export "VAR=$(call unquote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})" && if [ "$${VAR}" ]; then \
    if [ ! -d "$${VAR}" ]; then cd "$$(dirname "$${VAR}" 2>/dev/null)" 2>/dev/null && echo "$$(pwd -P)/$$(basename "$${VAR}" 2>/dev/null)"; \
    else cd "$${VAR}" 2>/dev/null && pwd -P; fi; fi))
  qndir = $(call quote,$(shell export "VAR=$(call unquote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})" && \
    if [ ! -d "$${VAR}" ]; then basename "$${VAR}" 2>/dev/null; fi))
  qdir = $(call quote,$(shell dirname "$(call unquote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})x" 2>/dev/null | $(SED) "s/\/\(\"*\)$$/\1/"))
else
  quote = $(strip        $(subst "",$(NULL),$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
  qapath = $(abspath     $(call quote,      $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
  qndir =  $(notdir      $(call quote,      $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
  qdir =   $(shell echo "$(dir $(call quote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))" | $(SED) "s/\/\(\"*\)$$/\1/")
endif
qsuffix = $(suffix   $(strip      $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
qbname =  $(basename $(call quote,$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}))
qxdir =   $(shell if [ -d "$1" ]; then echo "$1" | $(SED) "s/\/\(\"*\)$$/\1/"; else echo "$(call qdir,$1)"; fi)
qname =   $(basename $(call quote,$(shell echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}" | $(SED) "s/\.$(REVERSION0)//")))

# absolute directory paths
ABSDIR := $(call qapath,$(ROOTDIR))
HOMEDIR := $(call qapath,$(HOME))
HEREDIR := $(call qapath,.)

# pick the last of wildcard's sorted results
wildcard1 = $(strip $(lastword $(sort $(wildcard $1))))
# tail of list
tailwords = $(wordlist 2, $(words \
  $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}), \
  $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12})

# Pickup OpenMP library name if passed as OMP=libname|name
ifeq (NaN,$(call qnum,$(OMP),NaN))
  OMPRT := $(patsubst lib%,%,$(OMP))
endif
OMPRT ?= $(NULL)

uniqadd = $(if $($2),$(filter-out $($2),$($1)) $(if $3,$3,$($2)),$(filter-out $2,$($1)) $(if $3,$3,$2))
uniqmov = $(if $(filter $($2),$($1)),$(call uniqadd,$1,$2,$3),$($1))

ctest = $(if $1,$(if $(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c) && \
  echo "int main(void) { return 0; }" >$${INFILE} && if [ "$3" ]; \
  then $(firstword $1) $2 -c $${INFILE}   -o $${INFILE}.o >/dev/null && if [ "$(strip $3)" ]; then \
       $(firstword $1) $3    $${INFILE}.o -o $${INFILE}.x >/dev/null; fi; \
  else $(firstword $1) $2    $${INFILE}   -o $${INFILE}.x >/dev/null; fi 2>&1 || echo "FAILED"; \
  rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.c.* 2>/dev/null),$(NULL),$(strip $2 $3)))
# INFILE cannot use leading dot when used with certain F-compiler
ftest = $(if $1,$(if $(shell INFILE=$$($(MKTEMP) /tmp/_libxsmm_XXXXXX.f) && \
  printf "      PROGRAM test\n      END PROGRAM\n" >$${INFILE} && if [ "$3" ]; \
  then $(firstword $1) $2 -c $${INFILE}   -o $${INFILE}.o >/dev/null && if [ "$(strip $3)" ]; then \
       $(firstword $1) $3    $${INFILE}.o -o $${INFILE}.x >/dev/null; fi; \
  else $(firstword $1) $2    $${INFILE}   -o $${INFILE}.x >/dev/null; fi 2>&1 || echo "FAILED"; \
  rm -f /tmp/$$(basename $${INFILE} .f).* _libxsmm_??????.f.* 2>/dev/null),$(NULL),$(strip $2 $3)))

ldclib = $(if $1,$(call ctest,$1 $2,-l$(strip $3)))
ldflib = $(if $1,$(call ftest,$1 $2,-l$(strip $3)))

# extract from source file (1st arg) the value of the preprocessor definition (2nd arg)
preppat = $(if $1$2,s/[[:space:]]*$(CHAR_HASH)[[:space:]]*define[[:space:]][[:space:]]*$(if $(wildcard $1),$2,$1)[[:space:]][[:space:]]*\(..*\)[[:space:]]*/\1/p)
prepget = $(shell $(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) $(if $(wildcard $1),-c $1,- </dev/null) -dM -E 2>/dev/null \
          | $(SED) -n "$(call preppat,$1,$2);$(call preppat,$1,$3);$(call preppat,$1,$4)")

# Automatically pickup the environment (make -e is not required),
# or pickup the Intel Compiler (if available).
GNU ?= 0
ifeq (0,$(GNU))
  ifeq (,$(shell echo "$${CXX}"))
    ifneq (,$(filter icpc icpc1,$(call qndir,$(call which,icpc))$(INTEL)))
      CXX := icpc
    else ifneq (,$(filter icpx icpx2,$(call qndir,$(call which,icpx))$(INTEL)))
      CXX := icpx
    endif
  endif
  ifeq (,$(shell echo "$${CC}"))
    ifneq (,$(filter icc icc1,$(call qndir,$(call which,icc))$(INTEL)))
      CC := icc
    else ifneq (,$(filter icx icx2,$(call qndir,$(call which,icx))$(INTEL)))
      CC := icx
    endif
  endif
else ifneq (,$(filter-out default undefined,$(origin CC) $(origin CXX) $(origin FC)))
  $(info WARNING: GNU=$(GNU) was given but CC, CXX, or FC was specified!)
  FC := gfortran
  CC := gcc
  CXX := g++
endif

# adopt extra flags from C if not set individually
ECXXFLAGS ?= $(ECFLAGS)

COMPILER_VERSION_FLAG ?= $(ECXXFLAGS) $(EFLAGS) --version 2>/dev/null
CC_VERSION_FLAG ?= $(COMPILER_VERSION_FLAG)
FC_VERSION_FLAG ?= $(COMPILER_VERSION_FLAG)
CXX_VERSION_FLAG ?= $(CC_VERSION_FLAG)

# check if the Intel Development Tools are available
# 1: classic, 2: next-gen (opt-in)
INTEL ?= $(if $(filter 2,$(words $(filter icpc% icpx% icc% icx% Intel%, \
  $(shell $(CXX) $(CXX_VERSION_FLAG) | head -n1 | cut -d' ' -f1) \
  $(shell $(CC)  $(CC_VERSION_FLAG)  | head -n1 | cut -d' ' -f1)) )),1,0)

ifneq (0,$(INTEL))
  ifeq (,$(shell echo "$${FC}"))
    ifeq (ifx$(INTEL),$(call qndir,$(call which,ifx))$(filter-out 1,$(INTEL)))
      FC := ifx
    else ifneq (,$(call qndir,$(call which,ifort)))
      FC := ifort
    endif
  endif
else
  ifeq (,$(call which,$(CXX)))
    CXX := g++
  else ifneq (0,$(shell $(CXX) $(CXX_VERSION_FLAG) >/dev/null; echo "$$?"))
    CXX := g++
  else ifneq (gcc,$(call qndir,$(call which,$(CC))))
    ifeq (g++,$(call qndir,$(call which,$(CXX))))
      CC := gcc
    endif
  endif
  ifeq (,$(call which,$(CC)))
    CC := gcc
  else ifneq (0,$(shell $(CC) $(CC_VERSION_FLAG) >/dev/null; echo "$$?"))
    CC := gcc
  endif
  ifeq (Cray,$(shell $(CC) --version 2>&1 | head -n1 | cut -d' ' -f1))
    COMPILER_VERSION_FLAG ?= --version 2>&1
  else ifeq (Cray,$(shell $(CC) -V 2>&1 | head -n1 | cut -d' ' -f1))
    COMPILER_VERSION_FLAG ?= -V 2>&1
  endif
endif

ifneq (,$(filter-out __INTEL_LLVM_COMPILER,$(shell echo "__INTEL_LLVM_COMPILER" \
        | $(CC) $(call ctest,$(CC),$(ECFLAGS) $(EFLAGS)) -E -P - 2>/dev/null \
        | $(SED) "/^[[:space:]]*$$/d" 2>/dev/null)))
  ICX := 1
endif
ifneq (,$(filter-out __INTEL_LLVM_COMPILER,$(shell echo "__INTEL_LLVM_COMPILER" \
        | $(FC) $(call ftest,$(FC),$(EFCFLAGS) $(EFLAGS)) -E -P /dev/stdin 2>/dev/null \
        | $(SED) "/^[[:space:]]*$$/d" 2>/dev/null; rm -f stdin.f)))
  ICX := $(if $(filter 1,$(ICX)),2,1)
  FIXFC ?= 0
endif
ICX ?= 0

# linker setup
LD := $(CC)
XLD := $(CXX)

LNKSOFT ?= 1
ifeq (0,$(STATIC))
  ifeq (Windows_NT,$(UNAME))
    LNKSOFT := 0
  else ifeq (Darwin,$(UNAME))
    LNKSOFT := 0
  endif
endif

# BLAS is not used by default
ifneq (0,$(LNKSOFT))
  BLAS ?= 0
else
  BLAS ?= 2
endif

# Explicitly disable BLAS by user's intervention
# Makefile defines what happens (perhaps nothing)
NOBLAS ?= 0
DNOBLAS := -D__BLAS=0
ifneq (,$(filter environment% override command%,$(origin BLAS)))
ifeq (0,$(BLAS))
  NOBLAS := 1
endif
endif
ifneq (0,$(NOBLAS))
  BLAS_FLAGS += $(DNOBLAS)
endif

CXX_VERSION := $(shell $(CXX) $(CXX_VERSION_FLAG) | head -n1 | $(SED) -n "$(REVERSION1)p")
ifeq (,$(CXX_VERSION))
  CXX_VERSION := $(shell $(CXX) $(CXX_VERSION_FLAG) | head -n1 | $(SED) -n "$(REVERSION2)p")
endif
CXX_VERSION_MAJOR := $(shell echo "$(CXX_VERSION)" | cut -d. -f1)
CXX_VERSION_MINOR := $(shell echo "$(CXX_VERSION)" | cut -d. -f2)
CXX_VERSION_PATCH := $(shell echo "$(CXX_VERSION)" | cut -d. -f3)
ifeq (3,$(words $(CXX_VERSION_MAJOR) $(CXX_VERSION_MINOR) $(CXX_VERSION_PATCH)))
  CXX_VERSION_NUM := $(call version,$(CXX_VERSION_MAJOR),$(CXX_VERSION_MINOR),$(CXX_VERSION_PATCH))
else ifeq (2,$(words $(CXX_VERSION_MAJOR) $(CXX_VERSION_MINOR)))
  CXX_VERSION_NUM := $(call version,$(CXX_VERSION_MAJOR),$(CXX_VERSION_MINOR))
  CXX_VERSION_PATCH := 0
else
  CXX_VERSION := $(NULL)
  CXX_VERSION_NUM := 0
endif

CC_VERSION := $(shell $(CC) $(CC_VERSION_FLAG) | head -n1 | $(SED) -n "$(REVERSION1)p")
ifeq (,$(CC_VERSION))
  CC_VERSION := $(shell $(CC) $(CC_VERSION_FLAG) | head -n1 | $(SED) -n "$(REVERSION2)p")
endif
CC_VERSION_MAJOR := $(shell echo "$(CC_VERSION)" | cut -d. -f1)
CC_VERSION_MINOR := $(shell echo "$(CC_VERSION)" | cut -d. -f2)
CC_VERSION_PATCH := $(shell echo "$(CC_VERSION)" | cut -d. -f3)
ifeq (3,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR) $(CC_VERSION_PATCH)))
  CC_VERSION_NUM := $(call version,$(CC_VERSION_MAJOR),$(CC_VERSION_MINOR),$(CC_VERSION_PATCH))
else ifeq (2,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR)))
  CC_VERSION_NUM := $(call version,$(CC_VERSION_MAJOR),$(CC_VERSION_MINOR))
  CC_VERSION_PATCH := 0
else
  CC_VERSION := $(NULL)
  CC_VERSION_NUM := 0
endif

# disable Fortran per user-request
ifeq (0,$(FORTRAN))
  override FC := $(NULL)
endif

# fixup FC-default given by MAKE
ifneq (,$(strip $(FC)))
  ifeq (,$(call which,$(FC)))
    FIXFC ?= 1
  else ifneq (0,$(shell $(FC) $(FC_VERSION_FLAG) >/dev/null; echo "$$?"))
    FIXFC ?= 1
  else ifneq (gfortran,$(call qndir,$(call which,$(FC))))
    ifeq (g++,$(call qndir,$(call which,$(CXX))))
      FIXFC ?= 1
    endif
  endif
endif
FIXFC ?= 0
ifneq (0,$(FIXFC))
  ifneq (,$(call qndir,$(call which,gfortran)))
    MKL_FCRTL := gf
    GFC := gfortran
    FC := $(GFC)
  else
    FC := $(NULL)
  endif
endif

ifneq (,$(strip $(FC)))
  FC_VERSION := $(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | $(SED) -n "$(REVERSION1)p")
  ifeq (,$(FC_VERSION))
    FC_VERSION := $(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | $(SED) -n "$(REVERSION2)p")
  endif
  FC_VERSION_MAJOR := $(shell echo "$(FC_VERSION)" | cut -d. -f1)
  FC_VERSION_MINOR := $(shell echo "$(FC_VERSION)" | cut -d. -f2)
  FC_VERSION_PATCH := $(shell echo "$(FC_VERSION)" | cut -d. -f3)
  ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
    FC_VERSION_NUM := $(call version,$(FC_VERSION_MAJOR),$(FC_VERSION_MINOR),$(FC_VERSION_PATCH))
  else ifeq (2,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR)))
    FC_VERSION_NUM := $(call version,$(FC_VERSION_MAJOR),$(FC_VERSION_MINOR))
    FC_VERSION_PATCH := 0
  else
    FC_VERSION := $(NULL)
    FC_VERSION_NUM := 0
  endif
  ifeq (GNU,$(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | cut -d" " -f1))
    ifneq (0,$(shell echo "$$((40500>$(FC_VERSION_NUM)))"))
      ifneq (gfortran,$(call qndir,$(FC)))
        FC := gfortran
        FC_VERSION := $(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | $(SED) -n "$(REVERSION1)p")
        ifeq (,$(FC_VERSION))
          FC_VERSION := $(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | $(SED) -n "$(REVERSION2)p")
        endif
        FC_VERSION_MAJOR := $(shell echo "$(FC_VERSION)" | cut -d. -f1)
        FC_VERSION_MINOR := $(shell echo "$(FC_VERSION)" | cut -d. -f2)
        FC_VERSION_PATCH := $(shell echo "$(FC_VERSION)" | cut -d. -f3)
        ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
          FC_VERSION_NUM := $(call version,$(FC_VERSION_MAJOR),$(FC_VERSION_MINOR),$(FC_VERSION_PATCH))
        else ifeq (2,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR)))
          FC_VERSION_NUM := $(call version,$(FC_VERSION_MAJOR),$(FC_VERSION_MINOR))
          FC_VERSION_PATCH := 0
        else
          FC_VERSION_NUM := 0
        endif
        ifneq (0,$(shell echo "$$((40500>$(FC_VERSION_NUM)))"))
          override FC := $(NULL)
        endif
      else
        override FC := $(NULL)
      endif
    else ifneq (0,$(shell echo "$$((40600>$(FC_VERSION_NUM)))"))
      FORTRAN ?= 1
    endif
  else ifneq (0,$(INTEL))
    ifneq (0,$(shell echo "$$((130000>$(FC_VERSION_NUM)&&0<$(FC_VERSION_NUM)))"))
      override FC := $(NULL)
    endif
  endif
endif
FORTRAN ?= 2

ifeq (,$(strip $(FC)))
  ifeq (0,$(FORTRAN))
    FC_VERSION_NUM := 0
    FC_VERSION := $(NULL)
  else # keep FC_VERSION for message about outdated compiler
    FORTRAN := 0
  endif
endif

# compiler names
ifeq (0,$(PLATFORM))
  CXX_NAME := $(call qbname,$(shell $(CXX) $(CXX_VERSION_FLAG) | head -n1 \
              | $(SED) -e "s/^\([^0-9][^0-9]*\) ..*/\1/" -e "s/[[:space:]][[:space:]]*[Vv]ersion//" \
                    -e "s/[[:space:]][[:space:]]*$(CHAR_OPEN)..*//" -e "s/[[:space:]][[:space:]]*[[:punct:]]//" \
                    -e "s/[[:space:]][[:space:]]*$(REVERSION0)//" \
              | rev | cut -d" " -f1 | rev))
endif
ifeq (,$(strip $(CXX_NAME)))
  CXX_NAME := $(call qname,$(call qndir,$(CXX)))
endif
ifeq (0,$(PLATFORM))
  CC_NAME := $(call qbname,$(shell $(CC) $(CC_VERSION_FLAG) | head -n1 \
             | $(SED) -e "s/^\([^0-9][^0-9]*\) ..*/\1/" -e "s/[[:space:]][[:space:]]*[Vv]ersion//" \
                   -e "s/[[:space:]][[:space:]]*$(CHAR_OPEN)..*//" -e "s/[[:space:]][[:space:]]*[[:punct:]]//" \
                   -e "s/[[:space:]][[:space:]]*$(REVERSION0)//" \
             | rev | cut -d" " -f1 | rev))
endif
ifeq (,$(strip $(CC_NAME)))
  CC_NAME := $(call qname,$(call qndir,$(CC)))
endif
ifneq (,$(strip $(FC)))
  FC_NAME := $(firstword $(call qndir,$(FC)))
endif
ifeq (,$(strip $(FC_NAME)))
  FC_NAME := $(call qname,$(call qndir,$(FC)))
endif

ifneq (0,$(FORTRAN))
  ifeq (,$(strip $(GFC)))
    ifneq (,$(strip $(FC)))
      ifeq (GCC,$(shell $(FC) $(FC_VERSION_FLAG) | head -n1 | $(SED) "s/.* (\(..*\)) .*/\1/"))
        GFC := $(FC)
      else ifeq (0,$(shell $(FC) $(FC_VERSION_FLAG) | grep -q "Free Software Foundation"; echo "$$?"))
        GFC := $(FC)
      else ifneq (pgfortran,$(FC_NAME))
        ifneq (,$(findstring gfortran,$(FC_NAME)))
          GFC := $(FC)
        endif
      endif
      ifeq ($(strip $(FC)),$(strip $(GFC)))
        FC_NAME := gfortran
      endif
    else
      FORTRAN ?= 0
    endif
  endif
else
  FC := $(NULL)
endif
ifeq (,$(strip $(FC)))
  DFLAGS += -DLIBXSMM_NOFORTRAN
endif

# native GCC?
XSMM_GCC ?= 0
ifeq (0,$(shell $(CC) $(CC_VERSION_FLAG) | grep -q "Free Software Foundation"; echo "$$?"))
  XSMM_GCC := 1
else ifeq (GCC,$(shell $(CC) $(CC_VERSION_FLAG) | head -n1 | $(SED) "s/.* (\(..*\)) .*/\1/"))
  XSMM_GCC := 1
else ifneq (pgcc,$(CC_NAME))
  ifneq (,$(findstring pgcc,$(CC_NAME)))
    XSMM_GCC := 1
  endif
endif
ifeq (1,$(XSMM_GCC))
  ifeq (0,$(shell $(CXX) $(CXX_VERSION_FLAG) | grep -q "Free Software Foundation"; echo "$$?"))
    XSMM_GCC := 2
  else ifeq (GCC,$(shell $(CXX) $(CXX_VERSION_FLAG) | head -n1 | $(SED) "s/.* (\(..*\)) .*/\1/"))
    XSMM_GCC := 2
  else ifneq (,$(findstring g++,$(CXX_NAME)))
    XSMM_GCC := 2
  endif
  ifeq (2,$(XSMM_GCC))
  ifneq (,$(strip $(GFC)))
    MKL_FCRTL := gf
    XSMM_GCC := 3
  endif
  endif
endif
# Fortran runtime library
MKL_FCRTL ?= intel

# adopt fully equipped archiver
CCAR := $(call which,$(CC)-ar)
ifneq (,$(findstring -ar?,$(CCAR)?))
  ifeq (default,$(origin AR))
    AR := $(CCAR)
  else
    AR ?= $(CCAR)
  endif
endif

ifneq (0,$(ANALYZE))
  ifeq (1,$(ANALYZE))
    ANALYZE_FLAG := $(call ctest,$(CC),--analyze)
    ifeq (,$(ANALYZE_FLAG))
      ECFLAGS += $(call ctest,$(CC),-fanalyzer)
    else
      EFLAGS += $(ANALYZE_FLAG)
      FORCE_CXX := 1
    endif
  else
    FORCE_CXX := 1
    PEDANTIC := 3
  endif
endif

ifneq (,$(filter-out 0,$(FORCE_CXX)))
  override EFLAGS := $(EFLAGS) $(call ctest,$(CXX),-xc++)
  override CC := $(CXX)
endif

WCHECK ?= 0
WERROR_CFLAG := $(call ctest,$(CC),-Werror)
WERROR_FCFLAG := $(call ftest,$(FC),-Werror)
ifneq (0,$(WCHECK))
  WCHECK_CFLAG := $(call ctest,$(CC),-Wcheck)
  ifneq (,$(WCHECK_CFLAG))
    WCHECK_CFLAG := $(WCHECK_CFLAG) $(WERROR_CFLAG)
    ifneq (0,$(INTEL))
    ifeq (1,$(WCHECK))
      WCHECK_CFLAG := -diag-disable 2259 $(WCHECK_CFLAG)
    endif
    endif
  else
    WCHECK_CFLAG := $(WERROR_CFLAG)
  endif
endif

ifeq (0,$(ICX))
ifneq (0,$(INTEL))
  NOLIMITS_CFLAG := $(call ctest,$(CC),-qoverride-limits $(WERROR_CFLAG))
  ifeq (,$(strip $(GFC)))
  ifneq (,$(strip $(FC)))
    NOLIMITS_FCFLAG := $(call ftest,$(FC),-qoverride-limits $(WERROR_FCFLAG))
  endif
  endif
endif
endif

ifeq (Windows_NT,$(UNAME))
ifeq (MINGW64,$(MSYSTEM))
  MINGW := 64
else ifeq (MINGW32,$(MSYSTEM))
  MINGW := 32
else ifneq (,$(call prepget,__MINGW64__))
  MINGW := 64
else ifneq (,$(call prepget,__MINGW32__))
  MINGW := 32
endif
endif
MINGW ?= 0

# Library extension
ifneq (0,$(STATIC))
  LIBEXT ?= $(SLIBEXT)
  #ABSLIBS := 0
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
BLAS_STATIC ?= $(STATIC)

# PGI: ignore env. variable (same name)
override PGI := 0
ifeq (0,$(XSMM_GCC)) # not GCC
ifeq (0,$(INTEL)) # not Intel
ifneq (,$(call prepget,__clang__))
  CLANG := 1
else
  override PGI := $(shell $(CC) $(CC_VERSION_FLAG) | if grep -q "PGI"; then echo "1"; else echo "0"; fi)
endif
endif
endif
CLANG ?= 0

# Make GCC version number available even when not using GCC
ifneq (0,$(XSMM_GCC))
  GCC_VERSION := $(CXX_VERSION)
  GCC_VERSION_MAJOR := $(CXX_VERSION_MAJOR)
  GCC_VERSION_MINOR := $(CXX_VERSION_MINOR)
  GCC_VERSION_PATCH := $(CXX_VERSION_PATCH)
else ifeq (0,$(CLANG))
  GCCBIN := $(call qndir,$(call which,gcc))
  ifneq (,$(strip $(GCCBIN)))
    GCC_VERSION := $(shell $(GCCBIN) $(CXX_VERSION_FLAG) | head -n1 | $(SED) -n "$(REVERSION1)p")
    GCC_VERSION_MAJOR := $(shell echo "$(GCC_VERSION)" | cut -d. -f1)
    GCC_VERSION_MINOR := $(shell echo "$(GCC_VERSION)" | cut -d. -f2)
    GCC_VERSION_PATCH := $(shell echo "$(GCC_VERSION)" | cut -d. -f3)
  endif
endif
ifeq (3,$(words $(GCC_VERSION_MAJOR) $(GCC_VERSION_MINOR) $(GCC_VERSION_PATCH)))
  GCC_VERSION_NUM := $(call version,$(GCC_VERSION_MAJOR),$(GCC_VERSION_MINOR),$(GCC_VERSION_PATCH))
else ifeq (2,$(words $(GCC_VERSION_MAJOR) $(GCC_VERSION_MINOR)))
  GCC_VERSION_NUM := $(call version,$(GCC_VERSION_MAJOR),$(GCC_VERSION_MINOR))
  GCC_VERSION_PATCH := 0
else
  GCC_VERSION := $(NULL)
  GCC_VERSION_NUM := 0
endif

# Manually enable maintainer build (aka x86_64v1).
# Previously, it was automatically detected.
# Old logic:
#   ifeq (,$(filter Windows_NT Darwin,$(UNAME)))
#   ifeq (,$(SPACK_ENV_PATH)) # not under Spack
#   ifeq (3,$(XSMM_GCC)) # pure GNU toolchain
#   ifneq (0,$(shell echo "$$((100000<=$(GCC_VERSION_NUM)))"))
#
MAINTAINER ?= 0

# Select x86 code path (if not selected otherwise)
CPUFLAGS_X86 := $(strip $(shell if [ -e /proc/cpuinfo ]; then \
    grep -m1 flags /proc/cpuinfo | cut -d: -f2-; \
  elif [ "Darwin" = "$(UNAME)" ] && [ "x86_64" = "$(MNAME)" ]; then \
    sysctl -a \
      machdep.cpu.leaf7_features \
      machdep.cpu.extfeatures \
      machdep.cpu.features \
    | cut -d: -f2- | tr -s "\n" " " \
    | tr "[:upper:]." "[:lower:]_"; \
  fi))
SSE ?= 1
ifeq (0,$(SSE)) # discover AVX
  ifeq (1,$(words $(filter avx512f,$(CPUFLAGS_X86))))
    ifeq (2,$(words $(filter avx512pf avx512er,$(CPUFLAGS_X86)))) # KNL
      AVX ?= 3
      MIC ?= 1
    else
      AVX ?= 2
    endif
  else ifeq (1,$(words $(filter avx avx1.0,$(CPUFLAGS_X86))))
    ifeq (1,$(words $(filter fma,$(CPUFLAGS_X86))))
      AVX ?= 2
    else
      AVX ?= 1
    endif
  endif
else ifeq (1,$(SSE)) # discover SSE
  ifeq (1,$(words $(filter sse4_2 sse4.2,$(CPUFLAGS_X86))))
    SSE := 4
  else ifneq (0,$(words $(filter sse3 ssse3,$(CPUFLAGS_X86))))
    SSE := 3
  else ifneq (0,$(words $(filter sse2 ,$(CPUFLAGS_X86))))
    SSE := 2
  else ifneq (,$(CPUFLAGS_X86))
    SSE := 0
  endif
endif
AVX ?= 0
MIC ?= 0

# Select aarch64 code path
CPUFLAGS_AARCH64 := $(strip $(shell if [ -e /proc/cpuinfo ]; then \
    grep -m1 Features /proc/cpuinfo | cut -d: -f2-; \
  elif [ "Darwin" = "$(UNAME)" ] && [ "arm64" = "$(MNAME)" ]; then \
    sysctl -a hw.optional.neon | cut -d: -f2-; \
  fi))
ASIMD ?= 1
ifeq (1,$(ASIMD)) # discover ASIMD
  ifeq (1,$(words $(filter asimd,$(CPUFLAGS_AARCH64))))
    ASIMD := 1
  else ifneq (0,$(words $(filter 1,$(CPUFLAGS_AARCH64))))
    ASIMD := 1
  else ifneq (,$(CPUFLAGS_X86))
    ASIMD := 0
  endif
endif

ifneq (0,$(INTEL))
  SUITE := Intel Compiler
  MKL_OMPRTL := intel
else ifneq (0,$(XSMM_GCC))
  SUITE := GNU Compiler Collection
  MKL_OMPRTL := gnu
else ifneq (0,$(PGI))
  SUITE := $(if $(filter-out 0,$(PGI)),PGI $(NULL))Compiler
  MKL_OMPRTL := pgi
else
  ifeq (0,$(CLANG))
    COMPATIBLE := 1
  endif
  ifeq (Cray,$(shell $(CC) $(COMPILER_VERSION_FLAG) | head -n1 | cut -d' ' -f1))
    ifneq (0,$(COMPATIBLE))
      SUITE := Cray Compiler
      LDFLAGS += -hsystem_alloc
      # prevent codegen issues
      ifeq (0,$(OPT))
        override OPT := 1
      endif
    endif
    CRAY ?= 1
  endif
endif
SUITE ?= Compiler
MKL_OMPRTL ?= gnu
CRAY ?= 0
PGI ?= 0

ifeq (,$(strip $(GFC)))
ifneq (,$(filter-out 0,$(INTEL) $(ICX)))
ifneq (,$(call ftest,$(LD),$(NULL),-nofor-main))
  override LD := $(LD) -nofor-main
endif
endif
endif

ifneq (Windows_NT,$(UNAME))
ifneq (,$(strip $(PIC)))
  ifneq (,$(call ctest,$(CC),-f$(PIC)))
    PIC := PIC
  endif
  PICFLAG := -f$(PIC)
endif
endif

# prepend compiler-local library directory
LIBFIND := $(call qapath,$(call qdir,$(call which,$(LD)))/../lib) $(wildcard $(LIBFIND))

ifneq (,$(COMMON))
ifneq (0,$(COMMON))
  CFLAGS += $(call ctest,$(CC),-fcommon)
else
  CFLAGS += $(call ctest,$(CC),-fno-common)
endif
endif

ifneq (0,$(STATIC))
  ifeq (0,$(COMPATIBLE))
  ifneq (Darwin,$(UNAME))
  ifneq (Windows_NT,$(UNAME))
    ifneq (0,$(HARDEN))
      ifneq (,$(strip $(HARDEN))) # explicit
        DYNAMIC := 1
      else ifneq (0,$(SYM))
        DYNAMIC := 1
      endif
    else ifneq (0,$(SYM))
      DYNAMIC := 1
    endif
  endif
  endif
  endif
else
  DYNAMIC := 1
endif
DYNAMIC ?= 0

# enable MKL (if available)
ifeq (,$(strip $(MKLROOT)))
  BLAS_INCFILE := $(wildcard /opt/intel/oneapi/mkl/latest/include/mkl.h)
  ifeq (,$(BLAS_INCFILE))
    BLAS_INCFILE := $(call wildcard1,/opt/intel/compilers_and_libraries_*/$(MKL_PLATFORM)/mkl/include/mkl.h)
  endif
  ifneq (,$(BLAS_INCFILE))
    MKLROOT := $(call qapath,$(call qdir,$(BLAS_INCFILE))/..)
  endif
endif
ifeq (,$(strip $(MKLROOT)))
  BLAS_INCFILE := $(strip $(wildcard /usr/include/mkl/mkl.h))
  ifneq (,$(BLAS_INCFILE))
    MKLROOT := $(call qapath,$(call qdir,$(BLAS_INCFILE))/../..)
  endif
endif

# Compiler is used for link stage
ifneq (Darwin,$(UNAME))
  ifneq (ld,$(call qndir,$(LD)))
    XLNKOPT := -Wl,
  endif
  XLNKVERBOSE := --verbose
  linkopt = $(if $1,$(XLNKOPT)$(if $2,$1=$(call quote,$2),$1))
  abslibrpath = $(strip $(if $(findstring .$(ILIBEXT),$1)$(if $1,$(wildcard $1/)), \
    $(call linkopt,--rpath,$(call qxdir,$(call qapath,$1)))))
  XGROUP_BEGIN := $(call linkopt,--start-group)
  XGROUP_END := $(call linkopt,--end-group)
  WHOLE_BEGIN := $(call linkopt,--whole-archive)
  WHOLE_END := $(call linkopt,--no-whole-archive)
  XLIB_BEGIN_FORCE := $(call linkopt,--as-needed)
  XLIB_END_FORCE := $(call linkopt,--no-as-needed)
else # OSX
  ifneq (ld,$(call qndir,$(LD)))
    XLNKOPT := -Xlinker
  endif
  XLNKVERBOSE := -t
  linkopt = $(if $1,$(XLNKOPT) $(if $2,$1 $(XLNKOPT) $(call quote,$2),$1))
  abslibrpath = $(strip $(if $(findstring .$(ILIBEXT),$1)$(if $1,$(wildcard $1/)), \
    $(call linkopt,-rpath,$(call qxdir,$(call qapath,$1)))))
endif
ifneq (0,$(ASNEEDED))
  XLIB_BEGIN ?= $(XLIB_BEGIN_FORCE)
  XLIB_END ?= $(XLIB_END_FORCE)
endif

absliblpath = $(strip $(if $1,$(if $(findstring .$(ILIBEXT),$1)$(if $1,$(wildcard $1/)), \
  -L$(call qxdir,$(call qapath,$1)))))

ifneq (0,$(ABSLIBS))
  abslibpath = $(strip $(call abslibrpath,$1) $(call absliblpath,$1))
else
  abslibpath_force = $(strip $(call abslibrpath,$1))
endif
abslibpath ?= $(call absliblpath,$1)
abslibpath_force ?= $(NULL)

libpath = $(call qapath,$(if $1,$(shell $(FLUSH) $1 $2 -l$(strip $3) $(call linkopt,$(XLNKVERBOSE)) 2>&1 \
  | grep "lib$3" | tr " " "\n" | $(SED) -n "/\//p" | $(SED) "s/[$(CHAR_OPEN)]\(..*\)[$(CHAR_CLOSE)]/\1/" \
  | xargs -I {} sh -c "ls -pd {} 2>/dev/null || ls -pd {}.* 2>/dev/null" | grep -v /$$ | tail -n1)))

ifneq (Windows_NT1,$(UNAME)$(STATIC))
  abslibfile = $(strip $(if $(findstring .$(ILIBEXT),$1), \
    $(if $(patsubst lib%,%,$(call qname,$(call qndir,$1))), \
    $(if $(findstring .$(ILIBEXT).,$1),-l:$(call qndir,$1), \
       -l$(patsubst lib%,%,$(call qname,$(call qndir,$1))))), \
    $(if $(filter $(call qndir,$1),$(call qapath,$1)),$(call qapath,$1),$1)))
else
  abslibfile = $(strip $(if $(findstring .$(ILIBEXT),$1), \
    $(if $(call qname,$(call qndir,$1)), \
       -l$(call qname,$(call qndir,$1))), \
    $(if $(filter $(call qndir,$1),$(call qapath,$1)),$(call qapath,$1),$1)))
endif
abslib = $(strip $(call abslibpath,$1) $(call abslibfile,$1))

LIB_SOLD := $(LD) -shared $(PICFLAG)
ifeq (0,$(STATIC))
  LIB_XLD := $(XLD) -shared $(PICFLAG)
  LIB_LD := $(LIB_SOLD)
else
  LIB_XLD := $(XLD)
  LIB_LD := $(LD)
endif

ifeq (,$(strip $(FLD)))
ifneq (,$(strip $(FC)))
  FLD := $(FC) $(XLIB_END)
  LIB_SFLD := $(FLD) -shared $(PICFLAG)
  ifeq (0,$(STATIC))
    LIB_FLD := $(LIB_SFLD)
  else
    LIB_FLD := $(FLD)
  endif
endif
endif
LIB_SFLD ?= $(LIB_SOLD)
LIB_FLD ?= $(LIB_LD)
FLD ?= $(LD)

FREEFORM ?= 1
ifneq (0,$(INTEL))
ifeq (,$(strip $(GFC)))
  ifneq (,$(strip $(LIB_FLD)))
    LIB_FLD := $(call uniqadd,LIB_FLD,-nofor-main)
  endif
  ifneq (0,$(FREEFORM))
    FFORM_FLAG := -free
  endif
endif
endif
ifneq (0,$(FREEFORM))
  ifneq (,$(strip $(GFC)))
    FFORM_FLAG ?= -ffree-form
  else ifeq (0,$(PGI))
    FFORM_FLAG ?= $(call ftest,$(FC),-ffree-form)
  endif
endif

# CCE: resolve linker issue
ifneq (0,$(DYNAMIC))
  ifneq (,$(call ctest,$(LD),$(NULL),-dynamic))
    EXCLUDE_VALUE += -dynamic
    XLD := $(XLD) -dynamic
    LD := $(LD) -dynamic
  endif
  ifneq (,$(call ftest,$(FLD),$(NULL),-dynamic))
    EXCLUDE_VALUE += -dynamic
    FLD := $(FLD) -dynamic
  endif
endif

LIBDEP := $(LIBNAME).$(LIBEXT)
MAINLIB := $(call abslib,$(LIBDEP))

EXTDEP := $(LIBDEP) $(LIBNAME)ext.$(LIBEXT)
EXTLIB := $(call abslib,$(LIBNAME)ext.$(LIBEXT)) $(MAINLIB)

FORTDEP := $(EXTDEP) $(LIBNAME)f.$(LIBEXT)
FORTLIB := $(call abslib,$(LIBNAME)f.$(LIBEXT)) $(EXTLIB)

# provides libxsmmnoblas to satisfy BLAS symbols
NOBLASDEP := $(LIBNAME)noblas.$(LIBEXT)
NOBLASLIB := $(call abslib,$(NOBLASDEP))

ifeq (0,$(BLAS))
  FORTDEP := $(FORTDEP) $(NOBLASDEP)
  EXTDEP := $(EXTDEP) $(NOBLASDEP)
  LIBDEP := $(LIBDEP) $(NOBLASDEP)

  FORTLIB := $(FORTLIB) $(NOBLASLIB)
  EXTLIB := $(EXTLIB) $(NOBLASLIB)
  MAINLIB := $(MAINLIB) $(NOBLASLIB)
endif

ifneq (Darwin,$(UNAME))
  ifeq (0,$(shell ln -fs this-file-does-not-exist .ln 2>/dev/null && echo "$$?" && rm .ln 2>/dev/null))
    solink = -o "$(basename $1).$(DLIBEXT).$2.$3.$4" $(call linkopt,-soname,$(strip $(call qndir,$(basename $1))).$(DLIBEXT).$5)
    ifneq (0,$(SONAMELNK))
      solink += $(shell cd $(call qdir,$1) 2>/dev/null && \
        ln -fs $(call qndir,$(basename $1).$(DLIBEXT).$2.$3.$4) $(call qndir,$(basename $1).$(DLIBEXT).$5) 2>/dev/null \
          $(if $(filter-out 0,$(shell echo "$$((1<$(SONAMELNK)||0>$(SONAMELNK)))")), && \
        ln -fs $(call qndir,$(basename $1).$(DLIBEXT).$5) $(call qndir,$(basename $1).$(DLIBEXT)) 2>/dev/null))
    endif
  else # MinGW
    solink = -o $(call quote,$(basename $1).$(DLIBEXT)) $(call linkopt,-soname,$(strip $(call qndir,$(basename $1)).$(DLIBEXT).$5))
  endif
else # macOS
  solink = -o $(call qbname,$1).$2.$(DLIBEXT) \
           -install_name $(call qbname,$(call qapath,$1)).$2.$(DLIBEXT) \
           -current_version $2.$3.$4 -compatibility_version $5
  ifneq (0,$(SONAMELNK))
    solink += $(shell cd $(call qdir,$1) 2>/dev/null && \
      ln -fs $(call qndir,$(call qbname,$1).$2.$(DLIBEXT)) $(call qndir,$(basename $1)).$(DLIBEXT) 2>/dev/null)
  endif
endif

ifneq (0,$(INTEL))
  ifeq (1,$(STATIC))
    #SLDFLAGS += -Bstatic
    ifeq (0,$(ICX))
      SLDFLAGS += -no-intel-extensions -static-intel
      #SLDFLAGS += -static-libstdc++
    endif
    ifneq (Darwin,$(UNAME))
      SLDFLAGS += -static-libgcc
    endif
  else ifneq (0,$(STATIC))
    SLDFLAGS += -static
  endif
else # non-Intel
  ifeq (1,$(STATIC))
    ifeq (0,$(PGI))
      SLDFLAGS += -Bstatic
      ifeq (3,$(XSMM_GCC))
        #SLDFLAGS += -static-libstdc++
        ifneq (Darwin,$(UNAME))
          SLDFLAGS += -static-libgcc
        endif
      endif
    endif
  else ifneq (0,$(STATIC))
    ifeq (0,$(shell $(LD) -static -ldummydoesnotexist 2>&1 | grep -q "\-ldummydoesnotexist"; echo "$$?"))
      SLDFLAGS += -static
    endif
  endif
endif
SLDFLAGS ?= $(NULL)

LIBGFORTRAN := $(call libpath,$(FLD),$(SLDFLAGS),gfortran)
LIBPTHREAD := $(call ldclib,$(LD),$(SLDFLAGS),pthread)
QUADMATH := $(call ldclib,$(LD),$(SLDFLAGS),quadmath)
LIBCPP := $(call ldclib,$(LD),$(SLDFLAGS),stdc++)
LIBRT := $(call ldclib,$(LD),$(SLDFLAGS),rt)
LIBDL := $(call ldclib,$(LD),$(SLDFLAGS),dl)
ifneq (0,$(INTEL))
  LIBM := $(call ldclib,$(LD),$(SLDFLAGS),imf)
endif
LIBM ?= $(call ldclib,$(LD),$(SLDFLAGS),m)
LIBC := $(call ldclib,$(LD),$(SLDFLAGS),c)

# (default) runtime library dependencies
ifneq (0,$(FORCE_CXX)) # incl. undefined
  FCLDFLAGS += $(XLIB_BEGIN) $(call ldflib,$(FLD),$(SLDFLAGS),$(LIBCPP)) $(XLIB_END)
  CLDFLAGS += $(XLIB_BEGIN) $(LIBCPP) $(XLIB_END)
endif
FCLDFLAGS += $(XLIB_BEGIN) $(call ldflib,$(FLD),$(SLDFLAGS),c) $(XLIB_END)
CXXLDFLAGS += $(XLIB_BEGIN) $(LIBC) $(XLIB_END)

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
    ifeq (0,$(ICX))
      FCFLAGS += -g -traceback
    endif
    endif
  endif
  ifneq (0,$(shell echo "$$((170000<=$(CXX_VERSION_NUM)))"))
    CXXFLAGS += -std=c++14
  else ifneq (0,$(shell echo "$$((140000<=$(CXX_VERSION_NUM)))"))
    CXXFLAGS += -std=c++11
  endif
  CXXFLAGS += -Wall -diag-disable 1879,3415,3948,10006,10010,10411,10441,11074,11076,13003,13379
  CFLAGS += -Wall -diag-disable 1879,3415,3948,10006,10010,10411,10441,11074,11076,13003,13379
  ifneq (0,$(UNUSED))
    CXXFLAGS += -Wno-unused-function
    CFLAGS += -Wno-unused-function
  endif
  ifeq (,$(strip $(GFC)))
    ifneq (ld,$(call qndir,$(LD)))
      LDFLAGS += -diag-disable 1879,3415,10006,10010,10411,10441
    endif
    FCFLAGS += -diag-disable 10006,10010,10411,13003
    ifneq (0,$(THREADS))
      FCMTFLAGS += -threads
    endif
    FPEDANTIC += -warn all,notruncated_source -diag-disable 7025,7373,10237,10342,10382
  endif
  CPEDANTIC += $(WCHECK_CFLAG) -diag-disable 177,981,1419,1572,2547,10382
  ifeq (0,$(ICX))
    CPEDANTIC += -diag-disable 593,1599,2259,2415,2591
  endif
  CWARNEXTRA := -Wremarks
  ifeq (1,$(PEDANTIC))
    ifeq (,$(filter-out 0,$(FORCE_CXX)))
      CSTD := -std=c99
    endif
    CFLAGS += $(CSTD)
    ifeq (,$(strip $(GFC)))
      FSTD := -std$(if $(filter 1,$(FORTRAN)),03,08)
      FMFLAGS += $(FSTD) $(FPEDANTIC) -diag-disable 10010
      FCFLAGS += $(FFORM_FLAG)
    endif
    CXXFLAGS += $(WCHECK_CFLAG)
    CFLAGS += $(WCHECK_CFLAG)
  else ifneq (0,$(PEDANTIC))
    ifneq (,$(filter 0 1,$(ICX)))
    ifeq (,$(filter-out 0,$(FORCE_CXX)))
      CSTD := -std=c89
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
      FSTD := -std$(if $(filter 1,$(FORTRAN)),03,08)
      FCFLAGS += $(FSTD) $(FPEDANTIC)
      FMFLAGS += -fixed
    endif
  else
    ifneq (,$(filter 0 1,$(ICX)))
      ifeq (,$(filter-out 0,$(FORCE_CXX)))
        CSTD := -std=c89
      endif
    else ifneq (0,$(ICX))
      CXXFLAGS += -Wno-pass-failed
      CFLAGS += -Wno-pass-failed
    endif
    ifeq (,$(strip $(GFC)))
      FCFLAGS += $(FFORM_FLAG)
    endif
  endif
  CXXFLAGS += $(OPTFLAGS)
  CFLAGS += $(OPTFLAGS)
  FCFLAGS += $(OPTFLAGS)
  ifeq (,$(strip $(GFC)))
    # flag specifying output directory must be last
    FMFLAGS += -module
  else ifeq (0,$(ICX))
    FMFLAGS += -J
  endif
  # avoid turning OpenMP limits into an error (-Werror)
  ifneq (,$(WERROR_CFLAG))
    FCFLAGS += $(NOLIMITS_FCFLAG)
    CXXFLAGS += $(NOLIMITS_CFLAG)
    CFLAGS += $(NOLIMITS_CFLAG)
  endif
  ifeq (0,$(DBG))
    # consider more accurate -fp-model (C/C++: precise, Fortran: source)
    ifeq (0,$(ICX))
      #CXXFLAGS += -fp-model fast=2
      #CFLAGS += -fp-model fast=2
      CXXFLAGS += -fno-alias
      CFLAGS += -fno-alias
    endif
    CXXFLAGS += -ansi-alias
    CFLAGS += -ansi-alias
    ifeq (,$(strip $(GFC)))
    ifneq (,$(strip $(FC)))
      #FCFLAGS += -fp-model fast=2
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
  ifneq (0,$(shell echo "$$((3>$(DBG)))"))
    ifeq (0,$(COMPATIBLE))
    ifneq (,$(filter environment% override command%,$(origin COMPATIBLE))$(filter-out 1,$(INTEL)))
      ifneq (,$(filter 3,$(AVX))$(filter-out 0,$(VNNI) $(BF16)))
        ifeq (,$(MIC))
          CTARGET := -xCOMMON-AVX512
        else ifneq (0,$(MIC))
          CTARGET := -xMIC-AVX512
        else
          CTARGET := -xCORE-AVX512
          ifneq (0,$(VMAX))
            CTARGET += -qopt-zmm-usage=high
          endif
        endif
        ifneq (,$(filter-out 0,$(VNNI)))
          CTARGET += $(call ctest,$(CC),-mavx512vnni)
        else ifneq (,$(filter-out 0,$(BF16)))
          CTARGET += $(call ctest,$(CC),-mavx512vnni -mavx512bf16)
        endif
      else ifeq (2,$(AVX))
        CTARGET := -xCORE-AVX2
      else ifeq (1,$(AVX))
        CTARGET := -xAVX
      else ifneq (0,$(SSE))
        ifeq (1,$(SSE)) # default
          CTARGET := -xSSE4.2
        else ifeq (3,$(SSE))
          ifneq (Darwin,$(UNAME))
            CTARGET := -xSSE3
          else # no systems with less than SSE4.2
            CTARGET := -xSSE4.2
          endif
        else ifeq (4,$(SSE))
          CTARGET := -xSSE4.2
        else
          CTARGET := -xSSE$(SSE)
        endif
      else ifneq (0,$(AVX))
        CTARGET := -xHost
        ifneq (0,$(VMAX))
        ifeq (1,$(words $(filter avx512vl,$(CPUFLAGS_X86))))
          CTARGET += -qopt-zmm-usage=high
        endif
        endif
      endif
    endif
    endif
    ifneq (,$(filter 3,$(AVX))$(filter-out 0,$(VNNI) $(BF16)))
      ifeq (,$(MIC))
        CTARGET := -xCOMMON-AVX512
      else ifneq (0,$(MIC))
        CTARGET := -xMIC-AVX512
      else
        CTARGET := -xCORE-AVX512
      endif
      ifneq (,$(filter-out 0,$(VNNI)))
        CTARGET += $(call ctest,$(CC),-mavx512vnni)
      else ifneq (,$(filter-out 0,$(BF16)))
        CTARGET += $(call ctest,$(CC),-mavx512vnni -mavx512bf16)
      endif
    else ifeq (2,$(AVX))
      CTARGET := -march=core-avx2
    endif
  endif
  ifeq (,$(strip $(GFC)))
    ifneq (0,$(shell echo "$$((150000<=$(FC_VERSION_NUM)))"))
      OMPFLAG_FORCE := -qopenmp
    else
      OMPFLAG_FORCE := -fopenmp
    endif
  else
    OMPFLAG_FORCE := -fopenmp
  endif
  ifeq (,$(strip $(OMPRT)))
    OMPRT := iomp5
  endif
  ifneq (0,$(call qnum,$(OMP))) # NaN
    CXXFLAGS += $(OMPFLAG_FORCE)
    FCFLAGS += $(OMPFLAG_FORCE)
    CFLAGS += $(OMPFLAG_FORCE)
    ifneq (,$(strip $(GFC)))
      LDFLAGS += $(XLIB_BEGIN) $(call ldclib,$(LD),$(SLDFLAGS),$(OMPRT)) $(XLIB_END)
    else
      LDFLAGS := $(OMPFLAG_FORCE) $(LDFLAGS)
    endif
  endif
  ifneq (0,$(SIMD))
  ifneq (0,$(shell echo "$$((150000<=$(CXX_VERSION_NUM)))"))
    DFLAGS += -DLIBXSMM_OPENMP_SIMD
    ifeq (0,$(OMP))
      CXXFLAGS += -qopenmp-simd
      CFLAGS += -qopenmp-simd
      ifeq (,$(strip $(GFC)))
        FCFLAGS += -qopenmp-simd
      endif
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
  ifneq (0,$(shell echo "$$(((180000<=$(CC_VERSION_NUM)&&180001>$(CC_VERSION_NUM))||(170006>$(CC_VERSION_NUM)&&0!=$(CC_VERSION_NUM))))"))
    CFLAGS += -D_Float128=__float128
  endif
else # GCC assumed
  ifneq (0,$(SYM))
    ifeq (1,$(SYM))
      CXXFLAGS += -g
      FCFLAGS += -g
      CFLAGS += -g
    else
      ifneq (,$(filter 2 3,$(XSMM_GCC)))
        CXXFLAGS += -g3 -fno-omit-frame-pointer
        CFLAGS += -g3 -fno-omit-frame-pointer
      else
        CXXFLAGS += -g
        CFLAGS += -g
      endif
      ifeq (3,$(XSMM_GCC))
        FCFLAGS += -g3 -fno-omit-frame-pointer
      else
        FCFLAGS += -g
      endif
      ifneq (2,$(SYM))
        CXXFLAGS += -fsanitize=$(if $(SANITIZE),$(SANITIZE),thread)
        CFLAGS += -fsanitize=$(if $(SANITIZE),$(SANITIZE),thread)
        FCFLAGS += -fsanitize=$(if $(SANITIZE),$(SANITIZE),thread)
        LDFLAGS += -fsanitize=$(if $(SANITIZE),$(SANITIZE),thread)
        #LDFLAGS += $(call ldclib,$(LD),$(SLDFLAGS),tsan)
      endif
    endif
    ifneq (,$(SANITIZE))
    ifneq (,$(filter 1 2,$(SYM)))
      CXXFLAGS += -fsanitize=$(SANITIZE)
      CFLAGS += -fsanitize=$(SANITIZE)
      FCFLAGS += -fsanitize=$(SANITIZE)
      LDFLAGS += -fsanitize=$(SANITIZE)
      #LDFLAGS += $(call ldclib,$(LD),$(SLDFLAGS),tsan)
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
      ifneq (0,$(shell echo "$$((40000<=$(CC_VERSION_NUM)))"))
        CFLAGS += -Wno-pass-failed
      endif
    else ifneq (,$(filter-out 2 3,$(XSMM_GCC)))
      CXXFLAGS += -std=c++11
    endif
  endif
  ifeq (,$(filter-out 0,$(COMPATIBLE) $(PGI)))
    CXXFLAGS += -Wall
    CFLAGS += -Wall
    FSTD := -std=$(if $(filter 1,$(FORTRAN)),f2003,f2008)
    CPEDANTIC += -pedantic -Wextra -Wno-variadic-macros \
                 -Wno-unused-parameter -Wwrite-strings \
                 -Wmissing-field-initializers -Wwrite-strings \
                 -Wcast-qual -Wmissing-field-initializers \
                 -Wimplicit-fallthrough -Wmisleading-indentation
    FPEDANTIC += -pedantic -Wextra -Wunused-variable \
                 -Wimplicit-interface -Wimplicit-procedure \
                 -Wconversion -Wintrinsics-std \
                 -Wcharacter-truncation
    ifneq (0,$(shell echo "$$((40200<=$(CC_VERSION_NUM)))"))
      CPEDANTIC += -Wno-overlength-strings
      ifneq (0,$(shell echo "$$((40600<=$(CC_VERSION_NUM)))"))
        # can yield false positives with older GCC
        CPEDANTIC += -Wshadow
        ifneq (0,$(shell echo "$$((60000<=$(CC_VERSION_NUM)))"))
          ifneq (0,$(shell echo "$$((0==$(CLANG)||100000<=$(CC_VERSION_NUM)))"))
            CPEDANTIC += -Wmisleading-indentation
          endif
          ifneq (0,$(shell echo "$$((70000<=$(CC_VERSION_NUM)))"))
            CPEDANTIC += -Wimplicit-fallthrough
          endif
        endif
      endif
    else ifneq (0,$(CLANG))
      CPEDANTIC += -Wno-overlength-strings
    endif
    ifneq (0,$(CLANG))
      CPEDANTIC += -Wcovered-switch-default
      CPEDANTIC += -Wstring-conversion
    endif
    ifneq (,$(strip $(FC)))
    ifneq (0,$(shell echo "$$((50000<=$(FC_VERSION_NUM)))"))
      FWARNEXTRA := -Wuse-without-only -Wc-binding-type \
                    -Wrealloc-lhs -Wrealloc-lhs-all \
                    -Wreal-q-constant -Wconversion-extra \
                    -Wline-truncation
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
        CSTD := -std=c99
      endif
      ifeq (0,$(ANALYZE))
        CFLAGS += -Wstrict-prototypes
      endif
      CXXFLAGS += $(CPEDANTIC) -Wno-long-long
      #CXXFLAGS += -Wno-missing-field-initializers
      CFLAGS += $(CSTD) $(CPEDANTIC)
      FCFLAGS += $(FFORM_FLAG)
      FMFLAGS += $(FSTD) -pedantic -Wunused-variable $(FWARNEXTRA)
    else ifneq (0,$(PEDANTIC)) # 2<=PEDANTIC
      ifneq (2,$(PEDANTIC)) # 2<=PEDANTIC
      ifneq (0,$(CLANG))
        CPEDANTIC += -Weverything \
          -Wno-missing-variable-declarations \
          -Wno-implicit-int-float-conversion \
          -Wno-poison-system-directories \
          -Wno-cast-align -Wno-cast-qual \
          -Wno-disabled-macro-expansion \
          -Wno-atomic-implicit-seq-cst \
          -Wno-unknown-warning-option \
          -Wno-used-but-marked-unused \
          -Wno-covered-switch-default \
          -Wno-reserved-identifier \
          -Wno-padded -Wno-packed \
          -Wno-reserved-id-macro \
          -Wno-bad-function-cast \
          -Wno-double-promotion \
          -Wno-sign-conversion \
          -Wno-extra-semi-stmt \
          -Wno-unused-macros \
          -Wno-float-equal \
          -Wno-comma \
          -Wno-vla
      endif
      endif
      ifneq (Darwin,$(UNAME))
        ifeq (,$(filter-out 0,$(FORCE_CXX)))
          ifneq (0,$(shell echo "$$((0==$(CLANG)||130000>$(CC_VERSION_NUM)))"))
            CSTD := -std=c89
          else
            CSTD := -std=c99
          endif
        endif
        CPEDANTIC += -Wno-long-long
        CXXFLAGS += $(CPEDANTIC)
      else ifneq (0,$(XSMM_GCC))
        ifeq (,$(filter-out 0,$(FORCE_CXX)))
          CSTD := -std=c89
        endif
        CPEDANTIC += -Wno-long-long
        CXXFLAGS += $(CPEDANTIC)
      else # Clang may run into ICEs under macOS
        ifeq (,$(filter-out 0,$(FORCE_CXX)))
          CSTD := -std=c99
        endif
        CXXFLAGS += $(CPEDANTIC) -Wno-long-long
      endif
      ifeq (0,$(ANALYZE))
        CFLAGS += -Wstrict-prototypes
      endif
      #CXXFLAGS += -Wno-missing-field-initializers #-Wzero-as-null-pointer-constant
      CFLAGS += $(CSTD) $(CPEDANTIC)
      FCFLAGS += $(FSTD) $(FPEDANTIC)
    else ifeq (0,$(COMPATIBLE))
      CPEDANTIC += -Wno-long-long #-Wno-missing-field-initializers
      FCFLAGS += $(FFORM_FLAG)
      ifeq (,$(filter-out 0,$(FORCE_CXX)))
        CSTD := -std=c89
      endif
    endif
  endif
  # flag specifying output directory must be last
  ifeq (,$(filter-out 0,$(COMPATIBLE) $(PGI) $(ICX)))
    FMFLAGS += -J
  else # fallback
    FMFLAGS += -I
  endif
  CXXFLAGS += $(OPTFLAGS)
  CFLAGS += $(OPTFLAGS)
  FCFLAGS += $(OPTFLAGS)
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
  ifeq (Windows_NT,$(UNAME))
    LDFLAGS += $(XLIB_BEGIN) $(call ldclib,$(LD),$(SLDFLAGS),dbghelp) $(XLIB_END)
  else ifeq (FreeBSD,$(UNAME))
    LDFLAGS += $(XLIB_BEGIN) $(call ldclib,$(LD),$(SLDFLAGS),execinfo) $(XLIB_END)
  endif
  ifeq (0,$(COMPATIBLE))
    ifneq (0,$(PGI))
      OMPFLAG_FORCE := -mp
      OMPRT := omp
    else ifeq (0,$(INTEL))
      ifneq (Darwin,$(UNAME))
        ifneq (0,$(XSMM_GCC))
          OMPFLAG_FORCE := -fopenmp
        else ifneq (0,$(shell echo "$$((0!=$(CLANG)&&30900<=$(CC_VERSION_NUM)))"))
          OMPFLAG_FORCE := -fopenmp
          OMPRT := omp
        endif
      else # Darwin
        OMPFLAG_FORCE := -Xpreprocessor -fopenmp
        OMPRT := omp
      endif
    endif
  endif
  ifeq (,$(OMPFLAG_FORCE))
    ifneq (,$(filter environment% override command%,$(origin OMP)))
      OMPFLAG_FORCE := -fopenmp
    endif
  else ifeq (FreeBSD,$(UNAME))
    # avoid include path at begin of compile line
    OMPFLAG_FORCE += -I/usr/local/include
  else ifneq (0,$(CLANG))
    TESTRESB := $(call qapath,$(call qdir,$(call which,$(CC)))/../compiler)
    TESTRESB := $(wildcard $(TESTRESB)/include/omp.h)
    ifneq (,$(TESTRESB))
      OMPFLAG_FORCE += -I$(call qdir,$(TESTRESB))
    endif
  endif
  # account for missing TLS/OMP
  ifeq (,$(OMPFLAG_FORCE))
    THREADS ?= 0
    OMP := 0
  endif
  ifeq (,$(strip $(OMPRT))) # fallback
    OMPRT := gomp
  endif
  OMPLIBFILE := $(call libpath,$(LD),$(SLDFLAGS) $(OMPFLAG_FORCE) $(foreach LIB,$(LIBFIND),$(call abslibpath,$(LIB))),$(OMPRT))
  ifeq (0,$(CRAY))
  ifeq (,$(OMPLIBFILE))
  ifneq (,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c) && \
      printf "$(CHAR_HASH)include <omp.h>\nint main(void) { return omp_get_max_threads(); }\n" >$${INFILE} && \
      $(LD) $(SLDFLAGS) $(OMPFLAG_FORCE) $(foreach LIB,$(LIBFIND),$(call abslibpath,$(LIB))) -l$(OMPRT) $${INFILE} -o $${INFILE}.x 2>/dev/null >/dev/null && echo "OK"; \
      rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
    CLDFLAGS += $(foreach LIB,$(LIBFIND),$(call abslibpath,$(LIB)))
    OMPLIBFILE := -l$(OMPRT)
  endif
  endif
  endif
  ifneq (,$(OMPFLAG_FORCE))
    ifneq (0,$(call qnum,$(OMP))) # NaN
      ifneq (,$(OMPLIBFILE))
        ifneq (Darwin,$(UNAME))
          ifneq (0,$(CLANG))
            CXXLDFLAGS += $(call abslibpath,$(OMPLIBFILE))
            CLDFLAGS += $(call abslibpath,$(OMPLIBFILE))
          endif
          CXXLDFLAGS := $(OMPFLAG_FORCE) $(CXXLDFLAGS)
          CLDFLAGS := $(OMPFLAG_FORCE) $(CLDFLAGS)
        else # macOS
          CXXLDFLAGS += $(call abslib,$(OMPLIBFILE))
          CLDFLAGS += $(call abslib,$(OMPLIBFILE))
        endif
        CXXFLAGS += $(OMPFLAG_FORCE)
        CFLAGS += $(OMPFLAG_FORCE)
      else ifneq (Darwin,$(UNAME))
        CXXLDFLAGS := $(OMPFLAG_FORCE) $(CXXLDFLAGS)
        CLDFLAGS := $(OMPFLAG_FORCE) $(CLDFLAGS)
        CXXFLAGS += $(OMPFLAG_FORCE)
        CFLAGS += $(OMPFLAG_FORCE)
      endif
      ifeq (3,$(XSMM_GCC)) # pure GNU toolchain
        FLDFLAGS := -fopenmp $(FLDFLAGS)
        FCFLAGS += -fopenmp
      else # mixed toolchain
        TESTRESC := $(call libpath,$(FLD),$(SLDFLAGS) $(foreach LIB,$(LIBFIND),$(call abslibpath,$(LIB))),$(OMPRT))
        ifeq (,$(TESTRESC))
          TESTRESC := $(strip $(call ldflib,$(FLD),$(SLDFLAGS) $(foreach LIB,$(LIBFIND),$(call abslibpath,$(LIB))),$(OMPRT)))
        endif
        ifneq (,$(TESTRESC))
          FLDFLAGS += $(foreach LIB,$(LIBFIND),$(call abslibpath,$(LIB))) $(TESTRESC)
        else
          TESTRESC := $(call ftest,$(FLD),$(NULL),-fopenmp)
          ifneq (,$(TESTRESC))
            FLDFLAGS += $(TESTRESC)
          else # last try
            FLDFLAGS := $(OMPFLAG_FORCE) $(FLDFLAGS)
          endif
        endif
        TESTRESD := $(call ftest,$(FC),-fopenmp)
        ifneq (,$(TESTRESD))
          FCFLAGS += $(TESTRESD)
        else
          FCFLAGS := $(call ftest,$(FC),$(OMPFLAG_FORCE)) $(FCFLAGS)
        endif
      endif
    else ifneq (0,$(OMP)) # clang: OMP=libomp (NaN)
      ifneq (,$(OMPLIBFILE))
        ifneq (0,$(CLANG))
          CXXLDFLAGS += $(call abslibpath,$(OMPLIBFILE))
          CLDFLAGS += $(call abslibpath,$(OMPLIBFILE))
        endif
        CXXLDFLAGS := $(OMPFLAG_FORCE)=$(OMP) $(CXXLDFLAGS)
        CLDFLAGS := $(OMPFLAG_FORCE)=$(OMP) $(CLDFLAGS)
        CXXFLAGS += $(OMPFLAG_FORCE)=$(OMP)
        CFLAGS += $(OMPFLAG_FORCE)=$(OMP)
      endif
    endif
  endif
  ifneq (,$(strip $(R8)))
  ifneq (0,$(R8))
    FCFLAGS += -fdefault-real-8 -fdefault-double-8
  endif
  endif
endif
FCMTFLAGS ?= $(NULL)

ifeq (undefined,$(origin TARGET))
  ifneq (,$(CTARGET))
  ifeq (,$(call ctest,$(CC),$(CTARGET)))
    undefine CTARGET
  endif
  endif
  ifeq (,$(CTARGET))
  ifneq (0,$(shell echo "$$((3>$(DBG)))"))
    ifeq (x86_64,$(MNAME))
      ifneq (,$(filter 3,$(AVX))$(filter-out 0,$(VNNI) $(BF16)))
      ifneq (0,$(shell echo "$$(((0!=$(XSMM_GCC)&&80000<=$(GCC_VERSION_NUM))|| \
            (0!=$(CLANG)&&(40000<=$(CC_VERSION_NUM)||0==$(CC_VERSION_NUM)))|| \
            (0!=$(PGI)&&190000<=$(CC_VERSION_NUM))))"))
      ifneq (,$(call ctest,$(CC),-mfma -mavx512f -mavx512cd))
        CTARGET := -mfma -mavx512f -mavx512cd
        ifneq (0,$(lastword $(sort 0 $(MIC)))) # MIC
          ifneq (,$(call ctest,$(CC),-mavx512pf -mavx512er))
            CTARGET += -mavx512pf -mavx512er
          endif
        else ifneq (,$(call ctest,$(CC),-mavx512dq -mavx512bw -mavx512vl))
          ifneq (Darwin,$(UNAME))
            CTARGET += -mavx512dq -mavx512bw -mavx512vl
            #CTARGET += -mavx512ifma -mavx512vbmi
          else ifneq (,$(filter 1 2 3,$(INTEL) $(XSMM_GCC)))
            CTARGET += -mavx512dq -mavx512bw -mavx512vl
            #CTARGET += -mavx512ifma -mavx512vbmi
          else ifneq (0,$(shell echo "$$((0!=$(CLANG)&&80100<=$(CC_VERSION_NUM)))"))
            CTARGET += -mavx512dq -mavx512bw -mavx512vl
          endif
          ifneq (,$(filter-out 0,$(VNNI)))
            CTARGET += $(call ctest,$(CC),-mavx512vnni)
          else ifneq (,$(filter-out 0,$(BF16)))
            CTARGET += $(call ctest,$(CC),-mavx512vnni -mavx512bf16)
          endif
        endif
        ifneq (0,$(VMAX))
          CTARGET += -mprefer-vector-width=512
        endif
      endif
      endif
      endif
      ifeq (,$(CTARGET)) # fallback to AVX2
      ifneq (,$(filter 2 3,$(AVX)))
        ifneq (0,$(shell echo "$$((0!=$(INTEL)||0!=$(CLANG)||(40800<=$(GCC_VERSION_NUM)&&0==$(PGI))||0==$(CC_VERSION_NUM)))"))
          CTARGET ?= -march=core-avx2
        else
          CTARGET ?= -mavx2 -mfma
        endif
        ifneq (0,$(VMAX))
          CTARGET += -mprefer-vector-width=256
        endif
      endif
      endif
      ifeq (,$(CTARGET)) # fallback to AVX
      ifneq (,$(filter 1 2 3,$(AVX)))
        ifneq (0,$(shell echo "$$((0!=$(INTEL)||0!=$(CLANG)||40400<=$(GCC_VERSION_NUM)||0==$(CC_VERSION_NUM)))"))
          CTARGET ?= -mavx
          ifneq (0,$(VMAX))
            CTARGET += -mprefer-vector-width=256
          endif
        endif
      endif
      endif
      ifneq (,$(filter 1 2 3 4,$(SSE) $(AVX)))
        ifeq (,$(CTARGET)) # SSE-4.2 or default-SSE
          ifneq (,$(filter 01 02 03 10 11 12 13 40 41 42 43,$(SSE)$(AVX)))
            ifneq (0,$(XSMM_GCC))
              ifneq (0,$(shell echo "$$((40300<=$(CC_VERSION_NUM)))"))
                ifeq (0,$(MAINTAINER))
                  CTARGET := -msse4.2
                else # maintainer build
                  TARGET := -msse4.2
                  CTARGET := -msse2
                endif
              else
                CTARGET := -msse2
              endif
            endif
            CTARGET ?= -msse4.2
          else ifneq (,$(filter-out 0 1,$(SSE))) # better to use TARGET flag directly
            CTARGET := -msse$(SSE)
          endif
        endif
        ifeq (,$(CTARGET)) # SSE3
        ifneq (,$(filter 30 31 32 33,$(SSE)$(AVX)))
          ifneq (Darwin,$(UNAME))
            CTARGET ?= -msse2
          else # prevents Clang BE error (CRC32 and others)
            CTARGET ?= -msse4.2
          endif
        endif
        endif
        # stop here as SSE=2 is implicitly present (64-bit ABI)
      else ifneq (0,$(AVX))
        CTARGET := -march=native -mtune=native
        ifneq (0,$(VMAX))
          ifeq (1,$(words $(filter avx512f,$(CPUFLAGS_X86))))
            CTARGET += -mprefer-vector-width=512
          else ifeq (1,$(words $(filter avx2,$(CPUFLAGS_X86))))
            CTARGET += -mprefer-vector-width=256
          else ifeq (1,$(words $(filter avx,$(CPUFLAGS_X86))))
            CTARGET += -mprefer-vector-width=256
          endif
        endif
      endif
    else ifneq (,$(filter arm64 aarch64,$(MNAME)))
      ifeq (Darwin,$(UNAME))
      ifneq (,$(call ctest,$(CC),-target arm64-apple-macos -mcpu=apple-m1,$(NULL) $(NULL)))
        CTARGET := -target arm64-apple-macos -mcpu=apple-m1
      endif
      endif
      ifeq (,$(CTARGET))
      ifneq (,$(call ctest,$(CC),-march=armv8.1-a,$(NULL) $(NULL)))
        CTARGET := -march=armv8.1-a
      endif
      endif
      ifneq (,$(call ftest,$(FC),-march=armv8.1-a,$(NULL) $(NULL)))
        FTARGET := -march=armv8.1-a
      endif
    endif
  endif # DBG
  endif # CTARGET
else # take user's TARGET into account
  CTARGET := $(TARGET)
endif

override VNNI := $(if $(filter -mavx512vnni,$(CTARGET)),1,0)
override BF16 := $(if $(filter -mavx512bf16,$(CTARGET)),1,0)

ifneq (,$(call ftest,$(FC),$(CTARGET),$(NULL) $(NULL)))
  # inherit CTARGET flags
  FTARGET := $(CTARGET)
else ifneq (,$(call ftest,$(FC),$(patsubst -m%,-x%,$(CTARGET)),$(NULL) $(NULL)))
  FTARGET := $(patsubst -m%,-x%,$(CTARGET))
#else ifneq (0,$(FORTRAN)) # coherent flags (CC/CXX/FC)
  #override CTARGET := $(NULL)
endif
# The following logic is causing reset of .state flags when FORTRAN=0 is exported
#ifeq (,$(call ctest,$(CC),$(CFLAGS) $(CTARGET),$(NULL) $(NULL)))
#  ifneq (,$(call ctest,$(CC),-march=native -mtune=native,$(NULL) $(NULL)))
#    CTARGET := -march=native -mtune=native
#  else # revoke target flags
#    CTARGET := $(NULL)
#  endif
#endif
# avoid unnecessary state
ifeq ($(CTARGET),$(FTARGET))
  TARGET ?= $(CTARGET)
endif

# Intrinsics support level (0: None, 1: Static, 2: Dynamic)
# 1003<=INTRINSICS<1999: CPUID (see libxsmm_cpuid.h)
# For example, LIBXSMM_X86_AVX2=1006
ifneq (0,$(INTEL))
  INTRINSICS ?= 2
else ifneq (0,$(CLANG))
  INTRINSICS ?= 2
else ifneq (,$(GCC_VERSION))
  ifneq (0,$(shell echo "$$((100000<=$(GCC_VERSION_NUM)))"))
  ifeq (/usr/bin/,$(call qdir,$(call which,$(CC))))
    INTRINSICS ?= 2
  endif
  endif
endif
ifeq (x86_64,$(MNAME))
  INTRINSICS ?= 1006
  ifneq (0,$(INTRINSICS))
    ifeq (1,$(INTRINSICS))
      ifeq (0,$(INTEL))
        DFLAGS += -DLIBXSMM_INTRINSICS_STATIC
      else ifneq (,$(CTARGET))
        DFLAGS += -DLIBXSMM_INTRINSICS_STATIC
      endif
    else ifneq (2,$(INTRINSICS))
      DFLAGS += -DLIBXSMM_TARGET_ARCH=$(INTRINSICS)
    endif
  else
    DFLAGS += -DLIBXSMM_INTRINSICS_NONE
  endif
endif

ifeq (0,$(SYM))
ifneq (Darwin,$(UNAME))
  CLDFLAGS += $(call ctest,$(CC),$(NULL),-s)
  FLDFLAGS += $(call ftest,$(FC),$(NULL),-s)
endif
endif

ifneq (0,$(INSTRUMENT))
ifneq (Darwin1,$(UNAME)$(INSTRUMENT))
  DFLAGS += -D__TRACE=$(INSTRUMENT)
  ifeq (0,$(PGI))
    CXXFLAGS += -finstrument-functions $(call ctest,$(CXX),-fno-partial-inlining) $(call ctest,$(CXX),-fno-optimize-sibling-calls)
    CFLAGS += -finstrument-functions $(call ctest,$(CC),-fno-partial-inlining) $(call ctest,$(CC),-fno-optimize-sibling-calls)
    FCFLAGS += -finstrument-functions $(call ftest,$(FC),-fno-partial-inlining) $(call ftest,$(FC),-fno-optimize-sibling-calls)
    ifeq (0,$(INTEL)) # e.g. Intel, Clang, and others do not need/understand below flag
    ifneq (0,$(shell echo "$$((40300<=$(GCC_VERSION_NUM)))"))
      CFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
      ifneq (,$(filter 2 3,$(XSMM_GCC)))
        CXXFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
        ifeq (3,$(XSMM_GCC))
          FCFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
        endif
      endif
    endif
    endif
  endif
endif
endif

ifeq (0,$(COMPATIBLE))
  ifneq (0,$(OPT))
  ifeq (0,$(INTEL))
    ifneq (0,$(SIMD))
      ifneq (0,$(XSMM_GCC))
        ifneq (,$(CTARGET))
        ifneq (0,$(shell echo "$$((40900<=$(CC_VERSION_NUM)))"))
          DFLAGS += -DLIBXSMM_OPENMP_SIMD
          CFLAGS += -fopenmp-simd
          ifneq (1,$(XSMM_GCC))
          ifneq (0,$(shell echo "$$((40900<=$(CXX_VERSION_NUM)))"))
            CXXFLAGS += -fopenmp-simd
            ifneq (,$(FTARGET))
            ifneq (2,$(XSMM_GCC))
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
        ifneq (0,$(shell echo "$$((60000<=$(CC_VERSION_NUM)&&60000<=$(CXX_VERSION_NUM)))"))
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
      ifneq (,$(GFC)$(filter-out 0 1 2,$(XSMM_GCC)))
        FCFLAGS += -funroll-loops
      endif
    endif
    ifeq (,$(filter-out 0,$(PGI) $(INTEL) $(ICX) $(CRAY)))
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
      ifneq (,$(filter-out 0,$(STATIC) $(DYNAMIC)))
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
    ifeq (,$(filter-out 0,$(PGI) $(CRAY)))
    ifneq (0,$(OPT))
      CXXFLAGS += -fdata-sections -ffunction-sections
      CFLAGS += -fdata-sections -ffunction-sections
      ifeq (,$(filter-out 0,$(INTEL) $(ICX)))
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
  else # macOS
    SDKROOT := /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr
    ifneq (,$(wildcard $(SDKROOT)))
      ifeq (0,$(CLANG))
        IFLAGS += -I$(SDKROOT)/include
        LDFLAGS += -L$(SDKROOT)/lib
      endif
      ifneq (0,$(FORTRAN))
        FLDFLAGS += -L$(SDKROOT)/lib
      endif
    else  # SDK path not found
      LIBSYSTEM := $(call libpath,$(LD),$(SLDFLAGS),System)
      ifneq (,$(LIBSYSTEM))
        FLDFLAGS += -L$(dir $(LIBSYSTEM))
      endif
    endif
    ifneq (,$(OMPLIBFILE))
      FLDFLAGS += $(OMPLIBFILE)
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
  GINFO := $(SUITE): $(strip $(CC_NAME) $(CC_VERSION)), $(strip $(CXX_NAME) $(CXX_VERSION)), and $(strip $(FC_NAME) $(FC_VERSION))
  FINFO := Fortran target: $(if $(FTARGET),$(FTARGET),<compiler default>)
else
  GINFO := $(SUITE): $(strip $(CC_NAME) $(CC_VERSION)), and $(strip $(CXX_NAME) $(CXX_VERSION))
  FINFO := Fortran: <none>
endif
CINFO := C / C++ target: $(if $(CTARGET),$(CTARGET),<compiler default>)

ifeq (Darwin,$(UNAME))
  # avoid Homebrew based GCC AS; apply the flag only to the non-GCC components
  ifneq (0,$(XSMM_GCC))
    LDFLAGS += -Wa,-q
    CFLAGS += -Wa,-q
    ifneq (,$(filter 2 3,$(XSMM_GCC)))
      CXXFLAGS += -Wa,-q
    endif
  endif
  FLDFLAGS += -Wa,-q
  FCFLAGS += -Wa,-q
endif

ifeq (,$(filter-out 0,$(COMPATIBLE) $(PGI)))
  ifneq (,$(filter-out 0,$(THREADS) $(BLAS)))
    CXXFLAGS += -pthread
    CFLAGS += -pthread
  endif
  ifneq (Windows_NT,$(UNAME))
    ifneq (0,$(INTEL))
      FLDFLAGS += -pthread
      CLDFLAGS += -pthread
    else ifneq (Darwin,$(UNAME))
      FLDFLAGS += -pthread
      CLDFLAGS += -pthread
    else ifeq (0,$(CLANG))
      FLDFLAGS += -pthread
      CLDFLAGS += -pthread
    endif
  endif
else ifneq (0,$(PGI))
  LIBATOMIC ?= 1
endif

LIBATOMIC ?= 0
ifneq (0,$(THREADS))
ifneq (0,$(LIBATOMIC))
ifneq (,$(call ldclib,$(LD),$(SLDFLAGS),atomic))
  LDFLAGS += $(call ldclib,$(LD),$(SLDFLAGS),atomic)
  DFLAGS += -DLIBXSMM_LIBATOMIC
endif
endif
endif

ifeq (0,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c) && \
    printf "$(CHAR_HASH)include <features.h>\n$(CHAR_HASH)if !defined(__GNU_LIBRARY__) && !defined(__GLIBC__)\n0\n$(CHAR_HASH)endif\n" >$${INFILE} && \
    if [ "$$($(CC) -c $${INFILE} -o $${INFILE}.o 2>&1 || echo 'x')" ]; then echo "1"; else echo "0"; fi; \
    rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
  GLIBC := 1
endif
GLIBC ?= 0

OMPLIBFILE ?= $(call libpath,$(LD),$(SLDFLAGS) $(OMPFLAG_FORCE) $(foreach LIB,$(LIBFIND),$(call abslibpath,$(LIB))),$(OMPRT))
ifneq (,$(strip $(OMPLIBFILE)))
  OMPLIB ?= $(call abslib,$(OMPLIBFILE))
else ifneq (0,$(INTEL))
  OMPLIB ?= $(call ldclib,$(LD),$(SLDFLAGS),$(OMPRT))
endif

ifneq (,$(OMPFLAG_FORCE))
ifeq (0,$(shell INFILE=$$($(MKTEMP) /tmp/.libxsmm_XXXXXX.c) && \
    printf "$(CHAR_HASH)if defined(_OPENMP)\n$(CHAR_HASH) include <omp.h>\nint main() { return 0; }\n$(CHAR_HASH)endif\n" >$${INFILE} && \
    if [ "$$($(CC) $(OMPFLAG_FORCE) $${INFILE} -o $${INFILE}.x 2>&1 || echo 'x')" ]; then echo "1"; else echo "0"; fi; \
    rm -f /tmp/$$(basename $${INFILE} .c).* .libxsmm_??????.* 2>/dev/null))
  OMPFLAG := $(OMPFLAG_FORCE)
endif
endif
ifeq (0,$(OMP))
  EXTLIB += $(OMPLIB)
endif
OMPFLAG ?= $(NULL)

# some stand-alone libomp versions can cause warnings
ifneq (,$(filter 1 2,$(PEDANTIC)))
ifeq (Darwin,$(UNAME))
ifneq (,$(OMPFLAG))
ifneq (0,$(CLANG))
  CFLAGS += -Wno-strict-prototypes
endif
endif
endif
endif

ifneq (0,$(TBB_MALLOC))
ifneq (,$(TBBROOT))
  ifneq (Windows_NT,$(UNAME))
    TBBLIB_DIR := $(TBBROOT)/lib/intel64
    TBBLIB_DIRGCC := gcc$(GCC_VERSION_MAJOR).$(GCC_VERSION_MINOR)
    TBBLIB_MALLOC := $(wildcard $(TBBLIB_DIR)/$(TBBLIB_DIRGCC)/libtbbmalloc.$(ILIBEXT))
    ifeq (,$(TBBLIB_MALLOC))
      ifneq (0,$(TBB_OLDRTL))
        TBB_LIBDIRGCC := $(shell ls -1 $(call quote,$(TBB_LIBDIR)) | head -n1)
      else
        TBB_LIBDIRGCC := $(shell ls -1 $(call quote,$(TBB_LIBDIR)) | tail -n1)
      endif
      TBB_LIBMALLOC := $(wildcard $(TBB_LIBDIR)/$(TBB_LIBDIRGCC)/libtbbmalloc.$(ILIBEXT))
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

MAKE_ILP64 := 0
ifneq (,$(strip $(ILP64)))
ifneq (0,$(ILP64))
  MAKE_ILP64 := $(ILP64)
endif
endif
ifneq (0,$(MAKE_ILP64))
  BLAS_BITS := 64
  MKL_BITS := ilp64
else
  MKL_BITS := lp64
endif

ifneq (0,$(BLAS))
  ifneq (Darwin,$(UNAME))
    MKL_PLATFORM := linux
  else # macOS
    MKL_PLATFORM := mac
  endif
endif

ifneq (0,$(BLAS_STATIC))
  BLASLIBEXT ?= $(SLIBEXT)
else # shared (DLL)
  BLASLIBEXT ?= $(ILIBEXT)
endif

ifneq (,$(strip $(MKLROOT)))
  ifeq (0,$(PGI))
    MKL ?= $(BLAS)
    ifneq (,$(strip $(FC)))
      ifneq (0,$(shell echo "$$((0==$(XSMM_GCC)||40600<=$(GCC_VERSION_NUM)))"))
        MKL_DIRECT ?= 0
      else
        MKL_DIRECT := 0
      endif
    else
      MKL_DIRECT := 0
    endif
  else
    MKL ?= 0
  endif
else # disable
  MKL := 0
endif
ifneq (0,$(MKL))
  ifeq (1,$(MKL_DIRECT))
    ifeq (1,$(MKL))
      BLAS_FLAGS += -DMKL_DIRECT_CALL_SEQ
    else
      BLAS_FLAGS += -DMKL_DIRECT_CALL
    endif
    ifneq (0,$(XSMM_GCC))
      CXXFLAGS += -Wno-unused-value
      CFLAGS += -Wno-unused-value
    endif
  else ifneq (0,$(MKL_DIRECT))
    ifeq (1,$(MKL))
      BLAS_FLAGS += -DMKL_DIRECT_CALL_SEQ_JIT
    else
      BLAS_FLAGS += -DMKL_DIRECT_CALL_JIT
    endif
    ifneq (0,$(XSMM_GCC))
      CXXFLAGS += -Wno-unused-value
      CFLAGS += -Wno-unused-value
    endif
  endif
  BLAS_FLAGS += -D__CBLAS
endif
ifneq (Darwin,$(UNAME))
  ifneq (,$(wildcard $(MKLROOT)/lib/x86_64-linux-gnu/libmkl_rt.*))
    MKL_ARCH := x86_64-linux-gnu
  else
    MKL_ARCH := intel64
  endif
endif

ifeq (,$(wildcard $(BLAS_INCFILE) $(MKLROOT)/include/mkl.h))
  MKL := 0
endif
ifeq (1,$(MKL)) # sequential
  BLAS_FLAGS += -D__BLAS=1 -D__MKL=1
  MKL_THREADS := sequential
else ifneq (0,$(MKL)) # multi-threaded
  BLAS_FLAGS += -D__BLAS=$(MKL) -D__MKL=$(MKL)
  MKL_THREADS := $(MKL_OMPRTL)_thread
  ifeq (0,$(OMP))
    ifneq (gnu,$(MKL_OMPRTL))
      BLAS_LDFLAGS += $(OMPLIB)
    else
      BLAS_LDFLAGS += $(call ldclib,$(LD),$(SLDFLAGS),gomp)
    endif
  else ifeq (gnu,$(OMPLIB)$(MKL_OMPRTL))
    BLAS_LDFLAGS += $(call ldclib,$(LD),$(SLDFLAGS),gomp)
  endif
  ifneq (0,$(THREADS))
    BLAS_LDFLAGS += $(XLIB_BEGIN) $(LIBPTHREAD) $(XLIB_END)
  endif
endif

ifneq (0,$(MKL))
  ifeq (,$(BLAS_INCFILE))
    BLAS_IFLAGS += -I$(call quote,$(MKLROOT)/include)
  else
    BLAS_IFLAGS += -I$(call qdir,$(BLAS_INCFILE))
  endif
  BLAS_LDFLAGS += $(call abslibpath_force,$(MKLROOT)/lib/$(MKL_ARCH))
  ifeq (0,$(BLAS_STATIC)) # shared
    BLAS_LDFLAGS := $(call abslibpath,$(MKLROOT)/lib/$(MKL_ARCH)) \
      $(call abslibfile,libmkl_$(MKL_FCRTL)_$(MKL_BITS).$(ILIBEXT)) \
      $(call abslibfile,libmkl_core.$(ILIBEXT)) \
      $(call abslibfile,libmkl_$(MKL_THREADS).$(ILIBEXT)) \
      $(BLAS_LDFLAGS)
  else # static
    BLAS_LDFLAGS := $(XGROUP_BEGIN) \
      $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_$(MKL_FCRTL)_$(MKL_BITS).$(SLIBEXT) \
      $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_core.$(SLIBEXT) \
      $(MKLROOT)/lib/$(MKL_ARCH)/libmkl_$(MKL_THREADS).$(SLIBEXT) \
    $(XGROUP_END) $(BLAS_LDFLAGS)
  endif
else ifneq (0,$(BLAS)) # generic
  ifneq (,$(AOCL_ROOT))
    ifeq (0,$(BLAS_STATIC)) # shared
      ifeq (1,$(BLAS)) # sequential
        BLAS_LDFLAGS := -L$(AOCL_ROOT)/lib -lblis $(BLAS_LDFLAGS)
      else # multi-threaded
        BLAS_LDFLAGS := -L$(AOCL_ROOT)/lib -lblis-mt $(BLAS_LDFLAGS)
      endif
    else # static
      ifeq (1,$(BLAS)) # sequential
        BLAS_LDFLAGS := $(AOCL_ROOT)/lib/libblis.$(SLIBEXT) $(BLAS_LDFLAGS)
      else # multi-threaded
        BLAS_LDFLAGS := $(AOCL_ROOT)/lib/libblis-mt.$(SLIBEXT) $(BLAS_LDFLAGS)
      endif
    endif
  else ifeq (,$(strip $(BLASLIB)))
    ifneq (1,$(BLAS))
      ifneq (0,$(OMP))
        BLAS_THREADS := o
      else
        BLAS_THREADS := p
      endif
    endif
    BLASDIR ?= $(wildcard $(LIBFIND) /usr/lib /usr/lib64 $(subst :, ,$(LD_LIBRARY_PATH)))
    BLASROOT := $(wildcard $(patsubst %,%/..,$(BLASDIR)))
    ifeq (0,$(BLAS_STATIC)) # shared
      BLASTEST := $(filter-out -static,$(SLDFLAGS)) $(patsubst %,-L%,$(BLASDIR))
    else # static
      BLASTEST := $(SLDFLAGS) $(patsubst %,-L%,$(BLASDIR))
    endif
    BLASLIBFILE := $(call libpath,$(LD),$(BLASTEST),openblas$(BLAS_THREADS)$(BLAS_BITS))
    ifeq (,$(BLASLIBFILE)) # newer distributions symlink non-decorated library to threaded OpenBLAS
      BLASLIBFILE := $(call libpath,$(LD),$(BLASTEST),openblas$(BLAS_BITS))
    endif
    # most people expect to pickup OpenBLAS (if available) even when libblas/liblapack are available; use OPENBLAS=0 to avoid this
    ifeq (0,$(PGI))
      OPENBLAS := $(if $(BLASLIBFILE),1,0)
    else # PGI
      OPENBLAS := 0
    endif
    ifneq (0,$(OPENBLAS)) # OpenBLAS
      # OpenBLAS also carries the CBLAS bits
      BLAS_FLAGS += -D__BLAS=$(BLAS) -D__OPENBLAS
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
        BLASLIBFILE := $(call libpath,$(LD),$(BLASTEST),blas$(BLAS_BITS).dll)
      else
        BLASLIBFILE := $(call libpath,$(LD),$(BLASTEST),blas$(BLAS_BITS))
      endif
    endif
    ifneq (,$(BLASLIBFILE))
      ifeq (0,$(BLAS_STATIC)) # shared (END/BEGIN not BEGIN/END)
        BLAS_LDFLAGS += $(XLIB_END_FORCE) $(call abslib,$(BLASLIBFILE)) $(XLIB_BEGIN_FORCE)
      else # static
        BLAS_LDFLAGS += $(BLASLIBFILE)
      endif
      BLASREF ?= 1
    else ifneq (,$(call ctest,$(LD),-framework Accelerate)) # macOS
      BLAS_INCFILE := $(shell $(LD) $(call linkopt,-v) 2>&1 \
                     | grep -i frameworks 2>/dev/null \
                     | xargs -I {} find {} -name cblas.h 2>/dev/null \
                     | head -n1)
      ifneq (,$(BLAS_INCFILE))
        BLAS_FLAGS += -I$(call qapath,$(call qdir,$(BLAS_INCFILE)))
      endif
      #EXCLUDE_VALUE += -framework Accelerate
      BLAS_LDFLAGS += -framework Accelerate
      BLAS_FLAGS += -D__CBLAS
    else # fallback (END/BEGIN not BEGIN/END)
      #BLAS_LDFLAGS += $(XLIB_END_FORCE) $(call ldclib,$(LD),$(SLDFLAGS),blas) $(XLIB_BEGIN_FORCE)
      BLAS_LDFLAGS += -lblas
      BLASREF ?= 1
    endif
    BLASREF ?= 0
    ifneq (,$(filter-out 0,$(OPENBLAS) $(BLASREF))) # OpenBLAS or RefBLAS
    ifeq (0,$(OMP)) # Fortran compiler cannot link OpenMP runtime
      ifeq (,$(call libpath,$(FLD),$(SLDFLAGS) $(OMPFLAG_FORCE) $(foreach LIB,$(LIBFIND),$(call abslibpath,$(LIB))),$(OMPRT)))
        BLAS_CLDFLAGS += $(XLIB_BEGIN) $(call abslib,$(OMPLIBFILE)) $(XLIB_END)
      else # common link BLAS/OMP-link flags (all compilers)
        BLAS_LDFLAGS += $(XLIB_BEGIN) $(call abslib,$(OMPLIBFILE)) $(XLIB_END)
      endif
    endif
    endif
  else # BLAS library is specified via BLASLIB
    #BLASTEST := $(NULL)
    BLAS_FLAGS += -D__BLAS=$(BLAS)
    ifneq (,$(findstring openblas,$(call qndir,$(BLASLIB))))
      BLAS_FLAGS += -D__OPENBLAS
    endif
    ifneq (./,$(firstword $(BLASDIR))$(call qdir,$(BLASLIB)))
      ifeq (./,$(call qdir,$(BLASLIB)))
        BLAS_LDFLAGS += $(call abslib,$(firstword $(BLASDIR))/$(if $(call qsuffix,$(BLASLIB)),$(BLASLIB),lib$(BLASLIB).$(BLASLIBEXT)))
      else
        BLAS_LDFLAGS += $(call abslib,$(if $(call qsuffix,$(BLASLIB)),$(BLASLIB),$(BLASLIB).$(BLASLIBEXT)))
      endif
    else # fallback
      BLAS_LDFLAGS += $(call ldclib,$(LD),$(SLDFLAGS),$(BLASLIB))
    endif
  endif
endif
ifneq (0,$(BLAS))
ifneq (,$(GFC)$(filter-out 0,$(XSMM_GCC)))
  ifneq (,$(LIBGFORTRAN))
    ifneq (0,$(STATIC))
      BLAS_CLDFLAGS += $(XLIB_BEGIN) $(LIBGFORTRAN) $(XLIB_END)
    else
      BLAS_LDFLAGS += $(call abslibrpath,$(LIBGFORTRAN))
    endif
  endif
  ifneq (,$(QUADMATH))
    ifneq (0,$(STATIC))
      BLAS_CLDFLAGS += $(XLIB_BEGIN) $(QUADMATH) $(XLIB_END)
    else
      BLAS_LDFLAGS += $(call abslibrpath,$(QUADMATH))
    endif
  endif
  ifneq (,$(LIBM))
    ifneq (0,$(STATIC))
      BLAS_CLDFLAGS += $(XLIB_BEGIN) $(LIBM) $(XLIB_END)
    else
      BLAS_LDFLAGS += $(call abslibrpath,$(LIBM))
    endif
  endif
endif
endif

cleanup = $(foreach flag,$(subst //,/,$1),$(flag))
# cleanup eventually duplicated flags and slashes
cleanld = $(strip $(shell echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}" | $(SED) \
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
FLDFLAGS := $(call uniqadd,FLDFLAGS,LIBDL,$(XLIB_BEGIN) $(LIBDL) $(XLIB_END))
LDFLAGS := $(call uniqadd,LDFLAGS,LIBRT,$(XLIB_BEGIN) $(LIBRT) $(XLIB_END))
LDFLAGS := $(call uniqadd,LDFLAGS,LIBDL,$(XLIB_BEGIN) $(LIBDL) $(XLIB_END))

# no-BLAS flags: cleanup and extra flags; merged ("=" rather than ":=")
NOBLAS_CXXCLEAN := $(call cleanup,$(PICFLAG) $(CXXFLAGS) $(ECXXFLAGS) $(EFLAGS))
NOBLAS_CXXFLAGS  = $(NOBLAS_CXXCLEAN)
NOBLAS_FCCLEAN  := $(call cleanup,$(PICFLAG) $(FCFLAGS) $(EFCFLAGS) $(EFLAGS))
NOBLAS_FCFLAGS   = $(NOBLAS_FCCLEAN)
NOBLAS_CCLEAN   := $(call cleanup,$(PICFLAG) $(CFLAGS) $(ECFLAGS) $(EFLAGS))
NOBLAS_CFLAGS    = $(NOBLAS_CCLEAN)
NOBLAS_FLAGS     = $(call cleanup,$(filter-out -D__BLAS%,$(DFLAGS)))
NOBLAS_LDCLEAN  := $(call cleanld,$(LDFLAGS) $(ELDFLAGS))
NOBLAS_LDFLAGS   = $(NOBLAS_LDCLEAN)
NOBLAS_CLDCLEAN := $(call cleanld,$(CLDFLAGS))
NOBLAS_CLDFLAGS  = $(NOBLAS_CLDCLEAN)
NOBLAS_IFLAGS    = $(call cleanup,$(IFLAGS))

# regular flags: cleanup
CXXFLAGS   := $(call cleanup,$(BLAS_CXXFLAGS) $(NOBLAS_CXXFLAGS))
FCFLAGS    := $(call cleanup,$(BLAS_FCFLAGS) $(NOBLAS_FCFLAGS))
CFLAGS     := $(call cleanup,$(BLAS_CFLAGS) $(NOBLAS_CFLAGS))
DFLAGS     := $(call cleanup,$(BLAS_FLAGS) $(NOBLAS_FLAGS))
IFLAGS     := $(call cleanup,$(BLAS_IFLAGS) $(NOBLAS_IFLAGS))
CLDFLAGS   := $(call cleanld,$(BLAS_CLDFLAGS) $(NOBLAS_CLDFLAGS))
LDFLAGS    := $(call cleanld,$(BLAS_LDFLAGS) $(NOBLAS_LDFLAGS))
CXXLDFLAGS := $(call cleanld,$(CXXLDFLAGS))
FLDFLAGS   := $(call cleanld,$(FLDFLAGS))

.PRECIOUS: $(BLDDIR)/%-cpp.o $(BLDDIR)/%-c.o $(BLDDIR)/%-f.o \
           $(BLDDIR)/%-f90.o $(BLDDIR)/%-f90.o $(BLDDIR)/%-f77.o \
           %/.make

.SUFFIXES:

# applyif(A1,A2,A3,A4) evaluates to A4 if A1 (precondition) is non-zero and if the basenames (A2, A3) match
applyif = $(if $(filter 0,$1),$(NULL),$(if $(filter $2,$(call qname,$(call qndir,$3))),$(if $1,$4)))
# derives the extension of a filename
extname = $(subst .,,$(call qsuffix,$(1)))
# derives the name of an object files for a given source file
objname = $(foreach ARG,$(1),$(addprefix $(BLDDIR)/,$(patsubst %$(call qsuffix,$(ARG)),%-$(call extname,$(ARG)).o,$(call qndir,$(ARG)))))

STATEMAK := $(sort $(shell grep -how '$(VNAME)' $(ROOTDIR)/Makefile.inc $(MAKEFILE_LIST) 2>/dev/null))
ifneq (0,$(STATEFLT))
  STATEVAR := $(filter-out \
    $(EXCLUDE_STATE) EXCLUDE_% .% _% XSMM_% HAVE_% LIB_% %STATE MAKE_% MAKELEVEL MAKEFILE_LIST MAKEOVERRIDES MAKEFLAGS SHELL BLDDIR REVERSION% AVX SSE \
    DLIBEXT FLOCK LIBEXT ICX ILIBEXT LIBNAME LICFILE MAKEINC MV ROOTDIR SLIBEXT XLNK% CHAR_OPEN CHAR_CLOSE CHAR_HASH MKTEMP OMPLIBFILE BLAS_INCFILE \
    NOBLAS% DNOBLAS %LIB FLDFLAGS OPSYS %DEP LIBFIND LIBDL LIBRT LIBM LIBCPP BLAS_FLAGS BLAS_LDFLAGS BLAS_THREADS SUITE PEDANTIC CMAIN CLDFLAGS \
    GNU CCAR LIBGFORTRAN PGI FPEDANTIC FWARNEXTRA %_BEGIN% %_END% SLDFLAGS PICFLAG PIC OPTFLAGS OPT OMPFLAG_FORCE FORCE_CXX VNAME DYNAMIC \
    CPEDANTIC FREEFORM FIXFC INCDIR OUTDIR SRCDIR XLIB_% GFC CLANG INTEL CRAY GCCBIN FMFLAGS FCLDFLAGS PYTHON% HOST% TMUX %COLORS \
    HEREDIR HOMEDIR UNUSED TEST% MINGW FMAIN VERSION ACC% LESS% HOME% USER NAME LOGNAME LANG %PWD %PATH TERM% MOTD% SHLVL %_FLAG \
    CC_VERSION_% CXX_VERSION_% FC_VERSION_% GCC_VERSION_% CINFO CP DEPDIR FINFO ENVBIN FSTD GINFO XDG_DATA_DIRS ONEVERSION \
    LIBPTHREAD CSTD CWARNEXTRA QUADMATH BLASDIR BLASLIBEXT BLASTEST FYPPEXE INCLUDEMAKE LIBDIR OBJDIR \
    $(if $(filter-out $(FTARGET),$(CTARGET)),TARGET,CTARGET FTARGET) \
    $(if $(filter $(BLAS_STATIC),$(STATIC)),BLAS_STATIC) \
    $(if $(filter 0,$(TBB_MALLOC)),TBB_%) \
    $(if $(filter 0,$(MKL)),MKL_%), \
    $(.VARIABLES))
else # debug
  STATEVAR := $(filter-out FMAIN REVERSION%,$(.VARIABLES))
endif
STATE := $(sort $(foreach V,$(filter $(STATEMAK),$(STATEVAR)),$(if $(filter-out default automatic,$(origin $V)),$(if $($V),$V))))
ENVSTATE := $(strip $(foreach V,$(STATE),$(if $(filter environment%,$(origin $V)),$V)))
VALSTATE := $(foreach V,$(STATE),$V=$(subst \
  /$(USER),$(if $(USER),/$$USER),$(patsubst $(USER)%,$(if $(USER),$$USER%),$(subst \
  /$(HOME),$(if $(HOME),/$$HOME),$(patsubst $(HOME)%,$(if $(HOME),$$HOME%),$(subst \
  /$(HOMEDIR),$(if $(HOMEDIR),/$$HOME),$(patsubst $(HOMEDIR)%,$(if $(HOMEDIR),$$HOME%),$(filter-out \
  $(EXCLUDE_VALUE),$($V))))))))?)
DIRSTATE ?= .
PRESTATE := $(shell echo '$(VALSTATE)' | $(DEPDIR)/.state.sh $(DIRSTATE))

$(DIRSTATE)/.state: $(firstword $(PRESTATE))

%/.make:
	@$(MKDIR) -p $(call qdir,$@) && touch $@

.make:
	@-touch $@

.DEFAULT_GOAL := $(NULL)
