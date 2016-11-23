#############################################################################
# Copyright (c) 2015-2016, Intel Corporation                                #
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

ifeq (d,$(filter d,$(MAKEFLAGS)))
  SHELL = bash -xv
endif

ifeq (Windows_NT,$(OS))
  UNAME ?= Windows_NT
  ENV ?= $(NULL)
  # Cygwin based
  DLIBEXT ?= dll
  SLIBEXT ?= lib
else
  UNAME ?= $(shell uname)
  ENV ?= env
  ifneq (Darwin,$(UNAME))
    DLIBEXT ?= so
    SLIBEXT ?= a
  else
    DLIBEXT ?= dylib
    SLIBEXT ?= a
  endif
endif

# Regular expression to match "main" (good-enough pattern)
CMAIN = "main\s*(.*)"
FMAIN = "^\s*PROGRAM\s\s*\w\w*\(\s\s*\!.*\)*$$"

# PIC: PIC or pic
PIC=pic

# Optimization level
OPT ?= 2

# Optimization flag derived from OPT flag
ifeq (0,$(shell echo "$(OPT)" | grep -q "^[0-9]\+$$"; echo "$$?"))
  OPTFLAG = -O$(OPT)
else
  OPTFLAG = -$(OPT)
endif

# Trace level
TRACE ?= 0

# Avoid more sophisticated flags of the GCC tool chain,
# and improve compatibility with compilers supposed to be
# compatible with the GCC tool chain
COMPATIBLE ?= 0

# Number of repeated calls (tests),
# or used to scale the problem size
TESTSIZE ?= 1

# Static or shared binary
STATIC ?= 0

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

# Debugging and symbols (e.g., when profiling)
SYM ?= 0
DBG ?= 0

# Threading runtime
OMP ?= 0

# THREADS refers to foundational TRT (and not necessarily Posix Threads)
THREADS ?= 1

# Code conformance (beyond -Wall)
PEDANTIC ?= 0

# Embed InterProcedural Optimization information into libraries
IPO ?= 0

# ILP64=0 (LP64 with 32-bit integers), and ILP64=0 (64-bit integers)
ILP64 ?= 0
BLAS ?= 0

# Prefer to link statically against BLAS library
BLAS_STATIC ?= $(STATIC)
ifneq (0,$(BLAS_STATIC))
  BLASLIBEXT ?= $(SLIBEXT)
else
  BLASLIBEXT ?= $(DLIBEXT)
endif

# Explicitly disables BLAS by user's intervention
# Makefile defines what happens (perhaps nothing)
NOBLAS ?= 0
DNOBLAS = -D__BLAS=0
ifneq (,$(filter environment% override command%,$(origin BLAS)))
ifeq (0,$(BLAS))
  BLAS_DFLAGS += $(DNOBLAS)
  NOBLAS = 1
endif
endif

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

abslib = $(strip $(call abslibpath,$1) $(call abslibfile,$1))

EXTLIB = $(call abslib,$(LIBNAME)ext.$(LIBEXT)) #$(XLIB_BEGIN) -lm $(XLIB_END)
ifeq (0,$(BLAS))
  # provides a libxsmmext as a dummy library which satisfies BLAS symbols
  BLASLIB ?= $(EXTLIB)
else
  BLASLIB ?= $(NULL)
endif

# Embedd soname into shared library
ifneq (Darwin,$(UNAME))
  soname = -Wl,-soname=$(subst $(NULL) ,.,$(strip $(notdir $1) $2 $3 $4))
else
  soname = # TODO
endif

# Pickup OpenMP library name if passed as OMP=libname|name
ifneq (0,$(shell echo "$(OMP)" | grep -q "^-*[0-9]\+$$"; echo "$$?")) # NaN
  OMPRT = $(patsubst lib%,%,$(OMP))
endif

# Automatically pickup the environment (make -e not needed),
# or pickup the Intel Compiler (if available).
ifneq (,$(notdir $(shell which icpc 2> /dev/null)))
  CXX = icpc
endif
ifneq (,$(notdir $(shell which icc 2> /dev/null)))
  CC = icc
endif
ifneq (,$(notdir $(shell which ifort 2> /dev/null)))
  FC = ifort
endif

# check if the Intel Development Tools are available
INTEL ?= $(shell echo "$$((2==$(words $(filter icpc icc, \
  $(shell $(CXX) --version 2> /dev/null | head -n1 | cut -d' ' -f1) \
  $(shell $(CC) --version 2> /dev/null | head -n1 | cut -d' ' -f1)))))")

#FORTRAN ?= $(if $(FC), $(if $(notdir $(shell which $(FC) 2> /dev/null)), 1, 0), 0)
ifneq (0,$(INTEL))
  ifeq (0,$(OFFLOAD))
    CXXFLAGS += -qno-offload
    CFLAGS += -qno-offload
    ifneq (,$(strip $(FC)))
    ifeq (ifort,$(shell $(FC) --version 2> /dev/null | head -n1 | cut -d" " -f1))
      FCFLAGS += -qno-offload
    endif
    endif
  endif
else
  ifeq (,$(notdir $(shell which $(CXX) 2> /dev/null)))
    CXX = g++
  else ifneq (0,$(shell $(CXX) --version > /dev/null 2> /dev/null; echo "$$?"))
    CXX = g++
  endif
  ifeq (,$(notdir $(shell which $(CC) 2> /dev/null)))
    CC = gcc
  else ifneq (0,$(shell $(CC) --version > /dev/null 2> /dev/null; echo "$$?"))
    CC = gcc
  endif
  ifneq (0,$(FORTRAN))
    ifneq (,$(strip $(FC)))
    ifneq (0,$(shell $(FC) --version > /dev/null 2> /dev/null; echo "$$?"))
      FC = $(shell which gfortran 2> /dev/null)
    endif
    endif
  else
    FC = $(shell which gfortran 2> /dev/null)
  endif
endif
ifeq (0,$(FORTRAN))
  FC = $(NULL)
endif

CXX_VERSION_STRING = $(shell $(CXX) --version 2> /dev/null | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
CXX_VERSION_MAJOR = $(shell echo "$(CXX_VERSION_STRING)" | cut -d"." -f1)
CXX_VERSION_MINOR = $(shell echo "$(CXX_VERSION_STRING)" | cut -d"." -f2)
CXX_VERSION_PATCH = $(shell echo "$(CXX_VERSION_STRING)" | cut -d"." -f3)
ifeq (3,$(words $(CXX_VERSION_MAJOR) $(CXX_VERSION_MINOR) $(CXX_VERSION_PATCH)))
  CXX_VERSION = $(shell echo "$$(($(CXX_VERSION_MAJOR) * 10000 + $(CXX_VERSION_MINOR) * 100 + $(CXX_VERSION_PATCH)))")
else ifeq (2,$(words $(CXX_VERSION_MAJOR) $(CXX_VERSION_MINOR)))
  CXX_VERSION = $(shell echo "$$(($(CXX_VERSION_MAJOR) * 10000 + $(CXX_VERSION_MINOR) * 100))")
  CXX_VERSION_PATCH = 0
else
  CXX_VERSION_STRING = $(NULL)
  CXX_VERSION = 0
endif

CC_VERSION_STRING = $(shell $(CC) --version 2> /dev/null | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
CC_VERSION_MAJOR = $(shell echo "$(CC_VERSION_STRING)" | cut -d"." -f1)
CC_VERSION_MINOR = $(shell echo "$(CC_VERSION_STRING)" | cut -d"." -f2)
CC_VERSION_PATCH = $(shell echo "$(CC_VERSION_STRING)" | cut -d"." -f3)
ifeq (3,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR) $(CC_VERSION_PATCH)))
  CC_VERSION = $(shell echo "$$(($(CC_VERSION_MAJOR) * 10000 + $(CC_VERSION_MINOR) * 100 + $(CC_VERSION_PATCH)))")
else ifeq (2,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR)))
  CC_VERSION = $(shell echo "$$(($(CC_VERSION_MAJOR) * 10000 + $(CC_VERSION_MINOR) * 100))")
  CC_VERSION_PATCH = 0
else
  CC_VERSION_STRING = $(NULL)
  CC_VERSION = 0
endif

ifneq (,$(strip $(FC)))
  FC_VERSION_STRING := $(shell $(FC) --version 2> /dev/null | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
  FC_VERSION_MAJOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f1)
  FC_VERSION_MINOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f2)
  FC_VERSION_PATCH = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f3)
endif
ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
  FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR) * 10000 + $(FC_VERSION_MINOR) * 100 + $(FC_VERSION_PATCH)))")
else ifeq (2,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR)))
  FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR) * 10000 + $(FC_VERSION_MINOR) * 100))")
  FC_VERSION_PATCH = 0
else
  FC_VERSION_STRING = $(NULL)
  FC_VERSION = 0
endif

ifneq (,$(strip $(FC)))
ifeq (GNU,$(shell $(FC) --version 2> /dev/null | head -n1 | cut -d" " -f1))
  ifneq (0,$(shell echo "$$((40500 > $(FC_VERSION)))"))
  ifneq (gfortran,$(notdir $(FC)))
    FC = gfortran
    FC_VERSION_STRING := $(shell $(FC) --version 2> /dev/null | head -n1 | sed "s/..* \([0-9][0-9]*\.[0-9][0-9]*\.*[0-9]*\)[ \S]*.*/\1/")
    FC_VERSION_MAJOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f1)
    FC_VERSION_MINOR = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f2)
    FC_VERSION_PATCH = $(shell echo "$(FC_VERSION_STRING)" | cut -d"." -f3)
    ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
      FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR) * 10000 + $(FC_VERSION_MINOR) * 100 + $(FC_VERSION_PATCH)))")
    else ifeq (2,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR)))
      FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR) * 10000 + $(FC_VERSION_MINOR) * 100))")
      FC_VERSION_PATCH = 0
    else
      FC_VERSION = 0
    endif
    ifneq (0,$(shell echo "$$((40500 > $(FC_VERSION)))"))
      # keep FC_VERSION_STRING available for message about outdated compiler
      FC_VERSION = 0
      FC = $(NULL)
      FORTRAN = 0
    endif
  else
    # keep FC_VERSION_STRING available for message about outdated compiler
    FC_VERSION = 0
    FC = $(NULL)
    FORTRAN = 0
  endif
  endif
else ifneq (0,$(INTEL))
  ifneq (0,$(shell echo "$$((130000 > $(FC_VERSION) && 0 < $(FC_VERSION)))"))
    FC_VERSION = 0
    FC = $(NULL)
    FORTRAN = 0
  endif
endif
endif

# linker setup
LD = $(CC)
XLD = $(CXX)
ifneq (,$(strip $(FC)))
  FLD = $(FC)
else
  FLD = $(LD)
endif

# compiler names
CXX_NAME = $(shell $(CXX) --version 2> /dev/null | head -n1 | cut -d" " -f1)
ifeq (,$(strip $(CXX_NAME)))
  CXX_NAME = $(basename $(notdir $(CXX)))
endif
CC_NAME = $(shell $(CC) --version 2> /dev/null | head -n1 | cut -d" " -f1)
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
ifneq (,$(findstring gcc,$(CC_NAME)))
  GCC = 1
else ifeq (GCC,$(shell $(CC) --version 2> /dev/null | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
  GCC = 1
else ifeq (0,$(shell $(CC) --version 2> /dev/null | grep -q "Free Software Foundation"; echo "$$?"))
  GCC = 1
endif
ifeq (1,$(GCC))
  ifneq (,$(findstring g++,$(CXX_NAME)))
    GCC = 2
  else ifeq (GCC,$(shell $(CXX) --version 2> /dev/null | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
    GCC = 2
  else ifeq (0,$(shell $(CXX) --version 2> /dev/null | grep -q "Free Software Foundation"; echo "$$?"))
    GCC = 2
  endif
  ifeq (2,$(GCC))
    ifneq (,$(findstring gfortran,$(FC_NAME)))
      GCC = 3
    else ifneq (,$(strip $(FC)))
      ifeq (GCC,$(shell $(FC) --version 2> /dev/null | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
        GCC = 3
      else ifeq (0,$(shell $(FC) --version 2> /dev/null | grep -q "Free Software Foundation"; echo "$$?"))
        GCC = 3
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
    | cut -d: -f2- | tr "\n" " " | tr [:upper:] [:lower:]; \
  fi))
ifneq (0,$(AVX))
  ifeq (1,$(words $(filter avx512f,$(CPUFLAGS))))
    AVX ?= 3
    SSE ?= 0
    ifneq (0,$(words $(filter avx512er,$(CPUFLAGS)))) # KNL
      MIC ?= 1
    else # SKX
      MIC ?= 0
    endif
  else ifeq (2,$(words $(filter avx2 fma,$(CPUFLAGS))))
    AVX ?= 2
    SSE ?= 0
  else ifeq (1,$(words $(filter avx avx1.0,$(CPUFLAGS))))
    AVX ?= 1
    SSE ?= 0
  else #ifeq (,$(AVX))
    AVX = 0
  endif
endif
ifneq (0,$(SSE))
  ifeq (1,$(words $(filter sse4_2 sse4.2,$(CPUFLAGS))))
    SSE ?= 4
  else ifeq (1,$(words $(filter sse3,$(CPUFLAGS))))
    SSE ?= 3
  else #ifeq (,$(SSE))
    SSE ?= 0
  endif
else ifneq (0,$(KNC))
  MPSS = 1
endif

ifneq (0,$(INTEL))
  SUITE = Intel Compiler
else ifneq (0,$(GCC))
  SUITE = GNU Compiler Collection
else
  SUITE = Compiler
endif

ifneq (0,$(INTEL))
  AR ?= xiar
  CXXLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
  FCLDFLAGS += $(XLIB_BEGIN) -lc -lstdc++ $(XLIB_END)
  CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
  CXXFLAGS += -f$(PIC) -Wall -diag-disable 1879,3415,10006,10010,10411,13003
  CFLAGS += -f$(PIC) -Wall -diag-disable 1879,3415,10006,10010,10411,13003
  LDFLAGS += -f$(PIC) -diag-disable 1879,3415,10006,10010,10411
  FCFLAGS += -f$(PIC) -diag-disable 10006,10010,10411,13003
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
  ifeq (0,$(DBG))
    # consider more accurate -fp-model (C/C++: precise, Fortran: source)
    CXXFLAGS += $(OPTFLAG) -fno-alias -ansi-alias -fp-model fast=2
    CFLAGS += $(OPTFLAG) -fno-alias -ansi-alias -fp-model fast=2
    FCFLAGS += $(OPTFLAG) -fp-model fast=2
    ifneq (0,$(shell echo "$$((130000 <= $(FC_VERSION)))"))
      FCFLAGS += -align array64byte
    endif
    ifneq (0,$(IPO))
      CXXFLAGS += -ipo
      CFLAGS += -ipo
      FCFLAGS += -ipo
    endif
    ifneq (0,$(TRACE))
      SYM = $(TRACE)
    else ifeq (0,$(shell echo "$$((0 > $(SYM)))"))
      DFLAGS += -DNDEBUG
    endif
  else # debugging enabled
    ifneq (0,$(shell echo "$$((1 < $(DBG) || 0 > $(DBG)))"))
      DFLAGS += -D_DEBUG
    endif
    CXXFLAGS += -O0
    CFLAGS += -O0
    FCFLAGS += -O0 -check
    SYM = $(DBG)
  endif
  ifneq (0,$(TRACE))
    CXXFLAGS += -finstrument-functions
    CFLAGS += -finstrument-functions
    FCFLAGS += -finstrument-functions
    DFLAGS += -D__TRACE=$(TRACE)
    ifneq (0,$(DEPSTATIC))
    ifneq (Darwin,$(UNAME))
      LDFLAGS += -Wl,--export-dynamic
    endif
    endif
  endif
  ifneq (0,$(shell echo "$$((3 > $(DBG)))"))
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
          ifneq (Darwin,$(UNAME))
            CTARGET = -xSSE3
          else # no systems with less than SSE4.2
            CTARGET = -xSSE4.2
          endif
        else ifeq (3,$(SSE))
          CTARGET = -xSSE3
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
      CXXFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CXXFLAGS)
      CFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CFLAGS)
      FCFLAGS := -g -traceback $(FCFLAGS)
    endif
  endif
  ifeq (0,$(EXP))
    CXXFLAGS += -fno-exceptions
  endif
  OMPFLAG = -fopenmp -qoverride_limits
  ifeq (,$(strip $(OMPRT)))
    OMPRT = iomp5
  endif
  ifneq (0,$(OMP))
    CXXFLAGS += $(OMPFLAG)
    CFLAGS += $(OMPFLAG)
    FCFLAGS += $(OMPFLAG)
    LDFLAGS += $(OMPFLAG)
  endif
  ifeq (1,$(STATIC))
    SLDFLAGS += -no-intel-extensions -static-intel -static-libgcc -static-libstdc++
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
else # GCC assumed
  FCLDFLAGS += $(XLIB_BEGIN) -lc -lstdc++ $(XLIB_END)
  LIBGFORTRAN = $(shell $(FLD) $(SLDFLAGS) -lgfortran -Wl,-t 2> /dev/null | grep gfortran)
  ifneq (,$(shell echo "$(LIBGFORTRAN)"))
    ifneq (,$(shell echo "$(LIBGFORTRAN)" | grep succeeded))
      LIBGFORTRAN := $(abspath $(shell echo "$(LIBGFORTRAN)" | grep succeeded | sed "s/..* (\(..*\)).*/\1/" | head -n1))
    else
      LIBGFORTRAN := $(abspath $(shell echo "$(LIBGFORTRAN)" | sed "s/..* (\(..*\)).*/\1/" | head -n1))
    endif
    LDFLAGS := $(LDFLAGS) $(call abslibpath,$(LIBGFORTRAN))
    CXXLDFLAGS += $(XLIB_BEGIN) -lc $(call abslibfile,$(LIBGFORTRAN)) -lm $(XLIB_END)
    CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(call abslibfile,$(LIBGFORTRAN)) -lm $(XLIB_END)
  else
    CXXLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
    CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
  endif
  ifeq (0,$(COMPATIBLE))
    CXXFLAGS += -Wall -Wno-unused-function -Wno-attributes
    CFLAGS += -Wall -Wno-unused-function -Wno-attributes
  endif
  ifneq (Windows_NT,$(UNAME))
    CXXFLAGS += -f$(PIC)
    CFLAGS += -f$(PIC)
    FCFLAGS += -f$(PIC)
    LDFLAGS += -f$(PIC)
  endif
  ifeq (0,$(COMPATIBLE))
    FSTD = -std=f2003
    CPEDANTIC += -pedantic -Wno-variadic-macros -Wno-long-long -Wno-overlength-strings
    FPEDANTIC += -pedantic -Wunused-variable -Wc-binding-type \
                 -Wcharacter-truncation -Wline-truncation \
                 -Wconversion -Wconversion-extra -Wextra \
                 -Wimplicit-interface -Wimplicit-procedure \
                 -Wintrinsics-std -Wreal-q-constant \
                 -Wrealloc-lhs -Wrealloc-lhs-all
    ifneq (0,$(shell echo "$$((50000 <= $(FC_VERSION)))"))
      FWARNEXTRA = -Wuse-without-only
    endif
    FPEDANTIC += $(FWARNEXTRA)
  endif
  ifeq (1,$(PEDANTIC))
    CSTD = -std=c99
    CXXFLAGS += $(CPEDANTIC)
    CFLAGS += $(CSTD) $(CPEDANTIC)
    FCFLAGS += -ffree-form
    FMFLAGS += $(FSTD) -pedantic -Wunused-variable $(FWARNEXTRA)
  else ifneq (0,$(PEDANTIC))
    CSTD = -std=c89
    CXXFLAGS += $(CPEDANTIC)
    CFLAGS += $(CSTD) $(CPEDANTIC)
    FCFLAGS += $(FSTD) $(FPEDANTIC)
  else ifeq (0,$(COMPATIBLE))
    CSTD = -std=c89
    FCFLAGS += -ffree-form
  endif
  ifeq (0,$(COMPATIBLE))
    # flag specifying output directory must be last
    FMFLAGS += -J
  else # fallback
    FMFLAGS += -I
  endif
  ifeq (0,$(DBG))
    CXXFLAGS += $(OPTFLAG)
    CFLAGS += $(OPTFLAG)
    FCFLAGS += $(OPTFLAG)
    ifeq (0,$(COMPATIBLE))
      CXXFLAGS += -funroll-loops -ftree-vectorize
      CFLAGS += -funroll-loops -ftree-vectorize
      FCFLAGS += -funroll-loops -ftree-vectorize
      ifneq (0,$(GCC))
        CXXFLAGS += -ffast-math
        CFLAGS += -ffast-math
        FCFLAGS += -ffast-math
      else ifneq (3,$(AVX))
        CXXFLAGS += -ffast-math
        CFLAGS += -ffast-math
        FCFLAGS += -ffast-math
      endif
    endif
    ifneq (0,$(IPO))
      CXXFLAGS += -flto -ffat-lto-objects
      CFLAGS += -flto -ffat-lto-objects
      FCFLAGS += -flto -ffat-lto-objects
      LDFLAGS += -flto
    endif
    ifneq (0,$(TRACE))
      SYM = $(TRACE)
    else ifeq (0,$(shell echo "$$((0 > $(SYM)))"))
      DFLAGS += -DNDEBUG
    endif
  else
    ifneq (0,$(shell echo "$$((1 < $(DBG) || 0 > $(DBG)))"))
      DFLAGS += -D_DEBUG
    endif
    CXXFLAGS += -O0
    CFLAGS += -O0
    FCFLAGS += -O0
    SYM = $(DBG)
  endif
  ifneq (0,$(TRACE))
    CXXFLAGS += -finstrument-functions
    CFLAGS += -finstrument-functions
    FCFLAGS += -finstrument-functions
    # e.g. clang does not need/understand below flag
    ifneq (0,$(shell echo "$$((1 <= $(GCC)))"))
      CFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
      ifneq (0,$(shell echo "$$((2 <= $(GCC)))"))
        CXXFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
        ifneq (0,$(shell echo "$$((3 <= $(GCC)))"))
          FCFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_,__rdtsc
        endif
      endif
    endif
    ifneq (0,$(DEPSTATIC))
    ifneq (Darwin,$(UNAME))
      LDFLAGS += -Wl,--export-dynamic
    endif
    endif
    ifeq (Windows_NT,$(UNAME))
      LDFLAGS += -ldbghelp
    endif
    DFLAGS += -D__TRACE=$(TRACE)
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
      CXXFLAGS := -g3 -gdwarf-2 $(CXXFLAGS)
      CFLAGS := -g3 -gdwarf-2 $(CFLAGS)
      FCFLAGS := -g $(FCFLAGS)
    endif
  endif
  ifeq (0,$(EXP))
    CXXFLAGS += -fno-exceptions
  endif
  ifeq (0,$(COMPATIBLE))
    ifneq (0,$(shell echo "$$((1 <= $(GCC)))"))
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
  ifneq (0,$(STATIC))
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
ifneq (0,$(shell echo "$$((3 > $(DBG)))"))
  ifneq (0,$(shell echo "$$((2 <= $(AVX)))"))
    ifeq (3,$(AVX))
    ifeq (0,$(shell $(CC) -E -mavx512f -mavx512cd /dev/null 2> /dev/null > /dev/null; echo $$?))
      CTARGET = -mavx512f -mavx512cd
      ifneq (,$(MIC)) # specified
        ifneq (0,$(MIC)) # MIC
          ifeq (0,$(shell $(CC) -E -mavx512pf -mavx512er /dev/null 2> /dev/null > /dev/null; echo $$?))
            CTARGET += -mavx512pf -mavx512er
          endif
        else # SKX
          ifeq (0,$(shell $(CC) -E -mavx512dq -mavx512bw -mavx512vl /dev/null 2> /dev/null > /dev/null; echo $$?))
            CTARGET += -mavx512dq -mavx512bw -mavx512vl
            #CTARGET += -mavx512ifma -mavx512vbmi
          endif
        endif
      endif
    endif
    endif
    ifeq (,$(CTARGET)) # fall-back to AVX2
      CTARGET = -march=core-avx2
    endif
  else ifeq (1,$(AVX))
    CTARGET = -mavx
  else ifneq (0,$(SSE))
    ifeq (1,$(SSE)) # default
      ifneq (Darwin,$(UNAME))
        CTARGET = -msse3
      else
        # prevents Clang backend error with CRC32/other instructions
        CTARGET = -msse4.2
      endif
    else ifeq (2,$(SSE))
      # implicitly present (64-bit ABI)
    else ifeq (4,$(SSE))
      CTARGET = -msse4.2
    else # perhaps better to use TARGET flag directly
      CTARGET = -msse$(SSE)
    endif
  else ifneq (0,$(AVX))
    CTARGET = -march=native
  endif
endif
endif

# take user's TARGET into account
ifneq (undefined,$(origin TARGET))
  CTARGET = $(TARGET)
endif

# check target flags by feeding an ad-hoc C program into the compiler
ifneq (0,$(shell CINFILE=$$(mktemp checkXXXXXX.c); \
  echo "int main(void) { return 0; }" > $${CINFILE}; \
  CTEST=$$($(CC) $(CTARGET) -c $${CINFILE} 2>&1); \
  if [ "" = "$${CTEST}" ]; then echo $$?; else echo 1; fi; \
  rm -f $$(basename $${CINFILE} .c)*))
# revoke target flags
  CTARGET = $(NULL)
endif

ifneq (,$(strip $(FC))) # check target flags by feeding an ad-hoc Fortran program into the compiler
ifeq (0,$(shell FINFILE=$$(mktemp checkXXXXXX.f); \
  echo "      PROGRAM test" > $${FINFILE}; \
  echo "      END PROGRAM" >> $${FINFILE}; \
  FTEST=$$($(FC) $(CTARGET) -c $${FINFILE} 2>&1); \
  if [ "" = "$${FTEST}" ]; then echo $$?; else echo 1; fi; \
  rm -f $$(basename $${FINFILE} .f)*))
# inherit CTARGET flags
  FTARGET = $(CTARGET)
endif
endif

ifneq (0,$(THREADS))
  DFLAGS += -D_REENTRANT
  ifeq (0,$(COMPATIBLE))
    CXXFLAGS += -pthread
    CFLAGS += -pthread
    FLDFLAGS += -lpthread
    ifneq (Windows_NT,$(UNAME))
      CLDFLAGS += -pthread
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
  FINFO = Fortran: $(if $(FTARGET),$(FTARGET),<compiler default>)
else
  GINFO = $(SUITE): $(strip $(CC_NAME) $(CC_VERSION_STRING)), and $(strip $(CXX_NAME) $(CXX_VERSION_STRING))
  FINFO = Fortran: <none>
endif
CINFO = C / C++: $(if $(CTARGET),$(CTARGET),<compiler default>)

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
  ifneq (0,$(shell echo "$$((1 <= $(GCC)))"))
    LDFLAGS += -Wa,-q
    CFLAGS += -Wa,-q
    ifneq (0,$(shell echo "$$((2 <= $(GCC)))"))
      CXXFLAGS += -Wa,-q
    endif
  endif
  FLDFLAGS += -Wa,-q
  FCFLAGS += -Wa,-q
endif

ifneq (0,$(shell $(LD) $(SLDFLAGS) -l$(OMPRT) 2>&1 | grep -q "\-l$(OMPRT)"; echo "$$?"))
  OMPLIBFILE := $(shell $(LD) $(SLDFLAGS) -l$(OMPRT) -Wl,-t 2> /dev/null | grep $(OMPRT))
  ifneq (,$(shell echo "$(OMPLIBFILE)" | grep succeeded))
    OMPLIBFILE := $(abspath $(shell echo "$(OMPLIBFILE)" | grep succeeded | sed "s/..* (\(..*\)).*/\1/" | head -n1))
  else ifneq (,$(OMPLIBFILE))
    OMPLIBFILE := $(abspath $(shell echo "$(OMPLIBFILE)" | sed "s/..* (\(..*\)).*/\1/" | head -n1))
  endif
  ifneq (,$(strip $(OMPLIBFILE)))
    OMPLIB = $(strip $(call abslib,$(OMPLIBFILE)))
  else
    OMPLIB = -l$(OMPRT)
  endif
else
  OMPLIB = $(NULL)
endif

ifeq (Darwin,$(UNAME))
ifneq (,$(OMPLIBFILE))
  ifeq (0,$(ABSLIBS))
    LDFLAGS += $(call abslibrpath,$(OMPLIBFILE))
  endif
  LDFLAGS += $(call abslib,$(OMPLIBFILE))
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
  MKL_DIRECT ?= 0
else
  MKL = 0
endif
ifneq (0,$(MKL))
  BLAS_DFLAGS += -D__CBLAS
endif
ifeq (1,$(MKL_DIRECT))
  ifeq (1,$(MKL))
    BLAS_DFLAGS += -DMKL_DIRECT_CALL_SEQ
  else ifneq (0,$(MKL))
    BLAS_DFLAGS += -DMKL_DIRECT_CALL
  endif
endif
ifneq (Darwin,$(UNAME))
  MKL_ARCH = intel64
endif

ifeq (1,$(MKL)) # sequential
  BLAS_DFLAGS += -D__BLAS=1 -D__MKL=1
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
  BLAS_LDFLAGS += -lpthread -lm $(LIBDL)
  ifneq (0,$(OFFLOAD))
    BLAS_LDFLAGS += -qoffload-option,mic,ld,"-lm $(LIBDL)"
  endif
else ifneq (0,$(MKL)) # multi-threaded
  BLAS_DFLAGS += -D__BLAS=$(MKL) -D__MKL=$(MKL)
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
  BLAS_LDFLAGS += -lpthread -lm $(LIBDL)
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
    # most people expect to pickup OpenBLAS (if available) even when libblas/liblapack are available as well; use OPENBLAS=0 to avoid this
    OPENBLAS = $(shell $(LD) $(SLDFLAGS) $(call abslibpath,$(BLASDIR)) -lopenblas$(BLAS_THREADS)$(BLAS_BITS) 2>&1 | \
      grep -q "\-lopenblas$(BLAS_THREADS)$(BLAS_BITS)"; echo "$$?")

    ifneq (0,$(OPENBLAS))
      BLAS_LDFLAGS += -lopenblas$(BLAS_THREADS)$(BLAS_BITS)
      # OpenBLAS also carries the CBLAS bits
      BLAS_DFLAGS += -D__BLAS=$(BLAS) -D__CBLAS -D__OPENBLAS
    else
      BLAS_DFLAGS += -D__BLAS
      ifneq (Windows_NT,$(UNAME))
        BLAS_LDFLAGS += -lblas$(BLAS_BITS)
      else
        BLAS_LDFLAGS += -lblas$(BLAS_BITS).dll
      endif
      ifneq (0,$(STATIC))
      ifneq (,$(LIBGFORTRAN))
        BLAS_CLDFLAGS += $(XLIB_BEGIN) $(call abslibfile,$(LIBGFORTRAN)) -lm $(XLIB_END)
      endif
      endif
      QUADMATH = $(shell $(LD) $(SLDFLAGS) -lquadmath 2>&1 | grep -q "\-lquadmath"; echo "$$?")
      ifneq (0,$(QUADMATH))
        BLAS_CLDFLAGS += $(XLIB_BEGIN) -lquadmath $(XLIB_END)
      endif
    endif
  else # BLAS library is specified via BLASLIB
    BLAS_DFLAGS += -D__BLAS=$(BLAS)
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
NOBLAS_CXXFLAGS := $(strip $(CXXFLAGS) $(ECXXFLAGS) $(EFLAGS))
NOBLAS_FCFLAGS := $(strip $(FCFLAGS) $(EFCFLAGS) $(EFLAGS))
NOBLAS_CFLAGS := $(strip $(CFLAGS) $(ECFLAGS) $(EFLAGS))
NOBLAS_LDFLAGS := $(strip $(LDFLAGS) $(ELDFLAGS))
NOBLAS_CLDFLAGS := $(strip $(CLDFLAGS))
NOBLAS_DFLAGS := $(strip $(DFLAGS))
NOBLAS_IFLAGS := $(strip $(IFLAGS))

# regular flags: cleanup
CXXFLAGS := $(strip $(NOBLAS_CXXFLAGS) $(BLAS_CXXFLAGS))
FCFLAGS := $(strip $(NOBLAS_FCFLAGS) $(BLAS_FCFLAGS))
CFLAGS := $(strip $(NOBLAS_CFLAGS) $(BLAS_CFLAGS))
LDFLAGS := $(strip $(NOBLAS_LDFLAGS) $(BLAS_LDFLAGS))
CLDFLAGS := $(strip $(NOBLAS_CLDFLAGS) $(BLAS_CLDFLAGS))
DFLAGS := $(strip $(NOBLAS_DFLAGS) $(BLAS_DFLAGS))
IFLAGS := $(strip $(NOBLAS_IFLAGS) $(BLAS_IFLAGS))

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
  $(if $(filter-out .SHELLSTATUS .DEFAULT_GOAL MAKEFILE_LIST MAKEOVERRIDES MAKEFLAGS \
    $(EXCLUDE_STATE) CPUFLAGS TESTSIZE abslib soname extname objname,$V), \
    $V=$($V)?)))
.state: $(shell echo '$(STATE)' | $(DEPDIR)/.state.sh $(abspath .))

%/.make:
	@mkdir -p $(basename $@)
	@touch $@

.make:
	@touch $@

