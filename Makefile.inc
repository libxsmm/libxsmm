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

# Optimization level
OPT ?= 2

# Optimization flag derived from OPT flag
ifeq (0,$(shell echo $(OPT) | grep -q "^[0-9]\+$$"; echo $$?))
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

# Static or shared binary
STATIC ?= 0

OFFLOAD ?= 0
ifneq (0,$(OFFLOAD))
  MPSS ?= 1
  MIC ?= 1
else
  MPSS ?= 0
  MIC ?= 0
endif

DEPDIR ?= $(ROOTDIR)
ifeq (0,$(MIC))
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

# PTHREAD refers to foundational TRT (and not necessarily Posix Threads)
ifneq (0,$(OMP))
  PTHREAD ?= 1
else
  PTHREAD ?= 0
endif

# Threading runtime
OMP ?= 0

# Code conformance (beyond -Wall)
PEDANTIC ?= 0

# Embed InterProcedural Optimization information into libraries
IPO ?= 0

# ILP64=0 (LP64 with 32-bit integers), and ILP64=0 (64-bit integers)
ILP64 ?= 0
BLAS ?= 0

# Prefer to link statically against BLAS library
BLAS_STATIC ?= 1

# Conditional abspath/link-line generator function
ABSLIBS ?= 0
ifneq (0,$(DEPSTATIC))
  libdir = $(abspath $1)
else ifneq (0,$(ABSLIBS))
  libdir = $(abspath $1)
else
  libdir = -L$(dir $(abspath $1)) -l$(patsubst lib%.$(DLIBEXT),%,$(notdir $1))
endif

# Pickup OpenMP library name if passed as OMP=libname|name
ifneq (0,$(shell echo $(OMP) | grep -q "^-*[0-9]\+$$"; echo $$?)) # NaN
  OMPLIB = $(patsubst lib%,%,$(OMP))
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

FORTRAN ?= $(if $(notdir $(shell which $(FC) 2> /dev/null)), 1, 0)
ifneq (0,$(INTEL))
  ifeq (0,$(OFFLOAD))
    CXXFLAGS += -qno-offload
    CFLAGS += -qno-offload
    ifneq (0,$(FORTRAN))
      ifeq (ifort,$(shell $(FC) --version 2> /dev/null | head -n1 | cut -d" " -f1))
        FCFLAGS += -qno-offload
      endif
    endif
  endif
  XLIB_BEGIN = -Wl,--as-needed
  XLIB_END = -Wl,--no-as-needed
  # disable CPUID check
  COMPATIBLE = 1
else
  ifeq (,$(notdir $(shell which $(CXX) 2> /dev/null)))
    CXX = g++
  else ifneq (0,$(shell $(CXX) --version > /dev/null 2> /dev/null; echo $$?))
    CXX = g++
  endif
  ifeq (,$(notdir $(shell which $(CC) 2> /dev/null)))
    CC = gcc
  else ifneq (0,$(shell $(CC) --version > /dev/null 2> /dev/null; echo $$?))
    CC = gcc
  endif
  ifeq (0,$(FORTRAN))
    FC = $(shell which gfortran 2> /dev/null)
  else ifneq (0,$(shell $(FC) --version > /dev/null 2> /dev/null; echo $$?))
    FC = $(shell which gfortran 2> /dev/null)
  endif
endif

ifneq (,$(strip $(FC)))
  FORTRAN = 1
else
  FORTRAN = 0
endif
ifeq (0,$(FORTRAN))
  FC = $(NULL)
endif

# linker setup
LD = $(CC)

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

ifneq (0,$(FORTRAN))
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
endif

# compiler names
CXX_NAME = $(shell $(CXX) --version 2> /dev/null | head -n1 | cut -d" " -f1)
CC_NAME = $(shell $(CC) --version 2> /dev/null | head -n1 | cut -d" " -f1)
ifneq (,$(strip $(FC)))
  FC_NAME = $(firstword $(notdir $(FC)))
endif

# native GCC?
GCC ?= 0
ifneq (,$(findstring gcc,$(CC_NAME)))
  GCC = 1
else ifeq (GCC,$(shell $(CC) --version 2> /dev/null | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
  GCC = 1
else ifeq (0,$(shell $(CC) --version 2> /dev/null | grep -q "Free Software Foundation"; echo $$?))
  GCC = 1
endif
ifeq (1,$(GCC))
  # C Compiler is used for link stage
  ifneq (Darwin,$(UNAME))
    XLIB_BEGIN = -Wl,--as-needed
    XLIB_END = -Wl,--no-as-needed
  endif
  ifneq (,$(findstring g++,$(CXX_NAME)))
    GCC = 2
  else ifeq (GCC,$(shell $(CXX) --version 2> /dev/null | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
    GCC = 2
  else ifeq (0,$(shell $(CXX) --version 2> /dev/null | grep -q "Free Software Foundation"; echo $$?))
    GCC = 2
  endif
  ifeq (2,$(GCC))
    ifneq (,$(findstring gfortran,$(FC_NAME)))
      GCC = 3
    else ifneq (,$(strip $(FC)))
      ifeq (GCC,$(shell $(FC) --version 2> /dev/null | head -n1 | sed "s/.* (\(..*\)) .*/\1/"))
        GCC = 3
      else ifeq (0,$(shell $(FC) --version 2> /dev/null | grep -q "Free Software Foundation"; echo $$?))
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
    ifeq (1,$(words $(filter avx512er,$(CPUFLAGS))))
      MIC = 1
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
  ifeq (1,$(words $(filter sse3,$(CPUFLAGS))))
    SSE ?= 3
  else #ifeq (,$(SSE))
    SSE ?= 0
  endif
else ifneq (0,$(MIC))
  ifneq (3,$(AVX))
    MPSS = 1
  endif
endif

ifneq (0,$(INTEL))
  SUITE = Intel Compiler suite
else ifneq (0,$(GCC))
  SUITE = GNU Compiler Collection
else
  SUITE = Other compilers
endif

ifneq (,$(strip $(FC)))
  INFO = Using $(SUITE): $(strip $(CXX_NAME) $(CXX_VERSION_STRING)), $(strip $(CC_NAME) $(CC_VERSION_STRING)), and $(strip $(FC_NAME) $(FC_VERSION_STRING))
else
  INFO = Using $(SUITE): $(strip $(CXX_NAME) $(CXX_VERSION_STRING)), and $(strip $(CC_NAME) $(CC_VERSION_STRING))
endif

ifneq (0,$(INTEL))
  AR ?= xiar
  CXXLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
  FCLDFLAGS += $(XLIB_BEGIN) -lc -lstdc++ $(XLIB_END)
  CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
  CXXFLAGS += -fPIC -Wall -diag-disable 1879,3415,10010,10411,13003
  CFLAGS += -fPIC -Wall -diag-disable 1879,3415,10010,10411,13003
  LDFLAGS += -diag-disable 1879,3415,10010,10411
  FCFLAGS += -fPIC -diag-disable 10010,10411,13003
  LDFLAGS += -fPIC
  FCMTFLAGS += -threads
  ifeq (1,$(PEDANTIC))
    CFLAGS += -std=c89 -Wcheck
    FCFLAGS += -free
    FMFLAGS += -std03 -warn all,notruncated_source -diag-disable 7025,7373,10010,10237,10342,10382
  else ifneq (0,$(PEDANTIC))
    CFLAGS += -std=c89 -Wcheck -Wremarks -diag-disable 177,1419,2547,10010,10382
    FCFLAGS += -std03 -warn all,notruncated_source -diag-disable 7025,7373,10010,10237,10342,10382
    FMFLAGS += -fixed
  else
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
    else
      DFLAGS += -DNDEBUG
    endif
  else # debugging enabled
    ifneq (0,$(shell echo $$((1 < $(DBG)))))
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
      LDFLAGS += -rdynamic
    endif
  endif
  ifneq (0,$(shell echo $$((3 > $(DBG)))))
    ifeq (0,$(COMPATIBLE))
      ifeq (1,$(AVX))
        TARGET = -xAVX
      else ifeq (2,$(AVX))
        TARGET = -xCORE-AVX2
      else ifeq (3,$(AVX))
        ifeq (0,$(MIC))
          ifneq (file,$(origin MIC))
            TARGET = -xCORE-AVX512
          else
            TARGET = -xCOMMON-AVX512
          endif
        else
          TARGET = -xMIC-AVX512
        endif
      else ifneq (0,$(SSE))
        ifeq (1,$(SSE)) # default
          TARGET = -xSSE3
        else ifeq (3,$(SSE))
          TARGET = -xSSE3
        else ifeq (4,$(SSE))
          TARGET = -xSSE4.2
        else
          TARGET = -xSSE$(SSE)
        endif
      else ifneq (0,$(AVX))
        TARGET = -xHost
      endif
    else
      ifeq (1,$(AVX))
        TARGET = -mavx
      else ifeq (2,$(AVX))
        TARGET = -march=core-avx2
      else ifeq (3,$(AVX))
        ifeq (0,$(MIC))
          TARGET = -xCOMMON-AVX512
        else
          TARGET = -xMIC-AVX512
        endif
      else ifneq (0,$(SSE))
        ifeq (1,$(SSE)) # default
          TARGET = -msse3
        else ifeq (2,$(SSE))
          # implicitly present (64-bit ABI)
        else ifeq (3,$(SSE))
          TARGET = -msse3
        else ifeq (4,$(SSE))
          TARGET = -msse4.2
        else # perhaps better to use TARGET
          TARGET = -msse$(SSE)
        endif
      else ifneq (0,$(AVX))
        TARGET = -march=native
      endif
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
  ifeq (,$(strip $(OMPLIB)))
    OMPLIB = iomp5
  endif
  ifneq (0,$(OMP))
    CXXFLAGS += $(OMPFLAG)
    CFLAGS += $(OMPFLAG)
    FCFLAGS += $(OMPFLAG)
    LDFLAGS += $(OMPFLAG)
  endif
  ifneq (0,$(PTHREAD))
    DFLAGS += -D_REENTRANT
    CXXFLAGS += -pthread
    CFLAGS += -pthread
    ifneq (Windows_NT,$(UNAME))
      CLDFLAGS += -pthread
    endif
  else
    DFLAGS += -DLIBXSMM_NOSYNC
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
  ifneq (,$(strip $(FC)))
    LIBGFORTRAN = $(abspath $(shell $(FC) $(SLDFLAGS) -lgfortran -Wl,--verbose 2> /dev/null | grep libgfortran | grep succeeded | sed "s/..* \(..*\) succeeded/\1/" | head -n1))
  else
    LIBGFORTRAN = $(abspath $(shell $(LD) $(SLDFLAGS) -lgfortran -Wl,--verbose 2> /dev/null | grep libgfortran | grep succeeded | sed "s/..* \(..*\) succeeded/\1/" | head -n1))
  endif
  ifneq (,$(LIBGFORTRAN))
    LIBGFCPATH = $(dir $(LIBGFORTRAN))
    LDFLAGS := $(LDFLAGS) -L$(LIBGFCPATH) -Wl,-rpath=$(LIBGFCPATH)
    ifeq (3,$(GCC)) # prune link-line
      LIBGFORTRAN = -lgfortran
    endif
    CXXLDFLAGS += $(XLIB_BEGIN) -lc $(LIBGFORTRAN) -lm $(XLIB_END)
    CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(LIBGFORTRAN) -lm $(XLIB_END)
  else
    CXXLDFLAGS += $(XLIB_BEGIN) -lc $(XLIB_END)
    CLDFLAGS += $(XLIB_BEGIN) -lstdc++ $(XLIB_END)
  endif
  CXXFLAGS += -Wall -Wno-unused-function
  CFLAGS += -Wall -Wno-unused-function
  ifeq (0,$(COMPATIBLE))
    CXXFLAGS += -Wno-attributes
    CFLAGS += -Wno-attributes
  endif
  ifneq (Windows_NT,$(UNAME))
    CXXFLAGS += -fPIC
    CFLAGS += -fPIC
    FCFLAGS += -fPIC
    LDFLAGS += -fPIC
  endif
  ifeq (1,$(PEDANTIC))
    CFLAGS += -std=c89 -pedantic -Wno-variadic-macros -Wno-long-long -Wno-overlength-strings
    FCFLAGS += -ffree-form
    FMFLAGS += -std=f2003 -pedantic -Wunused-variable
    ifneq (0,$(shell echo "$$((50000 <= $(FC_VERSION)))"))
      FMFLAGS += -Wuse-without-only
    endif
  else ifneq (0,$(PEDANTIC))
    CFLAGS += -std=c89 -pedantic -Wno-variadic-macros -Wno-long-long -Wno-overlength-strings
    FCFLAGS += -std=f2003 -pedantic -Wunused-variable
    ifneq (0,$(shell echo "$$((50000 <= $(FC_VERSION)))"))
      FMFLAGS += -Wuse-without-only
    endif
  else ifeq (0,$(COMPATIBLE))
    FCFLAGS += -ffree-form
  endif
  ifeq (0,$(COMPATIBLE))
    # flag specifying output directory must be last
    FMFLAGS += -J
  else # fallback
    FMFLAGS += -I
  endif
  ifeq (0,$(DBG))
    CXXFLAGS += $(OPTFLAG) -funroll-loops
    CFLAGS += $(OPTFLAG) -funroll-loops
    FCFLAGS += $(OPTFLAG) -funroll-loops
    ifeq (0,$(COMPATIBLE))
      CXXFLAGS += -ftree-vectorize -ffast-math
      CFLAGS += -ftree-vectorize -ffast-math
      FCFLAGS += -ftree-vectorize -ffast-math
    endif
    ifneq (0,$(IPO))
      CXXFLAGS += -flto -ffat-lto-objects
      CFLAGS += -flto -ffat-lto-objects
      FCFLAGS += -flto -ffat-lto-objects
      LDFLAGS += -flto
    endif
    ifneq (0,$(TRACE))
      SYM = $(TRACE)
    else
      DFLAGS += -DNDEBUG
    endif
  else
    ifneq (0,$(shell echo $$((1 < $(DBG)))))
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
    ifneq (0,$(shell echo $$((1 <= $(GCC)))))
      CFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_
      ifneq (0,$(shell echo $$((2 <= $(GCC)))))
        CXXFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_
        ifneq (0,$(shell echo $$((3 <= $(GCC)))))
          FCFLAGS += -finstrument-functions-exclude-function-list=_mm_,_mm256_,_mm512_
        endif
      endif
    endif
    ifneq (Windows_NT,$(UNAME))
      ifneq (0,$(DEPSTATIC))
        ifeq (3,$(GCC))
          LDFLAGS += -rdynamic
        endif
      endif
    else
      LDFLAGS += -ldbghelp
    endif
    DFLAGS += -D__TRACE=$(TRACE)
  endif
  ifneq (0,$(shell echo $$((3 > $(DBG)))))
    ifeq (3,$(GCC))
      # downgrade to AVX2 due to outdated compiler
      ifneq (0,$(shell echo "$$((3 <= $(AVX)))"))
        ifneq (0,$(shell echo "$$((40900 > $(CC_VERSION)))"))
          AVX = 2
        endif
      endif
    endif
    ifeq (1,$(AVX))
      TARGET = -mavx
    else ifeq (2,$(AVX))
      TARGET = -mavx2
    else ifeq (3,$(AVX))
      TARGET = -mavx512f -mavx512cd
      ifneq (0,$(MIC))
        TARGET += -mavx512er -mavx512pf
      endif
    else ifneq (0,$(SSE))
      ifeq (1,$(SSE)) # default
        TARGET = -msse3
      else ifeq (2,$(SSE))
        # implicitly present (64-bit ABI)
      else ifeq (4,$(SSE))
        TARGET = -msse4.2
      else # perhaps better to use TARGET
        TARGET = -msse$(SSE)
      endif
    else ifneq (0,$(AVX))
      TARGET = -march=native
    endif
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
      LDFLAGS := -g $(LDFLAGS) -fsanitize=thread -fno-omit-frame-pointer
      ELDFLAGS := -pie
    else
      CXXFLAGS := -g3 -gdwarf-2 $(CXXFLAGS)
      CFLAGS := -g3 -gdwarf-2 $(CFLAGS)
      FCFLAGS := -g $(FCFLAGS)
    endif
  endif
  ifeq (0,$(EXP))
    CXXFLAGS += -fno-exceptions
  endif
  OMPFLAG = -fopenmp
  ifeq (,$(strip $(OMPLIB)))
    OMPLIB = gomp
  endif
  ifneq (0,$(OMP))
    # clang: OMP=libomp
    ifneq (0,$(shell echo $(OMP) | grep -q "^-*[0-9]\+$$"; echo $$?)) # NaN
      CXXFLAGS += $(OMPFLAG)=$(OMP)
      CFLAGS += $(OMPFLAG)=$(OMP)
      # omit passing special OMP into Fortran compiler
      FCFLAGS += $(OMPFLAG)
      CLDFLAGS += $(OMPFLAG)=$(OMP)
    else ifneq (0,$(shell $(LD) $(SLDFLAGS) -l$(OMPLIB) 2>&1 | grep -q "\-l$(OMPLIB)"; echo $$?))
      CXXFLAGS += $(OMPFLAG)
      CFLAGS += $(OMPFLAG)
      FCFLAGS += $(OMPFLAG)
      LDFLAGS += $(OMPFLAG)
    endif
  endif
  ifneq (0,$(PTHREAD))
    DFLAGS += -D_REENTRANT
    CXXFLAGS += -pthread
    CFLAGS += -pthread
    ifneq (Windows_NT,$(UNAME))
      CLDFLAGS += -pthread
    endif
  else
    DFLAGS += -DLIBXSMM_NOSYNC
  endif
  ifneq (0,$(STATIC))
    DFLAGS += -D__STATIC=$(STATIC)
    ifeq (0,$(shell $(LD) -static -ldummydoesnotexist 2>&1 | grep -q "\-ldummydoesnotexist"; echo $$?))
      SLDFLAGS += -static
    endif
  endif
  ifneq (,$(strip $(R8)))
    ifneq (0,$(R8))
      FCFLAGS += -fdefault-real-8 -fdefault-double-8
    endif
  endif
  OFFLOAD = 0
  MIC = 0
endif

ifneq (0,$(shell $(LD) $(SLDFLAGS) -lrt 2>&1 | grep -q "\-lrt"; echo $$?))
  LIBRT = -lrt
endif
ifneq (0,$(shell $(LD) $(SLDFLAGS) -ldl 2>&1 | grep -q "\-ldl"; echo $$?))
  LIBDL = -ldl
endif
ifneq (,$(strip $(LIBRT) $(LIBDL)))
  LDFLAGS += $(XLIB_BEGIN) $(LIBRT) $(LIBDL) $(XLIB_END)
endif
ifeq (Darwin,$(UNAME))
  # avoid Homebrew based GCC AS; apply the flag only to the non-GCC components
  ifneq (0,$(shell echo $$((1 <= $(GCC)))))
    LDFLAGS += -Wa,-q
    CFLAGS += -Wa,-q
    ifneq (0,$(shell echo $$((2 <= $(GCC)))))
      CXXFLAGS += -Wa,-q
    endif
  endif
  FLDFLAGS += -Wa,-q
  FCFLAGS += -Wa,-q
endif
ifeq (0,$(shell $(LD) $(SLDFLAGS) -l$(OMPLIB) 2>&1 | grep -q "\-l$(OMPLIB)"; echo $$?))
  OMPLIB = $(NULL)
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

ifeq (1,$(MKL_DIRECT))
  DFLAGS += -DMKL_DIRECT_CALL_SEQ
endif

ifeq (1,$(MKL)) # sequential
  DFLAGS += -D__MKL=1
  IFLAGS += -I$(MKLROOT)/include
  ifeq (0,$(BLAS_STATIC)) # shared
    LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_sequential
    ifneq (0,$(OFFLOAD))
      LDFLAGS += -qoffload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_sequential"
    endif
  else # static
    LDFLAGS += -Wl,--start-group \
      $(MKLROOT)/lib/intel64/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
      $(MKLROOT)/lib/intel64/libmkl_core.$(SLIBEXT) \
      $(MKLROOT)/lib/intel64/libmkl_sequential.$(SLIBEXT) \
    -Wl,--end-group
    ifneq (0,$(OFFLOAD))
      LDFLAGS += -qoffload-option,mic,ld,"--start-group \
        $(MKLROOT)/lib/mic/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
        $(MKLROOT)/lib/mic/libmkl_core.$(SLIBEXT) \
        $(MKLROOT)/lib/mic/libmkl_sequential.$(SLIBEXT) \
      --end-group"
    endif
  endif
  LDFLAGS += -lpthread -lm $(LIBDL)
  ifneq (0,$(OFFLOAD))
    LDFLAGS += -qoffload-option,mic,ld,"-lm $(LIBDL)"
  endif
else ifneq (0,$(MKL)) # multi-threaded
  DFLAGS += -D__MKL=$(MKL)
  IFLAGS += -I$(MKLROOT)/include
  ifeq (0,$(BLAS_STATIC)) # shared
    ifneq (0,$(INTEL))
      LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_intel_thread
      ifneq (0,$(OFFLOAD))
        LDFLAGS += -qoffload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_intel_thread"
      endif
    else # assuming GNU toolchain
      LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_gnu_thread
    endif
  else # static
    ifneq (0,$(INTEL))
      LDFLAGS += -Wl,--start-group \
        $(MKLROOT)/lib/intel64/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
        $(MKLROOT)/lib/intel64/libmkl_core.$(SLIBEXT) \
        $(MKLROOT)/lib/intel64/libmkl_intel_thread.$(SLIBEXT) \
      -Wl,--end-group
      ifneq (0,$(OFFLOAD))
        LDFLAGS += -qoffload-option,mic,ld,"--start-group \
          $(MKLROOT)/lib/mic/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
          $(MKLROOT)/lib/mic/libmkl_core.$(SLIBEXT) \
          $(MKLROOT)/lib/mic/libmkl_intel_thread.$(SLIBEXT) \
        --end-group"
      endif
    else # assuming GNU toolchain
      LDFLAGS += -Wl,--start-group \
        $(MKLROOT)/lib/intel64/libmkl_intel_$(MKL_BITS).$(SLIBEXT) \
        $(MKLROOT)/lib/intel64/libmkl_core.$(SLIBEXT) \
        $(MKLROOT)/lib/intel64/libmkl_gnu_thread.$(SLIBEXT) \
      -Wl,--end-group $(LIBDL)
    endif
  endif
  ifneq (0,$(INTEL))
    ifeq (0,$(OMP))
      ifneq (,$(strip $(OMPLIB)))
        LDFLAGS += -l$(OMPLIB)
        ifneq (0,$(OFFLOAD))
          LDFLAGS += -qoffload-option,mic,ld,"-l$(OMPLIB)"
        endif
      endif
    endif
    ifneq (0,$(OFFLOAD))
      LDFLAGS += -qoffload-option,mic,ld,"-lm $(LIBDL)"
    endif
  endif
  LDFLAGS += -lpthread -lm $(LIBDL)
else ifneq (0,$(BLAS)) # generic
  ifneq (1,$(BLAS))
    BLAS_THREADS = o
  endif
  OPENBLAS = $(shell $(LD) $(SLDFLAGS) -lopenblas$(BLAS_THREADS)$(BLAS_BITS) 2>&1 | grep -q "\-lopenblas$(BLAS_THREADS)$(BLAS_BITS)"; echo $$?)
  ifeq (1,$(OPENBLAS))
    LDFLAGS += -lopenblas$(BLAS_THREADS)$(BLAS_BITS)
  else
    ifneq (Windows_NT,$(UNAME))
      LDFLAGS += -lblas$(BLAS_BITS)
    else
      LDFLAGS += -lblas$(BLAS_BITS).dll
    endif
    ifneq (0,$(STATIC))
      ifneq (,$(LIBGFORTRAN))
        CLDFLAGS += $(XLIB_BEGIN) $(LIBGFORTRAN) -lm $(XLIB_END)
      endif
    endif
    QUADMATH = $(shell $(LD) $(SLDFLAGS) -lquadmath 2>&1 | grep -q "\-lquadmath"; echo $$?)
    ifneq (0,$(QUADMATH))
      CLDFLAGS += $(XLIB_BEGIN) -lquadmath $(XLIB_END)
    endif
  endif
endif

# cleanup and extra flags
CXXFLAGS := $(strip $(CXXFLAGS) $(ECXXFLAGS) $(EFLAGS))
FCFLAGS := $(strip $(FCFLAGS) $(EFCFLAGS) $(EFLAGS))
CFLAGS := $(strip $(CFLAGS) $(ECFLAGS) $(EFLAGS))
LDFLAGS := $(strip $(LDFLAGS) $(ELDFLAGS))

.PRECIOUS: $(BLDDIR)/%-cpp.o $(BLDDIR)/%-c.o $(BLDDIR)/%-f.o \
           $(BLDDIR)/%-f90.o $(BLDDIR)/%-f90.o $(BLDDIR)/%-f77.o \
           %/.make

.SUFFIXES:

STATE := $(foreach V,$(sort $(.VARIABLES)),$(if \
  $(filter-out environment% default automatic,$(origin $V)), \
  $(if $(filter-out .DEFAULT_GOAL BLAS_WARNING MAKEFILE_LIST MAKEOVERRIDES MAKEFLAGS PREFIX libdir,$V), \
    $V=$($V)?)))
.state: $(shell echo '$(STATE)' | $(DEPDIR)/.state.sh $(abspath .))

%/.make:
	@mkdir -p $(basename $@)
	@touch $@

.make:
	@touch $@

