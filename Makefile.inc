# Linux cut has features we use that do not work elsewhere. Mac, etc. users
# should install GNU coreutils and use "cut" from there.
# For example, if you use Homebrew, run "brew install coreutils" once and invoke:
# $ make CUT=/usr/local/Cellar/coreutils/8.24/libexec/gnubin/cut
CUT ?= cut

ifeq (Windows_NT,$(OS))
	ENV = $(NULL)
else
	ENV = env
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

# Debugging and symbols (e.g., when profiling)
SYM ?= 0
DBG ?= 0

# Threading runtime
OMP ?= 0
# PTHREAD refers to foundational TRT (and not necessarily Posix Threads)
ifneq (0,$(OMP))
	PTHREAD ?= 1
else
	PTHREAD ?= 0
endif

# Code conformance (beyond -Wall)
PEDANTIC ?= 0

# Embed InterProcedural Optimization information into libraries
IPO ?= 0

# ILP64=0 (LP64 with 32-bit integers), and ILP64=0 (64-bit integers)
ILP64 ?= 0
BLAS ?= 0

OFFLOAD ?= 0
ifneq (0,$(OFFLOAD))
	MPSS ?= 1
	MIC ?= 1
else
	MPSS ?= 0
	MIC ?= 0
endif

# Automatically pickup the Intel Compiler (if available)
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
INTEL ?= $(shell echo $$((2==$(words $(filter icpc icc, \
  $(shell $(CXX) --version 2> /dev/null | head -n1 | $(CUT) -d" " -f1) \
  $(shell $(CC) --version 2> /dev/null | head -n1 | $(CUT) -d" " -f1))))))

ifneq (0,$(INTEL))
	ifneq (,$(strip $(FC)))
		ifneq (ifort,$(shell $(FC) --version 2> /dev/null | head -n1 | $(CUT) -d" " -f1))
$(info ======================================)
$(info The Intel Fortran Compiler is missing.)
$(info Building without Fortran support!)
$(info ======================================)
		else ifeq (0,$(OFFLOAD))
			FCFLAGS += -no-offload
		endif
	endif
	ifeq (0,$(OFFLOAD))
		CXXFLAGS += -no-offload
		CFLAGS += -no-offload
	endif
else
	ifeq (,$(notdir $(shell which $(CXX) 2> /dev/null)))
		CXX = g++
	endif
	ifeq (,$(notdir $(shell which $(CC) 2> /dev/null)))
		CC = gcc
	endif
	ifeq (,$(notdir $(shell which $(FC) 2> /dev/null)))
		FC = $(shell which gfortran 2> /dev/null)
	endif
endif

# linker setup
LD = $(CC)

CC_VERSION_STRING = $(shell $(CC) --version 2> /dev/null | head -n1 | sed "s/.\+ \([0-9]\+\.[0-9]\+\.[0-9]\+\)\b.*/\1/")
CC_VERSION_MAJOR = $(shell echo "$(CC_VERSION_STRING)" | $(CUT) -d"." -f1)
CC_VERSION_MINOR = $(shell echo "$(CC_VERSION_STRING)" | $(CUT) -d"." -f2)
CC_VERSION_PATCH = $(shell echo "$(CC_VERSION_STRING)" | $(CUT) -d"." -f3)
ifeq (3,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR) $(CC_VERSION_PATCH)))
	CC_VERSION = $(shell echo "$$(($(CC_VERSION_MAJOR) * 10000 + $(CC_VERSION_MINOR) * 100 + $(CC_VERSION_PATCH)))")
else
	CC_VERSION_STRING = $(NULL)
	CC_VERSION = 0
endif

ifneq (,$(notdir $(shell which $(FC) 2> /dev/null)))
	FC_VERSION_STRING := $(shell $(FC) --version 2> /dev/null | head -n1 | sed "s/.\+ \([0-9]\+\.[0-9]\+\.[0-9]\+\)\b.*/\1/")
	FC_VERSION_MAJOR = $(shell echo "$(FC_VERSION_STRING)" | $(CUT) -d"." -f1)
	FC_VERSION_MINOR = $(shell echo "$(FC_VERSION_STRING)" | $(CUT) -d"." -f2)
	FC_VERSION_PATCH = $(shell echo "$(FC_VERSION_STRING)" | $(CUT) -d"." -f3)
endif
ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
	FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR) * 10000 + $(FC_VERSION_MINOR) * 100 + $(FC_VERSION_PATCH)))")
else
	FC_VERSION_STRING = $(NULL)
	FC_VERSION = 0
endif

ifneq (,$(strip $(FC)))
	ifeq (GNU,$(shell $(FC) --version 2> /dev/null | head -n1 | $(CUT) -d" " -f1))
		ifneq (0,$(shell echo "$$((40500 > $(FC_VERSION)))"))
			ifneq (gfortran,$(notdir $(FC)))
				FC = gfortran
				ifneq (,$(notdir $(shell which $(FC) 2> /dev/null)))
					FC_VERSION_STRING := $(shell $(FC) --version 2> /dev/null | head -n1 | sed "s/.\+ \([0-9]\+\.[0-9]\+\.[0-9]\+\)\b.*/\1/")
					FC_VERSION_MAJOR = $(shell echo "$(FC_VERSION_STRING)" | $(CUT) -d"." -f1)
					FC_VERSION_MINOR = $(shell echo "$(FC_VERSION_STRING)" | $(CUT) -d"." -f2)
					FC_VERSION_PATCH = $(shell echo "$(FC_VERSION_STRING)" | $(CUT) -d"." -f3)
				endif
				ifeq (3,$(words $(FC_VERSION_MAJOR) $(FC_VERSION_MINOR) $(FC_VERSION_PATCH)))
					FC_VERSION = $(shell echo "$$(($(FC_VERSION_MAJOR) * 10000 + $(FC_VERSION_MINOR) * 100 + $(FC_VERSION_PATCH)))")
					ifneq (0,$(shell echo "$$((40500 > $(FC_VERSION)))"))
						FC = $(NULL)
					endif
				else
					FC_VERSION_STRING = $(NULL)
					FC_VERSION = 0
					FC = $(NULL)
				endif
			else
				FC = $(NULL)
			endif
			ifeq (,$(strip $(FC)))
$(info ====================================)
$(info Outdated GNU Fortran $(FC_VERSION_STRING) detected.)
$(info Building without Fortran support!)
$(info ====================================)
			endif
		endif
	endif
endif

# compiler names
CXX_NAME = $(shell $(CXX) --version 2> /dev/null | head -n1 | $(CUT) -d" " -f1)
CC_NAME = $(shell $(CC) --version 2> /dev/null | head -n1 | $(CUT) -d" " -f1)
ifneq (,$(strip $(FC)))
	FC_NAME = $(notdir $(FC))
endif

# native GCC
GCC ?= $(shell echo $$((3==$(words $(filter g++ gcc gfortran,$(CXX_NAME) $(CC_NAME) $(FC_NAME))))))
ifneq (,$(strip $(FC)))
	ifeq (GCC,$(shell $(CXX) --version 2> /dev/null | head -n1 | sed "s/.* (\(.\+\)) .*/\1/"))
		ifeq (GCC,$(shell $(CC) --version 2> /dev/null | head -n1 | sed "s/.* (\(.\+\)) .*/\1/"))
			ifeq (GCC,$(shell $(FC) --version 2> /dev/null | head -n1 | sed "s/.* (\(.\+\)) .*/\1/"))
				GCC = 1
			endif
		endif
	endif
endif

# Select code path (if not selected otherwise)
CPUFLAGS = $(shell grep -m1 flags /proc/cpuinfo | $(CUT) -d: -f2-)
ifneq (0,$(AVX))
	ifeq (1,$(words $(filter avx512f,$(CPUFLAGS))))
		AVX ?= 3
	else ifeq (2,$(words $(filter avx2 fma,$(CPUFLAGS))))
		AVX ?= 2
	else ifeq (1,$(words $(filter avx,$(CPUFLAGS))))
		AVX ?= 1
	endif
else ifneq (0,$(SSE))
	ifeq (1,$(words $(filter sse3,$(CPUFLAGS))))
		SSE ?= 1
	endif
else ifneq (0,$(MIC))
	MPSS = 1
endif

ifneq (0,$(INTEL))
	SUITE = Intel Compiler suite
else ifneq (0,$(GCC))
	SUITE = GNU Compiler Collection
else
	SUITE = non-GNU compilers
endif

ifneq (,$(strip $(FC)))
	INFO = Using $(SUITE): $(CXX_NAME) $(CC_VERSION_STRING), $(CC_NAME) $(CC_VERSION_STRING), and $(FC_NAME) $(FC_VERSION_STRING)
else
	INFO = Using $(SUITE): $(CXX_NAME) $(CC_VERSION_STRING), and $(CC_NAME) $(CC_VERSION_STRING)
endif

ifneq (0,$(INTEL))
	AR ?= xiar
	CXXLDFLAGS += -Wl,--as-needed -lc -Wl,--no-as-needed
	FCLDFLAGS += -Wl,--as-needed -lc -lstdc++ -Wl,--no-as-needed
	CLDFLAGS += -Wl,--as-needed -lstdc++ -Wl,--no-as-needed
	LDFLAGS += -Wl,--as-needed -lrt -ldl -Wl,--no-as-needed
	CXXFLAGS += -fPIC -Wall -diag-disable 3415,10010
	CFLAGS += -fPIC -Wall -diag-disable 3415,10010
	FCFLAGS += -fPIC -diag-disable 10010
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
		CXXFLAGS += $(OPTFLAG) -fno-alias -ansi-alias
		CFLAGS += $(OPTFLAG) -fno-alias -ansi-alias
		FCFLAGS += $(OPTFLAG)
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
	else
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0 -check
		SYM = $(DBG)
	endif
	ifneq (0,$(TRACE))
		CXXFLAGS += -finstrument-functions
		CFLAGS += -finstrument-functions
		FCFLAGS += -finstrument-functions
		LDFLAGS += -rdynamic
		DFLAGS += -D__TRACE=$(TRACE)
	endif
	ifeq (1,$(shell echo $$((3 > $(DBG)))))
		ifeq (0,$(COMPATIBLE))
			ifeq (1,$(AVX))
				TARGET = -xAVX
			else ifeq (2,$(AVX))
				TARGET = -xCORE-AVX2
			else ifeq (3,$(AVX))
				ifeq (0,$(MIC))
					TARGET = -xCOMMON-AVX512
				else
					TARGET = -xMIC-AVX512
				endif
			else ifneq (0,$(SSE))
				TARGET = -xSSE3
			else ifneq (0,$(AVX))
				TARGET = -xHost
			endif
		else
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
				TARGET = -msse3
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
	ifneq (0,$(OMP))
		CXXFLAGS += -openmp
		CFLAGS += -openmp
		FCFLAGS += -openmp
		LDFLAGS += -openmp
	endif
	ifneq (0,$(PTHREAD))
		DFLAGS += -D_REENTRANT
		CXXFLAGS += -pthread
		CFLAGS += -pthread
		ifneq (Windows_NT,$(OS))
			CLDFLAGS += -pthread
		endif
	endif
	ifeq (1,$(STATIC))
		SLDFLAGS += -no-intel-extensions -static-intel -static-libgcc -static-libstdc++
		DFLAGS += -D__STATIC=1
	else ifneq (0,$(STATIC))
		SLDFLAGS += -static
		DFLAGS += -D__STATIC=$(STATIC)
	endif
	ifneq (,$(strip $(R8)))
		ifneq (0,$(R8))
			FCFLAGS += -autodouble
		endif
	endif
else # GCC assumed
	CXXLDFLAGS += -Wl,--as-needed -lc -lgfortran -Wl,--no-as-needed
	FCLDFLAGS += -Wl,--as-needed -lc -lstdc++ -Wl,--no-as-needed
	CLDFLAGS += -Wl,--as-needed -lstdc++ -lgfortran -Wl,--no-as-needed
	LDFLAGS += -Wl,--as-needed -lrt -ldl -Wl,--no-as-needed
	CXXFLAGS += -Wall -Wno-unused-function
	CFLAGS += -Wall -Wno-unused-function
	ifeq (0,$(COMPATIBLE))
		CXXFLAGS += -Wno-attributes
		CFLAGS += -Wno-attributes
	endif
	ifneq (Windows_NT,$(OS))
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
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0
		SYM = $(DBG)
	endif
	ifneq (0,$(TRACE))
		CXXFLAGS += -finstrument-functions
		CFLAGS += -finstrument-functions
		FCFLAGS += -finstrument-functions
		ifneq (0,$(GCC)) # e.g. clang does not need/understand below flag
			CXXFLAGS += -finstrument-functions-exclude-function-list=_mm_
			CFLAGS += -finstrument-functions-exclude-function-list=_mm_
			FCFLAGS += -finstrument-functions-exclude-function-list=_mm_
		endif
		ifneq (Windows_NT,$(OS))
			LDFLAGS += -rdynamic
		else
			LDFLAGS += -ldbghelp
		endif
		DFLAGS += -D__TRACE=$(TRACE)
	endif
	ifeq (0,$(GCC)) # fixup older Clang
		DFLAGS += -D__extern_always_inline=inline
	endif
	ifeq (1,$(shell echo $$((3 > $(DBG)))))
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
			TARGET = -msse3
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
	ifneq (0,$(OMP))
		ifeq (0,$(shell echo $(OMP) | grep -q "^-*[0-9]\+$$"; echo $$?))
			CXXFLAGS += -fopenmp
			CFLAGS += -fopenmp
			FCFLAGS += -fopenmp
			LDFLAGS += -fopenmp
		else ifneq (,$(strip $(OMP))) # clang: OMP=libomp
			CXXFLAGS += -fopenmp=$(OMP)
			CFLAGS += -fopenmp=$(OMP)
			# omit passing special OMP into Fortran compiler
			FCFLAGS += -fopenmp
			CLDFLAGS += -fopenmp=$(OMP)
		endif
	endif
	ifneq (0,$(PTHREAD))
		DFLAGS += -D_REENTRANT
		CXXFLAGS += -pthread
		CFLAGS += -pthread
		ifneq (Windows_NT,$(OS))
			CLDFLAGS += -pthread
		endif
	endif
	ifneq (0,$(STATIC))
		SLDFLAGS += -static
		DFLAGS += -D__STATIC=$(STATIC)
	endif
	ifneq (,$(strip $(R8)))
		ifneq (0,$(R8))
			FCFLAGS += -fdefault-real-8 -fdefault-double-8
		endif
	endif
	OFFLOAD = 0
	MIC = 0
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
	MKL_STATIC ?= 1
else
	MKL = 0
endif

ifeq (1,$(MKL_DIRECT))
	DFLAGS += -DMKL_DIRECT_CALL_SEQ
endif

ifeq (1,$(MKL)) # sequential
	DFLAGS += -D__MKL=1
	IFLAGS +=-I$(MKLROOT)/include
	ifeq (0,$(MKL_STATIC)) # shared
		LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_sequential
		ifneq (0,$(OFFLOAD))
			LDFLAGS += -offload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_sequential"
		endif
	else # static
		LDFLAGS += -Wl,--start-group \
			$(MKLROOT)/lib/intel64/libmkl_intel_$(MKL_BITS).a \
			$(MKLROOT)/lib/intel64/libmkl_core.a \
			$(MKLROOT)/lib/intel64/libmkl_sequential.a \
		-Wl,--end-group
		ifneq (0,$(OFFLOAD))
			LDFLAGS += -offload-option,mic,ld,"--start-group \
				$(MKLROOT)/lib/mic/libmkl_intel_$(MKL_BITS).a \
				$(MKLROOT)/lib/mic/libmkl_core.a \
				$(MKLROOT)/lib/mic/libmkl_sequential.a \
			--end-group"
		endif
	endif
	LDFLAGS += -lpthread -lm -ldl
	ifneq (0,$(OFFLOAD))
		LDFLAGS += -offload-option,mic,ld,"-lm -ldl"
		ifneq (,$(strip $(FC)))
			ifneq (0,$(PTHREAD))
				LDFLAGS += -offload-option,mic,ld,"--as-needed -lifcoremt --no-as-needed"
			else
				LDFLAGS += -offload-option,mic,ld,"--as-needed -lifcore --no-as-needed"
			endif
		endif
	endif
else ifneq (0,$(MKL)) # multi-threaded
	DFLAGS += -D__MKL=$(MKL)
	IFLAGS +=-I$(MKLROOT)/include
	ifeq (0,$(MKL_STATIC)) # shared
		ifneq (0,$(INTEL))
			LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_intel_thread
			ifneq (0,$(OFFLOAD))
				LDFLAGS += -offload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_intel_thread"
			endif
		else # assuming GNU toolchain
			LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_$(MKL_BITS) -lmkl_core -lmkl_gnu_thread
		endif
	else # static
		ifneq (0,$(INTEL))
			LDFLAGS += -Wl,--start-group \
				$(MKLROOT)/lib/intel64/libmkl_intel_$(MKL_BITS).a \
				$(MKLROOT)/lib/intel64/libmkl_core.a \
				$(MKLROOT)/lib/intel64/libmkl_intel_thread.a \
			-Wl,--end-group
			ifneq (0,$(OFFLOAD))
				LDFLAGS += -offload-option,mic,ld,"--start-group \
					$(MKLROOT)/lib/mic/libmkl_intel_$(MKL_BITS).a \
					$(MKLROOT)/lib/mic/libmkl_core.a \
					$(MKLROOT)/lib/mic/libmkl_intel_thread.a \
				--end-group"
			endif
		else # assuming GNU toolchain
			LDFLAGS += -Wl,--start-group \
				$(MKLROOT)/lib/intel64/libmkl_intel_$(MKL_BITS).a \
				$(MKLROOT)/lib/intel64/libmkl_core.a \
				$(MKLROOT)/lib/intel64/libmkl_gnu_thread.a \
			-Wl,--end-group -ldl
		endif
	endif
	ifneq (0,$(INTEL))
		ifeq (0,$(OMP))
			LDFLAGS += -liomp5
			ifneq (0,$(OFFLOAD))
				LDFLAGS += -offload-option,mic,ld,"-liomp5"
			endif
		endif
		ifneq (0,$(OFFLOAD))
			LDFLAGS += -offload-option,mic,ld,"-lm -ldl"
			ifneq (,$(strip $(FC)))
				LDFLAGS += -offload-option,mic,ld,"--as-needed -lifcoremt --no-as-needed"
			endif
		endif
	endif
	LDFLAGS += -lpthread -lm -ldl
else ifneq (0,$(BLAS)) # generic
	ifneq (1,$(BLAS))
		BLAS_THREADS = o
	endif
	OPENBLAS = $(shell $(LD) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) -lopenblas$(BLAS_THREADS)$(BLAS_BITS) 2>&1 | grep -q openblas$(BLAS_THREADS)$(BLAS_BITS); echo $$?)
	ifeq (1,$(OPENBLAS))
		LDFLAGS += -lopenblas$(BLAS_THREADS)$(BLAS_BITS)
	else
		ifneq (Windows_NT,$(OS))
			LDFLAGS += -lblas$(BLAS_BITS)
		else
			LDFLAGS += -lblas$(BLAS_BITS).dll
		endif
		CLDFLAGS += -Wl,--as-needed -lgfortran -Wl,--no-as-needed
		ifneq (0,$(shell echo "$$((40500 <= $(FC_VERSION)))"))
			CLDFLAGS += -Wl,--as-needed -lquadmath -Wl,--no-as-needed
		endif
	endif
endif

# cleanup and extra flags
CXXFLAGS := $(strip $(CXXFLAGS) $(ECXXFLAGS) $(EFLAGS))
FCFLAGS := $(strip $(FCFLAGS) $(EFCFLAGS) $(EFLAGS))
CFLAGS := $(strip $(CFLAGS) $(ECFLAGS) $(EFLAGS))
LDFLAGS := $(strip $(LDFLAGS) $(ELDFLAGS))

.SUFFIXES:

%/.make:
	@mkdir -p $(basename $@)
	@touch $@

.make:
	@touch $@

