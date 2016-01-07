ICPC    = $(notdir $(shell which icpc     2> /dev/null))
ICC     = $(notdir $(shell which icc      2> /dev/null))
IFORT   = $(notdir $(shell which ifort    2> /dev/null))
GPP     = $(notdir $(shell which g++      2> /dev/null))
GCC     = $(notdir $(shell which gcc      2> /dev/null))
GFC     = $(notdir $(shell which gfortran 2> /dev/null))

CXX_CHECK = $(notdir $(shell which $(CXX) 2> /dev/null))
CC_CHECK  = $(notdir $(shell which $(CC)  2> /dev/null))
ifneq (,$(FC)) # allow FC to be optional
	FC_CHECK = $(notdir $(shell which $(FC) 2> /dev/null))
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

# prefer Intel Compiler (if available)
ifneq (,$(ICPC))
	CXX = $(ICPC)
endif
ifneq (,$(IFORT))
	FC = $(IFORT)
endif
ifneq (,$(ICC))
	CC = $(ICC)
endif

CXX_NAME = $(shell $(CXX) --version 2> /dev/null | head -n1 | $(CUT) -d" " -f1)
CC_NAME = $(shell $(CC) --version 2> /dev/null | head -n1 | $(CUT) -d" " -f1)
ifneq (,$(FC)) # allow FC to be optional
	FC_NAME = $(shell $(FC) --version 2> /dev/null | head -n1 | $(CUT) -d" " -f1)
endif
ifeq (g++,$(CXX_NAME))
	GNU = 1
else
	GNU = 0
endif

CC_VERSION_STRING = $(shell $(CC) --version 2> /dev/null | head -n1 | sed "s/.\+ \([0-9]\+\.[0-9]\+\.[0-9]\+\) .*/\1/")
CC_VERSION_MAJOR = $(shell echo "$(CC_VERSION_STRING)" | $(CUT) -d"." -f1)
CC_VERSION_MINOR = $(shell echo "$(CC_VERSION_STRING)" | $(CUT) -d"." -f2)
CC_VERSION_PATCH = $(shell echo "$(CC_VERSION_STRING)" | $(CUT) -d"." -f3)

ifeq (3,$(words $(CC_VERSION_MAJOR) $(CC_VERSION_MINOR) $(CC_VERSION_PATCH)))
  CC_VERSION = $(shell echo "$$(($(CC_VERSION_MAJOR) * 10000 + $(CC_VERSION_MINOR) * 100 + $(CC_VERSION_PATCH)))")
else
  CC_VERSION = 0
endif

CPUFLAGS = $(shell grep -m1 flags /proc/cpuinfo | $(CUT) -d: -f2-)

# Select code path (if not selected otherwise)
ifneq (0,$(AVX))
	ifeq (1,$(words $(filter avx512f,$(CPUFLAGS))))
		AVX ?= 3
	else ifeq (2,$(words $(filter avx2 fma,$(CPUFLAGS))))
		AVX ?= 2
	else ifeq (1,$(words $(filter avx,$(CPUFLAGS))))
		AVX ?= 1
	endif
else ifneq (0,$(SSE))
	ifeq (1,$(words $(filter sse4_2,$(CPUFLAGS))))
		SSE ?= 1
	endif
endif

# check if the Intel Development Tools are available
NUM_INTEL_COMPILER = $(words $(filter icpc icc ifort,$(CXX_NAME) $(CC_NAME) $(FC_NAME)))
ifeq (1,$(shell echo $$((0<$(NUM_INTEL_COMPILER)&&3>$(NUM_INTEL_COMPILER)))))
$(info =============================================)
$(info Incomplete Intel Compiler toolchain detected!)
$(info =============================================)
	ifeq (0,$(OFFLOAD))
		ifeq (icpc,$(CXX_NAME))
			CXXFLAGS += -no-offload
		endif
		ifeq (icc,$(CC_NAME))
			CFLAGS += -no-offload
		endif
		ifeq (ifort,$(FC_NAME))
			FCFLAGS += -no-offload
		endif
	endif
endif

INTEL ?= $(shell echo $$((3==$(NUM_INTEL_COMPILER))))

ifneq (0,$(INTEL))
	AR ?= xiar
	CXXFLAGS += -fPIC -Wall -diag-disable 3415
	CFLAGS += -fPIC -Wall -diag-disable 3415
	FCFLAGS += -fPIC
	LDFLAGS += -fPIC -lrt
	FCMTFLAGS += -threads
	ifeq (1,$(PEDANTIC))
		CFLAGS += -std=c89 -Wcheck
		FCFLAGS += -free
		FMFLAGS += -std03 -warn all,notruncated_source -diag-disable 7025,7373,10237,10342,10382
	else ifneq (0,$(PEDANTIC))
		CFLAGS += -std=c89 -Wcheck -Wremarks -diag-disable 177,1419,2547,10382
		FCFLAGS += -std03 -warn all,notruncated_source -diag-disable 7025,7373,10237,10342,10382
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
		DFLAGS += -DNDEBUG
		ifneq (0,$(IPO))
			CXXFLAGS += -ipo
			CFLAGS += -ipo
			FCFLAGS += -ipo
		endif
		SYM = $(TRACE)
	else
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0
		SYM = $(DBG)
	endif
	ifneq (0,$(TRACE))
		CXXFLAGS += -finstrument-functions -pthread
		CFLAGS += -finstrument-functions -pthread
		FCFLAGS += -finstrument-functions
		CLDFLAGS += -pthread
		LDFLAGS += -rdynamic
		DFLAGS += -D__TRACE=$(TRACE)
	endif
	ifeq (1,$(shell echo $$((3 > $(DBG)))))
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
			TARGET = -xSSE4.2
		else ifneq (0,$(AVX))
			TARGET = -xHost
		endif
	endif
	ifneq (0,$(SYM))
		ifeq (1,$(SYM))
			CXXFLAGS := -g $(CXXFLAGS)
			CFLAGS := -g $(CFLAGS)
			FCFLAGS := -g -check -traceback $(FCFLAGS)
		else
			CXXFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CXXFLAGS)
			CFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CFLAGS)
			FCFLAGS := -g -check -traceback $(FCFLAGS)
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
	ifeq (0,$(OFFLOAD))
		CXXFLAGS += -no-offload
		CFLAGS += -no-offload
		FCFLAGS += -no-offload
	endif
	ifeq (1,$(STATIC))
		SLDFLAGS += -no-intel-extensions -static-intel -static-libgcc -static-libstdc++
		DFLAGS += -D__STATIC=1
	else ifneq (0,$(STATIC))
		SLDFLAGS += -static
		DFLAGS += -D__STATIC=$(STATIC)
	endif
	ifneq (,$(R8))
		ifneq (0,$(R8))
			FCFLAGS += -autodouble
		endif
	endif
else # GCC assumed
	ifeq (,$(CXX_CHECK))
		CXX = $(GPP)
	endif
	ifeq (,$(CC_CHECK))
		CC = $(GCC)
	endif
	ifeq (,$(FC_CHECK))
		FC = $(GFC)
	endif
	OFFLOAD = 0
	MIC = 0
	CXXFLAGS += -Wall -Wno-unused-function
	CFLAGS += -Wall -Wno-unused-function
	LDFLAGS += -lrt
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
		ifneq (0,$(shell echo "$$((50000 <= $(CC_VERSION)))"))
			FMFLAGS += -Wuse-without-only
		endif
	else ifneq (0,$(PEDANTIC))
		CFLAGS += -std=c89 -pedantic -Wno-variadic-macros -Wno-long-long -Wno-overlength-strings
		FCFLAGS += -std=f2003 -pedantic -Wunused-variable
		ifneq (0,$(shell echo "$$((50000 <= $(CC_VERSION)))"))
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
		DFLAGS += -DNDEBUG
		ifneq (0,$(IPO))
			CXXFLAGS += -flto -ffat-lto-objects
			CFLAGS += -flto -ffat-lto-objects
			FCFLAGS += -flto -ffat-lto-objects
			LDFLAGS += -flto
		endif
		SYM = $(TRACE)
	else
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0
		SYM = $(DBG)
	endif
	ifneq (0,$(TRACE))
		CXXFLAGS += -finstrument-functions -pthread
		CFLAGS += -finstrument-functions -pthread
		FCFLAGS += -finstrument-functions
		ifneq (0,$(GNU)) # e.g. clang does not need/understand below flag
			CXXFLAGS += -finstrument-functions-exclude-function-list=_mm_
			CFLAGS += -finstrument-functions-exclude-function-list=_mm_
			FCFLAGS += -finstrument-functions-exclude-function-list=_mm_
		endif
		ifneq (Windows_NT,$(OS))
			CLDFLAGS += -pthread
			LDFLAGS += -rdynamic
		else
			LDFLAGS += -ldbghelp
		endif
		DFLAGS += -D__TRACE=$(TRACE)
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
			TARGET = -msse4.2
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
		else ifneq (,$(OMP)) # clang: OMP=libomp
			CXXFLAGS += -fopenmp=$(OMP)
			CFLAGS += -fopenmp=$(OMP)
			# omit passing special OMP into Fortran compiler
			FCFLAGS += -fopenmp
			CLDFLAGS += -fopenmp=$(OMP)
		endif
	endif
	ifneq (0,$(STATIC))
		SLDFLAGS += -static
		DFLAGS += -D__STATIC=$(STATIC)
	endif
	ifneq (,$(R8))
		ifneq (0,$(R8))
			FCFLAGS += -fdefault-real-8 -fdefault-double-8
		endif
	endif
endif

ifneq (,$(CXX))
	LD = $(CXX)
endif
ifeq (,$(LD))
	LD = $(CC)
endif
ifeq (,$(LD))
	LD = $(FC)
endif

ifeq (,$(CXXFLAGS))
	CXXFLAGS = $(CFLAGS)
endif
ifeq (,$(CFLAGS))
	CFLAGS = $(CXXFLAGS)
endif
ifeq (,$(FCFLAGS))
	FCFLAGS = $(CFLAGS)
endif
ifeq (,$(LDFLAGS))
	LDFLAGS = $(CFLAGS)
endif

MAKE_ILP64 = 0
ifneq (,$(ILP64))
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

ifneq (,$(MKLROOT))
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
			LDFLAGS += -offload-option,mic,ld,"-lpthread -lm -ldl"
		endif
	endif
	LDFLAGS += -lpthread -lm -ldl
else ifneq (0,$(BLAS)) # generic
	ifneq (1,$(BLAS))
		BLAS_THREADS = o
	endif
	OPENBLAS = $(shell $(CC) -lopenblas$(BLAS_THREADS)$(BLAS_BITS) 2>&1 | grep -q openblas$(BLAS_THREADS)$(BLAS_BITS); echo $$?)
	ifeq (1,$(OPENBLAS))
		LDFLAGS += -lopenblas$(BLAS_THREADS)$(BLAS_BITS)
	else
		ifneq (Windows_NT,$(OS))
			LDFLAGS += -lblas$(BLAS_BITS)
		else
			LDFLAGS += -lblas$(BLAS_BITS).dll
		endif
		CLDFLAGS += -lgfortran
		ifeq (,$(CC_VERSION_STRING))
			LDFLAGS += -lquadmath
		else ifneq (0,$(shell echo "$$((40500 <= $(CC_VERSION)))"))
			CLDFLAGS += -lquadmath
		endif
	endif
endif

# append extra flags
CXXFLAGS  += $(ECXXFLAGS)
CXXFLAGS  += $(EFLAGS)
FCFLAGS   += $(EFCFLAGS)
FCFLAGS   += $(EFLAGS)
CFLAGS    += $(ECFLAGS)
CFLAGS    += $(EFLAGS)
LDFLAGS   += $(ELDFLAGS)

.SUFFIXES:

%/.make:
	@mkdir -p $(basename $@)
	@touch $@

.make:
	@touch $@

