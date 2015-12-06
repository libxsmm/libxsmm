ICPC    = $(notdir $(shell which icpc     2> /dev/null))
ICC     = $(notdir $(shell which icc      2> /dev/null))
IFORT   = $(notdir $(shell which ifort    2> /dev/null))
GPP     = $(notdir $(shell which g++      2> /dev/null))
GCC     = $(notdir $(shell which gcc      2> /dev/null))
GFC     = $(notdir $(shell which gfortran 2> /dev/null))

CXX_CHECK = $(notdir $(shell which $(CXX) 2> /dev/null))
CC_CHECK  = $(notdir $(shell which $(CC)  2> /dev/null))
FC_CHECK  = $(notdir $(shell which $(FC)  2> /dev/null))

# prefer Intel Compiler (if available)
CXX = $(ICPC)
FC = $(IFORT)
CC = $(ICC)

INTEL = $(shell echo $$((3==$(words $(filter icc icpc ifort,$(CC) $(CXX) $(FC))))))

ifneq (0,$(INTEL))
	AR = xiar
	CXXFLAGS += -fPIC -Wall -diag-disable 3415
	CFLAGS += -fPIC -Wall -diag-disable 3415
	FCFLAGS += -fPIC
	LDFLAGS += -fPIC -lrt
	FCMTFLAGS += -threads
	ifeq (1,$(PEDANTIC))
		CFLAGS += -std=c89 -Wcheck
		FCFLAGS += -free
		FMFLAGS += -e03
	else ifneq (0,$(PEDANTIC))
		CFLAGS += -std=c89 -Wcheck -Wremarks -diag-disable 177,2547
		FCFLAGS += -e03
	else
		FCFLAGS += -free
	endif
	# flag specifying output directory must be last
	FMFLAGS += -module
	ifeq (0,$(DBG))
		CXXFLAGS += -fno-alias -ansi-alias -O2
		CFLAGS += -fno-alias -ansi-alias -O2
		FCFLAGS += -O2
		DFLAGS += -DNDEBUG
		ifneq (0,$(IPO))
			CXXFLAGS += -ipo
			CFLAGS += -ipo
			FCFLAGS += -ipo
		endif
	else
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0
		SYM = $(DBG)
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
		else ifeq (1,$(shell echo $$((2 <= $(SSE)))))
			TARGET = -xSSE$(SSE)
		else ifeq (1,$(SSE))
			TARGET = -xSSE3
		else
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
		DFLAGS += -D__STATIC
	else ifneq (0,$(STATIC))
		SLDFLAGS += -static
		DFLAGS += -D__STATIC
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
	GCC_VERSION_STRING = $(shell $(CC) --version | grep "gcc (.\+) " | sed "s/gcc (.\+) \([0-9]\+\.[0-9]\+\.[0-9]\+\).*$$/\1/")
	GCC_VERSION_MAJOR = $(shell echo "$(GCC_VERSION_STRING)" | $(CUT) -d"." -f1)
	GCC_VERSION_MINOR = $(shell echo "$(GCC_VERSION_STRING)" | $(CUT) -d"." -f2)
	GCC_VERSION_PATCH = $(shell echo "$(GCC_VERSION_STRING)" | $(CUT) -d"." -f3)
	ifeq (3,$(words $(GCC_VERSION_MAJOR) $(GCC_VERSION_MINOR) $(GCC_VERSION_PATCH)))
		GCC_VERSION = $(shell echo "$$(($(GCC_VERSION_MAJOR) * 10000 + $(GCC_VERSION_MINOR) * 100 + $(GCC_VERSION_PATCH)))")
	else
		GCC_VERSION = 0
	endif
	MIC = 0
	CXXFLAGS += -Wall -Wno-unused-function -Wno-attributes
	CFLAGS += -Wall -Wno-unused-function -Wno-attributes
	LDFLAGS += -lrt
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
		ifneq (0,$(shell echo "$$((50000 <= $(GCC_VERSION)))"))
			FMFLAGS += -Wuse-without-only
		endif
	else ifneq (0,$(PEDANTIC))
		CFLAGS += -std=c89 -pedantic -Wno-variadic-macros -Wno-long-long -Wno-overlength-strings
		FCFLAGS += -std=f2003 -pedantic -Wunused-variable
		ifneq (0,$(shell echo "$$((50000 <= $(GCC_VERSION)))"))
			FMFLAGS += -Wuse-without-only
		endif
	else
		FCFLAGS += -ffree-form
	endif
	# flag specifying output directory must be last
	FMFLAGS += -J
	ifeq (0,$(DBG))
		CXXFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		CFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		FCFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		DFLAGS += -DNDEBUG
		ifneq (0,$(IPO))
			CXXFLAGS += -flto -ffat-lto-objects
			CFLAGS += -flto -ffat-lto-objects
			FCFLAGS += -flto -ffat-lto-objects
			LDFLAGS += -flto
		endif
	else
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0
		SYM = $(DBG)
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
		else ifeq (1,$(shell echo $$((2 <= $(SSE)))))
			TARGET = -msse$(SSE)
		else ifeq (1,$(SSE))
			TARGET = -msse3
		else
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
			CXXFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CXXFLAGS)
			CFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CFLAGS)
			FCFLAGS := -g -check -traceback $(FCFLAGS)
		endif
	endif
	ifeq (0,$(EXP))
		CXXFLAGS += -fno-exceptions
	endif
	ifneq (0,$(OMP))
		CXXFLAGS += -fopenmp
		CFLAGS += -fopenmp
		FCFLAGS += -fopenmp
		LDFLAGS += -fopenmp
	endif
	ifneq (0,$(STATIC))
		SLDFLAGS += -static
		DFLAGS += -D__STATIC
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
	ifneq (0,$(MKL_DIRECT))
		MKL_STATIC ?= 1
	else ifneq (0,$(STATIC))
		MKL_STATIC ?= 1
	else
		MKL_STATIC ?= 0
	endif
else
	MKL = 0
endif

ifeq (1,$(MKL_DIRECT))
	MKL_STATIC = 1
	DFLAGS += -DMKL_DIRECT_CALL_SEQ
endif

ifeq (1,$(MKL)) # sequential
	DFLAGS += -D__MKL
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
	LDFLAGS += -lpthread -lm
	ifneq (0,$(OFFLOAD))
		LDFLAGS += -offload-option,mic,ld,"-lm"
	endif
else ifneq (0,$(MKL)) # multi-threaded
	DFLAGS += -D__MKL
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
			LDFLAGS += -offload-option,mic,ld,"-lpthread -lm"
		endif
	endif
	LDFLAGS += -lpthread -lm
else ifneq (0,$(BLAS)) # generic
	ifneq (1,$(BLAS))
		BLAS_THREADS = o
	endif
	OPENBLAS = $(shell $(CC) -lopenblas$(BLAS_THREADS)$(BLAS_BITS) 2>&1 | grep openblas$(BLAS_THREADS)$(BLAS_BITS) > /dev/null ; echo $$?)
	ifeq (1,$(OPENBLAS))
		LDFLAGS += -lopenblas$(BLAS_THREADS)$(BLAS_BITS)
	else
		ifneq (Windows_NT,$(OS))
			LDFLAGS += -lblas$(BLAS_BITS)
		else
			LDFLAGS += -lblas$(BLAS_BITS).dll
		endif
		CLDFLAGS += -lgfortran
		ifeq (,$(GCC_VERSION_STRING))
			LDFLAGS += -lquadmath
		else ifneq (0,$(shell echo "$$((40500 <= $(GCC_VERSION)))"))
			CLDFLAGS += -lquadmath
		endif
	endif
endif

