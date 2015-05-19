NAME := dispatched
DBG := 0
MIC := 0

LIBXSMMROOT := ../..
CXXFLAGS :=
CFLAGS :=
DFLAGS :=
IFLAGS := -I$(LIBXSMMROOT)/include

ifeq ($(STATIC),0)
	ifeq ($(MIC),0)
		LIBXSMM := $(LIBXSMMROOT)/lib/intel64/libxsmm.so
		LDFLAGS := -L$(LIBXSMMROOT)/lib/intel64 -lxsmm
	else
		LIBXSMM := $(LIBXSMMROOT)/lib/mic/libxsmm.so
		LDFLAGS := -L$(LIBXSMMROOT)/lib/mic -lxsmm
	endif
else
	ifeq ($(MIC),0)
		LIBXSMM := $(LIBXSMMROOT)/lib/intel64/libxsmm.a
	else
		LIBXSMM := $(LIBXSMMROOT)/lib/mic/libxsmm.a
	endif
	LDFLAGS += $(LIBXSMM)
endif

ICPC := $(notdir $(shell which icpc 2> /dev/null))
ICC := $(notdir $(shell which icc 2> /dev/null))

ifneq (,$(ICPC))
	CXX := $(ICPC)
	ifeq (,$(ICC))
		CC := $(CXX)
	endif
endif
ifneq (,$(ICC))
	CC := $(ICC)
	ifeq (,$(ICPC))
		CXX := $(CC)
	endif
endif

LD := $(CXX)

ifneq (,$(filter $(CXX),icpc))
	CXXFLAGS += -fPIC -Wall
	ifeq (0,$(DBG))
		CXXFLAGS += -fno-alias -ansi-alias -mkl=sequential -DNDEBUG -O3 -ipo -openmp
		DFLAGS += -DNDEBUG
		ifeq ($(AVX),1)
			CXXFLAGS += -xAVX
		else ifeq ($(AVX),2)
			CXXFLAGS += -xCORE-AVX2
		else ifeq ($(AVX),3)
			CXXFLAGS += -xCOMMON-AVX512
		else ifneq ($(MIC),0)
			CXXFLAGS += -mmic
			LDFLAGS += -mmic
		else
			CXXFLAGS += -xHost
		endif
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CXXFLAGS += -O0 -g -openmp
	endif
	ifeq (0,$(OFFLOAD))
		CXXFLAGS += -no-offload
	else ifeq ($(MIC),0)
		CXXFLAGS += -offload-option,mic,compiler,"-O2 -opt-assume-safe-padding"
	endif
	LDFLAGS += -fPIC -mkl=sequential -openmp
	ifneq ($(STATIC),0)
		ifneq ($(STATIC),)
			LDFLAGS += -no-intel-extensions -static-intel
		endif
	endif
else ifneq (,$(filter $(CC),icc))
	CFLAGS += -fPIC -Wall
	ifeq (0,$(DBG))
		CFLAGS += -fno-alias -ansi-alias -mkl=sequential -DNDEBUG -O3 -ipo -openmp
		DFLAGS += -DNDEBUG
		ifeq ($(AVX),1)
			CFLAGS += -xAVX
		else ifeq ($(AVX),2)
			CFLAGS += -xCORE-AVX2
		else ifeq ($(AVX),3)
			CFLAGS += -xCOMMON-AVX512
		else ifneq ($(MIC),0)
			CFLAGS += -mmic
			LDFLAGS += -mmic
		else
			CFLAGS += -xHost
		endif
	else ifneq (1,$(DBG))
		CFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CFLAGS += -O0 -g -openmp
	endif
	ifeq (0,$(OFFLOAD))
		CFLAGS += -no-offload
	else ifeq ($(MIC),0)
		CFLAGS += -offload-option,mic,compiler,"-fPIC -O2 -opt-assume-safe-padding"
	endif
	LDFLAGS += -fPIC -mkl=sequential -openmp
	ifneq ($(STATIC),0)
		ifneq ($(STATIC),)
			LDFLAGS += -no-intel-extensions -static-intel
		endif
	endif
else # GCC assumed
	CXXFLAGS += -fPIC -Wall
	ifeq (0,$(DBG))
		CXXFLAGS += -O3 -fopenmp
		DFLAGS += -DNDEBUG
		ifeq ($(AVX),1)
			CXXFLAGS += -mavx
		else ifeq ($(AVX),2)
			CXXFLAGS += -mavx2
		else ifeq ($(AVX),3)
			CXXFLAGS += -mavx512f
		else
			CXXFLAGS += -march=native
		endif
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2
	else
		CXXFLAGS += -O0 -g -fopenmp
	endif
	ifneq ($(OS),Windows_NT)
		LDFLAGS += -fPIC -llapack -lblas
	else
		LDFLAGS += -llapack.dll -lblas.dll
	endif
	LDFLAGS += -lgfortran -lquadmath -fopenmp
	ifneq ($(STATIC),0)
		ifneq ($(STATIC),)
			LDFLAGS += -static
		endif
	endif
endif

ifeq (,$(CXXFLAGS))
	CXXFLAGS := $(CFLAGS)
endif
ifeq (,$(CFLAGS))
	CFLAGS := $(CXXFLAGS)
endif

MKL_DIRECT := 0
ifneq ($(MKL_DIRECT),0)
	CXXFLAGS := -DMKL_DIRECT_CALL_SEQ
	CFLAGS   := -DMKL_DIRECT_CALL_SEQ
endif


all: $(NAME)

$(NAME): $(NAME)-cpp.o

%: %-c.o $(LIBXSMM)
	$(LD) -o $@ $< $(LDFLAGS)

%: %-cpp.o $(LIBXSMM)
	$(LD) -o $@ $< $(LDFLAGS)


%-c.o: %.c
	$(CC) -o $@ -$(CFLAGS) $(DFLAGS) $(IFLAGS) -c $<

%-cpp.o: %.cpp
	$(CXX) -o $@ $(CXXFLAGS) $(DFLAGS) $(IFLAGS) -c $<

%-cpp.o: %.cxx
	$(CXX) -o $@ $(CXXFLAGS) $(DFLAGS) $(IFLAGS) -c $<

%-cpp.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) $(DFLAGS) $(IFLAGS) -c $<


clean:
	rm -f *.o *.i *.ii *.s

realclean: clean
	rm -f $(NAME) $(NAME).exe $(NAME).exe.stackdump

install: all clean
