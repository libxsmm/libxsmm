CXXFLAGS :=
CFLAGS :=
DFLAGS :=
IFLAGS := -I../include
LFLAGS := ../lib/intel64/libxsmm.a

ICPC := $(notdir $(shell which icpc 2> /dev/null))
ICC := $(notdir $(shell which icc 2> /dev/null))

ifneq (,$(ICPC))
	CXX := $(ICPC)
	ifeq (,$(ICC))
		CC := $(ICPC)
	endif
endif
ifneq (,$(ICC))
	CC := $(ICC)
	ifeq (,$(ICPC))
		CXX := $(ICC)
	endif
endif

LD := $(CXX)
DBG := 0

ifneq (,$(filter $(CXX),icpc))
	ifeq (0,$(DBG))
		CXXFLAGS += -Wall -fno-alias -ansi-alias -mkl=sequential -DNDEBUG -O3 -ipo -xHost -fopenmp -offload-option,mic,compiler,"-opt-assume-safe-padding"
		DFLAGS += -DNDEBUG
		ifeq ($(AVX),1)
			CXXFLAGS += -xAVX
		else ifeq ($(AVX),2)
			CXXFLAGS += -xCORE-AVX2
		else ifeq ($(AVX),3)
			CXXFLAGS += -xCOMMON-AVX512
		else
			CXXFLAGS += -xHost
		endif
	else ifneq (1,$(DBG))
		CXXFLAGS += -Wall -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CXXFLAGS += -Wall -O0 -g
	endif
	LFLAGS += -openmp -mkl=sequential
	ifeq (0,$(OFFLOAD))
		CXXFLAGS += -no-offload
		LFLAGS += -no-offload
	endif
else ifneq (,$(filter $(CC),icc))
	ifeq (0,$(DBG))
		CFLAGS += -Wall -fno-alias -ansi-alias -mkl=sequential -DNDEBUG -O3 -ipo -xHost -fopenmp -offload-option,mic,compiler,"-opt-assume-safe-padding"
		DFLAGS += -DNDEBUG
		ifeq ($(AVX),1)
			CFLAGS += -xAVX
		else ifeq ($(AVX),2)
			CFLAGS += -xCORE-AVX2
		else ifeq ($(AVX),3)
			CFLAGS += -xCOMMON-AVX512
		else
			CFLAGS += -xHost
		endif
	else ifneq (1,$(DBG))
		CFLAGS += -Wall -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CFLAGS += -Wall -O0 -g
	endif
	LFLAGS += -openmp -mkl=sequential
	ifeq (0,$(OFFLOAD))
		CFLAGS += -no-offload
		LFLAGS += -no-offload
	endif
else # GCC assumed
	ifeq (0,$(DBG))
		CXXFLAGS += -Wall -O3 -fopenmp
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
		CXXFLAGS += -Wall -O0 -g3 -gdwarf-2
	else
		CXXFLAGS += -Wall -O0 -g
	endif
	LFLAGS += -fopenmp -llapack -lblas
endif

ifeq (,$(CXXFLAGS))
	CXXFLAGS := $(CFLAGS)
endif
ifeq (,$(CFLAGS))
	CFLAGS := $(CXXFLAGS)
endif

MKLDIRECT := 0
ifneq ($(MKLDIRECT),0)
	CXXFLAGS := -DMKL_DIRECT_CALL_SEQ
	CFLAGS   := -DMKL_DIRECT_CALL_SEQ
endif


all: smm

smm: smm-cpp.o

%: %-c.o
	$(LD) -o $@ $< $(LFLAGS)

%: %-cpp.o
	$(LD) -o $@ $< $(LFLAGS)


%-c.o: %.c
	$(CC) -o $@ -$(CFLAGS) $(DFLAGS) $(IFLAGS) -c $<

%-cpp.o: %.cpp
	$(CXX) -o $@ $(CXXFLAGS) $(DFLAGS) $(IFLAGS) -c $<

%-cpp.o: %.cxx
	$(CXX) -o $@ $(CXXFLAGS) $(DFLAGS) $(IFLAGS) -c $<

%-cpp.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) $(DFLAGS) $(IFLAGS) -c $<


clean:
	@-rm -f *.o *.i *.ii *.s

realclean: clean
	@-rm -f smm smm.exe
	@-rm -f smm smm.exe

install: all clean
