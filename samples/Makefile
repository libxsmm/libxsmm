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
		CXXFLAGS += -openmp -O2 -ipo -fno-alias -ansi-alias -xHost -opt-assume-safe-padding -mkl=sequential
		DFLAGS += -DNDEBUG
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CXXFLAGS += -O0 -g
	endif
	LFLAGS += -mkl=sequential
	ifeq (0,$(OFFLOAD))
		CXXFLAGS += -no-offload
		LFLAGS += -no-offload
	endif
else ifneq (,$(filter $(CC),icc))
	ifeq (0,$(DBG))
		CFLAGS += -openmp -O2 -ipo -fno-alias -ansi-alias -xHost -opt-assume-safe-padding -mkl=sequential
		DFLAGS += -DNDEBUG
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CFLAGS += -O0 -g
	endif
	LFLAGS += -mkl=sequential
	ifeq (0,$(OFFLOAD))
		CXXFLAGS += -no-offload
		LFLAGS += -no-offload
	endif
else
	ifeq (0,$(DBG))
		CXXFLAGS += -fopenmp -O2 -march=native
		DFLAGS += -DNDEBUG
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2
	else
		CXXFLAGS += -O0 -g
	endif
	LFLAGS += -llapack -lblas
endif

ifeq (,$(CXXFLAGS))
	CXXFLAGS := $(CFLAGS)
endif
ifeq (,$(CFLAGS))
	CFLAGS := $(CXXFLAGS)
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
