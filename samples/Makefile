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
		CXXFLAGS += -O2 -fno-alias -ansi-alias -xHost -openmp
		DFLAGS += -DNDEBUG
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CXXFLAGS += -O0 -g
	endif
	LFLAGS += -mkl=sequential
else ifneq (,$(filter $(CC),icc))
	ifeq (0,$(DBG))
		CFLAGS += -O2 -fno-alias -ansi-alias -xHost -openmp
		DFLAGS += -DNDEBUG
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
	else
		CFLAGS += -O0 -g
	endif
	LFLAGS += -mkl=sequential
else
	ifeq (0,$(DBG))
		CXXFLAGS += -O2 -march=native -fopenmp
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
	$(LD) -o $@ $(CFLAGS) $< $(LFLAGS)

%: %-cpp.o
	$(LD) -o $@ $(CXXFLAGS) $< $(LFLAGS)


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
