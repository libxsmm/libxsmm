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
		CXXFLAGS += -O2 -xHost -ansi-alias
		DFLAGS += -DNDEBUG
	else
		CXXFLAGS += -O0 -g
	endif
	LFLAGS += -mkl=sequential
else ifneq (,$(filter $(CC),icc))
	ifeq (0,$(DBG))
		CFLAGS += -O2 -xHost -ansi-alias
		DFLAGS += -DNDEBUG
	else
		CFLAGS += -O0 -g
	endif
	LFLAGS += -mkl=sequential
else
        ifeq (0,$(DBG))
                CXXFLAGS += -O2 -march=native
                DFLAGS += -DNDEBUG
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


all: smm-cpp


smm-c: smm-c.o
	$(LD) -o $@ $(CFLAGS) $< $(LFLAGS)

smm-cpp: smm-cpp.o
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
	@-rm -f *.o

realclean: clean
	@-rm -f smm-c smm-c.exe
	@-rm -f smm-cpp smm-cpp.exe

install: all clean
