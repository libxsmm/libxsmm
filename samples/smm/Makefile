ROOTDIR = $(abspath $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
DEPDIR = $(ROOTDIR)/../..
INCDIR = $(ROOTDIR)
SRCDIR = $(ROOTDIR)
BLDDIR = build
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(DEPDIR)/include

OMP ?= 1
DBG ?= 0
IPO ?= 0

OFFLOAD ?= 0
ifneq ($(OFFLOAD),0)
	MIC ?= 1
else
	MIC ?= 0
endif

ifeq ($(MIC),0)
	LIBNAME = $(DEPDIR)/lib/intel64/libxsmm
else
	LIBNAME = $(DEPDIR)/lib/mic/libxsmm
endif

ifneq ("","$(wildcard $(LIBNAME).a)")
	STATIC ?= 1
	LIBEXT = a
else
	STATIC ?= 0
	LIBEXT = so
endif

BLAS ?= 1
ifneq ($(MKLROOT),)
	MKL ?= 1
	MKL_DIRECT ?= 0
	ifneq ($(MKL_DIRECT),0)
		MKL_STATIC ?= 1
	else ifneq ($(STATIC),0)
		MKL_STATIC ?= 1
	else
		MKL_STATIC ?= 0
	endif
else
	MKL ?= 0
endif

HEADERS = $(shell ls -1 $(INCDIR)/*.h   2> /dev/null | tr "\n" " ") \
          $(shell ls -1 $(SRCDIR)/*.hpp 2> /dev/null | tr "\n" " ") \
          $(shell ls -1 $(SRCDIR)/*.hxx 2> /dev/null | tr "\n" " ") \
          $(shell ls -1 $(SRCDIR)/*.hh  2> /dev/null | tr "\n" " ")
CPPSRCS = $(shell ls -1 $(SRCDIR)/*.cpp 2> /dev/null | tr "\n" " ")
CXXSRCS = $(shell ls -1 $(SRCDIR)/*.cxx 2> /dev/null | tr "\n" " ")
CCXSRCS = $(shell ls -1 $(SRCDIR)/*.cc  2> /dev/null | tr "\n" " ")
CSOURCS = $(shell ls -1 $(SRCDIR)/*.c   2> /dev/null | tr "\n" " ")
FTNSRCS = $(shell ls -1 $(SRCDIR)/*.f   2> /dev/null | tr "\n" " ")
F77SRCS = $(shell ls -1 $(SRCDIR)/*.F   2> /dev/null | tr "\n" " ")
F90SRCS = $(shell ls -1 $(SRCDIR)/*.f90 2> /dev/null | tr "\n" " ")
FTNINCS = $(shell ls -1 $(DEPDIR)/include/*.f   2> /dev/null | tr "\n" " ")
F77INCS = $(shell ls -1 $(DEPDIR)/include/*.F   2> /dev/null | tr "\n" " ")
F90INCS = $(shell ls -1 $(DEPDIR)/include/*.f90 2> /dev/null | tr "\n" " ")
FTNMODS = $(patsubst %,$(BLDDIR)/%,$(notdir $(FTNINCS:.f=-mod.o)))
F77MODS = $(patsubst %,$(BLDDIR)/%,$(notdir $(F77INCS:.F=-mod77.o)))
F90MODS = $(patsubst %,$(BLDDIR)/%,$(notdir $(F90INCS:.f90=-mod90.o)))
MODULES = $(FTNMODS) $(F77MODS) $(F90MODS)
SOURCES = $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS) $(FTNSRCS) $(F77SRCS) $(F90SRCS)
CPPOBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CPPSRCS:.cpp=-cpp.o)))
CXXOBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CXXSRCS:.cxx=-cxx.o)))
CCXOBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CCXSRCS:.cc=-cc.o)))
COBJCTS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
FTNOBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.f=-f.o)))
F77OBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.F=-f77.o)))
F90OBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.f90=-f90.o)))
OBJECTS = $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS) $(FTNOBJS) $(F77OBJS) $(F90OBJS)

ICPC    = $(notdir $(shell which icpc     2> /dev/null))
ICC     = $(notdir $(shell which icc      2> /dev/null))
IFORT   = $(notdir $(shell which ifort    2> /dev/null))
GPP     = $(notdir $(shell which g++      2> /dev/null))
GCC     = $(notdir $(shell which gcc      2> /dev/null))
GFC     = $(notdir $(shell which gfortran 2> /dev/null))

ifneq (,$(ICPC))
	CXX = $(ICPC)
	ifeq (,$(ICC))
		CC = $(CXX)
	endif
else
	CXX = $(GPP)
endif
ifneq (,$(ICC))
	CC = $(ICC)
	ifeq (,$(ICPC))
		CXX = $(CC)
	endif
else
	CC = $(GCC)
endif
ifneq (,$(IFORT))
	FC = $(IFORT)
else
	FC = $(GFC)
endif
ifneq ($(CXX),)
	LD = $(CXX)
endif
ifeq ($(LD),)
	LD = $(CC)
endif
ifeq ($(LD),)
	LD = $(FC)
endif

ifneq (,$(filter icpc icc ifort,$(CXX) $(CC) $(FC)))
	CXXFLAGS += -fPIC -Wall -std=c++0x
	CFLAGS += -fPIC -Wall -std=c99
	FCFLAGS += -fPIC
	LDFLAGS += -fPIC
	ifeq (0,$(DBG))
		CXXFLAGS += -fno-alias -ansi-alias -O2
		CFLAGS += -fno-alias -ansi-alias -O2
		FCFLAGS += -O2
		DFLAGS += -DNDEBUG
		ifneq ($(IPO),0)
			CXXFLAGS += -ipo
			CFLAGS += -ipo
			FCFLAGS += -ipo
		endif
		ifeq ($(AVX),1)
			TARGET = -xAVX
		else ifeq ($(AVX),2)
			TARGET = -xCORE-AVX2
		else ifeq ($(AVX),3)
			TARGET = -xCOMMON-AVX512
		else
			TARGET = -xHost
		endif
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
		CFLAGS += -O0 -g3 -gdwarf-2 -debug inline-debug-info
		FCFLAGS += -O0 -g
	else
		CXXFLAGS += -O0 -g
		CFLAGS += -O0 -g
		FCFLAGS += -O0 -g
	endif
	ifneq ($(OMP),0)
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
	ifneq ($(STATIC),0)
		SLDFLAGS += -no-intel-extensions -static-intel
	endif
else # GCC assumed
	VERSION = $(shell $(GCC) --version | grep "gcc (GCC)" | sed "s/gcc (GCC) \([0-9]\+\.[0-9]\+\.[0-9]\+\).*$$/\1/")
	VERSION_MAJOR = $(shell echo "$(VERSION)" | cut -d"." -f1)
	VERSION_MINOR = $(shell echo "$(VERSION)" | cut -d"." -f2)
	VERSION_PATCH = $(shell echo "$(VERSION)" | cut -d"." -f3)
	MIC = 0
	CXXFLAGS += -Wall -std=c++0x
	CFLAGS += -Wall
	ifneq ($(OS),Windows_NT)
		CXXFLAGS += -fPIC
		CFLAGS += -fPIC
		FCFLAGS += -fPIC
		LDFLAGS += -fPIC
	endif
	ifeq (0,$(DBG))
		CXXFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		CFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		FCFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		DFLAGS += -DNDEBUG
		ifeq ($(AVX),1)
			TARGET = -mavx
		else ifeq ($(AVX),2)
			TARGET = -mavx2
		else ifeq ($(AVX),3)
			TARGET = -mavx512f
		else
			TARGET = -march=native
		endif
	else ifneq (1,$(DBG))
		CXXFLAGS += -O0 -g3 -gdwarf-2
		CFLAGS += -O0 -g3 -gdwarf-2
		FCFLAGS += -O0 -g3 -gdwarf-2
	else
		CXXFLAGS += -O0 -g
		CFLAGS += -O0 -g
		FCFLAGS += -O0 -g
	endif
	ifneq ($(OMP),0)
		CXXFLAGS += -fopenmp
		CFLAGS += -fopenmp
		FCFLAGS += -fopenmp
		LDFLAGS += -fopenmp
	endif
	ifneq ($(STATIC),0)
		SLDFLAGS += -static
	endif
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

ifeq (1,$(MKL_DIRECT))
	MKL_STATIC = 1
	DFLAGS += -DMKL_DIRECT_CALL_SEQ
endif

ifeq (1,$(MKL)) # sequential
	DFLAGS += -D__MKL
	IFLAGS +=-I$(MKLROOT)/include
	ifeq (0,$(MKL_STATIC)) # shared
		LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential
		ifneq (0,$(OFFLOAD))
			LDFLAGS += -offload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_intel_lp64 -lmkl_core -lmkl_sequential"
		endif
	else # static
		LDFLAGS += \
			-Wl,--start-group \
				$(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
				$(MKLROOT)/lib/intel64/libmkl_core.a \
				$(MKLROOT)/lib/intel64/libmkl_sequential.a \
			-Wl,--end-group
		ifneq (0,$(OFFLOAD))
			LDFLAGS += -offload-option,mic,ld," \
				--start-group \
					$(MKLROOT)/lib/mic/libmkl_intel_lp64.a \
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
		LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread
		ifneq (0,$(OFFLOAD))
			LDFLAGS += -offload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread"
		endif
	else # static
		LDFLAGS += \
			-Wl,--start-group \
				$(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
				$(MKLROOT)/lib/intel64/libmkl_core.a \
				$(MKLROOT)/lib/intel64/libmkl_intel_thread.a \
			-Wl,--end-group
		ifneq (0,$(OFFLOAD))
			LDFLAGS += -offload-option,mic,ld," \
				--start-group \
					$(MKLROOT)/lib/mic/libmkl_intel_lp64.a \
					$(MKLROOT)/lib/mic/libmkl_core.a \
					$(MKLROOT)/lib/mic/libmkl_intel_thread.a \
				--end-group"
		endif
	endif
	ifeq (0,$(OMP))
		LDFLAGS += -liomp5
		ifneq (0,$(OFFLOAD))
			LDFLAGS += -offload-option,mic,ld,"-liomp5"
		endif
	endif
	LDFLAGS += -lpthread -lm
	ifneq (0,$(OFFLOAD))
		LDFLAGS += -offload-option,mic,ld,"-lpthread -lm"
	endif
else ifneq (0,$(BLAS)) # generic
	ifneq ($(OS),Windows_NT)
		LDFLAGS += -lblas
	else
		LDFLAGS += -lblas.dll
	endif
	LDFLAGS += -lgfortran
	ifeq (,$(VERSION))
		LDFLAGS += -lquadmath
	else ifneq (0,$(shell echo "$$((40500 <= $(VERSION_MAJOR) * 10000 + $(VERSION_MINOR) * 100 + $(VERSION_PATCH)))"))
		LDFLAGS += -lquadmath
	endif
endif

.PHONY: all
all: $(OUTDIR)/blas $(OUTDIR)/dispatched $(OUTDIR)/inlined $(OUTDIR)/specialized $(OUTDIR)/smm

$(OUTDIR)/blas: $(BLDDIR)/blas-cpp.o $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(SLDFLAGS) $(LDFLAGS)

$(OUTDIR)/dispatched: $(BLDDIR)/dispatched-cpp.o $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(SLDFLAGS) $(LDFLAGS)

$(OUTDIR)/inlined: $(BLDDIR)/inlined-cpp.o $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(SLDFLAGS) $(LDFLAGS)

$(OUTDIR)/specialized: $(BLDDIR)/specialized-cpp.o $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(SLDFLAGS) $(LDFLAGS)

$(OUTDIR)/smm: $(BLDDIR)/smm-f90.o $(MODULES) $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(LDFLAGS)

$(BLDDIR)/%-mod.o: $(DEPDIR)/include/%.f $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ -J $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.f90 $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ -J $(dir $@)

$(BLDDIR)/%-mod77.o: $(DEPDIR)/include/%.F $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ -J $(dir $@)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
else
	@rm -f $(OBJECTS) $(BLDDIR)/*.mod
endif
else
	@rm -f $(OBJECTS) $(BLDDIR)/*.mod
endif
	@rm -f fit.log *.dat

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
else
	@rm -f $(OUTDIR)/blas $(OUTDIR)/dispatched $(OUTDIR)/inlined $(OUTDIR)/specialized $(OUTDIR)/smm
endif
else
	@rm -f $(OUTDIR)/blas $(OUTDIR)/dispatched $(OUTDIR)/inlined $(OUTDIR)/specialized $(OUTDIR)/smm
endif
	@rm -f $(OUTDIR)/libxsmm.so

install: all clean
	@cp $(DEPDIR)/lib/intel64/libxsmm.so $(OUTDIR) 2> /dev/null || true
	@cp $(DEPDIR)/lib/mic/libxsmm.so $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/smm-perf.plt $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/smm-plot.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/smm-perf.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/blas.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/dispatched.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/inlined.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/specialized.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/smm.sh $(OUTDIR) 2> /dev/null || true

