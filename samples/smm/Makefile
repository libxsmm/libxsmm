ROOTDIR = $(abspath $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SRCDIR = $(ROOTDIR)
INCDIR = $(ROOTDIR)
BLDDIR = build
DEPDIR = ../..
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(DEPDIR)/include

STATIC ?= 0
BLAS ?= 1
OMP ?= 1
SYM ?= 0
DBG ?= 0
IPO ?= 0
SSE ?= 0
AVX ?= 0

OFFLOAD ?= 0
ifneq ($(OFFLOAD),0)
	MIC ?= 1
else
	MIC ?= 0
endif

ifeq ($(MIC),0)
	LIBNAME = $(DEPDIR)/lib/intel64/libxsmm
else ifeq ($(OFFLOAD),0)
	LIBNAME = $(DEPDIR)/lib/mic/libxsmm
else
	LIBNAME = $(DEPDIR)/lib/intel64/libxsmm
endif

ifneq (,$(wildcard $(LIBNAME).a))
	STATIC ?= 1
	LIBEXT = a
else
	STATIC ?= 0
	LIBEXT = so
endif

ifneq ($(MKLROOT),)
	MKL ?= $(BLAS)
	MKL_DIRECT ?= 0
	ifneq ($(MKL_DIRECT),0)
		MKL_STATIC ?= 1
	else ifneq ($(STATIC),0)
		MKL_STATIC ?= 1
	else
		MKL_STATIC ?= 0
	endif
else
	MKL = 0
endif

OUTNAME = $(shell basename $(ROOTDIR))
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
FTNOBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(FTNSRCS:.f=-f.o)))
F77OBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.F=-f77.o)))
F90OBJS = $(patsubst %,$(BLDDIR)/%,$(notdir $(F90SRCS:.f90=-f90.o)))
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
	CXXFLAGS += -fPIC -Wall
	CFLAGS += -fPIC -Wall -std=c99
	FCFLAGS += -fPIC
	LDFLAGS += -fPIC
	ifeq (0,$(DBG))
		CXXFLAGS += -fno-alias -ansi-alias -O2
		CFLAGS += -fno-alias -ansi-alias -O2
		FCMTFLAGS += -threads
		FCFLAGS += -O2
		DFLAGS += -DNDEBUG
		ifneq ($(IPO),0)
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
	ifeq (1,$(shell echo $$((2 > $(DBG)))))
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
		ifneq (1,$(SYM))
			CXXFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CXXFLAGS)
			CFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		else
			CXXFLAGS := -g $(CXXFLAGS)
			CFLAGS := -g $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		endif
	endif
	ifeq (0,$(EXP))
		CXXFLAGS += -fno-exceptions
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
	FCMODDIRFLAG = -module
else # GCC assumed
	VERSION = $(shell $(GCC) --version | grep "gcc (GCC)" | sed "s/gcc (GCC) \([0-9]\+\.[0-9]\+\.[0-9]\+\).*$$/\1/")
	VERSION_MAJOR = $(shell echo "$(VERSION)" | cut -d"." -f1)
	VERSION_MINOR = $(shell echo "$(VERSION)" | cut -d"." -f2)
	VERSION_PATCH = $(shell echo "$(VERSION)" | cut -d"." -f3)
	MIC = 0
	CXXFLAGS += -Wall
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
		ifneq ($(IPO),0)
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
	ifeq (1,$(shell echo $$((2 > $(DBG)))))
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
		ifneq (1,$(SYM))
			CXXFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CXXFLAGS)
			CFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		else
			CXXFLAGS := -g $(CXXFLAGS)
			CFLAGS := -g $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		endif
	endif
	ifeq (0,$(EXP))
		CXXFLAGS += -fno-exceptions
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
	FCMODDIRFLAG = -J
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
	CLDFLAGS += -lgfortran
	ifeq (,$(VERSION))
		LDFLAGS += -lquadmath
	else ifneq (0,$(shell echo "$$((40500 <= $(VERSION_MAJOR) * 10000 + $(VERSION_MINOR) * 100 + $(VERSION_PATCH)))"))
		CLDFLAGS += -lquadmath
	endif
endif

.PHONY: all
all: $(OUTDIR)/$(OUTNAME) $(OUTDIR)/blas $(OUTDIR)/dispatched $(OUTDIR)/inlined $(OUTDIR)/specialized

$(OUTDIR)/blas: $(BLDDIR)/blas-cpp.o $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/dispatched: $(BLDDIR)/dispatched-cpp.o $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/inlined: $(BLDDIR)/inlined-cpp.o $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/specialized: $(BLDDIR)/specialized-cpp.o $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/smm: $(BLDDIR)/smm-f90.o $(MODULES) $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(FC) -o $@ $^ $(LDFLAGS) $(FCMTFLAGS)

$(BLDDIR)/%-mod.o: $(DEPDIR)/include/%.f $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FCMODDIRFLAG) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.f90 $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FCMODDIRFLAG) $(dir $@)

$(BLDDIR)/%-mod77.o: $(DEPDIR)/include/%.F $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FCMODDIRFLAG) $(dir $@)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c $(HEADERS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

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
	@rm -f $(OUTDIR)/$(OUTNAME) $(OUTDIR)/blas $(OUTDIR)/dispatched $(OUTDIR)/inlined $(OUTDIR)/specialized
endif
else
	@rm -f $(OUTDIR)/$(OUTNAME) $(OUTDIR)/blas $(OUTDIR)/dispatched $(OUTDIR)/inlined $(OUTDIR)/specialized
endif
	@rm -f $(OUTDIR)/libxsmm.so

install: all clean
	@cp $(DEPDIR)/lib/intel64/libxsmm.so $(OUTDIR) 2> /dev/null || true
	@cp $(DEPDIR)/lib/mic/libxsmm.so $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME).sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-perf.plt $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-perf.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-plot.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/blas.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/dispatched.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/inlined.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/specialized.sh $(OUTDIR) 2> /dev/null || true

