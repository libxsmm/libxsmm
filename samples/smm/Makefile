# export all variables to sub-make processes
#.EXPORT_ALL_VARIABLES: #export
# Set MAKE_PARALLEL=0 for issues with parallel make (older GNU Make)
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif

# Linux cut has features we use that do not work elsewhere
# Mac, etc. users should install GNU coreutils and use cut from there.
#
# For example, if you use Homebrew, run "brew install coreutils" once
# and then invoke the LIBXSMM make command with
# CUT=/usr/local/Cellar/coreutils/8.24/libexec/gnubin/cut
CUT ?= cut

ROOTDIR = $(abspath $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SRCDIR = $(ROOTDIR)
INCDIR = $(ROOTDIR)
BLDDIR = build
DEPDIR = ../..
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -D__extern_always_inline=inline
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(DEPDIR)/include

# Request strongest code conformance
PEDANTIC ?= 0

STATIC ?= 0
BLAS ?= 1
OMP ?= 1
SYM ?= 0
DBG ?= 0
IPO ?= 0
SSE ?= 0
AVX ?= 0

OFFLOAD ?= 0
ifneq (0,$(OFFLOAD))
	MIC ?= 1
else
	MIC ?= 0
endif

ifeq (0,$(MIC))
	LIBNAME = $(DEPDIR)/lib/intel64/libxsmm
else ifeq (0,$(OFFLOAD))
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

OUTNAME := $(shell basename $(ROOTDIR))
HEADERS := $(shell ls -1 $(INCDIR)/*.h   2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.hpp 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.hxx 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.hh  2> /dev/null | tr "\n" " ")
CPPSRCS := $(shell ls -1 $(SRCDIR)/*.cpp 2> /dev/null | tr "\n" " ")
CXXSRCS := $(shell ls -1 $(SRCDIR)/*.cxx 2> /dev/null | tr "\n" " ")
CCXSRCS := $(shell ls -1 $(SRCDIR)/*.cc  2> /dev/null | tr "\n" " ")
CSOURCS := $(shell ls -1 $(SRCDIR)/*.c   2> /dev/null | tr "\n" " ")
FTNSRCS := $(shell ls -1 $(SRCDIR)/*.f   2> /dev/null | tr "\n" " ")
F77SRCS := $(shell ls -1 $(SRCDIR)/*.F   2> /dev/null | tr "\n" " ")
F90SRCS := $(shell ls -1 $(SRCDIR)/*.f90 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.F90 2> /dev/null | tr "\n" " ")
FTNINCS := $(shell ls -1 $(DEPDIR)/include/*.f   2> /dev/null | tr "\n" " ")
F77INCS := $(shell ls -1 $(DEPDIR)/include/*.F   2> /dev/null | tr "\n" " ")
F90INCS := $(shell ls -1 $(DEPDIR)/include/*.f90 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(DEPDIR)/include/*.F90 2> /dev/null | tr "\n" " ")
FTNMODS := $(patsubst %,$(BLDDIR)/%,$(notdir $(FTNINCS:.f=-mod.o)))
F77MODS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F77INCS:.F=-mod77.o)))
F90MODS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90INCS:.f90=-mod90.o)))
F90MODS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90MODS:.F90=-mod90.o)))
MODULES := $(FTNMODS) $(F77MODS) $(F90MODS)
SOURCES := $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS) $(FTNSRCS) $(F77SRCS) $(F90SRCS)
CPPOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CPPSRCS:.cpp=-cpp.o)))
CXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CXXSRCS:.cxx=-cxx.o)))
CCXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CCXSRCS:.cc=-cc.o)))
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
FTNOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(FTNSRCS:.f=-f.o)))
F77OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.F=-f77.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90SRCS:.f90=-f90.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90OBJS:.F90=-f90.o)))
OBJECTS := $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS) $(FTNOBJS) $(F77OBJS) $(F90OBJS)

# include common Makefile artifacts
include Makefile.inc

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
		LDFLAGS += -Wl,--start-group \
			$(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
			$(MKLROOT)/lib/intel64/libmkl_core.a \
			$(MKLROOT)/lib/intel64/libmkl_sequential.a \
		-Wl,--end-group
		ifneq (0,$(OFFLOAD))
			LDFLAGS += -offload-option,mic,ld,"--start-group \
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
		ifneq (0,$(INTEL))
			LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread
			ifneq (0,$(OFFLOAD))
				LDFLAGS += -offload-option,mic,ld,"-L$(MKLROOT)/lib/mic -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread"
			endif
		else # assuming GNU toolchain
			LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread
		endif
	else # static
		ifneq (0,$(INTEL))
			LDFLAGS += -Wl,--start-group \
				$(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
				$(MKLROOT)/lib/intel64/libmkl_core.a \
				$(MKLROOT)/lib/intel64/libmkl_intel_thread.a \
			-Wl,--end-group
			ifneq (0,$(OFFLOAD))
				LDFLAGS += -offload-option,mic,ld,"--start-group \
					$(MKLROOT)/lib/mic/libmkl_intel_lp64.a \
					$(MKLROOT)/lib/mic/libmkl_core.a \
					$(MKLROOT)/lib/mic/libmkl_intel_thread.a \
				--end-group"
			endif
		else # assuming GNU toolchain
			LDFLAGS += -Wl,--start-group \
				$(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
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
	ifneq (Windows_NT,$(OS))
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

$(OUTDIR)/smm: $(BLDDIR)/smm-f.o $(MODULES) $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(FC) -o $@ $^ $(SLDFLAGS) $(LDFLAGS) $(FCMTFLAGS)

$(BLDDIR)/%-mod.o: $(DEPDIR)/include/%.f $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FCMODDIRFLAG) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.f90 $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FCMODDIRFLAG) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.F90 $(ROOTDIR)/Makefile
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

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F $(MODULES) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR) *.mod
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
ifeq (0,$(OFFLOAD))
ifneq (0,$(MIC))
	@cp $(DEPDIR)/lib/mic/libxsmm.so $(OUTDIR) 2> /dev/null || true
endif
endif
	@cp $(ROOTDIR)/$(OUTNAME).sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-perf.plt $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-perf.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-plot.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-test.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/blas.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/dispatched.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/inlined.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/specialized.sh $(OUTDIR) 2> /dev/null || true

