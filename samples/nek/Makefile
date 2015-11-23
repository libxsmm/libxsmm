# Export all variables to sub-make processes.
#.EXPORT_ALL_VARIABLES: #export

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explcitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(MAKE_PARALLEL))
ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif
endif

# Linux cut has features we use that do not work elsewhere
# Mac, etc. users should install GNU coreutils and use cut from there.
#
# For example, if you use Homebrew, run "brew install coreutils" once
# and then invoke the LIBXSMM make command with
# CUT=/usr/local/Cellar/coreutils/8.24/libexec/gnubin/cut
CUT ?= cut

ROOTDIR = $(abspath $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = $(ROOTDIR)
INCDIR = $(DEPDIR)/include
BLDDIR = build
ifeq (0,$(MIC))
	LIBDIR = ../../lib/mic
else
 	LIBDIR = ../../lib/intel64
endif
DEPDIR = ../..
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -D__extern_always_inline=inline
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(DEPDIR)/include

STATIC ?= 0
OMP ?= 1
SYM ?= 0
DBG ?= 0

# Request strongest code conformance
PEDANTIC ?= 0

# Embed InterProcedural Optimization information into libraries
IPO ?= 0

# Select certain code path
SSE ?= 0
AVX ?= 0

# ILP64=0 (LP64 with 32-bit integers), and ILP64=0 (64-bit integers)
ILP64 ?= 0
BLAS ?= 1

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
           $(shell ls -1 $(INCDIR)/*.hpp 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(INCDIR)/*.hxx 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(INCDIR)/*.hh  2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.h   2> /dev/null | tr "\n" " ") \
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
include $(ROOTDIR)/Makefile.inc

.PHONY: all
all: $(OUTDIR)/grad $(OUTDIR)/rstr $(OUTDIR)/axhm $(LIBDIR)/libstream_update_kernels.a

$(BLDDIR)/stream_update_kernels-mod90.o: $(SRCDIR)/stream_update_kernels.f $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/grad-f.o: $(SRCDIR)/grad.f $(BLDDIR)/stream_update_kernels-mod90.o $(MODULES) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(OUTDIR)/grad: $(BLDDIR)/grad-f.o $(BLDDIR)/mxm_std-f.o $(LIBDIR)/libstream_update_kernels.a $(MODULES) $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(FC) -o $@ $^ $(SLDFLAGS) $(LDFLAGS) $(FCMTFLAGS) $(ELDFLAGS)

$(BLDDIR)/rstr-f.o: $(SRCDIR)/rstr.f $(BLDDIR)/stream_update_kernels-mod90.o $(MODULES) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(OUTDIR)/rstr: $(BLDDIR)/rstr-f.o $(BLDDIR)/mxm_std-f.o $(LIBDIR)/libstream_update_kernels.a $(MODULES) $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(FC) -o $@ $^ $(SLDFLAGS) $(LDFLAGS) $(FCMTFLAGS) $(ELDFLAGS)

$(BLDDIR)/axhm-f.o: $(SRCDIR)/axhm.f $(BLDDIR)/stream_update_kernels-mod90.o $(MODULES) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(OUTDIR)/axhm: $(BLDDIR)/axhm-f.o $(BLDDIR)/mxm_std-f.o $(LIBDIR)/libstream_update_kernels.a $(MODULES) $(LIBNAME).$(LIBEXT)
	@mkdir -p $(dir $@)
	$(FC) -o $@ $^ $(SLDFLAGS) $(LDFLAGS) $(FCMTFLAGS) $(ELDFLAGS)

$(LIBDIR)/libstream_update_kernels.a: $(BLDDIR)/stream_update_kernels-c.o
	@mkdir -p $(dir $@)
	$(AR) -rs $@ $^
	@cp stream_update_kernels.f $(DEPDIR)/include/

$(BLDDIR)/%-mod.o: $(DEPDIR)/include/%.f $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.f90 $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.F90 $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-mod77.o: $(DEPDIR)/include/%.F $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp $(HEADERS) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c $(HEADERS) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f $(MODULES) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 $(MODULES) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 $(MODULES) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F $(MODULES) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
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
	@rm -f $(OUTDIR)/grad $(OUTDIR)/axhm $(OUTDIR)/rstr
endif
else
	@rm -f $(OUTDIR)/grad $(OUTDIR)/axhm $(OUTDIR)/rstr
endif
	@rm -f $(OUTDIR)/libxsmm.so
	@rm -f $(LIBDIR)/libstream_update_kernels.a
	@rm -f $(DEPDIR)/include/stream_update_kernels.f

install: all clean
	@cp $(DEPDIR)/lib/intel64/libxsmm.so $(OUTDIR) 2> /dev/null || true
ifeq (0,$(OFFLOAD))
ifneq (0,$(MIC))
	@cp $(DEPDIR)/lib/mic/libxsmm.so $(OUTDIR) 2> /dev/null || true
endif
endif
	@cp $(ROOTDIR)/grad-perf.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/grad.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/rstr.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/axhm.sh $(OUTDIR) 2> /dev/null || true

