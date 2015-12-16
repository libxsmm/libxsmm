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

# Linux cut has features we use that do not work elsewhere. Mac, etc. users
# should install GNU coreutils and use "cut" from there.
# For example, if you use Homebrew, run "brew install coreutils" once and invoke:
# $ make CUT=/usr/local/Cellar/coreutils/8.24/libexec/gnubin/cut
CUT ?= cut

ROOTDIR = $(abspath $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = $(ROOTDIR)
INCDIR = $(DEPDIR)/include
BLDDIR = build
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
	LIBNAME = $(DEPDIR)/lib/libxsmm
else ifeq (0,$(OFFLOAD))
	LIBNAME = $(DEPDIR)/lib/mic/libxsmm
else
	LIBNAME = $(DEPDIR)/lib/libxsmm
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

.PHONY: all
all: $(OUTDIR)/grad $(OUTDIR)/rstr $(OUTDIR)/axhm $(OUTDIR)/libstream_update_kernels.a

$(BLDDIR)/stream_update_kernels-mod.o: $(SRCDIR)/stream_update_kernels.f $(BLDDIR)/.make $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(OUTDIR)

$(OUTDIR)/libstream_update_kernels.a: $(OUTDIR)/.make $(BLDDIR)/stream_update_kernels-c.o
	$(AR) -rs $@ $(BLDDIR)/stream_update_kernels-c.o

$(BLDDIR)/grad-f.o: $(SRCDIR)/grad.f $(BLDDIR)/.make $(BLDDIR)/stream_update_kernels-mod.o $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(OUTDIR)/grad: $(OUTDIR)/.make $(BLDDIR)/grad-f.o $(BLDDIR)/mxm_std-f.o $(OUTDIR)/libstream_update_kernels.a $(LIBNAME).$(LIBEXT)
	$(FC) -o $@ $(BLDDIR)/grad-f.o $(BLDDIR)/mxm_std-f.o $(OUTDIR)/libstream_update_kernels.a $(MODULES) $(LIBNAME).$(LIBEXT) $(SLDFLAGS) $(LDFLAGS) $(FCMTFLAGS) $(ELDFLAGS)

$(BLDDIR)/rstr-f.o: $(SRCDIR)/rstr.f $(BLDDIR)/.make $(BLDDIR)/stream_update_kernels-mod.o $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(OUTDIR)/rstr: $(OUTDIR)/.make $(BLDDIR)/rstr-f.o $(BLDDIR)/mxm_std-f.o $(OUTDIR)/libstream_update_kernels.a $(LIBNAME).$(LIBEXT)
	$(FC) -o $@ $(BLDDIR)/rstr-f.o $(BLDDIR)/mxm_std-f.o $(OUTDIR)/libstream_update_kernels.a $(MODULES) $(LIBNAME).$(LIBEXT) $(SLDFLAGS) $(LDFLAGS) $(FCMTFLAGS) $(ELDFLAGS)

$(BLDDIR)/axhm-f.o: $(SRCDIR)/axhm.f $(BLDDIR)/.make $(BLDDIR)/stream_update_kernels-mod.o $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(OUTDIR)/axhm: $(OUTDIR)/.make $(BLDDIR)/axhm-f.o $(BLDDIR)/mxm_std-f.o $(OUTDIR)/libstream_update_kernels.a $(LIBNAME).$(LIBEXT)
	$(FC) -o $@ $(BLDDIR)/axhm-f.o $(BLDDIR)/mxm_std-f.o $(OUTDIR)/libstream_update_kernels.a $(MODULES) $(LIBNAME).$(LIBEXT) $(SLDFLAGS) $(LDFLAGS) $(FCMTFLAGS) $(ELDFLAGS)

$(BLDDIR)/%-mod.o: $(DEPDIR)/include/%.f $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.f90 $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.F90 $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-mod77.o: $(DEPDIR)/include/%.F $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f $(BLDDIR)/.make $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 $(BLDDIR)/.make $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 $(BLDDIR)/.make $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F $(BLDDIR)/.make $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR) *.mod *.modmic
else
	@rm -f $(OBJECTS) $(BLDDIR)/*.mod $(BLDDIR)/*.modmic
endif
else
	@rm -f $(OBJECTS) $(BLDDIR)/*.mod $(BLDDIR)/*.modmic
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
	@rm -f $(OUTDIR)/libstream_update_kernels.a
	@rm -f $(OUTDIR)/libxsmm.so
	@rm -f $(OUTDIR)/*.bin
	@rm -f $(OUTDIR)/.make

install: all clean
	@cp $(DEPDIR)/lib/libxsmm.so $(OUTDIR) 2> /dev/null || true
ifeq (0,$(OFFLOAD))
ifneq (0,$(MIC))
	@cp $(DEPDIR)/lib/mic/libxsmm.so $(OUTDIR) 2> /dev/null || true
endif
endif
	@cp $(ROOTDIR)/grad-perf.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/grad.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/rstr.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/axhm.sh $(OUTDIR) 2> /dev/null || true

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

