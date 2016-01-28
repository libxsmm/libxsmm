# Export all variables to sub-make processes.
#.EXPORT_ALL_VARIABLES: #export

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explcitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(strip $(MAKE_PARALLEL)))
ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif
endif

ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = $(ROOTDIR)
INCDIR = $(DEPDIR)/include
BLDDIR = build
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(DEPDIR)/include

BLAS ?= 1
OMP ?= 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

ifeq (0,$(MIC))
	LIBNAME = $(DEPDIR)/lib/libxsmm
else ifeq (0,$(OFFLOAD))
	LIBNAME = $(DEPDIR)/lib/mic/libxsmm
else
	LIBNAME = $(DEPDIR)/lib/libxsmm
endif

ifneq (0,$(STATIC))
	ifeq (,$(strip $(wildcard $(LIBNAME).$(LIBEXT))))
		LIBEXT = so
	endif
else
	ifeq (,$(strip $(wildcard $(LIBNAME).$(LIBEXT))))
		LIBEXT = a
	endif
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
all: $(OUTDIR)/$(OUTNAME) $(OUTDIR)/blas $(OUTDIR)/dispatched $(OUTDIR)/inlined $(OUTDIR)/specialized

$(OUTDIR)/blas: $(BLDDIR)/blas-cpp.o $(OUTDIR)/.make $(LIBNAME).$(LIBEXT)
	$(LD) -o $@ $(BLDDIR)/blas-cpp.o $(abspath $(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
ifeq (0,$(STATIC))
	@cp -u $(LIBNAME).$(LIBEXT) $(OUTDIR) 2> /dev/null || true
endif

$(OUTDIR)/dispatched: $(OUTDIR)/.make $(BLDDIR)/dispatched-cpp.o $(LIBNAME).$(LIBEXT)
	$(LD) -o $@ $(BLDDIR)/dispatched-cpp.o $(abspath $(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
ifeq (0,$(STATIC))
	@cp -u $(LIBNAME).$(LIBEXT) $(OUTDIR) 2> /dev/null || true
endif

$(OUTDIR)/inlined: $(OUTDIR)/.make $(BLDDIR)/inlined-cpp.o $(LIBNAME).$(LIBEXT)
	$(LD) -o $@ $(BLDDIR)/inlined-cpp.o $(abspath $(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
ifeq (0,$(STATIC))
	@cp -u $(LIBNAME).$(LIBEXT) $(OUTDIR) 2> /dev/null || true
endif

$(OUTDIR)/specialized: $(OUTDIR)/.make $(BLDDIR)/specialized-cpp.o $(LIBNAME).$(LIBEXT)
	$(LD) -o $@ $(BLDDIR)/specialized-cpp.o $(abspath $(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
ifeq (0,$(STATIC))
	@cp -u $(LIBNAME).$(LIBEXT) $(OUTDIR) 2> /dev/null || true
endif

$(OUTDIR)/smm: $(OUTDIR)/.make $(BLDDIR)/smm-f.o $(MODULES) $(LIBNAME).$(LIBEXT)
	$(FC) -o $@ $(BLDDIR)/smm-f.o $(MODULES) $(abspath $(LIBNAME).$(LIBEXT)) $(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
ifeq (0,$(STATIC))
	@cp -u $(LIBNAME).$(LIBEXT) $(OUTDIR) 2> /dev/null || true
endif

$(BLDDIR)/%-mod.o: $(DEPDIR)/include/%.f $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.f90 $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-mod90.o: $(DEPDIR)/include/%.F90 $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-mod77.o: $(DEPDIR)/include/%.F $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@ $(FMFLAGS) $(dir $@)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f $(BLDDIR)/.make $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 $(BLDDIR)/.make $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 $(BLDDIR)/.make $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F $(BLDDIR)/.make $(MODULES) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

.PHONY: clean
clean:
	@rm -f $(OBJECTS)
	@rm -f fit.log *.dat

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
else
	@rm -f $(OUTDIR)/$(OUTNAME) $(OUTDIR)/blas $(OUTDIR)/dispatched $(OUTDIR)/inlined $(OUTDIR)/specialized
endif
else
	@rm -f $(OUTDIR)/$(OUTNAME) $(OUTDIR)/blas $(OUTDIR)/dispatched $(OUTDIR)/inlined $(OUTDIR)/specialized
endif
	@rm -f *.gcno *.gcda *.gcov
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
	@cp $(ROOTDIR)/$(OUTNAME).sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-perf.plt $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-perf.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-plot.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-test.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/blas.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/dispatched.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/inlined.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/specialized.sh $(OUTDIR) 2> /dev/null || true

