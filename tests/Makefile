# Export all variables to sub-make processes.
#.EXPORT_ALL_VARIABLES: #export

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explicitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(MAKE_PARALLEL))
ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif
endif

ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ..
SRCDIR = $(ROOTDIR)
INCDIR = .
BLDDIR = build
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(INCDIR) -I$(DEPDIR)/include -I$(DEPDIR)/src

BLAS ?= 1
OMP ?= 1
SYM ?= 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

OUTNAME := $(shell basename $(ROOTDIR))
HEADERS := $(shell ls -1 $(INCDIR)/*.h   2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(INCDIR)/*.hpp 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(INCDIR)/*.hxx 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(INCDIR)/*.hh  2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.h   2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.hpp 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.hxx 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.hh  2> /dev/null | tr "\n" " ")
CPPSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cpp 2> /dev/null | tr "\n" " ")
CPPSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cpp 2> /dev/null | tr "\n" " ")
CXXSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cxx 2> /dev/null | tr "\n" " ")
CXXSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cxx 2> /dev/null | tr "\n" " ")
CCXSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cc  2> /dev/null | tr "\n" " ")
CCXSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cc  2> /dev/null | tr "\n" " ")
CSOURCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.c   2> /dev/null | tr "\n" " ")
CSOURCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.c   2> /dev/null | tr "\n" " ")
FXXSRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.f   2> /dev/null | tr "\n" " ")
FXXSRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.f   2> /dev/null | tr "\n" " ")
F77SRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.F   2> /dev/null | tr "\n" " ")
F77SRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.F   2> /dev/null | tr "\n" " ")
F90SRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.f90 2> /dev/null | tr "\n" " ") \
           $(shell grep -L $(FMAIN) $(SRCDIR)/*.F90 2> /dev/null | tr "\n" " ")
F90SRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.f90 2> /dev/null | tr "\n" " ") \
           $(shell grep -l $(FMAIN) $(SRCDIR)/*.F90 2> /dev/null | tr "\n" " ")
MODULES := $(addsuffix    .mod,$(basename $(FXXSRCS) $(F77SRCS) $(F90SRCS))) \
           $(addsuffix .modmic,$(basename $(FXXSRCS) $(F77SRCS) $(F90SRCS)))
OBJECTS := $(call objname, $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS))
OBJECTX := $(call objname, $(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX))
FTNOBJS := $(call objname, $(FXXSRCS) $(F77SRCS) $(F90SRCS))
FTNOBJX := $(call objname, $(FXXSRCX) $(F77SRCX) $(F90SRCX))
XFILES  := $(addprefix $(OUTDIR)/,$(basename $(notdir \
           $(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX) \
           $(FXXSRCX) $(F77SRCX) $(F90SRCX))))

.PHONY: all
all: $(XFILES)

.PHONY: tests
tests: test

.PHONY: test
test: $(OUTDIR)/.make $(OUTDIR)/test.sh $(XFILES)
	@$(OUTDIR)/test.sh

define DEFINE_LINK_LD_RULE
$(1): $(2) $(call objname, $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)) $(LIBNAME).$(LIBEXT) $(dir $(1))/.make
	$(LD) -o $(1) $(2) \
		$(call objname, $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)) \
		$(call libdir, $(LIBNAME).$(LIBEXT)) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
endef

define DEFINE_LINK_FC_RULE
ifneq (,$(strip $(FC)))
$(1): $(2) $(call objname, $(FXXSRCS) $(F77SRCS) $(F90SRCS)) $(LIBNAME)f.$(LIBEXT) $(LIBNAME).$(LIBEXT) $(dir $(1))/.make
	$(FC) -o $(1) $(2) \
		$(call objname, $(FXXSRCS) $(F77SRCS) $(F90SRCS)) \
		$(call libdir, $(LIBNAME)f.$(LIBEXT)) \
		$(call libdir, $(LIBNAME).$(LIBEXT)) \
		$(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
else
.PHONY: $(1)
endif
endef

$(foreach SRC, $(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX),$(eval $(call DEFINE_LINK_LD_RULE, \
  $(basename $(notdir $(SRC))), $(call objname, $(SRC)))))

$(foreach SRC, $(FXXSRCX) $(F77SRCX) $(F90SRCX),$(eval $(call DEFINE_LINK_FC_RULE, \
  $(basename $(notdir $(SRC))), $(call objname, $(SRC)))))

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

.PHONY: clean-minimal
clean-minimal:
	@rm -f $(OBJECTS) $(OBJECTX)
	@rm -f $(FTNOBJS) $(FTNOBJX)
	@rm -f fit.log *.dat

.PHONY: clean
clean: clean-minimal
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
else
	@rm -f $(XFILES)
endif
else
	@rm -f $(XFILES)
endif
	@rm -f $(MODULES)
	@rm -f *.gcno *.gcda *.gcov
	@rm -f $(OUTDIR)/libxsmm.$(DLIBEXT)
	@rm -f $(OUTDIR)/*.bin
	@rm -f .make .state

