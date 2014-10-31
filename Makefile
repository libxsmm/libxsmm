
# Use ROW_MAJOR matrix representation if set to 1, COL_MAJOR otherwise 
ROW_MAJOR ?= 1

# M, N, K values of the generated matrices
INDICES_M ?= $(shell seq 1 8)
INDICES_N ?= $(shell seq 1 8)
INDICES_K ?= $(shell seq 1 8)

DIR_KNC   ?= .
SCRDIR_KNC = $(DIR_KNC)/scripts
OBJDIR_KNC = $(DIR_KNC)/build
INCDIR_KNC = $(DIR_KNC)/include
SRCDIR_KNC = $(DIR_KNC)/src
LIBDIR_KNC = $(DIR_KNC)/lib

INDICES ?= $(foreach m,$(INDICES_M),$(foreach n,$(INDICES_N),$(foreach k,$(INDICES_K),$m_$n_$k)))

TARGET_COMPILE_C_KNC := icc -std=c99 -mkl=sequential -fPIC -mmic
TARGET_COMPILE_C_HST := icc -std=c99 -mkl=sequential -fPIC -offload-attribute-target=mic
AR := xiar

SRCFILES_KNC = $(patsubst %,dc_small_dnn_%.c,$(INDICES))
OBJFILES_KNC = $(patsubst %,$(OBJDIR_KNC)/mic/dc_small_dnn_%.o,$(INDICES))
OBJFILES_HST = $(patsubst %,$(OBJDIR_KNC)/intel64/dc_small_dnn_%.o,$(INDICES))

LIB_KNC  ?= $(LIBDIR_KNC)/mic/libxsmm.a
LIB_HST  ?= $(LIBDIR_KNC)/intel64/libxsmm.a
INC_KNC   = $(INCDIR_KNC)/xsmm_knc.h
MAIN_KNC  = $(SRCDIR_KNC)/xsmm_knc.c


lib_all: lib_knc lib_hst

header_knc: $(INC_KNC)
$(INC_KNC):
	@cat $(INCDIR_KNC)/xsmm_knc.0 > $@
	@python $(SCRDIR_KNC)/xsmm_knc_gensrc.py $(ROW_MAJOR) 0 $(words $(INDICES_M)) $(words $(INDICES_N)) $(INDICES_M) $(INDICES_N) $(INDICES_K) >> $@
	@echo >> $@
	@cat $(INCDIR_KNC)/xsmm_knc.1 >> $@
	@echo >> $@
	@python $(SCRDIR_KNC)/xsmm_knc_geninc.py $(ROW_MAJOR) $(words $(INDICES_M)) $(words $(INDICES_N)) $(INDICES_M) $(INDICES_N) $(INDICES_K) >> $@
	@echo >> $@
	@cat $(INCDIR_KNC)/xsmm_knc.2 >> $@

source_knc: $(addprefix $(SRCDIR_KNC)/,$(SRCFILES_KNC))
$(SRCDIR_KNC)/%.c: $(INC_KNC)
	@mkdir -p $(SRCDIR_KNC)
	@python $(SCRDIR_KNC)/xsmm_knc_gensrc.py $(ROW_MAJOR) 1 `echo $* | awk -F_ '{ print $$4" "$$5" "$$6 }'` > $@

main_knc: $(MAIN_KNC)
$(MAIN_KNC): $(INC_KNC)
	@mkdir -p $(SRCDIR_KNC)
	@python $(SCRDIR_KNC)/xsmm_knc_genmain.py $(words $(INDICES_M)) $(words $(INDICES_N)) $(INDICES_M) $(INDICES_N) $(INDICES_K) > $@

compile_knc: $(OBJFILES_KNC)
$(OBJDIR_KNC)/mic/%.o: $(SRCDIR_KNC)/%.c
	@mkdir -p $(OBJDIR_KNC)/mic
	$(TARGET_COMPILE_C_KNC) -I$(INCDIR_KNC) -c $< -o $@

compile_hst: $(OBJFILES_HST)
$(OBJDIR_KNC)/intel64/%.o: $(SRCDIR_KNC)/%.c
	@mkdir -p $(OBJDIR_KNC)/intel64
	$(TARGET_COMPILE_C_HST) -I$(INCDIR_KNC) -c $< -o $@

lib_knc: $(LIB_KNC)
ifeq ($(origin NO_MAIN), undefined)
$(LIB_KNC): $(OBJFILES_KNC) $(patsubst $(SRCDIR_KNC)/%.c,$(OBJDIR_KNC)/mic/%.o,$(MAIN_KNC))
else
$(LIB_KNC): $(OBJFILES_KNC)
endif
	@mkdir -p $(LIBDIR_KNC)/mic
	$(AR) -rs $@ $^

lib_hst: $(LIB_HST)
ifeq ($(origin NO_MAIN), undefined)
$(LIB_HST): $(OBJFILES_HST) $(patsubst $(SRCDIR_KNC)/%.c,$(OBJDIR_KNC)/intel64/%.o,$(MAIN_KNC))
else
$(LIB_HST): $(OBJFILES_HST)
endif
	@mkdir -p $(LIBDIR_KNC)/intel64
	$(AR) -rs $@ $^

clean:
	rm -rf $(SRCDIR_KNC) $(OBJDIR_KNC) $(DIR_KNC)/*~ $(DIR_KNC)/*/*~

realclean: clean
	rm -rf $(LIBDIR_KNC) $(INC_KNC)

install: lib_all clean
