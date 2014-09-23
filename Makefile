INCLUDE_PATH := .
SOURCE_PATH := knc
BUILD_PATH := knc
LIB_PATH := .

SOURCES=$(wildcard $(SOURCE_PATH)/*.c)
OBJECTS=$(addprefix $(BUILD_PATH)/,$($(notdir $(SOURCES)):.c=.o))

knc: $(OBJECTS)
	xiar -rs -qoffload-build $(LIB_PATH)/libxsmmknc.a $(OBJECTS) 

.c.o:
	icc -offload-attribute-target=mic -mkl=sequential -std=c99 -I$(INCLUDE_PATH) -c $<

all: knc

