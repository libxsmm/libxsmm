INCLUDE_PATH := include
SOURCE_PATH := src/knc
LIB_PATH := lib

SOURCES=$(wildcard $(SOURCE_PATH)/*.c)
OBJECTS=$(SOURCES:.c=.o)

knc: $(OBJECTS)
	xiar -r -qoffload-build $(LIB_PATH)/libxsmmknc.a $(OBJECTS) 

.c.o:
	icc -offload-attribute-target=mic -mkl=sequential -std=c99 -I$(INCLUDE_PATH) -c $<

tests:
	icc -offload-attribute-target=mic -mkl=sequential -I$(INCLUDE_PATH) -c benchmark.cpp
	icc -offload-attribute-target=mic -mkl=sequential benchmark.o $(LIB_PATH)/libxsmmknc.a

all: knc tests

