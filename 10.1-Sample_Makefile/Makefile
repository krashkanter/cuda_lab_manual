SRCS := $(wildcard *.c)
OBJS := $(patsubst %.c, %.o, $(SRCS))
CUDA_SRCS := $(wildcard *.cu)
CUDA_OBJS := $(patsubst %.cu, %.o, $(CUDA_SRCS))
CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CC := gcc
LD := gcc -fopenmp
CUDA_LIB := -L$(CUDA_PATH)/lib64 -lcublas -lcufft -lcudart
CUDA_INC += -I$(CUDA_PATH)/include
CFLAGS += -std=c99
INCLUDES := 
GPU_CARD := -arch=sm_89
NVCC_FLAGS += -O3 -dc
NVCC_FLAGS += -Xcompiler -fopenmp
CUDA_LINK_FLAGS := -dlink
EXEC := test
CUDA_LINK_OBJ := cuLink.o

all: $(EXEC)

$(EXEC): $(OBJS) $(CUDA_OBJS)
	$(NVCC) $(GPU_CARD) $(CUDA_LINK_FLAGS) -o $(CUDA_LINK_OBJ) $(CUDA_OBJS)
	$(LD) -o $@ $(OBJS) $(CUDA_OBJS) $(CUDA_LINK_OBJ) $(CUDA_LIB)

%.o : %.c
	$(CC) -o $@ -c $(CFLAGS) $(INCLUDES) $<

%.o : %.cu
	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) -o $@ -c $< $(CUDA_INC)

clean:
	rm -f $(OBJS) $(EXEC) *.o a.out

install: