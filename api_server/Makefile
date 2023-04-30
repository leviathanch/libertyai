CXXFLAGS += -I koboldcpp
CXXFLAGS += -I koboldcpp/examples
CXXFLAGS += -g -O0

LDFLAGS += -lboost_system
LDFLAGS += -lboost_thread
LDFLAGS += -lboost_program_options
LDFLAGS += -lpthread

LDFLAGS_OPENCL = LDFLAGS
LDFLAGS_CUDA = LDFLAGS

LDFLAGS_OPENCL += -lOpenCL -lclblast
LDFLAGS_CUDA += -L/usr/local/cuda/lib64 -lcublas -lculibos -lcudart_static -lcublasLt

$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS: $(LDFLAGS))

llama_clblast:
	cd koboldcpp && make main LLAMA_CLBLAST=1 LLAMA_ACCELERATE=1 && cd ..

apiserver_opencl: llama_clblast apiserver.cpp
	$(CXX) $(CXXFLAGS) apiserver.cpp koboldcpp/ggml.o koboldcpp/ggml-opencl.o koboldcpp/llama.o koboldcpp/common.o -o apiserver $(LDFLAGS_OPENCL)

llama_cuda:
	cd koboldcpp && make main LLAMA_CUBLAS=1 LLAMA_ACCELERATE=1 && cd ..

apiserver_cuda: llama_cuda apiserver.cpp
	$(CXX) $(CXXFLAGS) apiserver.cpp koboldcpp/ggml.o koboldcpp/ggml-cuda.o koboldcpp/llama.o koboldcpp/common.o -o apiserver $(LDFLAGS_CUDA)

clean:
	rm -f apiserver
	cd koboldcpp && make clean && cd ..
