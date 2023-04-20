CXXFLAGS += -I koboldcpp
CXXFLAGS += -I koboldcpp/examples
CXXFLAGS += -g -O0

LDFLAGS += -lboost_system
LDFLAGS += -lboost_thread
LDFLAGS += -lboost_program_options
LDFLAGS += -lpthread

#LDFLAGS += -lclblast
LDFLAGS += -lOpenCL
LDFLAGS += -lopenblas

$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS: $(LDFLAGS))

all: apiserver

llama:
	cd koboldcpp && make main LLAMA_CLBLAST=1 && cd ..

apiserver: llama apiserver.cpp
	$(CXX) $(CXXFLAGS) apiserver.cpp koboldcpp/ggml.o koboldcpp/llama.o koboldcpp/common.o -o apiserver $(LDFLAGS)

clean:
	rm -f apiserver
	cd koboldcpp && make clean && cd ..
