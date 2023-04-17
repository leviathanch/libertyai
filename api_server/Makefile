CXXFLAGS += -I llama.cpp
CXXFLAGS += -I llama.cpp/examples
CXXFLAGS += -g -S0
LDFLAGS += -lboost_system
LDFLAGS += -lboost_thread
LDFLAGS += -lboost_program_options
LDFLAGS += -lpthread
LDFLAGS += -lclblast

$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS: $(LDFLAGS))

all: apiserver

llama:
	cd llama.cpp && make LDFLAGS="$(LDFLAGS)" LLAMA_OPENBLAS=1 && cd ..

apiserver: llama apiserver.cpp
	$(CXX) $(CXXFLAGS) apiserver.cpp llama.cpp/ggml.o llama.cpp/llama.o llama.cpp/common.o -o apiserver $(LDFLAGS)

clean:
	rm -f apiserver
	cd llama.cpp && make clean && cd ..
