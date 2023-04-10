CXXFLAGS += -I llama.cpp
CXXFLAGS += -I llama.cpp/examples
LDFLAGS += -lboost_system
LDFLAGS += -lboost_thread

$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS: $(LDFLAGS))

all: apiserver

apiserver:
	make -f llama.cpp/Makefile
	$(CXX) $(CXXFLAGS) apiserver.cpp llama.cpp/ggml.o llama.cpp/llama.o llama.cpp/common.o -o apiserver $(LDFLAGS)

clean:
	rm -f apiserver
	make -f llama.cpp/Makefile clean
