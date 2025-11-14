.PHONY: all main clean

CXX := g++
# Require C++17 for std::optional and generic lambdas
CXXFLAGS := -std=c++17 -O2 -I.
SOURCES := main.cc stubs.cpp
TARGET := main

all: main

main: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)