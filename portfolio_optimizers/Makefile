# Makefile — portfolio_optimizers/
# Kompilacija: make
# Čiščenje:    make clean

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall
TARGET   = portfolio_optimizer

SRCS = main.cpp

$(TARGET): $(SRCS) portfolio_common.h pso.cpp sa.cpp
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
