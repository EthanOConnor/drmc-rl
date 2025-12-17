CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
LDFLAGS = 

TARGET = drmario_engine
MONITOR = monitor
SRCS = main.cpp GameLogic.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET) $(MONITOR)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(MONITOR): monitor.cpp
	$(CXX) $(CXXFLAGS) monitor.cpp -o $(MONITOR) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
