CXX = g++
CXXFLAGS = -std=c++20 -g
TARGET = bin/main
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

run: $(TARGET)
	@echo "Running $(TARGET):\n"
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS) src/*.o

.PHONY: all clean run
