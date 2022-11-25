.PHONY: all

default: all

CC = clang++
STD = c++17

all:
	@$(CC) -I . -std=$(STD) -shared -fPIC -o your_dylib.so your_entry.cpp
