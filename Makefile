.PHONY: header library all usage exporters build_dir

default: all

CC = clang++
STD = c++17

QUOM = ../quom/proc

build_dir:
	mkdir -p build

header: build_dir
	@$(QUOM) -I . frontend/libmori.hpp build/libmori.hpp

library: build_dir
	@$(CC) -I . -std=$(STD) -shared -fPIC -o build/libmori.so backend/backend_entry.cpp

exporters: build_dir
	@$(MAKE) -f exporters/Makefile -e CC='${CC}' STD='${STD}'

all: header library exporters
	@$(CC) -I . -std=$(STD) main.cpp -Lbuild -lmori -o main

clean:
	@rm -rf build
	@rm -rf main main.dSYM

usage:
	@echo "Usages:"
	@echo "  CFLAGS=-Dmacros"
	@echo
	@echo "Macros:"
	@echo "  ENABLE_EXTERNAL_BACKEND: Enable external backend support. Including Dylib / Shared Memory / Unix Socket / HTTP backends."