.PHONY: header library all usage

default: all

CC = g++

header:
	@$(CC) -DSINGLE_HEADER_LIBRARY $(CFLAGS) -E frontend/libmori.hpp | grep -v "# " > libmori.hpp_definations
	@cat includes/stdlibs.hpp | grep -v "#define SINGLE_HEADER_LIBRARY" > libmori.hpp_includes
	@cat libmori.hpp_definations >> libmori.hpp_includes
	@mv libmori.hpp_includes libmori.hpp
	@rm -rf libmori.hpp_definations

library:
	@$(CC) -std=c++17 -shared -fPIC -o libmori.so backend/basic_backend.cpp

all: header library
	@$(CC) -std=c++17 main.cpp -L. -lmori

clean:
	@rm -rf libmori.so libmori.hpp a.out

usage:
	@echo "Usages:"
	@echo "  CFLAGS=-Dmacros"
	@echo
	@echo "Macros"
	@echo "  ENABLE_INTEGRATED_BACKEND"