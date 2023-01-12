cmake_minimum_required(VERSION 3.14.0)

# set the project name
project(mori)

# set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add the executable
add_library(libmori, backend/basic_backend.cpp)