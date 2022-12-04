cmake_minimum_required(VERSION 3.23)
project(DBN)

set(CMAKE_CXX_STANDARD 14)
include_directories(eigen-3.4.0)

add_executable(DBN main.cpp data.cpp data.h data_handler.cpp data_handler.h dbn.cpp dbn.h dataset.cpp dataset.h)

