cmake_minimum_required(VERSION 3.17)
project(neuralnet C)

set(CMAKE_C_STANDARD 11)
include_directories(visualizer/ src/)

add_executable(neuralnet
        src/main.c
        src/nn.c
        src/nn.h
        src/data.c
        src/data.h)

target_link_libraries(neuralnet PRIVATE m)
