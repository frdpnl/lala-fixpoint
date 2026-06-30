cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

if(MSVC)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Available configuration types to select")
endif()

option(GPU "GPU" ON)
if(GPU)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Targeted CUDA architectures")
  endif()
  project(lala-fixpoint
    HOMEPAGE_URL "https://github.com/lattice-land/lala-fixpoint"
    LANGUAGES CUDA CXX)
else()
  project(lala-fixpoint
    HOMEPAGE_URL "https://github.com/lattice-land/lala-fixpoint"
    LANGUAGES CXX)
endif()

option(WITH_ASAN "Compile with the Address Sanitizer to check for memory corruption errors" OFF)
option(LOCAL_DEPS "LOCAL_DEPS" OFF)
option(LALA_FIXPOINT_BUILD_TESTS "LALA_FIXPOINT_BUILD_TESTS" OFF)
option(LALA_FIXPOINT_BUILD_DOC "LALA_FIXPOINT_BUILD_DOC" OFF)

# Dependencies

include(FetchContent)

if(LOCAL_DEPS)
  FetchContent_Declare(cuda_battery SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../cuda-battery/")
  FetchContent_Declare(lala_interval SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-interval/")
else()
  FetchContent_Declare(
    cuda_battery
    GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
    GIT_TAG        v1.5.8
  )

  FetchContent_Declare(
    lala_interval
    GIT_REPOSITORY "https://github.com/lattice-land/lala-interval.git"
    GIT_TAG        v1.3.0
  )
endif()

FetchContent_MakeAvailable(cuda_battery lala_interval)

# Preparing the library

add_library(lala_fixpoint INTERFACE)
target_link_libraries(lala_fixpoint INTERFACE cuda_battery lala_interval)
target_include_directories(lala_fixpoint INTERFACE include)

# Benchmarking binaries

if(GPU)
  set_source_files_properties(benchmarks/minimum.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(benchmarks/bsf.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(minimum benchmarks/minimum.cpp)
target_include_directories(minimum PRIVATE include)
target_link_libraries(minimum PRIVATE cuda_battery lala_interval)
target_compile_options(minimum PRIVATE
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:SHELL:--ptxas-options=-v>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--keep$<$<BOOL:${TURBO_VERBOSE}>:;--verbose>>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>,$<CONFIG:Debug>>:-ftemplate-backtrace-limit=0>"
    # Address Sanitizer
    "$<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=address>"
    "$<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:Clang>>:-fsanitize=address>"
    "$<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:SHELL:--compiler-options -fsanitize=address>"
)

add_executable(bsf benchmarks/bsf.cpp)
target_include_directories(bsf PRIVATE include)
target_link_libraries(bsf PRIVATE cuda_battery lala_interval)
target_compile_options(bsf PRIVATE
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:SHELL:--ptxas-options=-v>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--keep$<$<BOOL:${TURBO_VERBOSE}>:;--verbose>>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>,$<CONFIG:Debug>>:-ftemplate-backtrace-limit=0>"
    # Address Sanitizer
    "$<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=address>"
    "$<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:Clang>>:-fsanitize=address>"
    "$<$<AND:$<BOOL:${WITH_ASAN}>,$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:SHELL:--compiler-options -fsanitize=address>"
)

