cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Note: update this to your new project's name and version
project(
  Ikarus
  VERSION 1.0
  LANGUAGES CXX
)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

add_definitions(-Wall -Wpedantic -Wextra -Werror -m64 -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(benchmarks)
enable_testing()
add_subdirectory(tests)
add_subdirectory(cmake/FormatTarget)

if(BUILD_DOCS)
  add_subdirectory(docs/BuildLocally)
endif()
