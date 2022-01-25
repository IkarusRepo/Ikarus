cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(IkarusTests LANGUAGES C CXX)

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

set(IWYU_ARGS "-Xiwyu --mapping_file=iwyu.imp")
include(../cmake/tools.cmake)

list(APPEND CMAKE_PREFIX_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/../../Ikarus_Dependencies/Dependencies_release/
)

include_directories(../libAddons/eigen/eigenAddons)

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

#find_package(GTest CONFIG REQUIRED)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.0.0-preview3
)

FetchContent_MakeAvailable(Catch2)

add_executable(IkarusTests ${sources})

target_link_libraries(
  IkarusTests
  PRIVATE Catch2::Catch2WithMain
  PUBLIC Ikarus::Ikarus
)
set_target_properties(IkarusTests PROPERTIES CXX_STANDARD 20)

target_compile_options(
  IkarusTests PUBLIC -Wall -Wpedantic -Wextra -Werror
                     -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable
)
target_compile_options(IkarusTests PUBLIC -fconcepts-diagnostics-depth=5)

if(MINGW)
  target_compile_options(IkarusTests PUBLIC -Wa,-mbig-obj -ftrack-macro-expansion=0)
endif()

target_compile_features(IkarusTests PRIVATE cxx_std_20)

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(IkarusTests)

if(ENABLE_TEST_COVERAGE)
  message("Enable Test Coverage")
  target_compile_options(IkarusTests PUBLIC -O0 -fprofile-arcs -ftest-coverage -fPIC)
  target_link_options(IkarusTests PUBLIC -fprofile-arcs -ftest-coverage)
endif()
