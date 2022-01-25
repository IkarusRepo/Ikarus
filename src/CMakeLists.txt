cmake_minimum_required(VERSION 3.18)

file(
  GLOB_RECURSE sources
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS include/ikarus/*.cpp
)

project(
  Ikarus
  VERSION 1.0
  LANGUAGES C CXX
)

add_library(${PROJECT_NAME} ${sources})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

list(APPEND CMAKE_PREFIX_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/../../Ikarus_Dependencies/Dependencies_release/
)
include(../cmake/AddDependencies.cmake)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>
)

add_library(Ikarus::Ikarus ALIAS Ikarus)

include(GNUInstallDirs)
install(TARGETS Ikarus EXPORT IkarusTargets)

target_compile_options(
        Ikarus PUBLIC -Wall -Wpedantic -Wextra -Werror
        -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable
)
# install_targets(Ikarus) install(DIRECTORY include/ikarus DESTINATION include COMPONENT Devel )

# configure_and_install(${CMAKE_SCRIPTS_ROOT}/simple-config.cmake.in Ikarus SameMajorVersion)
