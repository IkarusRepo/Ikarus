cmake_minimum_required(VERSION 3.18)
project(Ikarus-Docs)
add_custom_target(
  site
  COMMAND xvfb-run -a mkdocs build
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
