cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

file(GLOB programSourceFiles CONFIGURE_DEPENDS src/*.cpp)
include_directories(../libAddons/eigen/eigenAddons)
add_custom_target(AllExamples) # Create target that includes all examples
include(DuneMacros)
foreach(programSourceFile ${programSourceFiles})
  get_filename_component(programName ${programSourceFile} NAME_WLE)
  add_executable(${programName} ${programSourceFile})
  set_target_properties(${programName} PROPERTIES CXX_STANDARD 20 OUTPUT_NAME "Ikarus")
  add_dune_pythonlibs_flags(${programName})
  target_compile_options(
    ${programName} PUBLIC -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable
                          -Wno-deprecated-copy -ftemplate-backtrace-limit=0
  )

  target_link_libraries(${programName} PUBLIC Ikarus::Ikarus)
  add_dependencies(AllExamples ${programName})
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})
  target_link_libraries(${programName} ${PYTHON_LIBRARIES} PUBLIC)
endforeach()
