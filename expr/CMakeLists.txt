CPMAddPackage("gh:fmtlib/fmt#7.1.3")
CPMAddPackage(
  NAME CLI11
  VERSION 2.5.0
  GITHUB_REPOSITORY CLIUtils/CLI11
  OPTIONS
    "CLI11_INSTALL ON"
)
CPMAddPackage(
  NAME fkYAML
  VERSION 0.4.2
  GITHUB_REPOSITORY fktn-k/fkYAML
  OPTIONS
    "FK_YAML_INSTALL ON"
)

find_package(expr-utils REQUIRED)
add_executable(expr main.cc ../src/BoW3D.cpp ../src/Frame.cpp ../src/LinK3D_Extractor.cpp)
target_include_directories(expr PRIVATE ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(expr PRIVATE ${OpenCV_LIBS} ${PCL_LIBRARIES} Eigen3::Eigen expr-utils::dataloader
  fmt::fmt fkYAML::fkYAML CLI11::CLI11)

install(TARGETS expr DESTINATION lib/${PROJECT_NAME})
