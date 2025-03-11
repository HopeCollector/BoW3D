find_package(expr-utils REQUIRED)
add_executable(expr main.cc ../src/BoW3D.cpp ../src/Frame.cpp ../src/LinK3D_Extractor.cpp)
target_include_directories(expr PRIVATE ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(expr PRIVATE ${OpenCV_LIBS} ${PCL_LIBRARIES} Eigen3::Eigen expr-utils::dataloader)

install(TARGETS expr DESTINATION lib/${PROJECT_NAME})
