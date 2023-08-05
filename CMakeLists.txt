cmake_minimum_required(VERSION 3.21)
project(VideoPlayer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(NOT DEFINED CMAKE_BUILD_TYPE)
 set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED QT_INSTALL_PREFIX)
 set(QT_INSTALL_PREFIX "/Users/xiaopanzhang/Qt/6.2.4/macos")
 message("QT_INSTALL_PREFIX not defined, using the default setting")
endif()

LIST(APPEND CMAKE_PREFIX_PATH ${QT_INSTALL_PREFIX})


find_package(Qt6 COMPONENTS
        Core
        Gui
        Widgets
        Multimedia
        REQUIRED)

add_executable(VideoPlayer main.cpp WavReader.h WavReader.cpp)
target_link_libraries(VideoPlayer
        Qt::Core
        Qt::Gui
        Qt::Widgets
        Qt6::Multimedia
        )

if(BUILD_READ_RGB)
find_package( OpenCV CONFIG REQUIRED )
find_package(OpenMP)

add_executable(read_rgb read_rgb.cpp detect_nfl.h detect_nfl.cpp detect_mcdonald.h detect_mcdonald.cpp)
target_include_directories(read_rgb PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(read_rgb  ${OpenCV_LIBS} OpenMP::OpenMP_CXX Qt::Core Qt6::Multimedia)

add_executable(read_rgb_d2 read_rgb_d2.cpp detect_nfl.h detect_nfl.cpp detect_mcdonald.h detect_mcdonald.cpp)
target_include_directories(read_rgb_d2 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(read_rgb_d2  ${OpenCV_LIBS} OpenMP::OpenMP_CXX Qt::Core Qt6::Multimedia)

add_executable(read_rgb_d1 read_rgb_d1.cpp detect_subway.h detect_subway.cpp detect_starbuck.h detect_starbuck.cpp)
target_include_directories(read_rgb_d1 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(read_rgb_d1  ${OpenCV_LIBS} OpenMP::OpenMP_CXX Qt::Core Qt6::Multimedia)

add_executable(read_rgb_d11 read_rgb_d11.cpp detect_subway_v2.h detect_subway_v2.cpp detect_starbuck.h detect_starbuck.cpp)
target_include_directories(read_rgb_d11 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(read_rgb_d11  ${OpenCV_LIBS} OpenMP::OpenMP_CXX Qt::Core Qt6::Multimedia)

add_executable(read_rgb_d3 read_rgb_d3.cpp detect_hrc.h detect_hrc.cpp detect_ae.h detect_ae.cpp)
target_include_directories(read_rgb_d3 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(read_rgb_d3  ${OpenCV_LIBS} OpenMP::OpenMP_CXX Qt::Core Qt6::Multimedia)

add_executable(read_rgb_dt read_rgb_dt.cpp detect_subway.h detect_subway.cpp detect_starbuck.h detect_starbuck.cpp)
target_include_directories(read_rgb_dt PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(read_rgb_dt  ${OpenCV_LIBS} OpenMP::OpenMP_CXX Qt::Core Qt6::Multimedia)

add_executable(play_rgb play_rgb.cpp )
target_include_directories(play_rgb PUBLIC ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(play_rgb ${OpenCV_LIBS} OpenMP::OpenMP_CXX Qt::Core Qt6::Multimedia)
endif(BUILD_READ_RGB)

