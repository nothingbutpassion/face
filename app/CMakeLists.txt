# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.4.1)

# Set OpenCV_DIR as the directory contains OpenCVConfig.cmake
#set(OpenCV_DIR "D:/share/opencv_for_android/sdk/native/jni")
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(face ${OpenCV_LIBS})

# NOTES:
# Opencv shared libs can be built like the following
#/home/yuhao/cmake-3.6.3-Linux-x86_64/bin/cmake  \
#-DOPENCV_EXTRA_MODULES_PATH=/home/yuhao/opencv_contrib/modules  \
#-DCMAKE_INSTALL_PREFIX=/home/yuhao/opencv/build_for_android_arm64-v8a/install    \
#-DCMAKE_TOOLCHAIN_FILE=/mnt/hgfs/share/android-ndk-r17-beta2/build/cmake/android.toolchain.cmake \
#-DANDROID_NDK=/mnt/hgfs/share/android-ndk-r17-beta2 \
#-DANDROID_NATIVE_API_LEVEL=22   \
#-DBUILD_SHARED_LIBS=ON  \
#-DANDROID_ABI=arm64-v8a \
#-DBUILD_JAVA=OFF \
#-DBUILD_ANDROID_PROJECTS=OFF \
#-DBUILD_ANDROID_EXAMPLES=OFF \
#-DBUILD_DOCS=OFF \
#-DBUILD_PERF_TESTS=OFF \
#-DBUILD_TESTS=OFF \
#../

# NOTES:
# Set OPENCV_INC_DIR/OPENCV_LIB_DIR as OpenCV include/libs directory
# ${ANDROID_ABI} is set by gradle.
# See .externalNativeBuild/cmake/<debug or release>/<abi>/cmake_build_command.txt
set(OPENCV_INC_DIR "D:/share/opencv_for_android_${ANDROID_ABI}/sdk/native/jni/include")
set(OPENCV_LIB_DIR "D:/share/opencv_for_android_${ANDROID_ABI}/sdk/native/libs/${ANDROID_ABI}")
set(OPENCV_LIBS  opencv_core opencv_imgproc opencv_objdetect opencv_face)

foreach(name in ${OPENCV_LIBS})
    add_library(${name} SHARED IMPORTED )
    set_target_properties(${name} PROPERTIES
            IMPORTED_LOCATION ${OPENCV_LIB_DIR}/lib${name}.so)
endforeach()

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
add_library(face SHARED
        src/main/cpp/face_jni.cpp
        src/main/cpp/face_detector.cpp
        src/main/cpp/face_landmark.cpp
        src/main/cpp/utils.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.
#find_library(LOG_LIB log)
#find_library(ANDROID_LIB android)

# NOTES:
# PRIVATE is necessary, otherwise cmake build issue.
target_include_directories(face PRIVATE
        ${OPENCV_INC_DIR})

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.
target_link_libraries(face
        android
        log
        ${OPENCV_LIBS}
        )