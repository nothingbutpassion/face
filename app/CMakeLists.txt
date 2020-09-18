# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.4.1)

# NOTES:
# Set OpenCV_DIR as the directory contains OpenCVConfig.cmake
#set(OpenCV_DIR "D:/share/opencv_for_android/sdk/native/jni")
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(face ${OpenCV_LIBS})

# NOTES:
# Opencv shared libs can be built like as follows (ubuntu 16.04 & opencv 4.0 & NDK-r17 & cmake 3.6.3)
# After building completed, copy <install>/sdk/native/jni/include & <install-dir>/sdk/native/libs to libs/opencv/
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
# ${ANDROID_ABI} is set by gradle. see */cmake/<debug or release>/<abi>/cmake_build_command.txt
#set(OPENCV_INC_DIR "Z:/opencv/build_for_android_arm64-v8a/install/sdk/native/jni/include")
#set(OPENCV_LIB_DIR "Z:/opencv/build_for_android_arm64-v8a/install/sdk/native/libs/${ANDROID_ABI}")
#set(OPENCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_dnn opencv_calib3d opencv_features2d opencv_flann)
set(OPENCV_INC_DIR "D:/share/opencv-4.3.0-android-sdk/sdk/native/jni/include")
set(OPENCV_LIB_DIR "D:/share/opencv-4.3.0-android-sdk/sdk/native/libs/${ANDROID_ABI}")
set(OPENCV_LIBS opencv_java4)

foreach(name in ${OPENCV_LIBS})
    add_library(${name} SHARED IMPORTED )
    set_target_properties(${name} PROPERTIES
            IMPORTED_LOCATION ${OPENCV_LIB_DIR}/lib${name}.so)
endforeach()

# NOTES:
# We directly use dlib source, so set the dlib root dir for finding
set(DLIB_ROOT "Z:/dlib-19.19")

# NOTES:
# tensorflow lite head files research dirs include <tensorflow-source-root-dir> and <flatb-include-dir>
set(TFLITE_INC_DIR
        "Z:/tensorflow"
        "Z:/.cache/bazel/_bazel_yuhao/e2565a7e6afe4afd7195ad0efe2016e4/external/com_google_absl"
        "Z:/.cache/bazel/_bazel_yuhao/e2565a7e6afe4afd7195ad0efe2016e4/external/flatbuffers/include"
        )
set(TFLITE_LIB "Z:/tensorflow/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/libtensorflowlite.so" )
add_library(tensorflowlite SHARED IMPORTED )
set_target_properties(tensorflowlite PROPERTIES IMPORTED_LOCATION ${TFLITE_LIB})

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
add_library(face SHARED
        src/main/cpp/native_buffer_jni.cpp
        src/main/cpp/image_processor_jni.cpp
        src/main/cpp/image_processor.cpp
        src/main/cpp/hog_face_detector.cpp
        src/main/cpp/ssd_face_detector.cpp
        src/main/cpp/tflite_action_classifier.cpp
        src/main/cpp/caffe_action_classifier.cpp
        src/main/cpp/caffe_smoking_classifier.cpp
        src/main/cpp/caffe_calling_classifier.cpp
        src/main/cpp/hao_face_detector.cpp
        src/main/cpp/resnet_face_descriptor.cpp
        src/main/cpp/kazemi_face_landmark.cpp
        src/main/cpp/simple_pose_estimator.cpp
        src/main/cpp/utils.cpp
        ${DLIB_ROOT}/dlib/all/source.cpp        # for dlib
        )

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
        ${OPENCV_INC_DIR}
        ${DLIB_ROOT}                          # for dlib includes
        ${TFLITE_INC_DIR}                     # for tensorflow lite
        )

target_compile_definitions(face PRIVATE
        -DDLIB_NO_GUI_SUPPORT                 #  for building dlib with no GUI support
        -DCV__ENABLE_C_API_CTORS              #  for fixing convertion from cv::Mat to dilb::cv_image<>
        )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.
target_link_libraries(face
        android
        log
        ${OPENCV_LIBS}
        tensorflowlite
        )

add_custom_command(
        TARGET face PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/models/kazemi_face_landmark.dat
                ${CMAKE_CURRENT_SOURCE_DIR}/src/main/assets/kazemi_face_landmark.dat
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/models/smoking_classifier.prototxt
                ${CMAKE_CURRENT_SOURCE_DIR}/src/main/assets/smoking_classifier.prototxt
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/models/smoking_classifier.caffemodel
                ${CMAKE_CURRENT_SOURCE_DIR}/src/main/assets/smoking_classifier.caffemodel

        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/models/calling_classifier.prototxt
                ${CMAKE_CURRENT_SOURCE_DIR}/src/main/assets/calling_classifier.prototxt
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/models/calling_classifier.caffemodel
                ${CMAKE_CURRENT_SOURCE_DIR}/src/main/assets/calling_classifier.caffemodel
        )