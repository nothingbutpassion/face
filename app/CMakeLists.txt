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
# Set OPENCV_INC_DIR/OPENCV_LIB as OpenCV include directory/lib path
# ${ANDROID_ABI} is set by gradle. see */cmake/<debug or release>/<abi>/cmake_build_command.txt
set(OPENGC_DIR "D:/share/opencv-4.5.1-android-sdk")
set(OPENCV_INC_DIR "${OPENGC_DIR}/sdk/native/jni/include")
set(OPENCV_LIB "${OPENGC_DIR}/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so")
add_library(opencv SHARED IMPORTED )
set_target_properties(opencv PROPERTIES IMPORTED_LOCATION ${OPENCV_LIB})

# NOTES:
# We directly use dlib source, so set the dlib root dir for finding
set(DLIB_DIR "D:/share/dlib-19.21")

# NOTES:
# For tensorflow lite
set(TFLITE_DIR "D:/share/tensorflow-lite-2.9.0")
set(TFLITE_INC_DIR  "${TFLITE_DIR}/headers")
set(TFLITE_LIB "${TFLITE_DIR}/jni/${ANDROID_ABI}/libtensorflowlite_jni.so")
add_library(tensorflowlite SHARED IMPORTED )
set_target_properties(tensorflowlite PROPERTIES IMPORTED_LOCATION ${TFLITE_LIB})
# For tensoflow lite gpu delegate
set(TFLITE_GPU_DIR "D:/share/tensorflow-lite-gpu-2.9.0")
set(TFLITE_GPU_INC_DIR  "${TFLITE_GPU_DIR}/headers")
set(TFLITE_GPU_LIB "${TFLITE_GPU_DIR}/jni/${ANDROID_ABI}/libtensorflowlite_gpu_jni.so")
add_library(tensorflowlite_gpu SHARED IMPORTED )
set_target_properties(tensorflowlite_gpu PROPERTIES IMPORTED_LOCATION ${TFLITE_GPU_LIB})

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
add_library(face SHARED
        src/main/cpp/native_buffer_jni.cpp
        src/main/cpp/image_processor_jni.cpp
        src/main/cpp/image_processor.cpp
        src/main/cpp/hog_face_detector.cpp
        src/main/cpp/tflite_calling_classifier.cpp
        src/main/cpp/tflite_smoking_classifier.cpp
        src/main/cpp/resnet_face_descriptor.cpp
        src/main/cpp/kazemi_face_landmark.cpp
        src/main/cpp/simple_pose_estimator.cpp
        src/main/cpp/utils.cpp
        ${DLIB_DIR}/dlib/all/source.cpp        # for dlib
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
target_include_directories(
        face PRIVATE
        ${OPENCV_INC_DIR}
        ${DLIB_DIR}                          # for dlib includes
        ${TFLITE_INC_DIR}                     # for tensorflow lite
        ${TFLITE_GPU_INC_DIR}                 # for tensorflow lite gpu delegate
        )

target_compile_definitions(
        face PRIVATE
        -DDLIB_NO_GUI_SUPPORT                 #  for building dlib with no GUI support
        )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.
target_link_libraries(
        face
        android
        log
        opencv
        tensorflowlite
        tensorflowlite_gpu
        )

add_custom_command(
        TARGET face PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/models/kazemi_face_landmark.dat
                ${CMAKE_CURRENT_SOURCE_DIR}/src/main/assets/kazemi_face_landmark.dat
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/models/smoking_classifier.tflite
                ${CMAKE_CURRENT_SOURCE_DIR}/src/main/assets/smoking_classifier.tflite
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/models/calling_classifier.tflite
                ${CMAKE_CURRENT_SOURCE_DIR}/src/main/assets/calling_classifier.tflite
        )


add_executable(
        test_tflite
        src/main/cpp/test_tflite_inference.cpp
        src/main/cpp/tflite_wrapper.cpp
        )
target_include_directories(test_tflite PRIVATE
        ${OPENCV_INC_DIR}
        ${TFLITE_INC_DIR}
        ${TFLITE_GPU_INC_DIR}
        )
target_link_libraries(
        test_tflite
        log
        z
        mediandk
        jnigraphics
        EGL
        GLESv3
        opencv
        tensorflowlite
        tensorflowlite_gpu
        )
set_target_properties(
        test_tflite PROPERTIES RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/build/outputs/bin/${CMAKE_BUILD_TYPE}"
        )

add_executable(
        test_v4l2
        src/main/cpp/test_v4l2_capture.cpp
        src/main/cpp/v4l2_capture.cpp
        )
set_target_properties(
        test_v4l2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/build/outputs/bin/${CMAKE_BUILD_TYPE}"
        )