#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "tflite_wrapper.h"
#include "../../../../../opencv-4.3.0-android-sdk/sdk/native/jni/include/opencv2/core/mat.hpp"
#include "../../../../../opencv-4.3.0-android-sdk/sdk/native/jni/include/opencv2/core/types.hpp"

using namespace std;
using namespace cv;

namespace {
    void showUsage(const char* app) {
        LOGE("Usage: %s -m <model-file>  [-d <data-dir>] [-t <thread-num>] [-r <repeated-times>] [-n] [-g] [-f <gpu-flags>]\n"
             "       wherein:\n"
             "           -n  use NNAPI\n"
             "           -g  use GPU\n"
             "           -f  GPU flags (bitmask) : 0 - none, 1 - enable quant, 2 - only use OpenCL, 4 - only use OpenGL"
        , app);
    }

    vector<string> listFiles(const char* dirName) {
        vector<string> files;
        DIR* dir = opendir(dirName);
        if (!dir) {
            LOGE("can't open directory: %s", dir);
            return files;
        }
        dirent* ent = nullptr;
        while ((ent = readdir(dir)) != nullptr) {
            if (ent->d_name != string(".") && ent->d_name != string("..")) {
                files.push_back(ent->d_name);
            }
        }
        closedir(dir);
        return files;
    }

    void dumpOutput(TfLiteWrapper& tflite, const char* prefix) {
        int N =  tflite.outputBytes(0)/sizeof(float);
        float* result = new float[N];
        tflite.getOutput(0, result, N*sizeof(float));
        string str;
        for (int i=0; i < N; i++) {
            if (i != 0)
                str += ", ";
            str += std::to_string(result[i]);
        }
        delete result;
        LOGI("%s%s", prefix, str.c_str());
    }

    void testBufferCopying(TfLiteWrapper& tflite, int repeatedTimes) {
        LOGI("Test input buffer copying ...");
        for (int k = 0; k < tflite.inputCount(); ++k) {
            double duration = 0;
            int size = tflite.inputBytes(k);
            void * buf = operator new(size);
            for (int i = 0; i < repeatedTimes; ++i) {
                int64_t t = getTickCount();
                tflite.setInput(k, buf, size);
                double d = double(getTickCount()-t)/getTickFrequency();
                tflite.forward();
                duration += d;
                LOGI("copying %d bytes for input buffer %d: %dms", size, k, int(1000*d));
            }
            operator delete(buf);
            LOGI("Average duration for copying input buffer %d: %dms", k, int(1000*duration/repeatedTimes));
        }

        LOGI("Test output buffer copying ...");
        for (int k = 0; k < tflite.inputCount(); ++k) {
            double duration = 0;
            int size = tflite.outputBytes(k);
            void * buf = operator new(size);
            for (int i = 0; i < repeatedTimes; ++i) {
                int64_t t = getTickCount();
                tflite.getOutput(k, buf, size);
                double d = double(getTickCount()-t)/getTickFrequency();
                tflite.forward();
                duration += d;
                LOGI("copying %d bytes for output buffer %d: %dms", size, k, int(1000*d));
            }
            operator delete(buf);
            LOGI("Average duration for copying output buffer %d: %dms", k, int(1000*duration/repeatedTimes));
        }
    }
}


int main(int argc, char* argv[])
{
    int opt;
    const char* app = argv[0];
    const char* modelFile = nullptr;
    const char* dataDir = nullptr;
    int numThreads = 0;
    int repeatedTimes = 100;
    int gpuFlags = 0;
    bool useGPU = false;
    bool useNNAPI = false;
    while ((opt=getopt(argc, argv, "m:d:t:r:ngf:h")) != -1) {
        switch (opt) {
            case 'm':
                modelFile = optarg;
                break;
            case 'd':
                dataDir = optarg;
                break;
            case 't':
                numThreads = atoi(optarg);
                break;
            case 'r':
                repeatedTimes = atoi(optarg);
                break;
            case 'n':
                useNNAPI = true;
                break;
            case 'g':
                useGPU = true;
                break;
            case 'f':
                gpuFlags = atoi(optarg);
                break;
            case 'h':
                showUsage(app);
                return 0;
            default:
                showUsage(app);
                return -1;
        }
    }

    if (!modelFile) {
        showUsage(app);
        return -1;
    }

    vector<string> files;
    vector<Mat> imgs;
    if (dataDir) {
        files = listFiles(dataDir);
        if (files.size() == 0) {
            LOGE("no files found in %s", dataDir);
            return -1;
        }
        for (auto file: files) {
            string dir(dataDir);
            if (dir[dir.size()-1] == '/')
                dir = dir.substr(0, dir.size()-1);
            string path = dir + "/" + file;
            Mat img = imread(path, IMREAD_GRAYSCALE);
            if (!img.data) {
                LOGE("can't decode image file: %s", path.c_str());
                return -1;
            }
            imgs.push_back(img);
        }
    }


    TfLiteWrapper tflite;
    if (!tflite.load(modelFile, numThreads, useNNAPI, useGPU, gpuFlags))
        return -1;

    LOGI("Test forward inference ...");
    double duration = 0;
    if (imgs.size() > 0) {
        repeatedTimes = imgs.size();
        Size size(tflite.inputWidth(0), tflite.inputHeight(0));
        if (tflite.inputCount() != 1 || tflite.inputBytes(0) != size.width*size.height*sizeof(float)) {
            LOGE("only support one input tensor with one channel");
            return -1;
        }
        for (int i=0; i < imgs.size(); ++i) {
            Mat img = imgs[i];
            Mat input;
            resize(img, img, size);
            Scalar mean;
            Scalar stddev;
            meanStdDev(img, mean, stddev);
            stddev[0] += 1e-7;
            img.convertTo(input, CV_32F, 1./stddev[0], -mean[0]/stddev[0]);
            tflite.setInput(0, input.data, tflite.inputBytes(0));
            int64_t t = getTickCount();
            tflite.forward();
            double d = double(getTickCount()-t)/getTickFrequency();
            duration += d;
            LOGI("input:  %s", files[i].c_str());
            dumpOutput(tflite, "output: ");
        }
    } else {
        for (int i=0; i < repeatedTimes; ++i) {
            tflite.setRandomInput();
            int64_t t = getTickCount();
            tflite.forward();
            double d = double(getTickCount()-t)/getTickFrequency();
            duration += d;
            LOGI("forward duration: %dms", int(1000*d));
        }
    }
    LOGI("Average forward duration: %dms", int(1000*duration/repeatedTimes));
    // testBufferCopying(tflite, repeatedTimes);
    return 0;
}

