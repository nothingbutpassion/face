#include <cstdlib>
#include <cstdio>
#include "v4l2_capture.h"

int main(int argc, char* argv[])
{
    if (argc > 3) {
        LOGE("Usage: %s [output file] [output frames(defult=100)]\n", argv[0]);
        return  -1;
    }

    V4L2Capture capture;
    if (!capture.open()) {
        LOGE("failed to open video capture\n");
        return -1;
    }

    int32_t pixelformat = (int32_t) capture.get(V4L2Capture::GAP_PROP_FOURCC);
    int32_t width = (int32_t) capture.get(V4L2Capture::CAP_PROP_FRAME_WIDTH);
    int32_t height = (int32_t) capture.get(V4L2Capture::CAP_PROP_FRAME_HEIGHT);
    int32_t bufferSize = (int32_t) capture.get(V4L2Capture::CAP_PROP_BUFFERSIZE);
    int32_t fps = (int32_t) capture.get(V4L2Capture::CAP_PROP_FPS);
    LOGI("default properties: format=%c%c%c%c, width=%u, height=%u, buffers=%u, fps=%u\n",
         pixelformat&0xff, (pixelformat>>8)&0xff, (pixelformat>>16)&0xff, (pixelformat>>24)&0xff,
         width, height, bufferSize, fps);

    capture.set(V4L2Capture::GAP_PROP_FOURCC, v4l2_fourcc('R','G','B','3'));
    capture.set(V4L2Capture::CAP_PROP_FRAME_WIDTH, 320);
    capture.set(V4L2Capture::CAP_PROP_FRAME_HEIGHT, 240);
    capture.set(V4L2Capture::CAP_PROP_BUFFERSIZE, 4);
    capture.set(V4L2Capture::CAP_PROP_FPS, 20);
    pixelformat = (int32_t) capture.get(V4L2Capture::GAP_PROP_FOURCC);
    width = (int32_t) capture.get(V4L2Capture::CAP_PROP_FRAME_WIDTH);
    height = (int32_t) capture.get(V4L2Capture::CAP_PROP_FRAME_HEIGHT);
    bufferSize = (int32_t) capture.get(V4L2Capture::CAP_PROP_BUFFERSIZE);
    fps = (int32_t) capture.get(V4L2Capture::CAP_PROP_FPS);
    LOGI("now properties: format=%c%c%c%c, width=%u, height=%u, buffers=%u, fps=%u\n",
         pixelformat&0xff, (pixelformat>>8)&0xff, (pixelformat>>16)&0xff, (pixelformat>>24)&0xff,
         width, height, bufferSize, fps);

    const char* outFile =  argc > 1 ? argv[1] : nullptr;
    int outFrames = argc > 2 ? atoi(argv[2]) : 100;
    if (!outFile)
        return 0;

    FILE* fp = fopen(outFile, "wb");
    if (!fp) {
        LOGE("can't open %s\n", outFile);
        return -1;
    }
    for (int i=0; i < outFrames; ++i) {
        void* frame = nullptr;
        uint32_t lenght = 0;
        if (!capture.read(&frame, &lenght)) {
            LOGE("read video frame error\n");
            break;
        }
        fwrite(frame, 1, lenght, fp);
    }
    fclose(fp);
    return 0;
}


