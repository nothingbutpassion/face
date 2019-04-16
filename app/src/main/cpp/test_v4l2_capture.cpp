#include "v4l2_capture.h"

int main(int argc, char* argv[])
{
    V4L2Capture capture;
    capture.open();
    LOGD("v4l2 capture: opened=%s\n", capture.isOpened() ? "true" : "false");

    int32_t pixelformat = (int32_t) capture.get(V4L2Capture::CAP_PROP_FORMAT);
    int32_t width = (int32_t) capture.get(V4L2Capture::CAP_PROP_FRAME_WIDTH);
    int32_t height = (int32_t) capture.get(V4L2Capture::CAP_PROP_FRAME_HEIGHT);
    int32_t numBuffers = (int32_t) capture.get(V4L2Capture::CAP_PROP_BUFFERSIZE);
    int32_t fps = (int32_t) capture.get(V4L2Capture::CAP_PROP_FPS);

    LOGD("get properties: format=%c%c%c%c, width=%u, height=%u, buffers=%u, fps=%u\n",
         pixelformat&0xff, (pixelformat>>8)&0xff, (pixelformat>>16)&0xff, (pixelformat>>24)&0xff,
         width, height, numBuffers, fps);

    capture.set(V4L2Capture::CAP_PROP_FORMAT, pixelformat);
    capture.set(V4L2Capture::CAP_PROP_FRAME_WIDTH, width/2);
    capture.set(V4L2Capture::CAP_PROP_FRAME_HEIGHT, height/2);
    capture.set(V4L2Capture::CAP_PROP_BUFFERSIZE, numBuffers/2);
    capture.set(V4L2Capture::CAP_PROP_FPS, fps/2);

    pixelformat = (int32_t) capture.get(V4L2Capture::CAP_PROP_FORMAT);
    width = (int32_t) capture.get(V4L2Capture::CAP_PROP_FRAME_WIDTH);
    height = (int32_t) capture.get(V4L2Capture::CAP_PROP_FRAME_HEIGHT);
    numBuffers = (int32_t) capture.get(V4L2Capture::CAP_PROP_BUFFERSIZE);
    fps = (int32_t) capture.get(V4L2Capture::CAP_PROP_FPS);
    LOGD("get properties: format=%c%c%c%c, width=%u, height=%u, buffers=%u, fps=%u\n",
         pixelformat&0xff, (pixelformat>>8)&0xff, (pixelformat>>16)&0xff, (pixelformat>>24)&0xff,
         width, height, numBuffers, fps);
    return 0;
}


