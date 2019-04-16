#ifndef V4L2_CAPTURE_H
#define V4L2_CAPTURE_H

#include <linux/videodev2.h>
#include <cstdio>
#include <cstdint>

#define DEBUG_V4L2_CAPTURE
#ifdef  DEBUG_V4L2_CAPTURE

    #define LOGD(...)   ((void)fprintf(stdout, __VA_ARGS__))
    #define LOGI(...)   ((void)fprintf(stdout, __VA_ARGS__))
    #define LOGW(...)   ((void)fprintf(stdout, __VA_ARGS__))
    #define LOGE(...)   ((void)fprintf(stderr, __VA_ARGS__))
#else
    #define LOGD(...)
    #define LOGI(...)
    #define LOGW(...)
    #define LOGE(...)
#endif

#define MAX_CAMERAS         10
#define MAX_BUFFERS         8
#define DEFAULT_BUFFERS     4
#define DEFAULT_WIDTH       640
#define DEFAULT_HEIGHT      480
#define DEFAULT_FPS         30

class V4L2Capture {
public:
    enum {
        CAP_PROP_FORMAT,
        CAP_PROP_FRAME_WIDTH,
        CAP_PROP_FRAME_HEIGHT,
        CAP_PROP_FPS,
        CAP_PROP_BUFFERSIZE
    };

    V4L2Capture();
    V4L2Capture(int index);
    V4L2Capture(const char* deviceName);
    ~V4L2Capture();
    bool isOpened();
    bool open(int index=-1);
    bool open(const char* deviceName);
    bool set(int property, double value);
    double get(int property);
private:
    bool init_capture();
    // TODO: used for getProperty()/setProperty()
    bool reset_capture();
    bool try_capability();
    bool try_ioctl(unsigned long ioctlCode, void* parameter) const;
    bool try_enum_formats();
    bool try_set_format();
    bool try_set_fps();
    bool try_set_streaming(bool on);
    bool try_request_buffers(uint32_t numBuffers);
    bool try_create_buffers();
    bool try_release_buffers();
    bool try_queue_buffer(uint32_t index);
    bool try_dequeue_buffer(uint32_t& index);

private:
    bool opened = false;
    int deviceHandle = -1;
    uint32_t pixelformat;
    uint32_t width  = DEFAULT_WIDTH;
    uint32_t height = DEFAULT_HEIGHT;
    uint32_t fps = DEFAULT_FPS;
    // bufferSize is requested nums of buffers
    uint32_t requestBuffers = DEFAULT_BUFFERS;
    // numBuffers is the nums of allocated buffers
    uint32_t numBuffers = 0;
    int32_t bufferIndex = -1;
    struct Buffer {
        void *  start = 0;
        size_t  length = 0;
        // This is dequeued buffer. It used for to put it back in the queue.
        // The buffer is valid only if bufferIndex >= 0
        v4l2_buffer buffer = v4l2_buffer();
    } buffers[MAX_BUFFERS];

};

#endif // V4L2_CAPTURE_H