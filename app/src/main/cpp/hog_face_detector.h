#ifndef HOG_FACE_DETECTOR_H
#define HOG_FACE_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

class HOGFaceDetector {
public:
    ~HOGFaceDetector();
    bool load(const std::string& modelDir);
    void detect(const cv::Mat& gray, std::vector<cv::Rect>& faces, std::vector<float>& scores, std::vector<int>* indices = nullptr);
private:
    void* mDetector = nullptr;
};

#endif // HOG_FACE_DETECTOR_H