#include <opencv2/imgproc.hpp>
#include "utils.h"
#include "face_detector.h"

#undef  LOG_TAG
#define LOG_TAG "FaceDetector"

using namespace std;
using namespace cv;

static vector<String> getOutputsNames(const dnn::Net& net) {
    static std::vector<String> names;
    if (names.empty()) {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

bool FaceDetector::load(const string& modelDir) {
    string prototxt = modelDir + "/res10_300x300_ssd_iter_140000.prototxt";
    string caffeModel = modelDir + "/res10_300x300_ssd_iter_140000.caffemodel";
    mFaceNet = dnn::readNetFromCaffe(prototxt, caffeModel);
    if (mFaceNet.empty()) {
        LOGE("failed to load model files: %s, %s", prototxt.c_str(), caffeModel.c_str());
        return false;
    }
    return true;
}

void FaceDetector::detect(const Mat& image, vector<Rect>& objects, vector<float>& scores) {
    Mat rgbImage;
    Mat blob;
    vector<Mat> outs;

    cvtColor(image, rgbImage, COLOR_RGBA2BGR);
    dnn::blobFromImage(rgbImage, blob, 1.0, Size(300, 300), Scalar(104.0,177.0,123.0));
    mFaceNet.setInput(blob);
    mFaceNet.forward(outs, getOutputsNames(mFaceNet));

    // NOTES:
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    // CV_Assert(outs.size() == 1);
    constexpr float confidenceThreshold = 0.5f;
    vector<Rect> bboxes;
    vector<float> confidences;
    float* data = (float*)outs[0].data;
    for (size_t i = 0; i < outs[0].total(); i += 7)  {
        // int classId = int(data[i + 1]);
        float confidence = data[i + 2];
        if (confidence > confidenceThreshold) {
            int left = (int)(data[i + 3] * image.cols);
            int top = (int)(data[i + 4] * image.rows);
            int right = (int)(data[i + 5] * image.cols);
            int bottom = (int)(data[i + 6] * image.rows);
            int width = right - left + 1;
            int height = bottom - top + 1;
            if (0 < left && left < image.cols && 0 < top && top < image.rows &&
                left < right && right < image.cols &&
                top < bottom && bottom < image.rows) {
                bboxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }
    // Performs non maximum suppression given boxes and corresponding scores
    constexpr float nmsThreshold = 0.4f;
    vector<int> indices;
    dnn::NMSBoxes(bboxes, confidences, confidenceThreshold, nmsThreshold, indices);
    for (int index: indices) {
        objects.push_back(bboxes[index]);
        scores.push_back(confidences[index]);
    }
}
