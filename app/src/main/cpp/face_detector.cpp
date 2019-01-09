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

static void drawDetection(const Mat& frame, float confidence, int left, int top, int right, int bottom) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    string label = format("%.2f", confidence);

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}


bool FaceDetector::load(const string& modelDir) {
    string prototxt = modelDir + "/res10_300x300_ssd_iter_140000.prototxt";
    string caffeModel = modelDir + "/res10_300x300_ssd_iter_140000.caffemodel";
    string landmarkModel = modelDir + "/lbfmodel.yaml";

    LOGI("load model %s,%s", prototxt.c_str(), caffeModel.c_str());
    mFaceNet = dnn::readNetFromCaffe(prototxt, caffeModel);

    LOGI("load model %s", landmarkModel.c_str());
    return mFaceLandmark.load(landmarkModel);
}

void FaceDetector::detect(const Mat& image, vector<Rect>& objects) {
    int64 start = getTickCount();

    Mat rgbImage;
    Mat blob;
    vector<Mat> outs;

    resize(image, rgbImage, Size(300, 300));
    cvtColor(rgbImage, rgbImage, COLOR_RGBA2BGR);
    dnn::blobFromImage(rgbImage, blob, 1.0, Size(300, 300), Scalar(104.0,177.0,123.0));
    mFaceNet.setInput(blob);
    mFaceNet.forward(outs, getOutputsNames(mFaceNet));

    // NOTES:
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    CV_Assert(outs.size() == 1);
    constexpr float confidenceThreshold = 0.5f;
    constexpr float nmsThreshold = 0.4f;
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
                top < bottom && bottom < image.rows &&
                width < image.cols && height < image.rows) {
                confidences.push_back(confidence);
                bboxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    vector<int> indices;
    dnn::NMSBoxes(bboxes, confidences, confidenceThreshold, nmsThreshold, indices);
    for (int index: indices) {
        Rect box = bboxes[index];
        float confidence = confidences[index];
        objects.push_back(box);
        drawDetection(image, confidence, box.x, box.y, box.x + box.width, box.y + box.height);
        LOGI("detect face=[%d,%d,%d,%d] confidence=%.1f", box.x, box.y, box.x + box.width, box.y + box.height, confidence);
    }

    double duration = 1000*double(getTickCount() - start)/getTickFrequency();
    LOGI("detect face: total=%d, accepted=%d, duration=%.1fms", outs[0].total(), objects.size(), duration);
}

bool FaceDetector::fit(const Mat& image, const Rect& face, vector<Point2f>& landmarks) {
    vector<Rect> faces = { face };
    vector<vector<Point2f>> marks = { landmarks };
    if (!mFaceLandmark.fit(image, faces, marks)) {
        return  false;
    }
    landmarks = std::move(marks[0]);
    return true;
}
