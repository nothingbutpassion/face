#include <opencv2/imgproc.hpp>
#include <dlib/dnn.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include "utils.h"
#include "resnet_face_descriptor.h"

#define LOG_TAG "ResnetFaceDescriptor"

using namespace std;
using namespace dlib;
using namespace cv;

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                             alevel0<
                             alevel1<
                             alevel2<
                             alevel3<
                             alevel4<
                             max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                             input_rgb_image_sized<150>
                             >>>>>>>>>>>>;

ResnetFaceDescriptor::~ResnetFaceDescriptor() {
    if (mResNet) {
        delete static_cast<anet_type*>(mResNet);
        mResNet = nullptr;
    }
}

bool ResnetFaceDescriptor::load(const std::string& modelDir) {
    std::string modelFile = modelDir + "/resnet_face_descriptor.dat";
    anet_type* net = new anet_type;
    bool isOk = true;
    try {
        deserialize(modelFile) >> (*net);
        mResNet = net;
    } catch (...) {
        delete net;
        isOk = false;
        LOGE("failed to load model file: %s", modelFile.c_str());
    }
    return isOk;
}

cv::Mat ResnetFaceDescriptor::extract(const cv::Mat& rgb, const cv::Rect& face, std::vector<cv::Point2f>& landmark, cv::Mat* chip) {
    std::vector<dlib::point> parts;
    for (auto p: landmark)
        parts.push_back(dlib::point(p.x, p.y));
    dlib::rectangle rect(face.x, face.y, face.x + face.width, face.y + face.height);
    full_object_detection shape(rect, parts);
    std::vector<matrix<rgb_pixel>> face_chips(1);
    dlib::cv_image<rgb_pixel> cimg(rgb);
    extract_image_chip(cimg, get_face_chip_details(shape,150,0.25), face_chips[0]);
    anet_type& net = *static_cast<anet_type*>(mResNet);
    std::vector<matrix<float,0,1>> face_descriptors = net(face_chips);
    if (chip)
        dlib::toMat(face_chips[0]).copyTo(*chip);
    Mat descriptor;
    dlib::toMat(face_descriptors[0]).copyTo(descriptor);
    return descriptor;
}


float ResnetFaceDescriptor::distance(const cv::Mat& descriptor1, const cv::Mat& descriptor2) {
    LOGD("descriptor1: rows=%d, cols=%d, type=%x, data=%p", descriptor1.rows, descriptor1.cols, descriptor1.type(), descriptor1.data);
    LOGD("descriptor2: rows=%d, cols=%d, type=%x, data=%p", descriptor2.rows, descriptor2.cols, descriptor2.type(), descriptor2.data);
    return norm(descriptor1 - descriptor2, NORM_L2);
}
