#include "NeuralNetworkHelper.h"

NeuralNetworkHelper* NeuralNetworkHelper::Create(const HelperType helper_type)
{
    NeuralNetworkHelper* p = nullptr;
    switch (helper_type) {
#ifdef INFERENCE_HELPER_ENABLE_OPENCV
    case kOpencv:
    case kOpencvGpu:
        std::cout << "Use OpenCV \n"
        p = new InferenceHelperOpenCV();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE
    case kTensorflowLite:
        std::cout << "Use TensorflowLite\n"
        p = new InferenceHelperTensorflowLite();
        break;
#endif
    default:
        std::cout << "Unsupported inference helper type (%d)\n";
        break;
    }
    if (p == nullptr) {
        std::cout << "Failed to create inference helper\n";
    }
    else {
        p->helper_type_ = helper_type;
    }
    return p;
}

#ifdef NEURAL_NETWORK_ENABLEPRE_PROCESS_BY_OPENCV
void NeuralNetworkHelper::PreProcessByOpenCV(const InputTensorInfo& input_tensor_info, bool is_nchw, cv::Mat& img_blob)
{
    cv::Mat img_src = cv::Mat(cv::Size(input_tensor_info.image_info.width, input_tensor_info.image_info.height), (input_tensor_info.image_info.channel == 3) ? CV_8UC3 : CV_8UC1, input_tensor_info.data);

    if (input_tensor_info.image_info.width == input_tensor_info.image_info.crop_width && input_tensor_info.image_info.height == input_tensor_info.image_info.crop_height) {
        /* do nothing */
    }
    else {
        img_src = img_src(cv::Rect(input_tensor_info.image_info.crop_x, input_tensor_info.image_info.crop_y, input_tensor_info.image_info.crop_width, input_tensor_info.image_info.crop_height));
    }

    if (input_tensor_info.image_info.crop_width == input_tensor_info.GetWidth() && input_tensor_info.image_info.crop_height == input_tensor_info.GetHeight()) {
        /* do nothing */
    }
    else {
        cv::resize(img_src, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
}

    if (input_tensor_info.image_info.channel == input_tensor_info.GetChannel()) {
        if (input_tensor_info.image_info.channel == 3 && input_tensor_info.image_info.swap_color) {
            cv::cvtColor(img_src, img_src, cv::COLOR_BGR2RGB);
        }
    }
    else if (input_tensor_info.image_info.channel == 3 && input_tensor_info.GetChannel() == 1) {
        cv::cvtColor(img_src, img_src, (input_tensor_info.image_info.is_bgr) ? cv::COLOR_BGR2GRAY : cv::COLOR_RGB2GRAY);
    }
    else if (input_tensor_info.image_info.channel == 1 && input_tensor_info.GetChannel() == 3) {
        cv::cvtColor(img_src, img_src, cv::COLOR_GRAY2BGR);
    }

    if (input_tensor_info.tensorType == TensorInfo::kTensorTypeFp32) {
        if (input_tensor_info.GetChannel() == 3) {
            img_src.convertTo(img_src, CV_32FC3);
            cv::subtract(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.mean)), img_src);
            cv::multiply(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.norm)), img_src);
    }
        else {
            img_src.convertTo(img_src, CV_32FC1);
            cv::subtract(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.mean)), img_src);
            cv::multiply(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.norm)), img_src);
        }
    }
    else {
        }

    if (is_nchw) {
        img_src = cv::dnn::blobFromImage(img_src);
    }

    img_blob = img_src;
}
#else
void NeuralNetworkHelper::PreProcessByOpenCV(const InputTensorInfo& input_tensor_info, bool is_nchw, cv::Mat& img_blob)
{
    std::cout << "[PreProcessByOpenCV] Unsupported function called\n";
    exit(-1);
}
#endif

void NeuralNetworkHelper::ConvertNormalizeParameters(InputTensorInfo& tensor_info)
{
    if (tensor_info.data_type != InputTensorInfo::kDataTypeImage) return;

    for (int32_t i = 0; i < 3; i++) {
        tensor_info.normalize.mean[i] *= 255.0f;
        tensor_info.normalize.norm[i] *= 255.0f;
        tensor_info.normalize.norm[i] = 1.0f / tensor_info.normalize.norm[i];
    }
}

void NeuralNetworkHelper::PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, float* dst)
{
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    uint8_t* src = (uint8_t*)(input_tensor_info.data);
    if (input_tensor_info.isNchw == true) {
#pragma omp parallel for num_threads(num_thread)
        for (int32_t c = 0; c < img_channel; c++) {
            for (int32_t i = 0; i < img_width * img_height; i++) {
                dst[c * img_width * img_height + i] = (src[i * img_channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
            }
        }
    }
    else {
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[i * img_channel + c] = (src[i * img_channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
            }
        }
    }
}

void NeuralNetworkHelper::PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, uint8_t* dst)
{
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    uint8_t* src = (uint8_t*)(input_tensor_info.data);
    if (input_tensor_info.isNchw == true) {
#pragma omp parallel for num_threads(num_thread)
        for (int32_t c = 0; c < img_channel; c++) {
            for (int32_t i = 0; i < img_width * img_height; i++) {
                dst[c * img_width * img_height + i] = src[i * img_channel + c];
            }
        }
    }
    else {
        std::copy(src, src + input_tensor_info.GetElementNum(), dst);
    }
}

void NeuralNetworkHelper::PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, int8_t* dst)
{
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    uint8_t* src = (uint8_t*)(input_tensor_info.data);
    if (input_tensor_info.isNchw == true) {
#pragma omp parallel for num_threads(num_thread)
        for (int32_t c = 0; c < img_channel; c++) {
            for (int32_t i = 0; i < img_width * img_height; i++) {
                dst[c * img_width * img_height + i] = src[i * img_channel + c] - 128;
            }
        }
    }
    else {
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[i * img_channel + c] = src[i * img_channel + c] - 128;
            }
        }
    }
}

template<typename T>
void NeuralNetworkHelper::PreProcessBlob(int32_t num_thread, const InputTensorInfo& input_tensor_info, T* dst)
{
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    T* src = static_cast<T*>(input_tensor_info.data);
    if ((input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw && input_tensor_info.is_nchw) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc && !input_tensor_info.is_nchw)) {
        std::copy(src, src + input_tensor_info.GetElementNum(), dst);
    }
    else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw) {
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[i * img_channel + c] = src[c * (img_width * img_height) + i];
            }
        }
    }
    else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) {
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[c * (img_width * img_height) + i] = src[i * img_channel + c];
            }
        }
    }
}

template void NeuralNetworkHelper::PreProcessBlob<float>(int32_t num_thread, const InputTensorInfo& input_tensor_info, float* dst);
template void NeuralNetworkHelper::PreProcessBlob<int32_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, int32_t* dst);
template void NeuralNetworkHelper::PreProcessBlob<int64_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, int64_t* dst);
template void NeuralNetworkHelper::PreProcessBlob<uint8_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, uint8_t* dst);
template void NeuralNetworkHelper::PreProcessBlob<int8_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, int8_t* dst);
