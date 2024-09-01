#pragma once

#include <opencv2/opencv.hpp>

#include "TensorInfo.h"

class NeuralNetworkHelper
{
public:
	enum {
		kRetOk = 0,
		kRetErr = -1
	};

	typedef enum {
		kOpencv,
		kTensorFlowLite
	} HelperType;
public:
	virtual ~NeuralNetworkHelper() {}
	virtual int32_t SetNumThreads(const int32_t num_threads) = 0;
	virtual int32_t SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops) = 0;
	virtual int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
	virtual int32_t Finalize(void) = 0;
	virtual int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) = 0;
	virtual int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
public:
	static NeuralNetworkHelper* Create(const HelperType helper_type);
	static void PreProcessByOpenCV(const InputTensorInfo& input_tensor_info, bool is_nchw, cv::Mat& img_blob);
protected:
	void ConvertNormalizeParameters(InputTensorInfo& tensor_info);

	void PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, float* dst);
	void PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, uint8_t* dst);
	void PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, int8_t* dst);

	template<typename T>
	void PreProcessBlob(int32_t num_thread, const InputTensorInfo& input_tensor_info, T* dst);

protected:
	HelperType helper_type_;
};
