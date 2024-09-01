#pragma once

#include "../thirdparty/tensorflow/tensorflow/lite/c/c_api_types.h"

#include "NeuralNetworkHelper.h"

class NeuralNetworkHelperTFL : public NeuralNetworkHelper
{
public:
    NeuralNetworkHelperTFL();
    ~NeuralNetworkHelperTFL() override;
    int32_t SetNumThreads(const int32_t num_threads) override;
    int32_t SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops) override;
    int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t Finalize(void) override;
    int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) override;
    int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) override;
private:
    int32_t GetInputTensorInfo(InputTensorInfo& tensor_info);
    int32_t GetOutputTensorInfo(OutputTensorInfo& tensor_info);
    void DisplayModelInfo(const tflite::Interpreter& interpreter);

    int32_t SetBufferToTensor(int32_t index, void* data);
private:
    std::vector<char> model_buffer_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    TfLiteDelegate* delegate_;

    int32_t num_threads_;
};

