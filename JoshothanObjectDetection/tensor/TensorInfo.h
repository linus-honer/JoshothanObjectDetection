#pragma once

#include <cstdint>
#include <string>
#include <vector>

class TensorInfo
{
public:
    std::string          name;
    int32_t              id;
    int32_t              tensorType;
    std::vector<int32_t> tensorDims;
    bool                 isNchw;
public:
	enum {
		kTensorTypeNone,
		kTensorTypeUint8,
		kTensorTypeInt8,
		kTensorTypeFp32,
		kTensorTypeInt32,
		kTensorTypeInt64
	};

	TensorInfo() : name(""), id(-1), tensorType(kTensorTypeNone), isNchw(true) {}
	~TensorInfo() {}

	int32_t GetElementNum() const
	{
		int32_t element_num = 1;
		for (const auto& dim : tensorDims) {
			element_num *= dim;
		}
		return element_num;
	}

    int32_t GetBatch() const
    {
        if (tensorDims.size() <= 0) return -1;
        return tensorDims[0];
    }

    int32_t GetChannel() const
    {
        if (isNchw) {
            if (tensorDims.size() <= 1) return -1;
            return tensorDims[1];
        }
        else {
            if (tensorDims.size() <= 3) return -1;
            return tensorDims[3];
        }
    }

    int32_t GetHeight() const
    {
        if (isNchw) {
            if (tensorDims.size() <= 2) return -1;
            return tensorDims[2];
        }
        else {
            if (tensorDims.size() <= 1) return -1;
            return tensorDims[1];
        }
    }

    int32_t GetWidth() const
    {
        if (isNchw) {
            if (tensorDims.size() <= 3) return -1;
            return tensorDims[3];
        }
        else {
            if (tensorDims.size() <= 2) return -1;
            return tensorDims[2];
        }
    }
};

class InputTensorInfo : public TensorInfo {
public:
    enum {
        kDataTypeImage,
        kDataTypeBlobNhwc,  // data_ which already finished preprocess(color conversion, resize, normalize_, etc.)
        kDataTypeBlobNchw,
    };

public:
    InputTensorInfo()
        : data(nullptr)
        , data_type(kDataTypeImage)
        , image_info({ -1, -1, -1, -1, -1, -1, -1, true, false })
        , normalize({ 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f })
    {}

    InputTensorInfo(std::string name_, int32_t tensor_type_, bool is_nchw_ = true)
        : InputTensorInfo()
    {
        name = name_;
        tensorType = tensor_type_;
        isNchw = is_nchw_;
    }

    ~InputTensorInfo() {}

public:
    void* data;      // [In] Set the pointer to image/blob
    int32_t data_type; // [In] Set the type of data_ (e.g. kDataTypeImage)

    struct {
        int32_t width;
        int32_t height;
        int32_t channel;
        int32_t crop_x;
        int32_t crop_y;
        int32_t crop_width;
        int32_t crop_height;
        bool    is_bgr;        // used when channel == 3 (true: BGR, false: RGB)
        bool    swap_color;
    } image_info;              // [In] used when data_type_ == kDataTypeImage

    struct {
        float mean[3];
        float norm[3];
    } normalize;              // [In] used when data_type_ == kDataTypeImage
};


class OutputTensorInfo : public TensorInfo {
public:
    OutputTensorInfo()
        : data(nullptr)
        , quant({ 1.0f, 0 })
        , data_fp32_(nullptr)
    {}

    OutputTensorInfo(std::string name_, int32_t tensor_type_, bool is_nchw_ = true)
        : OutputTensorInfo()
    {
        name = name_;
        tensorType = tensor_type_;
        isNchw = is_nchw_;
    }

    ~OutputTensorInfo() {
        if (data_fp32_ != nullptr) {
            delete[] data_fp32_;
        }
    }

    float* GetDataAsFloat() {       /* Returned pointer should be with const, but returning pointer without const is convenient to create cv::Mat */
        if (tensorType == kTensorTypeUint8 || tensorType == kTensorTypeInt8) {
            if (data_fp32_ == nullptr) {
                data_fp32_ = new float[GetElementNum()];
            }
            if (tensorType == kTensorTypeUint8) {
#pragma omp parallel
                for (int32_t i = 0; i < GetElementNum(); i++) {
                    const uint8_t* val_uint8 = static_cast<const uint8_t*>(data);
                    float val_float = (val_uint8[i] - quant.zero_point) * quant.scale;
                    data_fp32_[i] = val_float;
                }
            }
            else {
#pragma omp parallel
                for (int32_t i = 0; i < GetElementNum(); i++) {
                    const int8_t* val_int8 = static_cast<const int8_t*>(data);
                    float val_float = (val_int8[i] - quant.zero_point) * quant.scale;
                    data_fp32_[i] = val_float;
                }
            }
            return data_fp32_;
        }
        else if (tensorType == kTensorTypeFp32) {
            return static_cast<float*>(data);
        }
        else {
            return nullptr;
        }
    }

public:
    void* data;
    struct {
        float   scale;
        int32_t zero_point;
    } quant;

private:
    float* data_fp32_;
};

