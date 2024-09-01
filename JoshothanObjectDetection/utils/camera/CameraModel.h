#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <vector>
#include <array>

#include <opencv2/opencv.hpp>

#include "../math/MathUtils.h"

static inline float FocalLength(int32_t image_size, float fov)
{
    return (image_size / 2) / std::tan(Deg2Rad(fov / 2));
}

class CameraModel
{
public:
    CameraModel();

    void SetIntrinsic(int32_t width, int32_t height, float focal_length);
    void SetExtrinsic(const std::array<float, 3>& rvec_deg, const std::array<float, 3>& tvec, bool is_t_on_world = true);
    void GetExtrinsic(std::array<float, 3>& rvec_deg, std::array<float, 3>& tvec, bool is_t_on_world = true);

    void SetCameraPos(float tx, float ty, float tz, bool is_on_world = true);
    void MoveCameraPos(float dtx, float dty, float dtz, bool is_on_world = true);
    void SetCameraAngle(float pitch_deg, float yaw_deg, float roll_deg);
    void RotateCameraAngle(float dpitch_deg, float dyaw_deg, float droll_deg);

    void SetDist(const std::array<float, 5>& dist);

    void UpdateNewCameraMatrix();

    void ConvertWorld2Image(const cv::Point3f& object_point, cv::Point2f& image_point);
    void ConvertWorld2Image(const std::vector<cv::Point3f>& object_point_list, std::vector<cv::Point2f>& image_point_list);
    void ConvertWorld2Camera(const std::vector<cv::Point3f>& object_point_in_world_list, std::vector<cv::Point3f>& object_point_in_camera_list);
    void ConvertCamera2World(const std::vector<cv::Point3f>& object_point_in_camera_list, std::vector<cv::Point3f>& object_point_in_world_list);
    void ConvertImage2GroundPlane(const std::vector<cv::Point2f>& image_point_list, std::vector<cv::Point3f>& object_point_list);
    void ConvertImage2Camera(std::vector<cv::Point2f>& image_point_list, const std::vector<float>& z_list, std::vector<cv::Point3f>& object_point_list);
    void ConvertImage2World(std::vector<cv::Point2f>& image_point_list, const std::vector<float>& z_list, std::vector<cv::Point3f>& object_point_list);

    float EstimatePitch(float vanishment_y);
    float EstimateYaw(float vanishment_x);

    int32_t EstimateVanishmentY();
    int32_t EstimateVanishmentX();

    static void RotateObject(float x_deg, float y_deg, float z_deg, std::vector<cv::Point3f>& object_point_list);
    static void MoveObject(float x, float y, float z, std::vector<cv::Point3f>& object_point_list);
public:
    cv::Mat K;
    cv::Mat K_new;

    int32_t width;
    int32_t height;

    cv::Mat dist_coeff;

    cv::Mat rvec;
    cv::Mat tvec;

    float& rx() { return rvec.at<float>(0); }
    float& ry() { return rvec.at<float>(1); }
    float& rz() { return rvec.at<float>(2); }
    float& tx() { return tvec.at<float>(0); }
    float& ty() { return tvec.at<float>(1); }
    float& tz() { return tvec.at<float>(2); }
    float& fx() { return K.at<float>(0); }
    float& cx() { return K.at<float>(2); }
    float& fy() { return K.at<float>(4); }
    float& cy() { return K.at<float>(5); }
public:
    template <typename T = float>
    static void PRINT_MAT_FLOAT(const cv::Mat& mat, int32_t size)
    {
        for (int32_t i = 0; i < size; i++) {
            printf("%d: %.3f\n", i, mat.at<T>(i));
        }
    }

    template <typename T = float>
    static cv::Mat MakeRotationMat(T x_deg, T y_deg, T z_deg)
    {
        {
            T x_rad = Deg2Rad(x_deg);
            T y_rad = Deg2Rad(y_deg);
            T z_rad = Deg2Rad(z_deg);

            cv::Mat rvec = (cv::Mat_<T>(3, 1) << x_rad, y_rad, z_rad);
            cv::Mat R;
            cv::Rodrigues(rvec, R);

            return R;
        }
    }
};

