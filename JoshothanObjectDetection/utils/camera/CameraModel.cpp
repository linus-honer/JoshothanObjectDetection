#include "CameraModel.h"

CameraModel::CameraModel()
{
    SetIntrinsic(1280, 720, 500.0f);
    SetDist({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
    SetExtrinsic({ 0, 0, 0 }, { 0, 0, 0 });
}

void CameraModel::SetIntrinsic(int32_t width, int32_t height, float focal_length)
{
    this->width = width;
    this->height = height;
    this->K = (cv::Mat_<float>(3, 3) << focal_length, 0, width / 2.f, 0, focal_length, height / 2.f, 0, 0, 1);
    UpdateNewCameraMatrix();
}

void CameraModel::SetExtrinsic(const std::array<float, 3>& rvec_deg, const std::array<float, 3>& tvec, bool is_t_on_world)
{
    this->rvec = (cv::Mat_<float>(3, 1) << Deg2Rad(rvec_deg[0]), Deg2Rad(rvec_deg[1]), Deg2Rad(rvec_deg[2]));
    this->tvec = (cv::Mat_<float>(3, 1) << tvec[0], tvec[1], tvec[2]);

    if (is_t_on_world) {
        cv::Mat R = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
        this->tvec = -R * this->tvec;
    }
}

void CameraModel::GetExtrinsic(std::array<float, 3>& rvec_deg, std::array<float, 3>& tvec, bool is_t_on_world)
{
    rvec_deg = { Rad2Deg(this->rvec.at<float>(0)), Rad2Deg(this->rvec.at<float>(1)) , Rad2Deg(this->rvec.at<float>(2)) };
    tvec = { this->tvec.at<float>(0), this->tvec.at<float>(1), this->tvec.at<float>(2) };
    if (is_t_on_world) {
        cv::Mat R = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
        cv::Mat R_inv;
        cv::invert(R, R_inv);
        cv::Mat T = -R_inv * this->tvec;
        tvec = { T.at<float>(0), T.at<float>(1), T.at<float>(2) };
    }
    else {
        tvec = { this->tvec.at<float>(0), this->tvec.at<float>(1), this->tvec.at<float>(2) };
    }
}

void CameraModel::SetCameraPos(float tx, float ty, float tz, bool is_on_world)
{
    this->tvec = (cv::Mat_<float>(3, 1) << tx, ty, tz);
    if (is_on_world) {
        cv::Mat R = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
        this->tvec = -R * this->tvec;
    }
    else {
        this->tvec *= -1;
    }
}

void CameraModel::MoveCameraPos(float dtx, float dty, float dtz, bool is_on_world)
{
    cv::Mat tvec_delta = (cv::Mat_<float>(3, 1) << dtx, dty, dtz);
    if (is_on_world) {
        cv::Mat R = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
        tvec_delta = -R * tvec_delta;
    }
    else {
        tvec_delta *= -1;
    }
    this->tvec += tvec_delta;
}

void CameraModel::SetCameraAngle(float pitch_deg, float yaw_deg, float roll_deg)
{
    cv::Mat R_old = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
    cv::Mat T = -R_old.inv() * this->tvec;
    this->rvec = (cv::Mat_<float>(3, 1) << Deg2Rad(pitch_deg), Deg2Rad(yaw_deg), Deg2Rad(roll_deg));
    cv::Mat R_new = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
    this->tvec = -R_new * T;
}

void CameraModel::RotateCameraAngle(float dpitch_deg, float dyaw_deg, float droll_deg)
{
    cv::Mat R_old = MakeRotationMat(Rad2Deg(rx()), Rad2Deg(ry()), Rad2Deg(rz()));
    cv::Mat T = -R_old.inv() * this->tvec;
    cv::Mat R_delta = MakeRotationMat(dpitch_deg, dyaw_deg, droll_deg);
    cv::Mat R_new = R_delta * R_old;
    this->tvec = -R_new * T;
    cv::Rodrigues(R_new, this->rvec);
}

void CameraModel::SetDist(const std::array<float, 5>& dist)
{
    this->dist_coeff = (cv::Mat_<float>(5, 1) << dist[0], dist[1], dist[2], dist[3], dist[4]);
    UpdateNewCameraMatrix();
}

void CameraModel::UpdateNewCameraMatrix()
{
    if (!this->K.empty() && !this->dist_coeff.empty()) {
        this->K_new = cv::getOptimalNewCameraMatrix(this->K, this->dist_coeff, cv::Size(this->width, this->height), 0.0);
    }
}

void CameraModel::ConvertWorld2Image(const cv::Point3f& object_point, cv::Point2f& image_point)
{
    std::vector<cv::Point3f> object_point_list = { object_point };
    std::vector<cv::Point2f> image_point_list;
    ConvertWorld2Image(object_point_list, image_point_list);
    image_point = image_point_list[0];
}

void CameraModel::ConvertWorld2Image(const std::vector<cv::Point3f>& object_point_list, std::vector<cv::Point2f>& image_point_list)
{
    cv::Mat K = this->K;
    cv::Mat R = MakeRotationMat(Rad2Deg(this->rx()), Rad2Deg(this->ry()), Rad2Deg(this->rz()));
    cv::Mat Rt = (cv::Mat_<float>(3, 4) <<
        R.at<float>(0), R.at<float>(1), R.at<float>(2), this->tx(),
        R.at<float>(3), R.at<float>(4), R.at<float>(5), this->ty(),
        R.at<float>(6), R.at<float>(7), R.at<float>(8), this->tz());

    image_point_list.resize(object_point_list.size());

    for (int32_t i = 0; i < object_point_list.size(); i++) {
        const auto& object_point = object_point_list[i];
        auto& image_point = image_point_list[i];
        cv::Mat Mw = (cv::Mat_<float>(4, 1) << object_point.x, object_point.y, object_point.z, 1);
        cv::Mat Mc = Rt * Mw;
        float Zc = Mc.at<float>(2);
        if (Zc <= 0) {
            image_point = cv::Point2f(-1, -1);
            continue;
        }

        cv::Mat XY = K * Mc;
        float x = XY.at<float>(0);
        float y = XY.at<float>(1);
        float s = XY.at<float>(2);
        x /= s;
        y /= s;

        if (this->dist_coeff.empty() || this->dist_coeff.at<float>(0) == 0) {
            image_point.x = x;
            image_point.y = y;
        }
        else {
            float u = (x - this->cx()) / this->fx();
            float v = (y - this->cy()) / this->fy();
            float r2 = u * u + v * v;
            float r4 = r2 * r2;
            float k1 = this->dist_coeff.at<float>(0);
            float k2 = this->dist_coeff.at<float>(1);
            float p1 = this->dist_coeff.at<float>(3);
            float p2 = this->dist_coeff.at<float>(4);
            u = u + u * (k1 * r2 + k2 * r4) + (2 * p1 * u * v) + p2 * (r2 + 2 * u * u);
            v = v + v * (k1 * r2 + k2 * r4) + (2 * p2 * u * v) + p1 * (r2 + 2 * v * v);
            image_point.x = u * this->fx() + this->cx();
            image_point.y = v * this->fy() + this->cy();
        }
    }
}

void CameraModel::ConvertWorld2Camera(const std::vector<cv::Point3f>& object_point_in_world_list, std::vector<cv::Point3f>& object_point_in_camera_list)
{
    cv::Mat K = this->K;
    cv::Mat R = MakeRotationMat(Rad2Deg(this->rx()), Rad2Deg(this->ry()), Rad2Deg(this->rz()));
    cv::Mat Rt = (cv::Mat_<float>(3, 4) <<
        R.at<float>(0), R.at<float>(1), R.at<float>(2), this->tx(),
        R.at<float>(3), R.at<float>(4), R.at<float>(5), this->ty(),
        R.at<float>(6), R.at<float>(7), R.at<float>(8), this->tz());

    object_point_in_camera_list.resize(object_point_in_world_list.size());

    for (int32_t i = 0; i < object_point_in_world_list.size(); i++) {
        const auto& object_point_in_world = object_point_in_world_list[i];
        auto& object_point_in_camera = object_point_in_camera_list[i];
        cv::Mat Mw = (cv::Mat_<float>(4, 1) << object_point_in_world.x, object_point_in_world.y, object_point_in_world.z, 1);
        cv::Mat Mc = Rt * Mw;
        object_point_in_camera.x = Mc.at<float>(0);
        object_point_in_camera.y = Mc.at<float>(1);
        object_point_in_camera.z = Mc.at<float>(2);
    }
}

void CameraModel::ConvertCamera2World(const std::vector<cv::Point3f>& object_point_in_camera_list, std::vector<cv::Point3f>& object_point_in_world_list)
{
    cv::Mat R = MakeRotationMat(Rad2Deg(this->rx()), Rad2Deg(this->ry()), Rad2Deg(this->rz()));
    cv::Mat R_inv;
    cv::invert(R, R_inv);
    cv::Mat t = this->tvec;


    object_point_in_world_list.resize(object_point_in_camera_list.size());

    for (int32_t i = 0; i < object_point_in_camera_list.size(); i++) {
        const auto& object_point_in_camera = object_point_in_camera_list[i];
        auto& object_point_in_world = object_point_in_world_list[i];
        cv::Mat Mc = (cv::Mat_<float>(3, 1) << object_point_in_camera.x, object_point_in_camera.y, object_point_in_camera.z);
        cv::Mat Mw = R_inv * (Mc - t);
        object_point_in_world.x = Mw.at<float>(0);
        object_point_in_world.y = Mw.at<float>(1);
        object_point_in_world.z = Mw.at<float>(2);
    }
}

void CameraModel::ConvertImage2GroundPlane(const std::vector<cv::Point2f>& image_point_list, std::vector<cv::Point3f>& object_point_list)
{
    if (image_point_list.size() == 0) return;

    cv::Mat K = this->K;
    cv::Mat R = MakeRotationMat(Rad2Deg(this->rx()), Rad2Deg(this->ry()), Rad2Deg(this->rz()));
    cv::Mat K_inv;
    cv::invert(K, K_inv);
    cv::Mat R_inv;
    cv::invert(R, R_inv);
    cv::Mat t = this->tvec;

    std::vector<cv::Point2f> image_point_undistort;
    if (this->dist_coeff.empty() || this->dist_coeff.at<float>(0) == 0) {
        image_point_undistort = image_point_list;
    }
    else {
        cv::undistortPoints(image_point_list, image_point_undistort, this->K, this->dist_coeff, this->K);    /* don't use K_new */
    }

    object_point_list.resize(image_point_list.size());
    for (int32_t i = 0; i < object_point_list.size(); i++) {
        const auto& image_point = image_point_list[i];
        auto& object_point = object_point_list[i];

        float x = image_point_undistort[i].x;
        float y = image_point_undistort[i].y;
        if (y < EstimateVanishmentY()) {
            object_point.x = 999;
            object_point.y = 999;
            object_point.z = 999;
            continue;
        }

        cv::Mat XY = (cv::Mat_<float>(3, 1) << x, y, 1);

        cv::Mat LEFT_WO_S = R_inv * K_inv * XY;
        cv::Mat RIGHT_WO_M = R_inv * t;
        float s = RIGHT_WO_M.at<float>(1) / LEFT_WO_S.at<float>(1);

        cv::Mat TEMP = R_inv * (s * K_inv * XY - t);

        object_point.x = TEMP.at<float>(0);
        object_point.y = TEMP.at<float>(1);
        object_point.z = TEMP.at<float>(2);
        if (object_point.z < 0) object_point.z = 999;
    }
}

void CameraModel::ConvertImage2Camera(std::vector<cv::Point2f>& image_point_list, const std::vector<float>& z_list, std::vector<cv::Point3f>& object_point_list)
{
    if (image_point_list.size() == 0) {
        if (z_list.size() != this->width * this->height) {
            printf("[ConvertImage2Camera] Invalid z_list size\n");
            return;
        }
        for (int32_t y = 0; y < this->height; y++) {
            for (int32_t x = 0; x < this->width; x++) {
                image_point_list.push_back(cv::Point2f(float(x), float(y)));
            }
        }
    }
    else {
        if (z_list.size() != image_point_list.size()) {
            printf("[ConvertImage2Camera] Invalid z_list size\n");
            return;
        }
    }

    std::vector<cv::Point2f> image_point_undistort;
    if (this->dist_coeff.empty() || this->dist_coeff.at<float>(0) == 0) {
        image_point_undistort = image_point_list;
    }
    else {
        cv::undistortPoints(image_point_list, image_point_undistort, this->K, this->dist_coeff, this->K);
    }

    object_point_list.resize(image_point_list.size());
    for (int32_t i = 0; i < object_point_list.size(); i++) {
        const auto& Zc = z_list[i];
        auto& object_point = object_point_list[i];

        float x = image_point_undistort[i].x;
        float y = image_point_undistort[i].y;

        float u = x - this->cx();
        float v = y - this->cy();
        float Xc = Zc * u / this->fx();
        float Yc = Zc * v / this->fy();
        object_point.x = Xc;
        object_point.y = Yc;
        object_point.z = Zc;
    }
}

void CameraModel::ConvertImage2World(std::vector<cv::Point2f>& image_point_list, const std::vector<float>& z_list, std::vector<cv::Point3f>& object_point_list)
{
    std::vector<cv::Point3f> object_point_in_camera_list;
    ConvertImage2Camera(image_point_list, z_list, object_point_in_camera_list);
    ConvertCamera2World(object_point_in_camera_list, object_point_list);
}

float CameraModel::EstimatePitch(float vanishment_y)
{
    float pitch = std::atan2(this->cy() - vanishment_y, this->fy());
    return Rad2Deg(pitch);
}

float CameraModel::EstimateYaw(float vanishment_x)
{
    float yaw = std::atan2(this->cx() - vanishment_x, this->fx());
    return Rad2Deg(yaw);
}

int32_t CameraModel::EstimateVanishmentY()
{
    float fy = this->fy();
    float cy = this->cy();
    float px_from_center = std::tan(this->rx()) * fy;
    float vanishment_y = cy - px_from_center;
    return static_cast<int32_t>(vanishment_y);
}

int32_t CameraModel::EstimateVanishmentX()
{
    float px_from_center = std::tan(this->ry()) * this->fx();
    float vanishment_x = this->cx() - px_from_center;
    return static_cast<int32_t>(vanishment_x);
}

void CameraModel::RotateObject(float x_deg, float y_deg, float z_deg, std::vector<cv::Point3f>& object_point_list)
{
    cv::Mat R = MakeRotationMat(x_deg, y_deg, z_deg);
    for (auto& object_point : object_point_list) {
        cv::Mat p = (cv::Mat_<float>(3, 1) << object_point.x, object_point.y, object_point.z);
        p = R * p;
        object_point.x = p.at<float>(0);
        object_point.y = p.at<float>(1);
        object_point.z = p.at<float>(2);
    }
}

void CameraModel::MoveObject(float x, float y, float z, std::vector<cv::Point3f>& object_point_list)
{
    for (auto& object_point : object_point_list) {
        object_point.x += x;
        object_point.y += y;
        object_point.z += z;
    }
}
