/*******************************************************
 * Author: Weizhe Liu
 * Date: 2019/3
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <deque>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "../utility/tic_toc.h"

#include "parameters.h"
#include "feature_manager.h"
#include "../featureTracker/feature_tracker.h"

class MyEstimator
{
public:
    struct FeaturePoint {
        Eigen::Vector3f xyz;
        Eigen::Vector2f uv;
    };

    struct FeatureFrame {
        double time;
        std::map<int,FeaturePoint> features;
    };


public:
    MyEstimator();
    void setPointsCloudCallback(std::function<void(const std::vector<Eigen::Vector3f>&, double)> callback) {
        points_cloud_callback_ = callback;
    }

    void setParameter();

    // interface
    void inputOdometry(double time, const Eigen::Vector3f& pose);
    void inputImage(double time, const cv::Mat& img);
    void processMeasurements();

    std::unique_ptr<Eigen::Vector3f> getInterpolatedOdomPose(double time);

private:
    Matrix3d ric_[2];
    Vector3d tic_[2];

    std::mutex feature_buf_mutex_;
    std::deque<FeatureFrame> feature_buf_;
    std::mutex odom_buf_mutex_;
    std::deque<std::pair<double, Eigen::Vector3f>> odom_buf_;

    Matrix3d        Rs_[(WINDOW_SIZE + 1)];

    FeatureTracker feature_tracker_;

    FeatureManager f_manager_;
    
    std::function<void(const std::vector<Eigen::Vector3f>&, double)> points_cloud_callback_;
};
