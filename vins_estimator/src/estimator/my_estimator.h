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
#include "my_feature_manager.h"
#include "../featureTracker/feature_tracker.h"
#include "../initial/my_aligment.h"
#include "../initial/initial_sfm.h"
#include "../initial/solve_5pts.h"

class MyEstimator
{
#if 0
public:
    struct FeaturePoint {
        Eigen::Vector3f xyz;
        Eigen::Vector2f uv;
    };

    struct FeatureFrame {
        double time;
        std::map<int,FeaturePoint> features;
    };
#endif

public:
    MyEstimator();
    void setPointsCloudCallback(std::function<void(const std::vector<Eigen::Vector3d>&, double)> callback) {
        points_cloud_callback_ = callback;
    }

    void setParameter();

    // interface
    void inputOdometry(double time, const Eigen::Vector3d& pose);
    void inputImage(double time, const cv::Mat& img);
    void processMeasurements();
    void processOdom(double dt, const Eigen::Vector3d &delta_pose);
    void processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, 
            const double header, const Eigen::Vector3d& odom_pose);

    

private:
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    bool failureDetection();
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Eigen::Matrix3d &Ri, Eigen::Vector3d &Pi, Eigen::Matrix3d &rici, Eigen::Vector3d &tici,
                                     Eigen::Matrix3d &Rj, Eigen::Vector3d &Pj, Eigen::Matrix3d &ricj, Eigen::Vector3d &ticj, 
                                     double depth, Eigen::Vector3d &uvi, Eigen::Vector3d &uvj);

    std::unique_ptr<Eigen::Vector3d> getInterpolatedOdomPose(double time);
    std::unique_ptr<std::pair<Eigen::Vector3d,Eigen::Vector3d>> getOdomInterval(double t1, double t2);

private:
    Matrix3d ric_[2];
    Vector3d tic_[2];

    bool initialized_;
    bool marginalize_old_; //marginalize new or old frame]

    std::mutex feature_buf_mutex_;
#if 1
    std::queue<pair<double, std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > > > feature_buf_;
#else
    std::deque<FeatureFrame> feature_buf_;
#endif
    std::mutex odom_buf_mutex_;
    std::deque<std::pair<double, Eigen::Vector3d>> odom_buf_;

    double prevTime_, curTime_;
    
    Eigen::Vector3d        Ps_[(WINDOW_SIZE + 1)];
    Eigen::Vector3d        Vs_[(WINDOW_SIZE + 1)];
    Eigen::Matrix3d        Rs_[(WINDOW_SIZE + 1)];
    double td_;

    Eigen::Matrix3d back_R0_, last_R_, last_R0_;
    Eigen::Vector3d back_P0_, last_P_, last_P0_;
    double headers_[(WINDOW_SIZE + 1)];

    int frame_count_;
    int sum_of_outlier_, sum_of_back_, sum_of_front_, sum_of_invalid_;
    
    std::map<double, ImageFrame> all_image_frame_;

    FeatureTracker feature_tracker_;

    MyFeatureManager f_manager_;
    MotionEstimator m_estimator_;

    bool failure_occur_;
    std::vector<Eigen::Vector3d> key_poses_;

    std::function<void(const std::vector<Eigen::Vector3d>&, double)> points_cloud_callback_;
};
