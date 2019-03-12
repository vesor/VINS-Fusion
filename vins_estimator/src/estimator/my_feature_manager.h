#pragma once

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

#include <eigen3/Eigen/Dense>

#include "parameters.h"
#include "../utility/tic_toc.h"

class MyFeaturePerFrame
{
  public:
    MyFeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double _td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        td = _td;
    }

    double td;
    Eigen::Vector3d point;
    Eigen::Vector2d uv;
    Eigen::Vector2d velocity;
};

class MyFeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    std::vector<MyFeaturePerFrame> feature_per_frame;
    int used_num;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    MyFeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame() {
        return start_frame + feature_per_frame.size() - 1;
    }
};

class MyFeatureManager
{
  public:
    MyFeatureManager(Eigen::Matrix3d Rs[]);

    void setRic(Eigen::Matrix3d ric[]);
    void clearState();
    int getFeatureCount();
    bool addFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    void setDepth(const Eigen::VectorXd &x);
    void removeFailures();
    void clearDepth();
    Eigen::VectorXd getDepthVector();
    void triangulate(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    // void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    // bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, 
    //                         vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier(std::set<int> &outlierIndex);
    
    std::list<MyFeaturePerId> feature_;

  private:
    double compensatedParallax2(const MyFeaturePerId &it_per_id, int frame_count);
    const Eigen::Matrix3d *Rs_;
    Eigen::Matrix3d ric_[2];

    int last_track_num_;
    double last_average_parallax_;
    int new_feature_num_;
    int long_track_num_;
};