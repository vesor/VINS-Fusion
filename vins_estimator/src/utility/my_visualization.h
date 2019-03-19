/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>

#include "../estimator/parameters.h"
#include <fstream>
#include <map>

class MyEstimator;
class MyVisualizer {
public:
    MyVisualizer(ros::NodeHandle &n, const MyEstimator &e);

    void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);

    void printStatistics(double t);

    void pubOdometry(double t);

    void pubInitialGuess(double t);

    void pubKeyPoses(double t);

    void pubCameraPose(double t);

    void pubPointCloud(double t);

    void pubTF(double t);

    void pubKeyframe();

    void pubRelocalization();

    void pubCar(double t);

    void pubFakeVelodyne(double t);

private:
    const MyEstimator &estimator;

    ros::Publisher pub_odometry, pub_latest_odometry;
    ros::Publisher pub_path;
    ros::Publisher pub_point_cloud, pub_margin_cloud;
    ros::Publisher pub_key_poses;
    ros::Publisher pub_camera_pose;
    ros::Publisher pub_camera_pose_visual;
    nav_msgs::Path path;

    ros::Publisher pub_keyframe_pose;
    ros::Publisher pub_keyframe_point;
    ros::Publisher pub_extrinsic;
    ros::Publisher pub_fake_velodyne_;

    std::map<int, Eigen::Vector3d> points_buf_;
};
