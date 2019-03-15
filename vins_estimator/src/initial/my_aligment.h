#pragma once

#include "initial_alignment.h"

bool VisualOdomAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::VectorXd &x);
bool VisualOdomAlignmentOpt(const int frame_count, const double headers[], const Eigen::Matrix3d Rs[], const Eigen::Vector3d Ps[], 
    const std::map<double, ImageFrame>& all_image_frame, Eigen::VectorXd &x);
