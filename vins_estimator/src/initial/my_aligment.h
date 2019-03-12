#pragma once

#include "initial_alignment.h"

bool VisualOdomAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::VectorXd &x);
