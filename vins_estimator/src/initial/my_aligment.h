#pragma once

#include "initial_alignment.h"

bool MyVisualOdomAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* poses, VectorXd &x);
