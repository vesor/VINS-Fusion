/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../estimator/parameters.h"

#include <ceres/ceres.h>

class OdomFactor
{
public:
    static ceres::CostFunction* Create(const Eigen::Vector3d& p, const Eigen::Quaterniond& q) {
        return new ceres::AutoDiffCostFunction<OdomFactor, 1, 7, 7>(new OdomFactor(p, q));
    }

public:
    OdomFactor() = delete;
    OdomFactor(const Eigen::Vector3d& p, const Eigen::Quaterniond& q):p_(p),q_(q)
    {
    }
    
    template <typename T>
    bool operator()(const T* const param1 , const T* const param2, T* residuals) const {
        Eigen::Matrix<T,3,1> Pi(param1[0], param1[1], param1[2]);
        Eigen::Quaternion<T> Qi(param1[6], param1[3], param1[4], param1[5]);
        Eigen::Matrix<T,3,1> Pj(param2[0], param2[1], param2[2]);
        Eigen::Quaternion<T> Qj(param2[6], param2[3], param2[4], param2[5]);
        Eigen::Matrix<T,3,1> delta_p = Pj - Qj * Pi;
        //Eigen::Quaternion<T> delta_q = Qj * Qi.inverse();
        // only consider relative distance err
        residuals[0] = delta_p.norm() - p_.norm();
        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);

private:
    Eigen::Vector3d p_;
    Eigen::Quaterniond q_;
};

