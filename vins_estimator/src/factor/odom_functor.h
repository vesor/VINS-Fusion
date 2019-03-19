
/*******************************************************
 * Author: Weizhe Liu
 * Date: 2019/3
 *******************************************************/

#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"

#include <ceres/ceres.h>

class OdomFunctor
{
public:
    static ceres::CostFunction* Create(const Eigen::Vector3d& p, const Eigen::Quaterniond& q) {
        return new ceres::AutoDiffCostFunction<OdomFunctor, 1, 7, 7>(new OdomFunctor(p, q));
    }

public:
    OdomFunctor() = delete;
    OdomFunctor(const Eigen::Vector3d& p, const Eigen::Quaterniond& q):p_(p),q_(q)
    {
    }
    
    template <typename T>
    bool operator()(const T* const param1 , const T* const param2, T* residuals) const {
    #if 0
        Eigen::Matrix<T,3,1> Pi(param1[0], param1[1], param1[2]);
        Eigen::Quaternion<T> Qi(param1[6], param1[3], param1[4], param1[5]);
        Eigen::Matrix<T,3,1> Pj(param2[0], param2[1], param2[2]);
        Eigen::Quaternion<T> Qj(param2[6], param2[3], param2[4], param2[5]);
        Eigen::Matrix<T,3,1> delta_p = Pj - Qj * Qi.inverse() * Pi;
        //Eigen::Quaternion<T> delta_q = Qj * Qi.inverse();

        // only consider relative distance err
        residuals[0] = /*T(1.0) */ (delta_p.norm() - p_.norm());
        //std::cout << "=== res " << residuals[0] << std::endl;
        return true;
    #else
        residuals[0] = T(0);
        return true;
    #endif
    }

private:
    Eigen::Vector3d p_;
    Eigen::Quaterniond q_;
};

