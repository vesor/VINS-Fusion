#pragma once
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

template<typename T, int N>
using EigenVector = Eigen::Matrix<T, N, 1>;

template<typename T>
using EigenVector3 = EigenVector<T, 3>;

template <typename FloatType> 
EigenVector3<FloatType> RPYFromQuat(const Eigen::Quaternion<FloatType>& quat)
{
    return quat.toRotationMatrix().eulerAngles(0, 1, 2);
}

template <typename FloatType> 
Eigen::Quaternion<FloatType> QuatFromRPY(const FloatType roll, const FloatType pitch,
                                const FloatType yaw) {
  const Eigen::AngleAxis<FloatType> roll_angle(roll, EigenVector3<FloatType>::UnitX());
  const Eigen::AngleAxis<FloatType> pitch_angle(pitch, EigenVector3<FloatType>::UnitY());
  const Eigen::AngleAxis<FloatType> yaw_angle(yaw, EigenVector3<FloatType>::UnitZ());
  return yaw_angle * pitch_angle * roll_angle;
}

template <typename FloatType>
void RigidMult(const EigenVector3<FloatType>& t1, const Eigen::Quaternion<FloatType>& r1, 
        const EigenVector3<FloatType>& t2, const Eigen::Quaternion<FloatType>& r2,
        EigenVector3<FloatType>& t, Eigen::Quaternion<FloatType>& r)
{
    t = r1 * t2 + t1;
    r = (r1 * r2).normalized();
}

template <typename FloatType>
void RigidInverse(const EigenVector3<FloatType>& t1, const Eigen::Quaternion<FloatType>& r1, 
        EigenVector3<FloatType>& t, Eigen::Quaternion<FloatType>& r)
{
    r = r1.conjugate();
    t = -(r * t1);
}

template <typename FloatType>
void RTFromTransform(Eigen::Matrix<FloatType, 3, 3>& R, 
    EigenVector3<FloatType>& T, const Eigen::Matrix<FloatType, 4, 4>& X)
{
    R = X.block(0, 0, 3, 3);
    T = X.block(0, 3, 3, 1);
}


template <typename FloatType>
Eigen::Matrix<FloatType, 4, 4> TransformFromRT(const Eigen::Matrix<FloatType, 3, 3>& R, 
    const EigenVector3<FloatType>& T)
{
    Eigen::Matrix<FloatType, 4, 4> X;
    X.setIdentity();
    X.block(0, 0, 3, 3) = R;
    X.block(0, 3, 3, 1) = T;
    return X;
}

// X must be in SE(3)
template <typename FloatType>
Eigen::Matrix<FloatType, 4, 4> TransformInverse(const Eigen::Matrix<FloatType, 4, 4>& X)
{
    Eigen::Matrix<FloatType, 3, 3> R;
    EigenVector3<FloatType> T;
    RTFromTransform(R, T, X);

    Eigen::Matrix<FloatType, 3, 3> R_inv = R.transpose();
    EigenVector3<FloatType> T_inv = -R_inv * T;
    return TransformFromRT(R_inv, T_inv);
}

template <typename FloatType>
Eigen::Matrix<FloatType, 4, 4> TransformFromRigid2(const EigenVector3<FloatType>& T)
{
    auto R = QuatFromRPY<FloatType>(0, 0, T[2]).toRotationMatrix();
    auto T0 = T;
    T0[2] = 0;
    return TransformFromRT(R, T0);
}
