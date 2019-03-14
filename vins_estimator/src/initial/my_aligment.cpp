
/*******************************************************
 * Author: Weizhe Liu
 * Date: 2019/3
 *******************************************************/

#include "my_aligment.h"

bool VisualOdomAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::VectorXd &x)
{
    // only use odom distance as constraint, absolute pose by odom is inaccurate.
    // so the problem is min || Tcam * scale - D_odom ||(2)
    // which is Tcam.solve(D_odom)
    TicToc ttc;
    int all_frame_count = all_image_frame.size();
    if (all_frame_count < 2) {
        return false;
    }

    int n_state = all_frame_count - 1; 
    
    Eigen::MatrixXd A(n_state, 1);
    A.setZero();
    Eigen::VectorXd b(n_state);
    b.setZero();

    int i = 0;
    for (auto iter_i = all_image_frame.begin(); std::next(iter_i) != all_image_frame.end(); ++iter_i, ++i)
    {
        auto iter_j = std::next(iter_i);
        const auto& frame_i = iter_i->second;
        const auto& frame_j = iter_j->second;
        //printf("i=%d, %d, %d\n",i,A.size(), b.size());
        // X2 = RelativePose * X1 => RelativePose = [R2*R1.inv -R2*T1+T2]
        A(i, 0) = (frame_j.T - frame_j.R.transpose() * frame_i.T).norm(); 
        b(i) = (frame_j.p_odom - frame_j.q_odom.inverse() * frame_i.p_odom).norm();
    }

    x = A.colPivHouseholderQr().solve(b);

    double s = x.tail(1)(0);
    ROS_DEBUG("estimated scale: %f", s);
    //g = x.segment<3>(n_state - 4);
    ROS_DEBUG("solver time: %f\n", ttc.toc());
    if(s <= 0.0)
        return false;   
    else
        return true;
}
