/*******************************************************
 * Author: Weizhe Liu
 * Date: 2019/3
 *******************************************************/

#include "my_estimator.h"
#include "../my_transform.h"
#include "../factor/odom_functor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"

constexpr int MIN_GOOD_DISPARITY = 10;
constexpr int MIN_GOOD_PAIRS = 15;

#if 0
struct FeaturePairInfo {
    std::pair<double,double> time;
    std::vector<std::pair<MyEstimator::FeaturePoint,MyEstimator::FeaturePoint>> feature_pairs;
};

static std::vector<std::pair<MyEstimator::FeaturePoint,MyEstimator::FeaturePoint>> 
filter_small_disparity(const std::map<int,MyEstimator::FeaturePoint>& features1, 
                    const std::map<int,MyEstimator::FeaturePoint>& features2)
{
    std::vector<std::pair<MyEstimator::FeaturePoint,MyEstimator::FeaturePoint>> feature_pairs;
    for (const auto& pr1 : features1) {
        const auto iter_found = features2.find(pr1.first);
        if (iter_found != features2.end()) {
            const auto& uv1 = pr1.second.uv;
            const auto& uv2 = iter_found->second.uv;
            Eigen::Vector2f uv_diff = uv1 - uv2;
            auto disparity = std::abs(uv_diff.x()) + std::abs(uv_diff.y()); //L1 norm is enough
            if (disparity > MIN_GOOD_DISPARITY) 
                feature_pairs.emplace_back(pr1.second, iter_found->second);
        }
    }
    return feature_pairs;
}

bool solve_relative_RT(const std::vector<std::pair<MyEstimator::FeaturePoint,MyEstimator::FeaturePoint>>& feature_pairs, 
    Eigen::Matrix3d &R, Eigen::Vector3d &T)
{
    if (feature_pairs.size() >= 15)
    {
        std::vector<cv::Point2f> ll, rr;
        for (const auto& pr : feature_pairs)
        {
            ll.push_back(cv::Point2f(pr.first.xyz(0), pr.first.xyz(1)));
            rr.push_back(cv::Point2f(pr.second.xyz(0), pr.second.xyz(1)));
        }
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;

        // Eigen::Matrix3d R;
        // Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        // Inverse it: Opencv provide RT as cam2 to cam1 transform
        // Rotation = R.transpose();
        // Translation = -R.transpose() * T;

        if(inlier_cnt > 12)
            return true;
        else
            return false;
    }
    return false;
}

void triangulate_points(const std::vector<std::pair<MyEstimator::FeaturePoint,MyEstimator::FeaturePoint>>& feature_pairs, 
    const Eigen::Matrix3d R, const Eigen::Vector3d T, std::vector<Eigen::Vector3d>& points)
{
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), T(0),
                                 R(1, 0), R(1, 1), R(1, 2), T(1),
                                 R(2, 0), R(2, 1), R(2, 2), T(2));

    std::vector<cv::Point2f> ll, rr;
    for (const auto& pr : feature_pairs)
    {
        ll.push_back(cv::Point2f(pr.first.xyz(0), pr.first.xyz(1)));
        rr.push_back(cv::Point2f(pr.second.xyz(0), pr.second.xyz(1)));
    }
    cv::Mat points_4d;
    cv::triangulatePoints(P, P1, ll, rr, points_4d);

    points.clear();
    for (int i = 0; i < points_4d.cols; i++)
    {
        cv::Mat x = points_4d.col(i);
        x /= x.at<float>(3,0); // divide by w
        points.emplace_back(x.at<float>(0,0), x.at<float>(1,0), x.at<float>(2,0));
    }
}
#endif

MyEstimator::MyEstimator(ros::NodeHandle &n): 
    last_marginalization_info_(nullptr), f_manager_(Rs_), visualizer_(n, *this) 
{
    ROS_INFO("init begins");
    clearState();

    prevTime_ = -1;
    curTime_ = 0;
} 

void MyEstimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic_[i] = TIC[i];
        ric_[i] = RIC[i];
        std::cout << " exitrinsic cam " << i << std::endl  << ric_[i] << endl << tic_[i].transpose() << std::endl;
    }
    td_ = TD;
    f_manager_.setRic(ric_);
    feature_tracker_.readIntrinsicParameter(CAM_NAMES);
    feature_tracker_.setCropRegion(0, 604-440, 3000, 604-(604-440)*2);
}

void MyEstimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs_[i].setIdentity();
        Ps_[i].setZero();
        Vs_[i].setZero();
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic_[i] = Vector3d::Zero();
        ric_[i] = Matrix3d::Identity();
    }

    initialized_ = false;
    marginalize_old_ = true;
    sum_of_back_ = 0;
    sum_of_front_ = 0;
    frame_count_ = 0;

    for(auto& fr : all_image_frame_) {
        delete fr.second.pre_integration;
        fr.second.pre_integration = nullptr;
    }
    all_image_frame_.clear();

    delete last_marginalization_info_;
    last_marginalization_info_ = nullptr;
    last_marginalization_parameter_blocks_.clear();

    f_manager_.clearState();

    failure_occur_ = false;
}

void MyEstimator::inputOdometry(double time, const Eigen::Vector3d& pose)
{
    std::lock_guard<std::mutex> lk(odom_buf_mutex_);
    odom_buf_.emplace_back(time, pose);
}

#if 1

void MyEstimator::inputImage(double t, const cv::Mat &_img)
{
    TicToc featureTrackerTime;
    auto featureFrame = feature_tracker_.trackImage(t, _img);

    //printf("featureTracker time: %f\n", featureTrackerTime.toc());
    
    {
        std::lock_guard<std::mutex> lk(feature_buf_mutex_);
        feature_buf_.push(make_pair(t, featureFrame));
    }
    
    TicToc processTime;
    processMeasurements();
    auto time_cost = processTime.toc();
    if (time_cost > 100.)
        ROS_WARN("process time: %f ms", processTime.toc());
}

void MyEstimator::processMeasurements()
{
    //printf("process measurments\n");
    std::pair<double, std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
    Eigen::Vector3d delta_pose;
    Eigen::Vector3d odom_pose;
    {
        std::lock_guard<std::mutex> lk(feature_buf_mutex_);
        if (feature_buf_.empty()) return;

        feature = feature_buf_.front();
        feature_buf_.pop();
        curTime_ = feature.first + td_;
    }
    
    {
        std::unique_lock<std::mutex> lk(odom_buf_mutex_);  
        if (odom_buf_.empty() || (curTime_ < odom_buf_.begin()->first))
        {
            return; //skip image
        }  
        else
        {
            while(curTime_ > odom_buf_.back().first) { //waiting for more odom data
                lk.unlock(); //unlock to allow new data be inserted
                std::chrono::milliseconds dura(5);
                std::this_thread::sleep_for(dura);
                lk.lock();
            }

            if (prevTime_ > 0) 
            {
                auto pose_result = getOdomInterval(prevTime_, curTime_);
                if (pose_result) {
                    odom_pose = pose_result->first;
                    delta_pose = pose_result->second;
                }
                else {
                    ROS_WARN("Cannot interpolate odom pose!");
                    return;
                }
            }
        }
    }

    if (prevTime_ > 0) 
    {
        auto t = feature.first;
        processOdom(curTime_ - prevTime_, delta_pose);

        processImage(feature.second, t, odom_pose);

        visualizer_.printStatistics(0);

        visualizer_.pubOdometry(t);
        visualizer_.pubKeyPoses(t);
        visualizer_.pubCameraPose(t);
        visualizer_.pubPointCloud(t);
        visualizer_.pubKeyframe();
        visualizer_.pubTF(t);
    }

    prevTime_ = curTime_;
}

std::unique_ptr<std::pair<Eigen::Vector3d,Eigen::Vector3d>> MyEstimator::getOdomInterval(double t1, double t2)
{
    //asume buffer lock aquired.

    auto pose1 = getInterpolatedOdomPose(t1);
    auto pose2 = getInterpolatedOdomPose(t2);
    if (!pose1 || !pose2) return nullptr;

    Eigen::Vector3d delta_pose = pose2->cast<double>() - pose1->cast<double>();
    return std::unique_ptr<std::pair<Eigen::Vector3d,Eigen::Vector3d>>(new std::pair<Eigen::Vector3d,Eigen::Vector3d>(*pose2, delta_pose));
}

void MyEstimator::processOdom(double dt, const Eigen::Vector3d &delta_pose)
{
    if (frame_count_ != 0)
    {
        int j = frame_count_;
        Rs_[j] *= QuatFromRPY(0.,0.,delta_pose[2]).normalized().toRotationMatrix();
        Ps_[j] += Eigen::Vector3d(delta_pose[0], delta_pose[1], 0.);
        //Vs_[j] += dt * un_acc;
    }
}

void MyEstimator::processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, 
        const double header, const Eigen::Vector3d& odom_pose)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager_.addFeatureCheckParallax(frame_count_, image, td_))
    {
        marginalize_old_ = true;
        //printf("keyframe\n");
    }
    else
    {
        marginalize_old_ = false;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalize_old_ ? "Keyframe" : "Non-keyframe");
    ROS_DEBUG("Solving %d", frame_count_);
    ROS_DEBUG("number of feature: %d", f_manager_.getFeatureCount());
    headers_[frame_count_] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = nullptr;//new IntegrationBase;
    imageframe.p_odom = Eigen::Vector3d(odom_pose[0], odom_pose[1], 0);
    imageframe.q_odom = QuatFromRPY(0.,0.,odom_pose[2]);
    all_image_frame_.insert(std::make_pair(header, imageframe));

    if (!initialized_)
    {
        if (frame_count_ == WINDOW_SIZE)
        {
            if(initialStructure())
            {
                initialized_ = true;
                optimization();
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
            else
                slideWindow();
        }

        if(frame_count_ < WINDOW_SIZE)
        {
            frame_count_++;
            int prev_frame = frame_count_ - 1;
            Ps_[frame_count_] = Ps_[prev_frame];
            Vs_[frame_count_] = Vs_[prev_frame];
            Rs_[frame_count_] = Rs_[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;

        // if(!USE_IMU)
        //     f_manager_.initFramePoseByPnP(frame_count_, Ps_, Rs_, tic_, ric_);
        
        // Weizhe Liu: align odom to avoid scale drift, similar as we do when init.
        // Odom is not used when do BA, because odom is not well modeled as IMU.
        // So this is just a loosely coupled optmization like in initialization.
        // visualOptimizeAlign();

        f_manager_.triangulate(frame_count_, Ps_, Rs_, tic_, ric_);

        optimization();
        std::set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager_.removeOutlier(removeIndex);
        //if (! MULTIPLE_THREAD)
        {
            // feature_tracker_.removeOutliers(removeIndex);
            // predictPtsInNextFrame();
        }
            
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur_ = true;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager_.removeFailures();
        // prepare output of VINS
        key_poses_.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses_.push_back(Ps_[i]);

        last_R_ = Rs_[WINDOW_SIZE];
        last_P_ = Ps_[WINDOW_SIZE];
        last_R0_ = Rs_[0];
        last_P0_ = Ps_[0];
        //updateLatestStates();
    }  
}

bool MyEstimator::initialStructure()
{
    TicToc t_sfm;
    
    // global sfm
    Eigen::Quaterniond Q[frame_count_ + 1];
    Eigen::Vector3d T[frame_count_ + 1];
    std::map<int, Eigen::Vector3d> sfm_tracked_points;
    std::vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager_.feature_)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Eigen::Matrix3d relative_R;
    Eigen::Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count_ + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalize_old_ = true;
        return false;
    }

    // std::cout << "==== TIC " << TIC[0].transpose() << std::endl 
    //             << " RIC " << RPYFromQuat(Eigen::Quaterniond(RIC[0])).transpose() << std::endl;

    // for (int i = 0; i <= frame_count_;++i) {
    //     std::cout << "==== i " << i << " Ts " << T[i].transpose() << std::endl 
    //             << " Qs " << RPYFromQuat(Eigen::Quaterniond(Q[i])).transpose() << std::endl;
    // }
    //printf("t_sfm %i time: %f\n", (int)sfm_f.size(), t_sfm.toc());
    //solve pnp for all frame
    std::map<double, ImageFrame>::iterator frame_it;
    std::map<int, Eigen::Vector3d>::iterator it;
    frame_it = all_image_frame_.begin( );
    for (int i = 0; frame_it != all_image_frame_.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == headers_[i])
        {
            frame_it->second.is_key_frame = true;
            // frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            // frame_it->second.T = T[i];

            // Weizhe Liu: convert R/t to be in body (IMU) coordinate
            // which means R/t is the camera pose in frame0's body coord 
            frame_it->second.R = Eigen::Quaterniond(RIC[0]) * Q[i];
            frame_it->second.T = RIC[0] * T[i] + TIC[0];
            i++;
            continue;
        }
        if((frame_it->first) > headers_[i])
        {
            i++;
        }
        Eigen::Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Eigen::Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        std::vector<cv::Point3f> pts_3_vector;
        std::vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Eigen::Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Eigen::Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        Eigen::MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        Eigen::MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        // frame_it->second.R = R_pnp * RIC[0].transpose();
        // frame_it->second.T = T_pnp;
        frame_it->second.R = RIC[0] * R_pnp;
        frame_it->second.T = RIC[0] * T_pnp + TIC[0];
    }
    bool align_ok = visualInitialAlign();
    if (align_ok)
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}


bool MyEstimator::visualInitialAlign()
{
    TicToc t_g;
    Eigen::VectorXd x;
    //solve scale
    bool result = VisualOdomAlignment(all_image_frame_, x);
    if(!result)
    {
        ROS_DEBUG("visual init align failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count_; i++)
    {
        Eigen::Matrix3d Ri = all_image_frame_ [headers_[i]].R;
        Eigen::Vector3d Pi = all_image_frame_ [headers_[i]].T;
        Ps_[i] = Pi;
        Rs_[i] = Ri;
        all_image_frame_ [headers_[i]].is_key_frame = true;
    }

    // for (int i = 0; i <= frame_count_;++i) {
    //     std::cout << "==== i " << i << " Ts " << Ps_[i].transpose() << std::endl 
    //             << " Rs " << RPYFromQuat(Eigen::Quaterniond(Rs_[i])).transpose() << std::endl;
    // }

    double s = (x.tail<1>())(0);
    // for (int i = frame_count_; i >= 0; i--)
    //     Ps_[i] = s * Ps_[i] - Rs_[i] * TIC[0] - (s * Ps_[0] - Rs_[0] * TIC[0]);

    // Weizhe Liu: update pose according to scale
    for (int i = 1; i <= frame_count_; i++)
        Ps_[i] = Rs_[i] * Rs_[i-1].transpose() * Ps_[i-1] + s * (Ps_[i] - Rs_[i] * Rs_[i-1].transpose() * Ps_[i-1]);
    // and set Pose i be relative pose of Pose i to Pose 0,
    // thus make all coord base on first frame's coord. 
    for (int i = 1; i <= frame_count_; i++) {
        Ps_[i] = s * (Ps_[i] - Rs_[i] * Rs_[0].transpose() * Ps_[0]);
        Rs_[i] = Rs_[i] * Rs_[0].transpose();
    }
    Rs_[0].setIdentity();
    Ps_[0].setZero();

    ROS_INFO("visual init align: scale %f", s); 

    // for (int i = 0; i <= frame_count_;++i) {
    //     std::cout << "==== i " << i << " Ts " << Ps_[i].transpose() << std::endl 
    //             << " Rs " << RPYFromQuat(Eigen::Quaterniond(Rs_[i])).transpose() << std::endl;
    // }
    
    f_manager_.clearDepth();
    f_manager_.triangulate(frame_count_, Ps_, Rs_, tic_, ric_);

    return true;
}

// different from visualInitialAlign(), this function align Rs/Ps instead of camera pose which is in camera 0's coord.
bool MyEstimator::visualOptimizeAlign()
{
    TicToc t_g;
    Eigen::VectorXd x;
    //solve scale
    bool result = VisualOdomAlignmentOpt(frame_count_, headers_, Rs_, Ps_, all_image_frame_, x);
    if(!result)
    {
        ROS_INFO("visual opt align failed!");
        return false;
    }

    double s = (x.tail<1>())(0);
    for (int i = 1; i <= frame_count_; i++)
        Ps_[i] = Rs_[i] * Rs_[i-1].transpose() * Ps_[i-1] + s * (Ps_[i] - Rs_[i] * Rs_[i-1].transpose() * Ps_[i-1]);

    ROS_INFO("visual optimize align: scale %f", s); 

    f_manager_.clearDepth();
    //f_manager_.triangulate(frame_count_, Ps_, Rs_, tic_, ric_); //no need as it will be called by outside caller.
    return true;
}

bool MyEstimator::relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
        corres = f_manager_.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator_.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}


void MyEstimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose_[i][0] = Ps_[i].x();
        para_Pose_[i][1] = Ps_[i].y();
        para_Pose_[i][2] = Ps_[i].z();
        Eigen::Quaterniond q{Rs_[i]};
        para_Pose_[i][3] = q.x();
        para_Pose_[i][4] = q.y();
        para_Pose_[i][5] = q.z();
        para_Pose_[i][6] = q.w();
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose_[i][0] = tic_[i].x();
        para_Ex_Pose_[i][1] = tic_[i].y();
        para_Ex_Pose_[i][2] = tic_[i].z();
        Eigen::Quaterniond q{ric_[i]};
        para_Ex_Pose_[i][3] = q.x();
        para_Ex_Pose_[i][4] = q.y();
        para_Ex_Pose_[i][5] = q.z();
        para_Ex_Pose_[i][6] = q.w();
    }


    Eigen::VectorXd dep = f_manager_.getDepthVector();
    for (int i = 0; i < f_manager_.getFeatureCount(); i++)
        para_Feature_[i][0] = dep(i);

    para_Td_[0][0] = td_;
}

void MyEstimator::double2vector()
{
    Eigen::Vector3d origin_R0 = Utility::R2ypr(Rs_[0]);
    Eigen::Vector3d origin_P0 = Ps_[0];

    if (failure_occur_)
    {
        origin_R0 = Utility::R2ypr(last_R0_);
        origin_P0 = last_P0_;
        failure_occur_ = false;
    }

    //if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose_[0][6],
                                                          para_Pose_[0][3],
                                                          para_Pose_[0][4],
                                                          para_Pose_[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs_[0] * Eigen::Quaterniond(para_Pose_[0][6],
                                           para_Pose_[0][3],
                                           para_Pose_[0][4],
                                           para_Pose_[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs_[i] = rot_diff * Eigen::Quaterniond(para_Pose_[i][6], para_Pose_[i][3], para_Pose_[i][4], para_Pose_[i][5]).normalized().toRotationMatrix();
            
            Ps_[i] = rot_diff * Eigen::Vector3d(para_Pose_[i][0] - para_Pose_[0][0],
                                    para_Pose_[i][1] - para_Pose_[0][1],
                                    para_Pose_[i][2] - para_Pose_[0][2]) + origin_P0;
            
        }
    }


    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic_[i] = Eigen::Vector3d(para_Ex_Pose_[i][0],
                            para_Ex_Pose_[i][1],
                            para_Ex_Pose_[i][2]);
        ric_[i] = Eigen::Quaterniond(para_Ex_Pose_[i][6],
                                para_Ex_Pose_[i][3],
                                para_Ex_Pose_[i][4],
                                para_Ex_Pose_[i][5]).toRotationMatrix();
    }

    Eigen::VectorXd dep = f_manager_.getDepthVector();
    for (int i = 0; i < f_manager_.getFeatureCount(); i++)
        dep(i) = para_Feature_[i][0];
    f_manager_.setDepth(dep);

    td_ = para_Td_[0][0];

}


bool MyEstimator::failureDetection()
{
    return false;
}

void MyEstimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count_ + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose_[i], SIZE_POSE, local_parameterization);
    }

    Eigen::Vector3d V0 = (Ps_[1] - Rs_[1] * Ps_[0]) / (headers_[1] - headers_[0]);
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose_[i], SIZE_POSE, local_parameterization);
        if (!(ESTIMATE_EXTRINSIC && frame_count_ == WINDOW_SIZE && V0.norm() > 0.2))
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose_[i]);
        }
    }
    problem.AddParameterBlock(para_Td_[0], 1);

    if (!ESTIMATE_TD || V0.norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td_[0]);

    if (last_marginalization_info_ && last_marginalization_info_->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info_);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks_);
    }

    for (int i = 0; i < frame_count_; i++)
    {
        int j = i + 1;
        auto p1 = all_image_frame_[headers_[j]].p_odom;
        auto p2 = all_image_frame_[headers_[i]].p_odom;
        auto q1 = all_image_frame_[headers_[j]].q_odom;
        auto q2 = all_image_frame_[headers_[i]].q_odom;
        Eigen::Vector3d delta_p = p2 - q2 * q1.inverse() * p1;
        Eigen::Quaterniond delta_q = q2 * q1.inverse();
        problem.AddResidualBlock(OdomFunctor::Create(delta_p, delta_q), NULL, para_Pose_[i], para_Pose_[j]);
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager_.feature_)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Eigen::Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].td, it_per_frame.td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose_[imu_i], para_Pose_[imu_j], para_Ex_Pose_[0], para_Feature_[feature_index], para_Td_[0]);
            }

            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalize_old_)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    double2vector();
    //printf("frame_count_: %d \n", frame_count_);

    if(frame_count_ < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    if (marginalize_old_)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info_ && last_marginalization_info_->valid)
        {
            std::vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks_.size()); i++)
            {
                if (last_marginalization_parameter_blocks_[i] == para_Pose_[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info_);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks_,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            int i = 0;
            int j = 1;
            auto p1 = all_image_frame_[headers_[j]].p_odom;
            auto p2 = all_image_frame_[headers_[i]].p_odom;
            auto q1 = all_image_frame_[headers_[j]].q_odom;
            auto q2 = all_image_frame_[headers_[i]].q_odom;
            Eigen::Vector3d delta_p = p2 - q2 * q1.inverse() * p1;
            Eigen::Quaterniond delta_q = q2 * q1.inverse();
            ceres::CostFunction* odom_factor = OdomFunctor::Create(delta_p, delta_q);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(odom_factor, NULL,
                                                                           std::vector<double *>{para_Pose_[i], para_Pose_[j]},
                                                                           std::vector<int>{i, j});
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager_.feature_)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Eigen::Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].td, it_per_frame.td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        std::vector<double *>{para_Pose_[imu_i], para_Pose_[imu_j], para_Ex_Pose_[0], para_Feature_[feature_index], para_Td_[0]},
                                                                                        std::vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose_[i])] = para_Pose_[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose_[i])] = para_Ex_Pose_[i];

        addr_shift[reinterpret_cast<long>(para_Td_[0])] = para_Td_[0];

        std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info_)
            delete last_marginalization_info_;
        last_marginalization_info_ = marginalization_info;
        last_marginalization_parameter_blocks_ = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info_ &&
            std::count(std::begin(last_marginalization_parameter_blocks_), std::end(last_marginalization_parameter_blocks_), para_Pose_[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info_ && last_marginalization_info_->valid)
            {
                std::vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks_.size()); i++)
                {
                    //ROS_ASSERT(last_marginalization_parameter_blocks_[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks_[i] == para_Pose_[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info_);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks_,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose_[i])] = para_Pose_[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose_[i])] = para_Pose_[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose_[i])] = para_Ex_Pose_[i];

            addr_shift[reinterpret_cast<long>(para_Td_[0])] = para_Td_[0];

            
            std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info_)
                delete last_marginalization_info_;
            last_marginalization_info_ = marginalization_info;
            last_marginalization_parameter_blocks_ = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void MyEstimator::slideWindow()
{
    TicToc t_margin;
    if (marginalize_old_)
    {
        double t_0 = headers_[0];
        back_R0_ = Rs_[0];
        back_P0_ = Ps_[0];
        if (frame_count_ == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                headers_[i] = headers_[i + 1];
                Rs_[i].swap(Rs_[i + 1]);
                Ps_[i].swap(Ps_[i + 1]);
                
                Vs_[i].swap(Vs_[i + 1]);
            }
            headers_[WINDOW_SIZE] = headers_[WINDOW_SIZE - 1];
            Ps_[WINDOW_SIZE] = Ps_[WINDOW_SIZE - 1];
            Rs_[WINDOW_SIZE] = Rs_[WINDOW_SIZE - 1];

            Vs_[WINDOW_SIZE] = Vs_[WINDOW_SIZE - 1];

            if (true || !initialized_)
            {
                std::map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame_.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
                all_image_frame_.erase(all_image_frame_.begin(), it_0);


                if (!all_image_frame_.empty())
                {
                    double t = all_image_frame_.begin()->first;
                    std::lock_guard<std::mutex> lk(odom_buf_mutex_);
                    while (!odom_buf_.empty() && odom_buf_.front().first < t)
                        odom_buf_.pop_front();
                }
            }

            slideWindowOld();
        }
    }
    else
    {
        if (frame_count_ == WINDOW_SIZE)
        {
            headers_[frame_count_ - 1] = headers_[frame_count_];
            Ps_[frame_count_ - 1] = Ps_[frame_count_];
            Rs_[frame_count_ - 1] = Rs_[frame_count_];

            Vs_[frame_count_ - 1] = Vs_[frame_count_];

            slideWindowNew();
        }
    }
}

void MyEstimator::slideWindowNew()
{
    sum_of_front_++;
    f_manager_.removeFront(frame_count_);
}

void MyEstimator::slideWindowOld()
{
    sum_of_back_++;

    bool shift_depth = initialized_;
    if (shift_depth)
    {
        Eigen::Matrix3d R0, R1;
        Eigen::Vector3d P0, P1;
        R0 = back_R0_ * ric_[0];
        R1 = Rs_[0] * ric_[0];
        P0 = back_P0_ + back_R0_ * tic_[0];
        P1 = Ps_[0] + Rs_[0] * tic_[0];
        f_manager_.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager_.removeBack();
}



void MyEstimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs_[frame_count_];
    T.block<3, 1>(0, 3) = Ps_[frame_count_];
}

void MyEstimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs_[index];
    T.block<3, 1>(0, 3) = Ps_[index];
}

void MyEstimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count_ < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count_ - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    std::map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager_.feature_)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count_, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count_)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric_[0] * (depth * it_per_id.feature_per_frame[0].point) + tic_[0];
                Vector3d pts_w = Rs_[firstIndex] * pts_j + Ps_[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric_[0].transpose() * (pts_local - tic_[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    feature_tracker_.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double MyEstimator::reprojectionError(Eigen::Matrix3d &Ri, Eigen::Vector3d &Pi, Eigen::Matrix3d &rici, Eigen::Vector3d &tici,
                                 Eigen::Matrix3d &Rj, Eigen::Vector3d &Pj, Eigen::Matrix3d &ricj, Eigen::Vector3d &ticj, 
                                 double depth, Eigen::Vector3d &uvi, Eigen::Vector3d &uvj)
{
    Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return std::sqrt(rx * rx + ry * ry);
}

void MyEstimator::outliersRejection(std::set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager_.feature_)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs_[imu_i], Ps_[imu_i], ric_[0], tic_[0], 
                                                    Rs_[imu_j], Ps_[imu_j], ric_[0], tic_[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}


#else

void MyEstimator::inputImage(double time, const cv::Mat& img)
{
    TicToc featureTrackerTime;

    auto fm = feature_tracker_.trackImage(time, img);
    
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());
    
    
    {
        FeatureFrame ff;
        ff.time = time;
        for (const auto& pr: fm) {
            assert(pr.second.size() == 1); //only for 1 camera for now
            const auto& m7 = pr.second.begin()->second;
            FeaturePoint fp;
            fp.xyz.x() = m7[0];
            fp.xyz.y() = m7[1];
            fp.xyz.z() = m7[2];
            fp.uv.x() = m7[3];
            fp.uv.y() = m7[4];
            ff.features.emplace(pr.first, fp);
        }
        std::lock_guard<std::mutex> lk(feature_buf_mutex_);
        feature_buf_.emplace_back(ff);
    }
    
    
    TicToc processTime;
    processMeasurements();
    //printf("process time: %f\n", processTime.toc());
    
}

void MyEstimator::processMeasurements()
{
    
    // find some pairs within some prev frames
    double frame_time; 
    std::vector<FeaturePairInfo> featurePairs;
    double lost_track_time = 0;
    {
        std::lock_guard<std::mutex> lk(feature_buf_mutex_);
        auto iter_last = feature_buf_.rbegin();
        frame_time = iter_last->time;
        for (auto iter = std::next(iter_last); iter != feature_buf_.rend(); ++iter) {
            auto goodPairs = filter_small_disparity(iter_last->features, iter->features);
            if (goodPairs.size() >= MIN_GOOD_PAIRS) {
                std::pair<double,double> tp(iter_last->time, iter->time);
                featurePairs.push_back({tp, goodPairs});
            }
            else {
                if (!featurePairs.empty()) {
                    // lost track
                    lost_track_time = iter->time;
                    break;
                }
            }
        }
    }
    if (featurePairs.empty()) {
        std::cout << "no pairs" << std::endl;
        return;
    }
    
    //std::cout << "good pairs " << featurePairs.size() << std::endl;

    //auto iter_best = std::prev(featurePairs.end());
    auto iter_best = std::max_element(featurePairs.begin(), featurePairs.end(),
                 [](const FeaturePairInfo& a, const FeaturePairInfo& b) { 
                     return a.feature_pairs.size() < b.feature_pairs.size(); });

    //std::cout << "max " << iter_best->feature_pairs.size() << " off-end " << std::distance(iter_best, featurePairs.end()) << std::endl;

    // calculate transform of pairs
    Eigen::Matrix3d Rotation;
    Eigen::Vector3d Translation;
    if (!solve_relative_RT(iter_best->feature_pairs, Rotation, Translation)) {
        std::cout << "solve RT fail" << std::endl;
        return;
    }
    //std::cout << "T: " << Translation.z() << std::endl;

    // find related odom translation for pairs
    // use odom trans to correct visual scale
    {
        std::unique_ptr<Eigen::Vector3d> pose1,pose2;
        {
            std::lock_guard<std::mutex> lk(odom_buf_mutex_);
    
            pose1 = getInterpolatedOdomPose(iter_best->time.first);
            pose2 = getInterpolatedOdomPose(iter_best->time.second);
        }
        if (!pose1 || !pose2) {
            std::cout << "no odom interpolate" << std::endl;
            return;
        }

        Eigen::Matrix4f CIB = TransformFromRT(ric_[0], tic_[0]).cast<float>();
        
        auto X1 = TransformFromRigid2(*pose1);
        auto X2 = TransformFromRigid2(*pose2);

        // Local coord to world coord: https://zhuanlan.zhihu.com/p/35943426
        // consider camera in odom coord as local coord, consider odom pose as the transform of local coord in world coord.
        X1 = X1 * CIB;
        X2 = X2 * CIB;

        Eigen::Vector3d cam_pose_translation(X2(0,3) - X1(0,3), X2(1,3) - X1(1,3), X2(2,3) - X1(2,3));
        float dist = cam_pose_translation.norm();
        Translation = dist * Translation; //correct the scale

        // auto odom_trans = (*pose2 - *pose1).head<2>();
        // std::cout << " trans " << odom_trans.norm() << " " << dist << std::endl;
    }

    // calculate 3D position of pairs, publish as point cloud
    std::vector<Eigen::Vector3d> points_3d;
    triangulate_points(iter_best->feature_pairs, Rotation, Translation, points_3d);

    if (points_cloud_callback_) {
        points_cloud_callback_(points_3d, frame_time);
    }

    // prune old feature/odom data
    if (lost_track_time > 0) 
    {
        {
            std::lock_guard<std::mutex> lk(feature_buf_mutex_);
            while (!feature_buf_.empty() && feature_buf_.front().time < lost_track_time)
                feature_buf_.pop_front();
        }
    
        {
            std::lock_guard<std::mutex> lk(odom_buf_mutex_);
            while (!odom_buf_.empty() && odom_buf_.front().first < lost_track_time)
                odom_buf_.pop_front();
        }
    }
}

#endif

std::unique_ptr<Eigen::Vector3d> MyEstimator::getInterpolatedOdomPose(double time)
{
    if (odom_buf_.size() < 2) return nullptr;
    if (odom_buf_.front().first > time || odom_buf_.back().first < time) return nullptr;

    auto iter = std::lower_bound(odom_buf_.begin(), odom_buf_.end(), time, 
        [](const std::pair<double, Eigen::Vector3d>& pr, double time) {
            return pr.first < time;
        });

    if (iter == odom_buf_.end()) return nullptr;

    if (iter == odom_buf_.begin()) {
        return std::unique_ptr<Eigen::Vector3d>(new Eigen::Vector3d(iter->second));
    } else {
        auto iter_start = std::prev(iter);
        const double start_time = iter_start->first;
        const double end_time = iter->first;
        const Eigen::Vector2d start_trans = iter_start->second.head<2>();
        const Eigen::Vector2d end_trans = iter->second.head<2>();
        const Eigen::Quaterniond start_rot = QuatFromRPY(0.,0.,iter_start->second.z());
        const Eigen::Quaterniond end_rot = QuatFromRPY(0.,0.,iter->second.z());
        const double factor = (time - start_time) / (end_time - start_time);
        const Eigen::Vector2d origin = start_trans + (end_trans - start_trans) * factor;
        const Eigen::Quaterniond rotation = start_rot.slerp(factor, end_rot);

        Eigen::Vector3d new_pose;
        new_pose.x() = origin.x();
        new_pose.y() = origin.y();
        new_pose.z() = RPYFromQuat(rotation)[2];

        return std::unique_ptr<Eigen::Vector3d>(new Eigen::Vector3d(new_pose));
    }
}


