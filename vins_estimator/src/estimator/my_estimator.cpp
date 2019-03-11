/*******************************************************
 * Author: Weizhe Liu
 * Date: 2019/3
 *******************************************************/

#include "my_estimator.h"
#include "../my_transform.h"

constexpr int MIN_GOOD_DISPARITY = 10;
constexpr int MIN_GOOD_PAIRS = 15;

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
    const Eigen::Matrix3d R, const Eigen::Vector3d T, std::vector<Eigen::Vector3f>& points)
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

MyEstimator::MyEstimator(): f_manager_{Rs_}
{
    ROS_INFO("init begins");
}

void MyEstimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic_[i] = TIC[i];
        ric_[i] = RIC[i];
        std::cout << " exitrinsic cam " << i << std::endl  << ric_[i] << endl << tic_[i].transpose() << std::endl;
    }
    f_manager_.setRic(ric_);
    feature_tracker_.readIntrinsicParameter(CAM_NAMES);
    feature_tracker_.setCropRegion(0, 604-440, 3000, 604-(604-440)*2);
}

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

void MyEstimator::inputOdometry(double time, const Eigen::Vector3f& pose)
{
    std::lock_guard<std::mutex> lk(odom_buf_mutex_);
    odom_buf_.emplace_back(time, pose);
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
        auto pose1 = getInterpolatedOdomPose(iter_best->time.first);
        auto pose2 = getInterpolatedOdomPose(iter_best->time.second);
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

        Eigen::Vector3f cam_pose_translation(X2(0,3) - X1(0,3), X2(1,3) - X1(1,3), X2(2,3) - X1(2,3));
        float dist = cam_pose_translation.norm();
        Translation = dist * Translation; //correct the scale

        // auto odom_trans = (*pose2 - *pose1).head<2>();
        // std::cout << " trans " << odom_trans.norm() << " " << dist << std::endl;
    }

    // calculate 3D position of pairs, publish as point cloud
    std::vector<Eigen::Vector3f> points_3d;
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

std::unique_ptr<Eigen::Vector3f> MyEstimator::getInterpolatedOdomPose(double time)
{
    std::lock_guard<std::mutex> lk(odom_buf_mutex_);
    if (odom_buf_.size() < 2) return nullptr;
    if (odom_buf_.front().first >= time || odom_buf_.back().first < time) return nullptr;

    auto iter = std::lower_bound(odom_buf_.begin(), odom_buf_.end(), time, 
        [](const std::pair<double, Eigen::Vector3f>& pr, double time) {
            return pr.first < time;
        });

    if (iter != odom_buf_.end() && iter != odom_buf_.begin()) {
        auto iter_start = std::prev(iter);
        const double start_time = iter_start->first;
        const double end_time = iter->first;
        const Eigen::Vector2f start_trans = iter_start->second.head<2>();
        const Eigen::Vector2f end_trans = iter->second.head<2>();
        const Eigen::Quaternionf start_rot = QuatFromRPY(0.f,0.f,iter_start->second.z());
        const Eigen::Quaternionf end_rot = QuatFromRPY(0.f,0.f,iter->second.z());
        const double factor = (time - start_time) / (end_time - start_time);
        const Eigen::Vector2f origin = start_trans + (end_trans - start_trans) * factor;
        const Eigen::Quaternionf rotation = start_rot.slerp(factor, end_rot);

        Eigen::Vector3f new_pose;
        new_pose.x() = origin.x();
        new_pose.y() = origin.y();
        new_pose.z() = RPYFromQuat(rotation)[2];

        return std::unique_ptr<Eigen::Vector3f>(new Eigen::Vector3f(new_pose));
    }

    return nullptr;
}


