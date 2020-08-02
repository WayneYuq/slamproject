#include <iostream>
#include <vector>
#include <fstream>

#include <sophus/se3.h>

using namespace std;
using Sophus::SE3;

#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

// parameters
const int boarder = 20; // 边缘宽度
const int width = 640;  // 宽度
const int height = 480; // 高度
const double fx = 481.2f; // 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 2; // NCC取的窗口半宽度
const int ncc_area = (2*ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1; // 收敛判定：最小方差
const double max_cov = 10;  // 发散判定：最大方差

bool readDatasetFiles(
const string& path,
vector<string>& color_image_files,
vector<SE3>& poses
);


