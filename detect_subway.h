#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace detect_subway {
cv::MatND GetColorHistogram(cv::Mat input_hsv, cv::Mat mask);

double CosineSimilarity(const std::vector<float> &A,
                        const std::vector<float> &B);

std::vector<float> GetColorHistogramForCosine(cv::Mat input_hsv, cv::Mat mask);

cv::Mat PlotHistogram(cv::MatND hist);

int FeatureMatchingLogoDetection(cv::Mat logo_frame, cv::Mat frame,
                                 std::string debug_str);

std::pair<double, cv::Rect>
HistogramObjectDetection(cv::Mat logo_rgb, cv::Mat logo_hsv,
                         cv::MatND logo_hist, cv::Mat frame, cv::Mat logo_mask);

std::unordered_map<size_t, cv::Rect>
FindLogo(const std::vector<cv::Mat> &rgb_frames, cv::Mat logo,
         std::string mask_path);
void FindLogo2(const std::vector<cv::Mat> &rgb_frames, cv::Mat logo,
               std::string mask_path);
void FindLogoSingleFrame(const cv::Mat &rgb_frame, cv::Mat logo,
                         std::string mask_path);
} // namespace detect_subway