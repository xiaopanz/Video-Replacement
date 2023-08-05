#include "brand_detection.h"
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace detect_subway {
#define DEBUG_OUTPUT false

#define HIST_2D true

#if HIST_2D

// Quantize the hue to 30 levels
// and the saturation to 32 levels
constexpr int hbins = 30, sbins = 32;
constexpr int histSize[] = {hbins, sbins};
// hue varies from 0 to 179, see cvtColor
constexpr float hranges[] = {0, 180};
// saturation varies from 0 (black-gray-white) to
// 255 (pure spectrum color)
constexpr float sranges[] = {0, 256};
const float *ranges[] = {hranges, sranges};
// we compute the histogram from the 0-th and 1-st channels
constexpr int channels[] = {0, 1};

cv::MatND GetColorHistogram(cv::Mat input_hsv, cv::Mat mask) {

  cv::MatND hist;
  cv::calcHist(&input_hsv, 1, channels, mask, hist, 2, histSize, ranges,
               true, // the histogram is uniform
               false);
  for (int h = 0; h < hbins; h++) {
    for (int s = 0; s < sbins; s++) {

      // float binVal = hist.at<float>(h);
      float binVal = hist.at<float>(h, s);
      if (binVal < 0.01) {
        hist.at<float>(h, s) = 0;
      }
    }
  }
  cv::MatND normalized_hist;
  normalize(hist, normalized_hist, 1, 0, cv::NORM_L1, -1, cv::Mat());

  return normalized_hist;
}

cv::Mat PlotHistogram(cv::MatND hist) {
  double maxVal = 0;
  minMaxLoc(hist, 0, &maxVal, 0, 0);
  int scale = 10;
  cv::Mat hist_img = cv::Mat::zeros(sbins * scale, hbins * 10, CV_8UC3);
  for (int h = 0; h < hbins; h++) {
    for (int s = 0; s < sbins; s++) {

      // float binVal = hist.at<float>(h);
      float binVal = hist.at<float>(h, s);
      int intensity = cvRound(binVal * 255 / maxVal);

      rectangle(hist_img, cv::Point(h * scale, s * scale),
                cv::Point((h + 1) * scale - 1, (s + 1) * scale - 1),
                cv::Scalar::all(intensity), -1);
    }
  }
  return hist_img;
}

#else

// Quantize the hue to 30 levels
// and the saturation to 32 levels
constexpr int hbins = 30;
constexpr int histSize[] = {hbins};
// hue varies from 0 to 179, see cvtColor
constexpr float hranges[] = {0, 180};
const float *ranges[] = {hranges};
// we compute the histogram from the 0-th and 1-st channels
constexpr int channels[] = {0};

cv::MatND GetColorHistogram(cv::Mat input_hsv, cv::Mat mask) {

  cv::MatND hist;
  cv::calcHist(&input_hsv, 1, channels, mask, hist, 1, histSize, ranges,
               true, // the histogram is uniform
               false);
  for (int h = 0; h < hbins; h++) {

    // float binVal = hist.at<float>(h);
    float binVal = hist.at<float>(h);
    // if (binVal < 0.01) {
    //   hist.at<float>(h) = 0;
    // }
  }
  cv::MatND normalized_hist;
  normalize(hist, normalized_hist, 1, 0, cv::NORM_L1, -1, cv::Mat());

  return normalized_hist;
}

cv::Mat PlotHistogram(cv::MatND hist) {
  double maxVal = 0;
  minMaxLoc(hist, 0, &maxVal, 0, 0);
  int scale = 10;
  cv::Mat hist_img = cv::Mat::zeros(10 * scale, hbins * 10, CV_8UC3);
  for (int h = 0; h < hbins; h++) {
    for (int s = 0; s < 10; s++) {

      // float binVal = hist.at<float>(h);
      float binVal = hist.at<float>(h);
      int intensity = cvRound(binVal * 255 / maxVal);

      rectangle(hist_img, cv::Point(h * scale, s * scale),
                cv::Point((h + 1) * scale - 1, (s + 1) * scale - 1),
                cv::Scalar::all(intensity), -1);
    }
  }
  for (int h = 0; h < hbins; h++) {
    float binVal = hist.at<float>(h);
    std::cout << binVal << " ";
  }
  std::cout << std::endl;
  return hist_img;
}
#endif

int FeatureMatchingLogoDetection(cv::Mat logo_frame, cv::Mat frame,
                                 std::string debug_str) {
  // auto sift_detector = cv::SIFT::create(2000, 5);
  auto sift_detector = cv::SIFT::create();

  std::vector<cv::KeyPoint> logo_keypoints;
  cv::Mat logo_descriptors;
  sift_detector->detectAndCompute(logo_frame, cv::noArray(), logo_keypoints,
                                  logo_descriptors);

  std::vector<cv::KeyPoint> frame_keypoints;
  cv::Mat frame_descriptors;
  sift_detector->detectAndCompute(frame, cv::noArray(), frame_keypoints,
                                  frame_descriptors);

  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  if (logo_descriptors.empty() || frame_descriptors.empty()) {
    return 0;
  }

  matcher->knnMatch(frame_descriptors, logo_descriptors, knn_matches, 2);

  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance <
        ratio_thresh * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }

  if (!debug_str.empty()) {
    cv::Mat match_vis;
    cv::drawMatches(frame, frame_keypoints, logo_frame, logo_keypoints,
                    good_matches, match_vis);
    cv::imwrite("matching_output/m_" + debug_str + ".png", match_vis);
  }

  if (good_matches.size() < 4) {
    return 0;
  }
  std::vector<cv::Point2f> good_logo_points;
  std::vector<cv::Point2f> good_frame_points;
  for (const auto &match : good_matches) {
    good_logo_points.push_back(logo_keypoints[match.trainIdx].pt);
    good_frame_points.push_back(frame_keypoints[match.queryIdx].pt);
  }

  cv::Mat find_homo_mask;
  cv::Mat homography = cv::findHomography(good_logo_points, good_frame_points,
                                          cv::RANSAC, 2, find_homo_mask);
  int inlier_count = 0;
  for (int i = 0; i < find_homo_mask.rows; i++) {
    if (find_homo_mask.at<uchar>(i) != 0) {
      inlier_count++;
    }
  }

  return inlier_count;
}
std::pair<double, cv::Rect> HistogramObjectDetection(cv::Mat logo_rgb,
                                                     cv::Mat logo_hsv,
                                                     cv::MatND logo_hist,
                                                     cv::Mat frame,
                                                     cv::Mat logo_mask) {

  cv::Mat frame_hsv;
  cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

  int height = logo_hsv.rows;
  int width = logo_hsv.cols;
  double aspect_ratio =
      static_cast<double>(width) / static_cast<double>(height);

  int init_height = 60;
  int init_width = 60 * aspect_ratio;

  int curr_height = init_height;
  int curr_width = init_width;
  double scale_factor = 1.2;

  double frame_score = 0;
  cv::Rect frame_rect;
  while (curr_height <= height) {
    curr_width = std::min(curr_width, width);

    cv::Mat scaled_mask = logo_mask.clone();
    cv::resize(logo_mask, scaled_mask, cv::Size(curr_width, curr_height));

    double best_score = std::numeric_limits<double>::min();
    int max_i = -1, max_j = -1;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < frame.rows - curr_height; i += 1) {
      for (int j = 0; j < frame.cols - curr_width; j += 1) {
        cv::Mat sub_img_hsv = frame_hsv(cv::Range(i, i + curr_height),
                                        cv::Range(j, j + curr_width));
        cv::MatND sub_img_hist = GetColorHistogram(sub_img_hsv, cv::Mat());

        double intersect_score =
            cv::compareHist(logo_hist, sub_img_hist, cv::HISTCMP_INTERSECT);

        double bhtt_score = (1 - cv::compareHist(logo_hist, sub_img_hist,
                                                 cv::HISTCMP_BHATTACHARYYA));

        double score = bhtt_score;
#pragma omp critical
        if (score > best_score) {
          best_score = score;
          max_i = i;
          max_j = j;
        }
      }
    }

    if (DEBUG_OUTPUT) {
      std::cout << best_score << std::endl;
      if (max_i != -1) {

        cv::Mat sub_img = frame(cv::Range(max_i, max_i + curr_height),
                                cv::Range(max_j, max_j + curr_width));
        cv::Mat sub_img_scaled;
        cv::resize(sub_img, sub_img_scaled, logo_rgb.size());
        int num_matches = FeatureMatchingLogoDetection(
            sub_img_scaled, logo_rgb,
            std::to_string(curr_height) + "_" + std::to_string(curr_width));
        std::cout << sub_img.size << std::endl;
        std::cout << num_matches << std::endl;

        // std::cout << num_matches << std::endl;
        cv::imwrite("matching_output/" + std::to_string(curr_height) + "_" +
                        std::to_string(curr_width) + ".png",
                    sub_img);

        cv::MatND sub_img_hist = GetColorHistogram(sub_img, cv::Mat());
        cv::Mat histImg = PlotHistogram(sub_img_hist);

        cv::imwrite("matching_output/" + std::to_string(curr_height) + "_" +
                        std::to_string(curr_width) + "_hist.png",
                    histImg);
      }
    }
    if (frame_score < best_score && max_i != -1) {
      cv::Mat sub_img = frame(cv::Range(max_i, max_i + curr_height),
                              cv::Range(max_j, max_j + curr_width));
      cv::Mat sub_img_scaled;
      cv::resize(sub_img, sub_img_scaled, logo_rgb.size());
      int num_matches = FeatureMatchingLogoDetection(
          sub_img_scaled, logo_rgb,
          std::to_string(curr_height) + "_" + std::to_string(curr_width));
      // std::cout << num_matches << std::endl;
      if (num_matches >= 12) {

        frame_score = best_score;
        frame_rect = cv::Rect(cv::Point2d(max_j, max_i),
                              cv::Size(curr_width, curr_height));
      }
    }

    // Update the size of the sliding windows
    curr_height = curr_height * scale_factor;
    curr_width = curr_height * aspect_ratio;
  }

  return {frame_score, frame_rect};
}

cv::Mat CreateMaskForSubway(cv::Mat logo) {
  auto green = logo.at<cv::Vec3b>(1, 1);
  cv::Mat mask(logo.size(), CV_8UC1, cv::Scalar(0));

  for (int i = 0; i < logo.rows; i++) {
    for (int j = 0; j < logo.cols; j++) {
      auto pixel_rgb = logo.at<cv::Vec3b>(i, j);
      if (std::abs(pixel_rgb[0] - green[0]) < 5 &&
          std::abs(pixel_rgb[1] - green[1]) < 5 &&
          std::abs(pixel_rgb[2] - green[2]) < 5) {
        continue;
      }
      mask.at<uchar>(i, j) = 255;
    }
  }

  // cv::imwrite("subway_mask.png", mask);
  return mask;
}

std::unordered_map<size_t, cv::Rect>
MeanshiftTrackAcrossFrame(const std::vector<cv::Mat> &rgb_frames,
                          size_t frame_id, cv::Rect rect) {

  cv::Mat found_frame = rgb_frames[frame_id];

  cv::Rect tracking_window = rect;
  tracking_window.height = rect.height * 2;
  tracking_window.width = rect.width * 2.5;
  tracking_window.x = rect.x - rect.width / 2.5;
  tracking_window.y = rect.y - rect.height / 2;

  rect = tracking_window;
  cv::Mat roi = found_frame(tracking_window);
  cv::Mat roi_hsv;
  cv::cvtColor(roi, roi_hsv, cv::COLOR_BGR2HSV);

  std::unordered_map<size_t, cv::Rect> result;
  result[frame_id] = tracking_window;
  cv::Mat frame_clone = found_frame.clone();
  cv::rectangle(frame_clone, tracking_window, 255, 2);
  cv::imwrite("matching_output/track_" + std::to_string(frame_id) + ".png",
              frame_clone);

  cv::Mat mask;
  cv::inRange(roi_hsv, cv::Scalar(0, 60, 32), cv::Scalar(180, 255, 255), mask);
  float range_[] = {0, 180};
  const float *range[] = {range_};
  cv::Mat roi_hist;
  int histSize[] = {180};
  int channels[] = {0};
  cv::calcHist(&roi_hsv, 1, channels, mask, roi_hist, 1, histSize, range);
  cv::normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);
  // Setup the termination criteria, either 10 iteration or move by atleast 1 pt
  cv::TermCriteria term_crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                             10, 1);

  // Backward loop
  for (size_t i = frame_id - 1; i >= 0; i--) {
    cv::Mat frame = rgb_frames[i];
    cv::Mat frame_hsv;
    cv::Mat dst;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
    calcBackProject(&frame_hsv, 1, channels, roi_hist, dst, range);
    meanShift(dst, tracking_window, term_crit);

    cv::Scalar avg = cv::mean(dst);
    int matching_count = FeatureMatchingLogoDetection(frame(tracking_window),
                                                      found_frame(rect), "");
    // std::cout << matching_count << std::endl;
    if (matching_count < 50) {
      break;
    }

    result[i] = tracking_window;

    std::cout << i << " " << matching_count << std::endl;
    cv::Mat frame_clone = frame.clone();
    cv::rectangle(frame_clone, tracking_window, 255, 2);
    cv::imwrite("matching_output/track_" + std::to_string(i) + ".png",
                frame_clone);
  }

  // Reset the tracking window
  tracking_window = rect;

  // Forward loop
  for (size_t i = frame_id + 1; i < rgb_frames.size(); i++) {
    cv::Mat frame = rgb_frames[i];
    cv::Mat frame_hsv;
    cv::Mat dst;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
    calcBackProject(&frame_hsv, 1, channels, roi_hist, dst, range);
    meanShift(dst, tracking_window, term_crit);

    cv::Scalar avg = cv::mean(dst);
    int matching_count = FeatureMatchingLogoDetection(frame(tracking_window),
                                                      found_frame(rect), "");

    // std::cout << matching_count << std::endl;
    if (matching_count < 50) {
      break;
    }
    result[i] = tracking_window;
    std::cout << i << " " << matching_count << std::endl;
    cv::Mat frame_clone = frame.clone();
    cv::rectangle(frame_clone, tracking_window, 255, 2);
    cv::imwrite("matching_output/track_" + std::to_string(i) + ".png",
                frame_clone);
  }
  return result;
}

std::unordered_map<size_t, cv::Rect>
FindLogo(const std::vector<cv::Mat> &rgb_frames, cv::Mat logo,
         std::string mask_path) {
  // cv::Mat mask = CreateMaskForSubway(logo);
  cv::Mat logo_hsv;
  cv::cvtColor(logo, logo_hsv, cv::COLOR_BGR2HSV);
  cv::Mat mask = cv::Mat(logo.size(), CV_8UC1, cv::Scalar(1));
  if (!mask_path.empty()) {
    mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
  }

  cv::MatND logo_hist = GetColorHistogram(logo_hsv, mask);

  cv::Mat logo_hist_plot = PlotHistogram(logo_hist);
  cv::imwrite("logo_hist.png", logo_hist_plot);
  std::vector<size_t> found_frames;

  std::unordered_map<size_t, cv::Rect> logo_box;
  for (size_t i = 0; i < rgb_frames.size(); i += 60) {
    auto [score, rect] = HistogramObjectDetection(logo, logo_hsv, logo_hist,
                                                  rgb_frames[i], mask);
    std::cout << "Frame: " << i << " " << score << std::endl;
    if (score > 0.29) {
      // found_frames.push_back(i);
      logo_box = MeanshiftTrackAcrossFrame(rgb_frames, i, rect);

      // cv::Mat frame_clone = rgb_frames[i].clone();

      // cv::rectangle(frame_clone, rect, cv::Scalar(0, 255, 0));
      // cv::imwrite("matching_output/" + std::to_string(i) + "_rect.png",
      //             frame_clone);
    }
    // break;
  }
  return logo_box;
}

} // namespace detect_subway