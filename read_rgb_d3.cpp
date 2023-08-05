#include "WavReader.h"
#include "detect_ae.h"
#include "detect_hrc.h"

#include <QProcess>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
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
#include <optional>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

constexpr int height = 270;
constexpr int width = 480;
constexpr int frame_length = 3 * height * width;

cv::Mat ReadRGB(char *buf) {

  cv::Mat image(height, width, CV_8UC3);

  cv::Mat channelR(height, width, CV_8UC1, buf);
  cv::Mat channelG(height, width, CV_8UC1, buf + height * width);
  cv::Mat channelB(height, width, CV_8UC1, buf + 2 * height * width);
  std::vector<cv::Mat> channels{channelB, channelG, channelR};
  merge(channels, image);

  return image;
}

void WriteRGB(const cv::Mat &image, char *buf) {
  if (image.channels() != 3) {
    std::cerr << "Unexpected number of channels" << std::endl;
    exit(-1);
  }
  if (image.type() != CV_8UC3) {
    std::cerr << "Unexpected data type" << std::endl;
    exit(-1);
  }
  cv::Mat channels[3];
  cv::split(image, channels);

  // buf is in order of R, G, B
  // channels is in order of B, G, R

  memcpy(buf, channels[2].data, height * width);
  memcpy(buf + height * width, channels[1].data, height * width);
  memcpy(buf + 2 * height * width, channels[0].data, height * width);
}

std::vector<cv::Mat> ReadRGBFrames(const std::string &input_path) {

  std::ifstream input(input_path, std::ios::binary);
  std::vector<cv::Mat> frames;
  char *buf = new char[frame_length];
  uint64_t curr_len = 0;
  while (input.good() && input.peek() != EOF) {
    input.read(buf, frame_length);
    frames.push_back(ReadRGB(buf));
    curr_len += frame_length;
  }

  return frames;
}

void WriteRGBFrames(const std::vector<cv::Mat> &frames,
                    const std::string &output_path) {

  std::ofstream ofile(output_path, std::ios::binary);
  char *buf = new char[frame_length];
  for (const auto &frame : frames) {
    WriteRGB(frame, buf);
    ofile.write(buf, frame_length);
  }
  ofile.close();
}

std::vector<cv::Mat>
FilterRGBFrames(const std::vector<cv::Mat> &frames,
                const std::vector<std::pair<size_t, size_t>> &filter_intervals,
                std::map<size_t, size_t> &insert_frame_ids) {

  std::vector<cv::Mat> result;
  result.reserve(frames.size());

  bool within_ads = false;

  std::map<size_t, size_t> insert_frames;

  for (size_t i = 0; i < frames.size(); i++) {
    bool do_filter = false;
    // If not in any filter interval, then we keep the frame
    for (size_t j = 0; j < filter_intervals.size(); j++) {
      if (i >= filter_intervals[j].first && i <= filter_intervals[j].second) {
        do_filter = true;
      }
    }
    if (!do_filter) {
      result.push_back(frames[i]);
      within_ads = false;
    } else {
      // This is the start of an ad
      if (!within_ads) {
        insert_frames[i] = result.size();
        within_ads = true;
      }
    }
  }
  insert_frame_ids = insert_frames;
  return result;
}

void PlayRGBFrames(const std::vector<cv::Mat> &frames, int fps) {
  for (const auto &frame : frames) {
    cv::imshow("Video", frame);
    cv::waitKey(33);
  }
}

int CountFeatureMatching(cv::Mat descriptors_1, cv::Mat descriptors_2) {

  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  if (descriptors_1.empty() || descriptors_2.empty()) {
    return 0;
  }

  matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance <
        ratio_thresh * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  return good_matches.size();
}

std::string FrameToTimeStr(size_t frame_id) {
  size_t total_sec = frame_id / 30;

  size_t minute = total_sec / 60;
  size_t second = total_sec % 60;

  return std::to_string(minute) + ":" + std::to_string(second);
}

void OutputFrames(const std::vector<cv::Mat> &frames) {

  size_t frame_id = 0;
  for (const auto &frame : frames) {
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << frame_id;
    cv::imwrite("./output/" + ss.str() + ".png", frame);
    frame_id++;
  }
}

void DivideShots(const std::vector<cv::Mat> &frames) {

  auto sift_detector = cv::SiftFeatureDetector::create();

  std::optional<std::vector<cv::KeyPoint>> prev_keypoints;
  std::optional<cv::Mat> prev_descriptors;

  size_t frame_id = 0;

  for (const auto &frame : frames) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift_detector->detectAndCompute(frame, cv::noArray(), keypoints,
                                    descriptors);

    if (prev_descriptors.has_value()) {
      int num_matched_features =
          CountFeatureMatching(descriptors, *prev_descriptors);
      std::cout << frame_id << " " << FrameToTimeStr(frame_id) << " "
                << num_matched_features << std::endl;
    }

    prev_keypoints = keypoints;
    prev_descriptors = descriptors;
    frame_id++;
  }
}

double GetAvgSaturation(cv::Mat rgb_frame) {
  cv::Mat hsv_frame;
  cv::cvtColor(rgb_frame, hsv_frame, cv::COLOR_BGR2HSV);
  return cv::mean(hsv_frame).val[1];
}

struct FrameInfo {
  size_t num_keypoints;
  size_t num_matches;
};

std::vector<size_t> DivideShotsOMP(const std::vector<cv::Mat> &frames,
                                   std::vector<FrameInfo> &frame_info_vec) {

  size_t steps_completed = 0;
  size_t total_steps = frames.size();
  frame_info_vec.resize(frames.size());

#pragma omp parallel for num_threads(64)
  for (size_t frame_id = 1; frame_id < frames.size(); frame_id++) {
    const cv::Mat &frame = frames[frame_id];
    const cv::Mat &prev_frame = frames[frame_id - 1];
    auto sift_detector = cv::SiftFeatureDetector::create();

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift_detector->detectAndCompute(frame, cv::noArray(), keypoints,
                                    descriptors);
    frame_info_vec[frame_id].num_keypoints = keypoints.size();

    std::vector<cv::KeyPoint> prev_keypoints;
    cv::Mat prev_descriptors;
    sift_detector->detectAndCompute(prev_frame, cv::noArray(), prev_keypoints,
                                    prev_descriptors);
    if (frame_id == 1) {
      frame_info_vec[frame_id - 1].num_keypoints = prev_keypoints.size();
      frame_info_vec[frame_id - 1].num_matches = 0;
    }

    int num_matched_features =
        CountFeatureMatching(descriptors, prev_descriptors);

    frame_info_vec[frame_id].num_matches = num_matched_features;
#pragma omp atomic
    ++steps_completed;

    if (steps_completed % 100 == 1) {
#pragma omp critical
      std::cout << "Progress: " << steps_completed << " of " << total_steps
                << " (" << std::fixed << std::setprecision(1)
                << (100.0 * steps_completed / total_steps) << "%)\n";
    }
  }

  std::ofstream ofile("num_matches.csv");
  ofile << "frame_id,num_matches,num_keypoints" << std::endl;
  for (int i = 0; i < frame_info_vec.size(); i++) {
    ofile << i << "," << frame_info_vec[i].num_matches << ","
          << frame_info_vec[i].num_keypoints << std::endl;
  }
  ofile.close();

  std::vector<size_t> shots_frame_id;
  shots_frame_id.push_back(0);

  for (int i = 1; i < frame_info_vec.size(); i++) {
    double ratio = static_cast<double>(frame_info_vec[i].num_matches) /
                   static_cast<double>(frame_info_vec[i - 1].num_matches);
    if (ratio < 0.1 && frame_info_vec[i].num_matches < 20) {
      std::cout << "Frame: " << i << " time " << FrameToTimeStr(i) << std::endl;
      shots_frame_id.push_back(i);
    }
  }
  return shots_frame_id;
}

std::vector<std::pair<size_t, size_t>>
FindAdsShots(const std::vector<size_t> &shots_start,
             const std::vector<cv::Mat> &rgb_frames,
             const std::vector<FrameInfo> &frame_info_vec) {

  std::vector<size_t> avg_num_features;

  std::vector<bool> ads_candidates(shots_start.size(), false);
  // First we use a lower threshold to find short shots
  for (size_t i = 0; i < shots_start.size(); i++) {
    size_t shot_length;
    if (i == shots_start.size() - 1) {
      shot_length = rgb_frames.size() - shots_start[i];
    } else {
      shot_length = shots_start[i + 1] - shots_start[i];
    }

    size_t total_num_features = 0;
    for (size_t j = 0; j < shot_length; j++) {
      total_num_features += frame_info_vec[shots_start[i] + j].num_keypoints;
    }
    size_t avg_num = total_num_features / shot_length;
    avg_num_features.push_back(avg_num);

    // 4 second and low number of feature points
    if (shot_length < 120 || avg_num < 200) {
      ads_candidates[i] = true;
    }
  }

  for (size_t i = 0; i < ads_candidates.size(); i++) {
    std::cout << ads_candidates[i] << " ";
  }
  std::cout << std::endl;

  for (size_t i = 0; i < shots_start.size(); i++) {
    size_t shot_length;
    if (i == shots_start.size() - 1) {
      shot_length = rgb_frames.size() - shots_start[i];
    } else {
      shot_length = shots_start[i + 1] - shots_start[i];
    }

    // If the current shot is not shot enough, but still short, we check if the
    // surrounding shots are characterized as ads
    if (!ads_candidates[i] && shot_length < 240) {
      // If the prev and next are both ads, and this shot is short, it's very
      // likely an ad
      if (i > 0 && i < shots_start.size() - 1 && ads_candidates[i - 1] &&
          ads_candidates[i + 1]) {
        ads_candidates[i] = true;
      } else if (i < shots_start.size() - 1 && ads_candidates[i + 1]) {
        double curr_saturation = GetAvgSaturation(rgb_frames[shots_start[i]]);
        double next_saturation =
            GetAvgSaturation(rgb_frames[shots_start[i + 1]]);
        // If it's a middle shot, we can compare the prev and next
        bool cond_1 =
            (i > 0) &&
            ((std::abs(curr_saturation - next_saturation) <
              std::abs(curr_saturation -
                       GetAvgSaturation(rgb_frames[shots_start[i - 1]]))));
        // If it's a start shot, we can only compare the curr and next by a
        // threshold
        bool cond_2 =
            (i == 0) && (std::abs(curr_saturation - next_saturation) < 20);

        if (cond_1 || cond_2) {
          ads_candidates[i] = true;
        }
      }
    }
  }
  for (size_t i = 0; i < ads_candidates.size(); i++) {
    std::cout << ads_candidates[i] << " ";
  }
  std::cout << std::endl;

  std::vector<std::pair<size_t, size_t>> ads_intervals;
  // Find the sequences of continuous 1s
  for (size_t i = 0; i < ads_candidates.size(); i++) {
    if (ads_candidates[i]) {
      size_t start_frame = shots_start[i];
      size_t end_frame;
      i++;
      while (i <= ads_candidates.size()) {
        if (i == ads_candidates.size()) {
          end_frame = rgb_frames.size() - 1;
          break;
        }
        end_frame = shots_start[i] - 1;

        if (!ads_candidates[i]) {
          // the next shot is not a ads shot
          break;
        }
        i++;
      }

      ads_intervals.push_back({start_frame, end_frame});
    }
  }
  for (const auto &[s, e] : ads_intervals) {
    std::cout << s << "-" << e << std::endl;
  }

  return ads_intervals;
}

void AddLogoRect(std::vector<cv::Mat> &rgb_frames,
                 const std::unordered_map<size_t, cv::Rect> &logo_boxes,
                 const std::string &text) {
  for (auto &[frame_id, rect] : logo_boxes) {
    if (frame_id >= rgb_frames.size()) {
      std::cerr << "Frame id larger than the number of frames" << std::endl;
    }
    std::cout << "add rect at " << frame_id << std::endl;
    cv::rectangle(rgb_frames[frame_id], rect, 255, 2);
    cv::putText(rgb_frames[frame_id], text, cv::Point2f(rect.x, rect.y),
                cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255, 255, 0));
  }
}

void InsertAds(std::map<size_t, size_t> &insert_frame_id,
               std::vector<std::vector<cv::Mat>> &brand_videos,
               std::vector<cv::Mat> &rgb_frames) {
  size_t brand_index = 0;
  if (insert_frame_id.size() != brand_videos.size()) {
    std::cerr << "Insert slot not equals to the number of brand videos"
              << std::endl;
    return;
  }
  for (auto it = insert_frame_id.rbegin(); it != insert_frame_id.rend();
       it++, brand_index++) {
    // std::copy(brand_videos[brand_index].begin(),
    //           brand_videos[brand_index].end(),
    //           std::inserter(rgb_frames, rgb_frames.begin() + it->second));
    rgb_frames.insert(rgb_frames.begin() + it->second,
                      brand_videos[brand_index].begin(),
                      brand_videos[brand_index].end());
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "read_rgb input.rgb input.wav output.rgb output.wav"
              << std::endl;
    return -1;
  }
  struct stat st;
  u_int64_t file_size = 0;
  std::string input_rgb_path = argv[1];
  if (stat(input_rgb_path.c_str(), &st) == 0) {
    file_size = st.st_size;
  } else {
    std::cerr << "Error in stat the file" << std::endl;
    return -1;
  }

  if (file_size % frame_length != 0) {
    std::cout << "There might be redundant bytes in the rgb file" << std::endl;
  }
  std::vector<cv::Mat> rgb_frames = ReadRGBFrames(input_rgb_path);

  std::vector<FrameInfo> frame_info_vec;
  std::vector<size_t> shot_starts = DivideShotsOMP(rgb_frames, frame_info_vec);
  auto ads_intervals = FindAdsShots(shot_starts, rgb_frames, frame_info_vec);
  // std::vector<std::pair<size_t, size_t>> ads_intervals = {{4501, 4893},
  //                                                         {8494, 9004}};

  std::map<size_t, size_t> insert_frame_id;
  std::vector<cv::Mat> rgb_frames_no_ads =
      FilterRGBFrames(rgb_frames, ads_intervals, insert_frame_id);

  const std::string hrc_logo_path =
      "data/dataset-003/dataset3/Brand Images/hrc_logo.rgb";
  const std::string ae_logo_path =
      "data/dataset-003/dataset3/Brand Images/ae_logo.rgb";

  const std::string ae_ads_path = "data/dataset-003/dataset3/Ads/ae_ad_15s.rgb";
  const std::string hrc_ads_path =
      "data/dataset-003/dataset3/Ads/hrc_ad_15s.rgb";

  // // Read Logo
  std::vector<cv::Mat> hrc_logo_frames = ReadRGBFrames(hrc_logo_path);
  std::vector<cv::Mat> ae_logo_frames = ReadRGBFrames(ae_logo_path);
  std::vector<cv::Mat> hrc_ads_frames = ReadRGBFrames(hrc_ads_path);
  std::vector<cv::Mat> ae_ads_frames = ReadRGBFrames(ae_ads_path);

  auto hrc_logo_boxes =
      detect_hrc::FindLogo(rgb_frames_no_ads, hrc_logo_frames.at(0), "");
  auto ae_logo_boxes =
      detect_ae::FindLogo(rgb_frames_no_ads, ae_logo_frames.at(0), "");

  AddLogoRect(rgb_frames_no_ads, hrc_logo_boxes, "HRC");
  AddLogoRect(rgb_frames_no_ads, ae_logo_boxes, "AE");

  std::vector<std::vector<cv::Mat>> all_ads({hrc_ads_frames, ae_ads_frames});
  InsertAds(insert_frame_id, all_ads, rgb_frames_no_ads);

  WriteRGBFrames(rgb_frames_no_ads, argv[3]);

  // OutputFrames(rgb_frames);
  // DivideShots(rgb_frames);
  // writeVideo(rgb_frames, "data/dataset-002/dataset2/Videos/data_test2.wav");
}
