#include "WavReader.h"
#include <QProcess>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

void PlayRGBFrames(const std::vector<cv::Mat> &frames, int start_frame = 0) {
  for (int i = start_frame; i < frames.size(); i++) {
    cv::imshow("Video", frames[i]);
    cv::waitKey(33);
  }
}

int main(int argc, char **argv) {

  struct stat st;
  u_int64_t file_size = 0;
  std::string input_rgb_path = argv[1];
  if (stat(input_rgb_path.c_str(), &st) == 0) {
    file_size = st.st_size;
  } else {
    std::cerr << "Error in stat the file" << std::endl;
    return -1;
  }
  int start_frame;
  if (argc > 2) {
    start_frame = std::stoi(argv[2]);
  }

  if (file_size % frame_length != 0) {
    std::cout << "There might be redundant bytes in the rgb file" << std::endl;
  }
  std::vector<cv::Mat> rgb_frames = ReadRGBFrames(input_rgb_path);

  PlayRGBFrames(rgb_frames, start_frame);
}