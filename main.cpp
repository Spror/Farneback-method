#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

void opticalFlow(std::string device, std::string videoName) {}

void opticalFlow(std::string device) {

  cv::VideoCapture capture;

  int deviceID = 0; // 0 = Deafult camera
  int apiID = 0;    // 0 = autodetect defaul API

  capture.open(deviceID, apiID);

  if (!capture.isOpened()) {
    std::cerr << "ERROR! Unable to open camera" << std::endl;
    exit(-1);
  }

  std::unordered_map<std::string, std::vector<double>> timers;

  double fps = capture.get(cv::CAP_PROP_FPS); // Default video FPS
  // int numFrames= int(capture.get(cv::CAP_PROP_FRAME_COUNT)); //

  cv::Mat frame, previousFrame;
  capture >> frame;

  if (device == "cpu") {

    // resizing frame
    cv::resize(frame, frame, cv::Size(960, 540), 0, 0, cv::INTER_LINEAR);

    // converto to gray
    cv::cvtColor(frame, previousFrame, cv::COLOR_BGR2GRAY);

    //outputs to optical flow
  }
}



int main(int argc, const char *argv[]) {

  if (argc > 3) {
    std::cerr << "ERROR! To much options" << std::endl;
    return (-1);
  }

  std::string device = "cpu";

  if (argc >= 2) {
    if (argv[1] == std::string("cpu") || argv[1] == std::string("gpu")) {
      device = argv[1];
    } else {
      std::cerr << "ERROR! Wrong option during called" << std::endl;
      return (-1);
    }
  }

  std::cout << "Configuration:" << std::endl;
  std::cout << "device: " << device << std::endl;

  if (argc == 3) {
    std::string videoName = argv[2];
    std::cout << "File name: " << videoName << std::endl;
    opticalFlow(device, videoName);
  } else {
    opticalFlow(device);
  }

  return 0;
}
