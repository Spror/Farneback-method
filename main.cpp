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

void opticalFlow(std::string device) {

  cv::Mat frame;
  cv::VideoCapture cap;

  int deviceID = 0; // 0 = Deafult camera
  int apiID = 0;    // 0 = autodetect defaul API

  cap.open(deviceID, apiID);

  if (!cap.isOpened()) {
    std::cerr << "ERROR! Unable to open camera" << std::endl;
    exit(-1);
  }

  std::cout << "Start grabbing" << std::endl
            << "Press any key to terminate" << std::endl;

  for (;;) {
    // wait for a new frame from camera and store it into 'frame'
    cap.read(frame);
    // check if we succeeded
    if (frame.empty()) {
      std::cerr << "ERROR! blank frame grabbed\n";
      break;
    }
    // show live and wait for a key with timeout long enough to show images
    imshow("Live", frame);
    if (cv::waitKey(5) >= 0)
      break;
  }
}

int main(int argc, const char *argv[]) {

  std::string device = "cpu";

  if (argc == 2) {
    std::cout << argv[1] << std::endl;
    if (argv[1] == std::string("cpu") || argv[1] == std::string("gpu")) {
      device = argv[1];
    } else {
      std::cerr << "ERROR! Wrong option during called" << std::endl;
      return (-1);
    }
  }

  if (argc > 2) {
    std::cerr << "ERROR! To much options" << std::endl;
    return (-1);
  }

  std::cout << "Configuration:" << std::endl;
  std::cout << "device: " << device << std::endl;
  opticalFlow(device);

  return 0;
}
