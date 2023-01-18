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

using TimeMap = std::unordered_map<std::string, std::vector<double>>;

void printStats(TimeMap timers, double fps, int numFrames) {

  std::cout << "Elapsed time" << std::endl;
  for (auto const &timer : timers) {
    std::cout << "- " << timer.first << " : "
              << std::accumulate(timer.second.begin(), timer.second.end(), 0.0)
              << " seconds" << std::endl;
  }

  std::cout << "Default video FPS : " << fps << std::endl;
  auto optical_flow_fps =
      (numFrames - 1) / std::accumulate(timers["optical flow"].begin(),
                                        timers["optical flow"].end(), 0.0);
  std::cout << "Optical flow FPS : " << optical_flow_fps << std::endl;

  auto full_pipeline_fps =
      (numFrames - 1) / std::accumulate(timers["full pipeline"].begin(),
                                        timers["full pipeline"].end(), 0.0);
  std::cout << "Full pipeline FPS : " << full_pipeline_fps << std::endl;
}

void opticalFlow(std::string device) {

  cv::VideoCapture capture;

  auto deviceID = "/dev/video0"; // 0 = Deafult camera
  auto apiID = 0;                // 0 = autodetect defaul API

  capture.open(deviceID, apiID);

  if (!capture.isOpened()) {
    std::cerr << "ERROR! Unable to open camera" << std::endl;
    exit(-1);
  }

  TimeMap timers;

  double fps = capture.get(cv::CAP_PROP_FPS); // Default FPS
  int numFrames = 1;                          // number of frames

  cv::Mat frame, previousFrame;
  capture >> frame;

  if (device == "cpu") {

    // convert to gray
    cv::cvtColor(frame, previousFrame, cv::COLOR_BGR2GRAY);

    // declare outputs for optical flow
    cv::Mat magnitude, normalizedMagnitude, angle;
    cv::Mat hsv[3], merged_hsv, bgr;

    // set saturation to 1
    hsv[1] = cv::Mat::ones(frame.size(), CV_32F);

    while (true) {

      auto startFullTime = std::chrono::high_resolution_clock::now();
      auto startReadTime = std::chrono::high_resolution_clock::now();

      // capture frame-by-frame
      capture >> frame;
      numFrames++;

      if (frame.empty())
        break;

      auto endReadTime = std::chrono::high_resolution_clock::now();
      timers["reading"].push_back(
          double(std::chrono::duration_cast<std::chrono::milliseconds>(
                     endReadTime - startReadTime)
                     .count()) /
          1000.0);

      auto startPreTime = std::chrono::high_resolution_clock::now();

      cv::Mat currentFrame;
      cv::cvtColor(frame, currentFrame, cv::COLOR_BGR2GRAY);

      auto endPreTime = std::chrono::high_resolution_clock::now();
      timers["pre-process"].push_back(
          double(std::chrono::duration_cast<std::chrono::milliseconds>(
                     endPreTime - startPreTime)
                     .count()) /
          1000.0);

      auto startOfTime = std::chrono::high_resolution_clock::now();

      // calculate optical flow
      cv::Mat flow(currentFrame.size(), CV_32FC2);
      calcOpticalFlowFarneback(previousFrame, currentFrame, flow, 0.5, 5, 15, 3,
                               5, 1.2, 0);

      auto endOfTime = std::chrono::high_resolution_clock::now();
      timers["optical flow"].push_back(
          double(std::chrono::duration_cast<std::chrono::milliseconds>(
                     endOfTime - startOfTime)
                     .count()) /
          1000.0);

      // auto startPostTime = std::chrono::high_resolution_clock::now();

      flow.convertTo(flow, CV_32F, 1.0f / (1 << 3));
      cv::Mat magnitude, angle;

      cv::Mat flowChannels[2];
      split(flow, flowChannels);
      cv::cartToPolar(flowChannels[0], flowChannels[1], magnitude, angle, true);

      float clip = 5;
      cv::threshold(magnitude, magnitude, clip, clip, cv::THRESH_TRUNC);
      hsv[0] = angle;
      hsv[2] = magnitude / clip;
      merge(hsv, 3, merged_hsv);
      cv::cvtColor(merged_hsv, bgr, cv::COLOR_HSV2BGR);
      bgr.convertTo(bgr, CV_8U, 255.0);

      // update previous_frame value
      previousFrame = currentFrame;

      auto endFullTime = std::chrono::high_resolution_clock::now();
      auto pipelineTime =
          double(std::chrono::duration_cast<std::chrono::milliseconds>(
                     endFullTime - startFullTime)
                     .count()) /
          1000.0;
      timers["full pipeline"].push_back(pipelineTime);

      // visualization
      imshow("original", frame);
      imshow("result", bgr);

      std::cout << "FPS: " << 1.0 / pipelineTime << std::endl;

      int keyboard = cv::waitKey(1);
      if (keyboard == 113 || keyboard == 'q')
        break;
    }
  } else {

    // convert to gray
    cv::cvtColor(frame, previousFrame, cv::COLOR_BGR2GRAY);

    // upload pre-processed frame to GPU
    cv::cuda::GpuMat gpuPrevious;
    gpuPrevious.upload(previousFrame);

    // declare cpu outputs for optical flow
    cv::Mat hsv[3], angle, bgr;

    // declare gpu outputs for optical flow
    cv::cuda::GpuMat gpuMagnitude, gpuNormalizedMagnitude, gpuAngle;
    cv::cuda::GpuMat gpu_hsv[3], gpuMerged_hsv, gpu_hsv_8u, gpu_bgr;

    // set saturation to 1
    hsv[1] = cv::Mat::ones(frame.size(), CV_32F);
    gpu_hsv[1].upload(hsv[1]);

    while (true) {

      auto startFullTime = std::chrono::high_resolution_clock::now();

      auto startReadTime = std::chrono::high_resolution_clock::now();

      // capture frame-by-frame
      capture >> frame;
      numFrames++;

      if (frame.empty())
        break;

      // upload frame to GPU
      cv::cuda::GpuMat gpuFrame;
      gpuFrame.upload(frame);

      auto endReadTime = std::chrono::high_resolution_clock::now();
      timers["reading"].push_back(
          double(std::chrono::duration_cast<std::chrono::milliseconds>(
                     endReadTime - startReadTime)
                     .count()) /
          1000.0);

      auto startPreTime = std::chrono::high_resolution_clock::now();

      // convert to gray
      cv::cuda::GpuMat gpuCurrent;
      cv::cuda::cvtColor(gpuFrame, gpuCurrent, cv::COLOR_BGR2GRAY);

      auto endPreTime = std::chrono::high_resolution_clock::now();
      timers["pre-process"].push_back(
          double(std::chrono::duration_cast<std::chrono::milliseconds>(
                     endPreTime - startPreTime)
                     .count()) /
          1000.0);

      auto startOfTime = std::chrono::high_resolution_clock::now();

      // create optical flow instance
      cv::Ptr<cv::cuda::FarnebackOpticalFlow> ptr_calc =
          cv::cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2,
                                                 0);
      // calculate optical flow
      cv::cuda::GpuMat gpuFlow;
      ptr_calc->calc(gpuPrevious, gpuCurrent, gpuFlow);

      auto endOfTime = std::chrono::high_resolution_clock::now();
      timers["optical flow"].push_back(
          double(std::chrono::duration_cast<std::chrono::milliseconds>(
                     endOfTime - startOfTime)
                     .count()) /
          1000.0);

      auto startPostTime = std::chrono::high_resolution_clock::now();

      gpuFlow.cv::cuda::GpuMat::convertTo(gpuFlow, CV_32F, 1.0f / (1 << 3));

      cv::cuda::GpuMat flowChannels[2];
      cv::cuda::split(gpuFlow, flowChannels);
      cv::cuda::cartToPolar(flowChannels[0], flowChannels[1], gpuMagnitude,
                            gpuAngle, true);

      cv::cuda::threshold(gpuMagnitude, gpuMagnitude, 5, 5, cv::THRESH_TRUNC);
      gpu_hsv[0] = gpuAngle;

      cv::cuda::divide(gpuMagnitude, 5, gpu_hsv[2]);
      cv::cuda::merge(gpu_hsv, 3, gpuMerged_hsv);
      cv::cuda::cvtColor(gpuMerged_hsv, gpu_bgr, cv::COLOR_HSV2BGR);
      gpu_bgr.cv::cuda::GpuMat::convertTo(gpu_bgr, CV_8U, 255.0);

      // send original frame from GPU back to CPU
      gpuFrame.download(frame);

      // send result from GPU back to CPU
      gpu_bgr.download(bgr);

      // update previous_frame value
      gpuPrevious = gpuCurrent;

      auto endPostTime = std::chrono::high_resolution_clock::now();
      timers["post-process"].push_back(
          double(std::chrono::duration_cast<std::chrono::milliseconds>(
                     endPostTime - startPostTime)
                     .count()) /
          1000.0);

      auto endFullTime = std::chrono::high_resolution_clock::now();
      auto pipelineTime =
          double(std::chrono::duration_cast<std::chrono::milliseconds>(
                     endFullTime - startFullTime)
                     .count()) /
          1000.0;

      timers["full pipeline"].push_back(pipelineTime);

      // visualization
      imshow("original", frame);
      imshow("result", bgr);

      std::cout << "FPS: " << 1.0 / pipelineTime << std::endl;

      int keyboard = cv::waitKey(1);
      if (keyboard == 27 || keyboard == 'q')
        break;
    }
  }

  // release the capture
  capture.release();

  // destroy all windows
  cv::destroyAllWindows();

  printStats(timers, fps, numFrames);
}

int main(int argc, const char *argv[]) {

  if (argc == 2) {
    if (argv[1] == std::string("cpu") || argv[1] == std::string("gpu")) {
      std::string device = argv[1];

      std::cout << "Configuration:" << std::endl;
      std::cout << "device: " << device << std::endl;

      opticalFlow(device);
    } else {
      std::cerr << "ERROR! Wrong option during called" << std::endl;
      exit(-1);
    }
  } else {
    std::cerr << "ERROR! Wrong number of called arguments" << std::endl;
    exit(-1);
  }

  return 0;
}
