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


void printStats(TimeMap timers, double fps, int numFrames){

    // elapsed time at each stage
    std::cout << "Elapsed time" << std::endl;
    for (auto const& timer : timers)
    {
        std::cout << "- " << timer.first << " : " << std::accumulate( timer.second.begin(), timer.second.end() ,0.0)<< " seconds"<< std::endl;
    }
      // calculate frames per second
    std::cout << "Default video FPS : "  << fps << std::endl;
    float optical_flow_fps  = (numFrames - 1) / std::accumulate(timers["optical flow"].begin(),timers["optical flow"].end(), 0.0);
    std::cout << "Optical flow FPS : "   << optical_flow_fps  << std::endl;

    float full_pipeline_fps = (numFrames - 1) / std::accumulate(timers["full pipeline"].begin(), timers["full pipeline"].end(), 0.0);
    std::cout << "Full pipeline FPS : "  << full_pipeline_fps << std::endl;
    
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
  int numFrames = 1; //number of frames

  cv::Mat frame, previousFrame;
  capture >> frame;
if (device == "cpu")
    {

        // convert to gray
        cv::cvtColor(frame, previousFrame, cv::COLOR_BGR2GRAY);

        // declare outputs for optical flow
        cv::Mat magnitude, normalizedMagnitude, angle;
        cv::Mat hsv[3], merged_hsv, bgr;

        // set saturation to 1
        hsv[1] = cv::Mat::ones(frame.size(), CV_32F);

        while (true)
        {
            // start full pipeline timer
            auto startFullTime = std::chrono::high_resolution_clock::now();

            // start reading timer
            auto startReadTime = std::chrono::high_resolution_clock::now();

            // capture frame-by-frame
            capture >> frame;
            numFrames++;

            if (frame.empty())
                break;

            // end reading timer
            auto endReadTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["reading"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endReadTime - startReadTime).count() / 1000.0);

            // start pre-process timer
            auto startPreTime = std::chrono::high_resolution_clock::now();



            // convert to gray
            cv::Mat currentFrame;
            cv::cvtColor(frame, currentFrame, cv::COLOR_BGR2GRAY);

            // end pre-process timer
            auto endPreTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["pre-process"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endPreTime - startPreTime).count() / 1000.0);

            // start optical flow timer
            auto startOfTime = std::chrono::high_resolution_clock::now();

            // calculate optical flow
            cv::Mat flow;
            calcOpticalFlowFarneback(previousFrame, currentFrame, flow, 0.5, 5, 15, 3, 5, 1.2, 0);

            // end optical flow timer
            auto endOfTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["optical flow"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endOfTime - startOfTime).count() / 1000.0);

            // start post-process timer
            auto startPostTime = std::chrono::high_resolution_clock::now();

    
            flow.convertTo(flow, CV_32F, 1.0f / (1 << 3));
            cv::Mat magnitude, angle;
            {
                cv::Mat flowChannels[2];
                split(flow, flowChannels);
                cv::cartToPolar(flowChannels[0], flowChannels[1], magnitude, angle, true);
            }
            float clip = 5;
            cv::threshold(magnitude, magnitude, clip, clip, cv::THRESH_TRUNC);
            hsv[0] = angle;
            hsv[2] = magnitude / clip;
            merge(hsv, 3, merged_hsv);
            cv::cvtColor(merged_hsv, bgr, cv::COLOR_HSV2BGR);
            bgr.convertTo(bgr, CV_8U, 255.0);

            // update previous_frame value
            previousFrame = currentFrame;

            // end post pipeline timer
            auto endPostTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["post-process"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endPostTime - startPostTime).count() / 1000.0);

            // end full pipeline timer
            auto endFullTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["full pipeline"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endFullTime - startFullTime).count() / 1000.0);

            // visualization
            imshow("original", frame);
            imshow("result", bgr);

            std::cout << "FPS: " << 1.0/(std::chrono::duration_cast<std::chrono::milliseconds>(endFullTime - startFullTime).count() / 1000.0) << std::endl;


            int keyboard = cv::waitKey(1);
            if (keyboard == 27)
                break;
        }
    }
    else
    {

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

        while (true)
        {
            // start full pipeline timer
            auto startFullTime = std::chrono::high_resolution_clock::now();

            // start reading timer
            auto startReadTime = std::chrono::high_resolution_clock::now();

            // capture frame-by-frame
            capture >> frame;
            numFrames++;

            if (frame.empty())
                break;

            // upload frame to GPU
            cv::cuda::GpuMat gpuFrame;
            gpuFrame.upload(frame);

            // end reading timer
            auto endReadTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["reading"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endReadTime - startReadTime).count() / 1000.0);

            // start pre-process timer
            auto startPreTime = std::chrono::high_resolution_clock::now();


            // convert to gray
            cv::cuda::GpuMat gpu_current;
            cv::cuda::cvtColor(gpuFrame, gpu_current, cv::COLOR_BGR2GRAY);

            // end pre-process timer
            auto endPreTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["pre-process"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endPreTime - startPreTime).count() / 1000.0);

            // start optical flow timer
            auto startOfTime = std::chrono::high_resolution_clock::now();

            // create optical flow instance
            cv::Ptr<cv::cuda::FarnebackOpticalFlow> ptr_calc = cv::cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, 0);
            // calculate optical flow
            cv::cuda::GpuMat gpuFlow;
            ptr_calc->calc(gpuPrevious, gpu_current, gpuFlow);

            // end optical flow timer
            auto endOfTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["optical flow"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endOfTime - startOfTime).count() / 1000.0);

            // start post-process timer
            auto startPostTime = std::chrono::high_resolution_clock::now();
           
            // // split the output flow into 2 vectors
            // cv::cuda::GpuMat gpuFlow_xy[2];
            // cv::cuda::split(gpuFlow, gpuFlow_xy);

            // // convert from cartesian to polar coordinates
            // cv::cuda::cartToPolar(gpuFlow_xy[0], gpuFlow_xy[1], gpuMagnitude, gpuAngle, true);

            // // normalize magnitude from 0 to 1
            // cv::cuda::normalize(gpuMagnitude, gpuNormalizedMagnitude, 0.0, 1.3, cv::NORM_MINMAX, -1);

            // // get angle of optical flow
            // gpuAngle.download(angle);
            // angle *= ((1 / 360.0) * (180 / 255.0));

            // // build hsv image
            // gpu_hsv[0].upload(angle);
            // gpu_hsv[2] = gpuNormalizedMagnitude;
            // cv::cuda::merge(gpu_hsv, 3, gpuMerged_hsv);

            // // multiply each pixel value to 255
            // gpuMerged_hsv.cv::cuda::GpuMat::convertTo(gpu_hsv_8u, CV_8U, 255.0);

            // // convert hsv to bgr
            // cv::cuda::cvtColor(gpu_hsv_8u, gpu_bgr, cv::COLOR_HSV2BGR);
            
    
            gpuFlow.cv::cuda::GpuMat::convertTo(gpuFlow, CV_32F, 1.0f / (1 << 5));
            cv::cuda::GpuMat magnitude, angle;
            {
                cv::cuda::GpuMat flowChannels[2];
                cv::cuda::split(gpuFlow, flowChannels);
                cv::cuda::cartToPolar(flowChannels[0], flowChannels[1], magnitude, angle, true);
            }
            
            cv::cuda::threshold(magnitude, magnitude, 5, 5, cv::THRESH_TRUNC);
            gpu_hsv[0] = angle;
            gpu_hsv[2] = magnitude;
            cv::cuda::merge(gpu_hsv, 3, gpuMerged_hsv);
            cv::cuda::cvtColor(gpuMerged_hsv, gpu_bgr, cv::COLOR_HSV2BGR);
            gpu_bgr.cv::cuda::GpuMat::convertTo(gpu_bgr, CV_8U, 255.0);

            // send original frame from GPU back to CPU
            gpuFrame.download(frame);

            // send result from GPU back to CPU
            gpu_bgr.download(bgr);

            // update previous_frame value
            gpuPrevious = gpu_current;

            // end post pipeline timer
            auto endPostTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["post-process"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endPostTime - startPostTime).count() / 1000.0);

            // end full pipeline timer
            auto endFullTime = std::chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["full pipeline"].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(endFullTime - startFullTime).count() / 1000.0);

            // visualization
            imshow("original", frame);
            imshow("result", bgr);

            std::cout << "FPS: " << 1.0/(std::chrono::duration_cast<std::chrono::milliseconds>(endFullTime - startFullTime).count() / 1000.0) << std::endl;

            int keyboard = cv::waitKey(1);
            if (keyboard == 27)
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
    
  } else {
    opticalFlow(device);
  }

  return 0;
}
