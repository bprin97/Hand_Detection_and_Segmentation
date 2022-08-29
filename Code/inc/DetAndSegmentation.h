/*

Authors = Bruno Principe , Piermarco Giustini

*/
#ifndef CVPROJECT__DETANDSEGMENTATION__H
#define CVPROJECT__DETANDSEGMENTATION__H
// Libreries definitions :
// 
// C++
#include<tuple>
#include <fstream>
// OpenCV 
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/shape.hpp>
// 
// Functions prototypes :
//
// Basic operations : 
// Print images on screen
void showImage(cv::Mat img,const cv::String title);
// Load images from a folder
int retrieveImage(size_t count,std::vector<cv::String> file_name,std::vector<cv::Mat> &images);
// Save images in jpg format in a specified folder
void saveImg(const std::vector<cv::Mat> img, const cv::String path);
// Write and Save Labels in txt format in a specified folder
void writeBoundBoxTxt(const std::vector<std::tuple<int,std::vector<cv::Rect>>> bounding_boxes,const std::string path);
// checkPath takes as input the current path and completes it , if is not a valid path returns -1
void checkPath(int &flag,const std::string path, std::string &final_path);
// Preprocessing and Controls :
//
// Check if an image is gray scale
bool isGrayImage( cv::Mat img );
// Thresholding based on the BGR images to create and return a skin mask
void skinDetection(cv::Mat image,cv::Mat &partial_mask);
// Task 1 : Hand Detection 
// 
// Takes in input the images arrays and returns the images with the hands detected and the corresponting bounding boxes
void handDetection(std::vector<cv::Mat> images, cv::dnn::Net net,std::vector<cv::Mat>& hand_detected_img,std::vector<std::tuple<int,std::vector<cv::Rect>>>& bounding_boxes, std::vector<cv::Scalar> colors);
// Pre-process steps takes the images in input and adapt them before apply the DNN and insert them in input to DNN returns the output
std::vector<cv::Mat> preProcess(cv::Mat &input_image,cv::Mat &resized_img, cv::dnn::Net &net , const float INPUT_WIDTH,const float INPUT_HEIGHT);
// After the pre-process adapt the results , create the images with the bounding boxes and save them and the boxes either.
cv::Mat postProcess(cv::Mat &input_image,cv::Mat &resized_img, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name, 
    const float INPUT_WIDTH,const float INPUT_HEIGHT, const float CONFIDENCE_THRESHOLD,const float SCORE_THRESHOLD,
	const float NMS_THRESHOLD,const int THICKNESS, std::vector<cv::Scalar> colors,
	std::vector<std::tuple<int,std::vector<cv::Rect>>>& bounding_boxes);
// Task 2 : Hand Segmentation
//
// Takes in input the images arrays and returns the segmented images and the masks obtaines
void handSegmentation(std::vector<cv::Mat> images,std::vector<cv::Mat>& hand_segmented_img,std::vector<cv::Mat>& hand_masks_img,
    std::vector<std::tuple<int,std::vector<cv::Rect>>>& bounding_boxes, std::vector<cv::Scalar> colors);
// Colors the mask in input from one of the color chosen
void colorHandMask(cv::Mat mask, cv::Mat &col_mask , cv::Scalar color);
// Merge the mask to the original image to build segmentation
void colorHandSegmentation(cv::Mat img,cv::Mat &dest, cv::Mat mask);
// Preprocess to build the hand mask for gray scale images
void createMaskForGrayImage(cv::Mat img,cv::Mat &partial_mask,cv::Rect box);
// Given the input image returns the image with only the contents inside the bounding boxes
void extractMaskFromBoundingBox(cv::Mat img,cv::Mat &partial_box_area,cv::Rect box);
// Takes in input the image, the current buonding box , the number of iterations it extract the foreground from the background producing as output the colored section
// and binarized counterpart.
void removeBackgroundfromImg(cv::Mat img,cv::Mat &dest,cv::Rect box,unsigned int iteration,cv::Mat &mask2);

#endif // CVPROJECT__DETANDSEGMENTATION__H
