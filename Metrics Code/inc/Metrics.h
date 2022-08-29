/*

Authors = Bruno Principe , Piermarco Giustini

*/
#ifndef CVPROJECT__METRICS__H
#define CVPROJECT__METRICS__H
// Libraries Declarations :
//
// C++
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/shape.hpp>
// Functions prototypes :
// 
// Basic Operations :
//
// retrieveBoxes reads the .txt files where are stored the labels and puts them into a vector of vector of Boxes
// if the files failed to open return -1 otherwise 0
int retrieveBoxes(size_t count,std::vector<cv::String> file_name,std::vector<std::vector<cv::Rect>> &boxes);
// fileName creates the final name of the file and merge it with the path given the path to store a file and the current image number 
void fileName(int num,std::string final_path, std::string &file_path_name);
// checkPath takes as input the current path and completes it , if is not a valid path returns -1
void checkPath(int &flag,const std::string path, std::string &final_path);
// retrieveImage loads the images given its numbers of files from the folder and saves them in a vector of images
int retrieveImage(size_t count,std::vector<cv::String> file_name,std::vector<cv::Mat> &images);
// sortVectorOfBox takes the current vector containing the vectors of bounding boxes of each image and sort them in ascending order from the Box's x coordinate
void sortVectOfBox(std::vector<std::vector<cv::Rect>> &sorted_box);
// comp defines the ordering rules of the boxes in the vector
static bool comp(const cv::Rect& a, const cv::Rect& b);
// swapArrayBoxPos swap a box from an initial position to the final one in the vector of boxes of an image specified in the input
void swapArrayBoxPos(std::vector<std::vector<cv::Rect>> &sorted_box,int img_pos,int box_pos_in,int box_pos_fin);
// Task 1 : IoU Accuracy 
//
// boundBoxAccuracy takes as input the vectors of ground truths and derived labels and save the results of the accuracies of the single box in a vector od vector of double 
// and the average accuracy of an image in a vector of double
void boundBoxAccuracy(std::vector<std::vector<cv::Rect>> true_box,std::vector<std::vector<cv::Rect>> derived_box,std::vector<std::vector<double>> &iou_single_accuracy,
    std::vector<double> &iou_accuracy,std::vector<cv::Mat> &correct_mask);
// finUnmatchedIndex controls if the derived labels has False Positives or lacks of unmatched objects with the respect to the ground truth 
void findUnmatchedIndex(int &diff_size,std::vector<cv::Rect> true_box,std::vector<cv::Rect> derived_box,std::vector<int> &k,std::vector<int> &di);
// makeMaskBoundingBox creates the box mask in order to performs IoU
void makeMaskBoundingBox(const cv::Mat &img,cv::Mat &partial_box_area,cv::Rect box);
// Task 2 : Pixel Accuracy
//
// pixelAccuracy takes as input the masks sets and returns the hand segmentation and the background segmentation accuracies 
void pixelAccuracy(std::vector<cv::Mat> correct_mask,std::vector<cv::Mat> derived_mask,std::vector<double> &foreground_accuracy
    ,std::vector<double> &background_accuracy,std::vector<double> &average_accuracy);
// intersectOverUnionSegm computes the IoU/pixel accuracy and returns the value
void intersectOverUnionSegm(cv::Mat correct,cv::Mat derived,double &pixel_accuracy); 
// Storing Functions :
// writePixelAccTxt write the pixels accuracies and the average pixel accuracies in files per image to a specified path
void writePixelAccTxt(const std::vector<double> foreground_accuracy,const std::vector<double> background_accuracy,
    const std::vector<double> average_accuracy,const std::string path);
// writeBoxAccTxt write the boxes accuracies and the average boxes accuracies in files per image to a specific path
void writeBoxAccTxt(const std::vector<std::vector<double>> &iou_single_accuracy,const std::vector<double> &iou_accuracy,const std::string path);


#endif // CVPROJECT__METRICS__H