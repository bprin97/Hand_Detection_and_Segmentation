/*

Authors = Bruno Principe ,Piermarco Giustini

*/
#include <iostream>
#include <string>
#include <fstream>
#include "DetAndSegmentation.h"

int main(int argc, char** argv){
	// CMD parser definitions
	cv::CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this message }"
        "{ TestImgPath i        |      | (required) path to reference folders of Test Dataset }"
		"{ model m        |      | (required) DNN .onnx model with path to reach it }"
		"{ DetImgSave fd        |      | (optional) path where storing Detected Hands Imgs }"
		"{ BoxCoordSave fd        |      | (optional) path where storing coordinates of Bounding Boxes Detected }"
        "{ SegImgSave fs        |      | (optional) path where store Segmented Hands Imgs }"
		"{ MaskImgSave fs       |      | (optional) path where store Masks of Segm Hands Imgs }"
    );
	if(argc < 3)
	{
        std::cout << "Not enough CML parameters" << std::endl;
        parser.printMessage();
		return -1;
	}
	// Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
	// Check if the CMDL has the mandatory keywords for running the executable
	if(!parser.has("TestImgPath") && !parser.has("model"))
	{
		std::cout << "" <<std::endl;
		std::cout << "-TestImgPath and -model keywords are required for running the executable!" <<std::endl;
		std::cout << "" <<std::endl;
		std::cout << "The correct syntax : .\\name of the executable -keyword=\"argument required ..." << std::endl;
        std::cout << "" <<std::endl;
		parser.printMessage();
		return -1;
	}
	std::string test_imgs_path_fold = (parser.get<std::string>("TestImgPath")); 
	std::string model(parser.get<std::string>("model")); 
	std::string det_img_path_fold(parser.get<std::string>("DetImgSave")); 
	std::string box_coord_path_fold(parser.get<std::string>("BoxCoordSave"));
	std::string seg_img_path_fold(parser.get<std::string>("SegImgSave"));
	std::string mask_img_path_fold(parser.get<std::string>("MaskImgSave"));
	// Check the CMDL syntax if is correct
	if(!parser.check())
	{
		std::cout << "The correct syntax : .\\name of the executable -keyword=\"argument required ..." << std::endl;
        parser.printMessage();
		return -1;
	}
	if((test_imgs_path_fold == "true") || (model == "true") ||  (det_img_path_fold == "true")
	    || (box_coord_path_fold == "true") || (seg_img_path_fold == "true") ||  (mask_img_path_fold == "true"))
	{
		std::cout << "The correct syntax : name of the executable -keyword=\"path to the folder\" and the model has to end in .onnx format" << std::endl;
		return -1;
	}
	// Check the validity of the paths 
	int flag = 0;
    checkPath(flag,test_imgs_path_fold,test_imgs_path_fold);
	if(flag == -1){
		return -1;
	}
	// Load the images and the templates of left and right hands
	std::vector<cv::String> file_name;
	// check if the image format is correct
	// Insert your path to the image folder in the " " of glob
	cv::glob(test_imgs_path_fold+"*.jpg",file_name,false);
	std::vector<cv::Mat> images;
	size_t count = file_name.size();
    std::cout << "0 - Pre Load Images " << std::endl;
	flag = 0;
	flag = retrieveImage(count,file_name,images);
	if(flag == -1){
		return -1;
	}
	// Load the DNN model 
	cv::dnn::Net net = cv::dnn::readNet(model);
	if(net.empty())
	{
		std::cout << "the DNN model has to be .onnx format exemple : path_to_the_model\\model.onnx" << std::endl;
		return -1;
	}
    // Set the colors list to use for boxes and segmentations
    static std::vector<cv::Scalar> colors{
        cv::Scalar(0 ,0 ,255),
        cv::Scalar(255, 178, 50),
        cv::Scalar(0 ,255 ,255),
        cv::Scalar(0 ,255 ,0)
    };
	// Step 1 :
	// Dnn test , draw bounding boxes and create the hand mask from original images
	std::cout << "1 - Detection Task " << std::endl;
	std::vector<cv::Mat> hand_detected_img;
	std::vector<std::tuple<int,std::vector<cv::Rect>>> bounding_boxes;
	handDetection(images,net,hand_detected_img,bounding_boxes,colors);
    // Step 2 :
	// Save the resulting images and write a txt file for each image with the coordinate of the bounding boxes found
	std::cout << "2 - Save the Results of Detection " << std::endl;
	// if the paths for save the results are empty do not do anything
	if(!det_img_path_fold.empty() && !box_coord_path_fold.empty())
	{
		saveImg(hand_detected_img, det_img_path_fold);
        writeBoundBoxTxt(bounding_boxes,box_coord_path_fold);
	}
    // Step 3 :
	// Hands Segmentation, create a mask and segment the original images
	std::cout << "3 - Segmentation Task " << std::endl;
    std::vector<cv::Mat> hand_segmented_img;
	std::vector<cv::Mat> hand_masks_img;
	handSegmentation(images,hand_segmented_img,hand_masks_img,bounding_boxes,colors);
	// Step 4 :
	// Save the resulting results and masks
	std::cout << "4 - Save the Results of Segmentation " << std::endl;
	// if the paths for save the results are empty do not do anything
	if(!seg_img_path_fold.empty() && !box_coord_path_fold.empty())
	{
	    saveImg(hand_segmented_img, seg_img_path_fold);
        saveImg(hand_masks_img,mask_img_path_fold);
	}
	return 0;
}
