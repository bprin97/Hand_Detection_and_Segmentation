/*

Authors = Bruno Principe , Piermarco Giustini 

*/
#include "Metrics.h"
int main(int argc, char** argv){
	// CMD parser definitions
	cv::CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this message }"
        "{ MasksGroundTruth i        |      | (required) path to reference folders of Ground Truth Masks }"
		"{ MasksDerived i        |      | (required) path to reference folders of Derived Masks }"
		"{ BoxesGroundTruth f        |      | (required) path to reference folders of Ground Truth Boxes }"
		"{ BoxesDerived f        |      | (required) path to reference folders of Derived Boxes }"
        "{ PixelAccPath p        |      | (optional) path for storing the Pixel Accuracy in a .txt file for each image }"
		"{ BoxAccPath p        |      | (optional) path for storing the Boxes Accuracy in a .txt file for each image }"
    );
	if(argc < 5)
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
	if(!parser.has("MasksGroundTruth") && !parser.has("MasksDerived") && !parser.has("BoxesGroundTruth") && !parser.has("BoxesDerived"))
	{
		std::cout << "" <<std::endl;
		std::cout << "-MasksGroundTruth , -MasksDerived , BoxesGroundTruth and BoxesDerived keywords are required for running the executable!" <<std::endl;
		std::cout << "" <<std::endl;
		std::cout << "The correct syntax : .\\name of the executable -keyword=\"argument required ..." << std::endl;
        std::cout << "" <<std::endl;
		parser.printMessage();
		return -1;
	}
	std::string ground_truth_mask_path_fold = (parser.get<std::string>("MasksGroundTruth")); 
	std::string der_mask_path_fold(parser.get<std::string>("MasksDerived")); 
	std::string ground_truth_box_path_fold(parser.get<std::string>("BoxesGroundTruth")); 
	std::string der_box_path_fold(parser.get<std::string>("BoxesDerived")); 
	std::string pixel_acc_path(parser.get<std::string>("PixelAccPath"));
	std::string box_acc_path(parser.get<std::string>("BoxAccPath"));
	// Check the CMDL syntax if is correct
	if(!parser.check())
	{
		std::cout << "The correct syntax : .\\name of the executable -keyword=\"argument required ..." << std::endl;
        parser.printMessage();
		return -1;
	}
	if(ground_truth_mask_path_fold == "true" || der_mask_path_fold == "true" ||  ground_truth_box_path_fold == "true"
	    || der_box_path_fold == "true" || pixel_acc_path == "true" ||  box_acc_path == "true")
		{
			std::cout << "The correct syntax : is name of the executable -keyword=\"path to the folder\"" << std::endl;
			return -1;
		}
	// Check the validity of the paths 
	int flag = 0;
	std::string ground_truth_mask_path_fold_f;
    checkPath(flag,ground_truth_mask_path_fold,ground_truth_mask_path_fold_f);
	if(flag == -1){
		return -1;
	}
	std::string der_mask_path_fold_f;
    checkPath(flag,der_mask_path_fold,der_mask_path_fold_f);
	if(flag == -1){
		return -1;
	}
	std::string ground_truth_box_path_fold_f;
    checkPath(flag,ground_truth_box_path_fold,ground_truth_box_path_fold_f);
	if(flag == -1){
		return -1;
	}
	std::string der_box_path_fold_f;
    checkPath(flag,der_box_path_fold,der_box_path_fold_f);
	if(flag == -1){
		return -1;
	}
	// 0 - Load the ground truth masks and the resulting masks
	std::vector<cv::String> file_name_mask_ground_truth;
	std::vector<cv::String> file_name_mask_derived;
	// check if the image format is correct
	// Insert your path to the image folder in the " " of glob
	cv::glob(ground_truth_mask_path_fold_f+"*.png",file_name_mask_ground_truth,false);
	cv::glob(der_mask_path_fold_f+"*.jpg",file_name_mask_derived,false);
	std::vector<cv::Mat> correct_mask;
	std::vector<cv::Mat> derived_mask;
	size_t count_file_corr = file_name_mask_ground_truth.size();
	size_t count_file_der = file_name_mask_derived.size();
    std::cout << "Pre Load Images " << std::endl;
	flag = 0;
	flag = retrieveImage(count_file_corr,file_name_mask_ground_truth,correct_mask);
	if(flag == -1){
		return -1;
	}
	flag = 0;
	flag = retrieveImage(count_file_der,file_name_mask_derived,derived_mask);
	if(flag == -1){
		return -1;
	}
	// 1 - Computing the Pixel Accuracy
	std::vector<double> foreground_accuracy;
    std::vector<double> background_accuracy;
	std::vector<double> average_accuracy;
    pixelAccuracy(correct_mask,derived_mask,foreground_accuracy,background_accuracy,average_accuracy);
	std::cout << "PixelAccuracy " << std::endl;
	// 2 - Load Correct Bounding Boxes and Derived Buonding Boxes
    std::vector<cv::String> file_name_boxes_ground_truth;
	std::vector<cv::String> file_name_boxes_derived;
	cv::glob(ground_truth_box_path_fold_f+"*.txt",file_name_boxes_ground_truth,false);
	cv::glob(der_box_path_fold_f+"*.txt",file_name_boxes_derived,false);
	count_file_corr = file_name_boxes_ground_truth.size();
	count_file_der = file_name_boxes_derived.size();
	std::vector<std::vector<cv::Rect>> corr_boxes;
	std::vector<std::vector<cv::Rect>> der_boxes;
	// Retrieve the ground truth boxes
	flag = retrieveBoxes(count_file_corr,file_name_boxes_ground_truth,corr_boxes);
	if(flag == -1){
		return -1;
	}
	// Retrieve the derived boxes
	flag = retrieveBoxes(count_file_der,file_name_boxes_derived,der_boxes);
	if(flag == -1){
		return -1;
	}
	// 3 - Sorting the coordinates orders
	// sorting in ascendent order of the x coordinate the boxes of ground truth and derived
    sortVectOfBox(corr_boxes);
	sortVectOfBox(der_boxes);
	// Swap 1 Bounding Box of image 12 from the 3rd to the 4th position, in this way all boxes from ground truth and derived are aligned 
	int img_pos = 11;
	int box_pos_in = 2;
	int box_pos_fin = 3;
	swapArrayBoxPos(der_boxes,img_pos,box_pos_in,box_pos_fin);
	// 4 - Computing the Intersection over Union (IoU) of Bouding Boxes
	std::vector<std::vector<double>> iou_single_accuracy;
	std::vector<double> iou_accuracy;
	boundBoxAccuracy(corr_boxes,der_boxes,iou_single_accuracy,iou_accuracy,correct_mask);
	// if the paths for save the results are empty do not do anything
	if(!pixel_acc_path.empty() && !box_acc_path.empty())
	{
		// 5 - Store the Pixel Accuracy in a .txt file for each image
	    writePixelAccTxt(foreground_accuracy,background_accuracy,average_accuracy,pixel_acc_path);
		// 6 - Store the IoU Bounding Boxes Accuracy in .txt format for each image
        writeBoxAccTxt(iou_single_accuracy,iou_accuracy,box_acc_path);
	}
    
	return 0;
}