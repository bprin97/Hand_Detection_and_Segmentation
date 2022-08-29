/*

Authors = Bruno Principe , Piermarco Giustini

*/
// Libraries definitions
#include "DetAndSegmentation.h"
#include <iostream>
#include <string>
#include <algorithm>
// Pre Processing functions :
//
// isGrayImage takes as input an image and check if is in gray scale format , if it is returns true otherwise false
bool isGrayImage( cv::Mat img )
{
    // Variable initializations
    cv::Mat dst;
    cv::Mat bgr[3];
    // split the image in a multiple images with the corresponding channels
    cv::split( img, bgr );
    // Compute the absolute difference with the first and the second channels
    cv::absdiff( bgr[0], bgr[1], dst );
    // if there are some non zero pixels it returns false
    if(cv::countNonZero( dst ))
        return false;
    // Compute the absolute difference with the first and the third channels
    cv::absdiff( bgr[0], bgr[2], dst );
    // if all the pixels are 0 return true
    return !countNonZero( dst );
}
// skinDetection produce the mask from the color thresholding, it stores the skin masks at the ends.
void skinDetection(cv::Mat image,cv::Mat &partial_mask)
{
    // Create a structuring element
    int morph_size_1 = 3;
    cv::Mat element_1 = getStructuringElement(cv::MORPH_RECT,cv::Size(morph_size_1 + 1, morph_size_1 + 1),
            cv::Point(morph_size_1,morph_size_1));
    //converting from gbr to YCbCr color space
    cv::Mat img_YCrCb = cv::Mat::zeros(image.rows, image.cols, image.type());
    cv::cvtColor(image, img_YCrCb, cv::COLOR_BGR2YCrCb);
    //skin color range for YCbCr color space , picks only color between :  Y >= 0 , 135 <= Cr <= 180 , 85 <= Cb <= 135 
    partial_mask = cv::Mat::zeros(image.rows, image.cols, image.type());
    cv::inRange(img_YCrCb, cv::Scalar(0, 135, 85), cv::Scalar(255,180,135),partial_mask); 
    // Smooth the resulting mask by morphological OPEN operation
    cv::morphologyEx(partial_mask ,partial_mask ,cv::MORPH_OPEN ,element_1 ,cv::Point(-1,-1) ,1);
}
// Task 1 : Hands Detection :
//
// handDetection takes in input a vector of images and the DNN model , the vector of colors to use to draw the results, it save the outcomes in a vector of images and a vector of tuples <number of boxes,vector of bounding boxes>
void handDetection(std::vector<cv::Mat> images, cv::dnn::Net net,std::vector<cv::Mat>& hand_detected_img,std::vector<std::tuple<int,std::vector<cv::Rect>>>& bounding_boxes,std::vector<cv::Scalar> colors)
{
    // Definition of the Constants
    // Image Size
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    // Score, Confidence and NMS thresholds
    const float SCORE_THRESHOLD = 0.25;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.25;
    // Define the class of objects to finds
    std::vector<std::string> class_name = {"Hands"};
    // Text parameters
    const int THICKNESS = 1;
    for(int i=0 ; i<images.size();i++)
    {
        cv::Mat img_det = images[i].clone();
        cv::Mat input_image=cv::Mat::zeros(images[i].rows, images[i].cols, CV_8UC3);;
        // Pre Processing :
        std::vector<cv::Mat> detections = preProcess(images[i],input_image,net,INPUT_WIDTH,INPUT_HEIGHT);
        // Post Processing
        cv::Mat img = postProcess(img_det,input_image, detections, class_name,INPUT_WIDTH,INPUT_HEIGHT,CONFIDENCE_THRESHOLD,SCORE_THRESHOLD,
            NMS_THRESHOLD,THICKNESS,colors,bounding_boxes);
        showImage(img_det,"Current Image Hand Detection Output");
        hand_detected_img.push_back(img_det);
    }
}
// preProcess takes in input the image , the models and the image size
std::vector<cv::Mat> preProcess(cv::Mat &input_image,cv::Mat &resized_img, cv::dnn::Net &net,const float INPUT_WIDTH,const float INPUT_HEIGHT)
{
    // Make the image square 
    int _max = MAX(input_image.cols, input_image.rows);
    resized_img = cv::Mat::zeros(_max, _max, CV_8UC3);
    input_image.copyTo(resized_img(cv::Rect(0, 0, input_image.cols, input_image.rows)));
    // Switch the B and R channels , resize the image into the requested size and normalize it to [0,1[
    cv::Mat blob;
    cv::dnn::blobFromImage(resized_img, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    // Makes the predictions
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}
// postProcess takes in input the input image the resized image , the class name , the size of the settings defined in handDetection : image size , score , NMS and confidence thresholds
// and process the image in order to obtain the bounding boxes coordinates ,check the correctness of the predictions and draw them to the image
cv::Mat postProcess(cv::Mat &input_image,cv::Mat &resized_img, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name, const float INPUT_WIDTH,const float INPUT_HEIGHT ,
    const float CONFIDENCE_THRESHOLD,const float SCORE_THRESHOLD,const float NMS_THRESHOLD
    ,const int THICKNESS, std::vector<cv::Scalar> colors,std::vector<std::tuple<int,std::vector<cv::Rect>>>& bounding_boxes)
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    // Resizing factor.
    float x_factor = resized_img.cols / INPUT_WIDTH;
    float y_factor = resized_img.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;
    // 25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * class_scores = data + 5;
            // Create a 1x1 Mat and store class score of 1 class.
            cv::Mat scores(1, class_name.size(), CV_32FC1, class_scores);
            // Perform minMaxLoc and acquire the index of best class score.
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 6;
    }
    // Perform Non-Maximum Suppression and draw predictions.
    std::vector<int> indices;
    std::vector<cv::Rect> img_boxes;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Store the Bounding Boxes for the current image
        img_boxes.push_back(box);
        // Draw bounding box.
        cv::rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), colors[i], 3*THICKNESS);
        // Get the label for the class name and its confidence.
        std::string label = cv::format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
    }
    std::tuple<int,std::vector<cv::Rect>> coord = std::make_tuple(indices.size(),img_boxes);
    bounding_boxes.push_back(coord);
    return input_image;
    
}
// Task 2 : Hands Segmentation
//
// handSegmentation takes as input the vector of images , the vector of colors and the vector of buonding boxes , it saves the masks and the segmented images in different arrays
void handSegmentation(std::vector<cv::Mat> images,std::vector<cv::Mat>& hand_segmented_img,std::vector<cv::Mat>& hand_masks_img,std::vector<std::tuple<int,std::vector<cv::Rect>>>& bounding_boxes,
     std::vector<cv::Scalar> colors)
{
    for(int i=0;i<bounding_boxes.size();i++)
    {
        cv::Mat col_mask_f = cv::Mat::zeros(images[i].rows,images[i].cols,images[i].type());
        cv::Mat img_final_mask = cv::Mat::zeros(images[i].rows,images[i].cols,CV_8UC1);
        cv::Mat img_final_segm = images[i].clone();
        int num_boxes = std::get<1>(bounding_boxes[i]).size();
        for(int j=0;j<num_boxes;j++)
        {
            // Initialize some requirements
            cv::Mat partial_mask = cv::Mat::zeros(images[i].rows,images[i].cols,images[i].type());
            cv::Rect box = std::get<1>(bounding_boxes[i])[j];
            cv::Mat dest= cv::Mat::zeros(images[i].rows,images[i].cols,images[i].type());
            cv::Mat img_norm= cv::Mat::zeros(images[i].rows,images[i].cols,images[i].type());
            // Check if an image is gray scale
            bool gray = isGrayImage(images[i]);
            if(gray == true){
                // Retrieve the mask of the image in gray scale format
                createMaskForGrayImage(images[i],dest,box);
            // if BGR image do
            }else
            {
                // 1 - backbround removal from the contents inside the box
                // 2 - sking thresholding
                unsigned int iteration = 7;
                removeBackgroundfromImg(images[i],dest,box,iteration,partial_mask); 
                skinDetection(dest,dest);
            }
            // Merge the previous image masks together
            cv::bitwise_or(img_final_mask,dest,img_final_mask);
            // Color the hand Area with Trasparency
            cv::Mat col_mask = cv::Mat::zeros(images[i].rows,images[i].cols,images[i].type());
            colorHandMask(dest,col_mask, colors[j]);
            cv::bitwise_or(col_mask_f,col_mask,col_mask_f);
        }
        // Merge the previous Segmentation together
        colorHandSegmentation(img_final_segm,img_final_segm,col_mask_f); 
        // Store the masks and the segmented images derived in the final vectors
        hand_masks_img.push_back(img_final_mask);
        hand_segmented_img.push_back(img_final_segm);
        showImage(img_final_mask,"Current Image Hand Segmentation Mask Output");
        showImage(img_final_segm,"Current Image Hand Segmentation Final Output");
    }
}
// colorHandMask takes as input the mask and colors the area where there are white pixels from a color chosen
void colorHandMask(cv::Mat mask, cv::Mat &col_mask , cv::Scalar color)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL  , cv::CHAIN_APPROX_NONE);
    // Draw the Hand Areas of the given color
    cv::drawContours(col_mask, contours, -1, color, -1);
}
// colorHandSegmentation takes as input the img, the mask and merge them with trasparency effects (causing a ghosting effects of the original image)
// the image segmented is saved in the dest cv::Mat
void colorHandSegmentation(cv::Mat img,cv::Mat &dest, cv::Mat mask)
{
    // Add transparency to the mask color
    double alpha = 0.5;
    double beta = 1 - alpha;
    cv::addWeighted(img,alpha,mask,beta,0,dest);
}
// createMaskForGrayImage takes as input the image and the current buonding box and return the mask
void createMaskForGrayImage(cv::Mat img,cv::Mat &partial_mask,cv::Rect box)
{
    // Convert the image into a gray scale version
    cv::Mat gray_img = cv::Mat::zeros(img.rows,img.cols,img.type());
    cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
    // Initialization of some variables to store partial results of each pre-process
    cv::Mat morph_img = cv::Mat::zeros(gray_img.rows,gray_img.cols,gray_img.type());
    cv::Mat thres_img = cv::Mat::zeros(gray_img.rows,gray_img.cols,gray_img.type());
    cv::Mat canny_img = cv::Mat::zeros(gray_img.rows,gray_img.cols,gray_img.type());
    cv::Mat equaliz_img = cv::Mat::zeros(img.rows,img.cols,gray_img.type());
    cv::Mat box_img = cv::Mat::zeros(img.rows,img.cols,gray_img.type());
    // Equalize the gray scale image to accentuate contrast
    cv::equalizeHist(gray_img,equaliz_img);
    // Thresholding of the equalized image by OTSU approach and BINARIZATION
    cv::threshold(equaliz_img,thres_img,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
    // Perform erosion for 2 iterations and dilatation to 1 iteration to enhence contouring results
    cv::erode(thres_img, morph_img,cv::Mat(),cv::Point(-1,-1),2);
    cv::dilate(morph_img, morph_img,cv::Mat(),cv::Point(-1,-1),1);
    // Extract the interested area from the buonding box
    extractMaskFromBoundingBox(morph_img,box_img,box);
    // Perform Canny to retrieve edges from the image
    cv::Canny(box_img,canny_img,100,150,3);
    // Dilate the results for closing the poligones
    cv::dilate(canny_img, canny_img,cv::Mat(),cv::Point(-1,-1),1);
    // Find the countours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(canny_img, contours, hierarchy, cv::RETR_TREE  , cv::CHAIN_APPROX_SIMPLE);
    // Compute the approximation of polygones to close openes lines through convex hull technique
    std::vector<std::vector<cv::Point> >hull( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        convexHull( contours[i], hull[i] );
    }
    // Draw the area of the convex hull with area greater than the (bounding box area)/6 to filters unwanted objects
    cv::Mat drawing = cv::Mat::zeros( img.size(), CV_8UC1 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        double cont_area = cv::contourArea(hull[i]);
        if(cont_area >= (box.area())/6 )
        {
            drawContours( drawing, hull, (int)i, cv::Scalar(255,255,255),-1 );
        }
    }
    // Save the results
    partial_mask = drawing;
}
// extractMaskFromBoundingBox takes as input the image and the current bounding box and cross out all the informations out of the box
// and save the result
void extractMaskFromBoundingBox(cv::Mat img,cv::Mat &partial_box_area,cv::Rect box)
{
    partial_box_area = img.clone();
    // Create a mask to separate only the area of the bounding boxes
    cv::Mat bound_box_mask = cv::Mat::zeros(img.rows,img.cols,img.type());
    cv::rectangle(bound_box_mask,box,cv::Scalar(255,255,255),-1);
    // Select the area of the current bounding box
    cv::bitwise_and(img,bound_box_mask,partial_box_area);
}
// removeBackgroundFromImg takes as input the img , the current buonding box and the number of iteration to performs the task, it separates the backgrounds
// to the foregrounds inside the boxes and save the final outcomes both on BGR format "dest" and Binary format "mask"
void removeBackgroundfromImg(cv::Mat img,cv::Mat &dest,cv::Rect box,unsigned int iteration,cv::Mat &mask2)
{
    // Initialize the mask
    cv::Mat mask = cv::Mat::zeros(img.rows,img.cols,CV_8UC1);
    // Initialize the background mask and foreground mask
    //cv::Mat bgModel,fgModel;
    // Extract the Foreground from the Background in the image
    cv::grabCut(img,mask,box,cv::Mat(),cv::Mat(),iteration,cv::GC_INIT_WITH_RECT);
	// combine Exact Foreground to Probable Foreground
    mask2 = (mask == 1) + (mask == 3);
    // Create the Final Mask
    img.copyTo(dest, mask2);
}
// Basic Operations :
// retrieveImage obtain the images from a folder, reads them and saves them into a vector of images
int retrieveImage(size_t count,std::vector<cv::String> file_name,std::vector<cv::Mat> &images)
{
    // Check if there are images or exit
	if(count == 0)
	{
		std::cout << "No Image Files in the folder chosen." << std::endl;
		return -1;
	}
    // Retriving the images
	for(int i=0;i<count;i++){
		cv::Mat img=cv::imread(file_name[i]);
		if(!img.empty()){
		    images.push_back(img);
		}
		else{
			return -1;
		}
	}
    return 0;
}
// saveImg saves the images given in input from a vector to a specified folder in .jpg format
void saveImg(const std::vector<cv::Mat> img, const cv::String path)
{
    for(int i=0 ; i<img.size();i++)
    {
        // derive the current image name
        cv::String current_img = "0";
        // if the image is less than the 10th write the current indx with a 0 before
        // otherwise write just the current indx 
        if(i<9)
        {
            current_img = '0'+std::to_string(i+1);
        }else{
            current_img = std::to_string(i+1);
        }
        // Check the path given in input , if contains as last string \ or / adds just the name of the file.jpg and save it
        // else if the path does not contains at the final character \ or / search the first occurency in the string 
        // and adds it then save the current image or if does not contains any \ or / not save anything
        if(path[path.size()-1] == '/')
        {
            cv::imwrite(path+"/"+current_img+".jpg",img[i]);
        }else if(path[path.size()-1] == '\\'){
            cv::imwrite(path+"\\"+current_img+".jpg",img[i]);
        }else{
            size_t found = path.find_first_of("\\");
            if(found != cv::String::npos)
            {
                cv::imwrite(path+"\\"+current_img+".jpg",img[i]);
            }else{
                size_t found = path.find_first_of('/');
                if(found != cv::String::npos)
                {
                    cv::imwrite(path+"/"+current_img+".jpg",img[i]);
                }else{
                    std::cout << "Not a full path , rewrite the full path where the files will be then stored" << std::endl;
                }
            }
        }
    }
}
// writeBounBoxTxt takes the vector of tuples<number of boxes,vector of bounding boxes> and the path, it write a txt file
// at each row the coordinate of one bounding box "x    y    width    height" to the path given
void writeBoundBoxTxt(const std::vector<std::tuple<int,std::vector<cv::Rect>>> bounding_boxes,const std::string path)
{
    // Initialization of the stream and the final path of the .txt file
    std::fstream file;
    std::string final_path;
    // Check the path given in input , if contains as last string \ or / then it will be the final path,
    // else if the path does not contains at the final character \ or / search the first occurency in the string 
    // and adds it to form the final path or if does not contains any \ or / not save anything
    if(path[path.size()-1] == '/')
    {
        final_path = path;  
    }else if(path[path.size()-1] == '\\'){
        final_path = path; 
    }else{
        size_t found = path.find_first_of("\\");
        if(found != cv::String::npos)
        {
            final_path = path+"\\";       
        }else{
            size_t found = path.find_first_of('/');
            if(found != cv::String::npos)
            {
                final_path = path+"/"; 
            }else{
                std::cout << "Not a full path , rewrite the full path where the files will be then stored" << std::endl;
            }
        }
    }
    for(int i=0;i<bounding_boxes.size();i++)
    {
        // derive the current image name
        std::string current_img = "0";
        // if the image is less than the 10th write the current indx with a 0 before
        // otherwise write just the current indx 
        if(i<9)
        {
            current_img = '0'+std::to_string(i+1);
        }else{
            current_img = std::to_string(i+1);
        }
        int num_boxes = std::get<1>(bounding_boxes[i]).size();
        // Complete the final name of the .txt file
        std::string file_path_name = final_path+current_img+".txt";
        // Open the file in w mode            
        file.open(file_path_name, std::ios_base::out);
        // Check if the file is open , then if is confirmed it proceeds to write at each line the coordinates of a bounding box of the current image
        if (!file.is_open()) 
        {
            std::cout << "failed to open " << file_path_name << '\n';
        }else
        {
            for(int j=0;j<num_boxes;j++)
            {
                std::string x = std::to_string(std::get<1>(bounding_boxes[i])[j].x);
                std::string y = std::to_string(std::get<1>(bounding_boxes[i])[j].y);
                std::string width = std::to_string(std::get<1>(bounding_boxes[i])[j].width);
                std::string height = std::to_string(std::get<1>(bounding_boxes[i])[j].height);
                file <<  x +"     "+y +"     "+width+"     "+height<< std::endl;
            }
        }
        file.close();
    }
}
// checkPath takes as input the current path and completes it , if is not a valid path returns -1
void checkPath(int &flag,const std::string path, std::string &final_path)
{
    flag = 0;
    // Check the path given in input , if contains as last string \ or / then it will be the final path,
    // else if the path does not contains at the final character \ or / search the first occurency in the string 
    // and adds it to form the final path or if does not contains any \ or / not save anything
    if(path[path.size()-1] == '/')
    {
        final_path = path;  
    }else if(path[path.size()-1] == '\\'){
        final_path = path; 
    }else{
        size_t found = path.find_first_of("\\");
        if(found != cv::String::npos)
        {
            final_path = path+"\\";       
        }else{
            size_t found = path.find_first_of('/');
            if(found != cv::String::npos)
            {
                final_path = path+"/"; 
            }else{
                std::cout << "Not a full path , rewrite the full path where the files will be then stored" << std::endl;
                flag = -1;
            }
        }
    }
}
// showImage plots the image given in input with a specific windows name
void showImage(cv::Mat img,const cv::String title)
{
    cv::namedWindow(title);
    cv::imshow(title,img);
    cv::waitKey(0);
}
