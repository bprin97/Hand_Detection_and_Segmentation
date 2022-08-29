/*

Authors = Bruno Principe , Piermarco Giustini

*/
// Library definition :
#include "Metrics.h"
//
// Functions :
//
// pixelAccuracy takes as input the masks sets and returns the hand segmentation and the background segmentation accuracies 
void pixelAccuracy(std::vector<cv::Mat> correct_mask,std::vector<cv::Mat> derived_mask,std::vector<double> &foreground_accuracy
    ,std::vector<double> &background_accuracy,std::vector<double> &average_accuracy)
{
    for(int i=0;i<correct_mask.size();i++){
        // Foreground Accuracy 
        double f_pixel_accuracy = 0;
        intersectOverUnionSegm(correct_mask[i],derived_mask[i],f_pixel_accuracy);
        foreground_accuracy.push_back(f_pixel_accuracy);
        // Background accuracy
        double b_pixel_accuracy = 0;
        cv::Mat b_correct_mask = cv::Mat::zeros(correct_mask[i].size(),correct_mask[i].type());
        cv::Mat b_derived_mask = cv::Mat::zeros(correct_mask[i].size(),correct_mask[i].type());
        cv::bitwise_not(correct_mask[i],b_correct_mask);
        cv::bitwise_not(derived_mask[i],b_derived_mask);
        intersectOverUnionSegm(b_correct_mask,b_derived_mask,b_pixel_accuracy);
        background_accuracy.push_back(b_pixel_accuracy);
        // Average pixel accuracy
        double avg_pixel_accuracy = (f_pixel_accuracy+b_pixel_accuracy)/2;
        average_accuracy.push_back(avg_pixel_accuracy);
    }
}
// intersectOverUnionSegm computes the IoU/pixel accuracy and returns the value
void intersectOverUnionSegm(cv::Mat correct,cv::Mat derived,double &pixel_accuracy)
{
    // Compute the intersect of the derived mask and the correct mask
    cv::Mat intersect = cv::Mat::zeros(correct.size(),correct.type());
    cv::bitwise_and(correct,derived,intersect);
    double intesect_value = cv::countNonZero(intersect);
    // Compute the union of the derived mask and the correct mask
    cv::Mat mask_union = cv::Mat::zeros(correct.size(),correct.type());
    cv::bitwise_or(correct,derived,mask_union);
    double union_value = cv::countNonZero(mask_union);
    // Compute the percentage
    double iou = intesect_value/ union_value;
    pixel_accuracy = (iou*100);
}
// retrieveBoxes reads the .txt files where are stored the labels and puts them into a vector of vector of Boxes
// if the files failed to open return -1 otherwise 0
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
		cv::Mat img=cv::imread(file_name[i],cv::IMREAD_GRAYSCALE);
		if(!img.empty()){
		    images.push_back(img);
		}
		else{
			return -1;
		}
	}
    return 0;
}
// writePixelAccTxt write the pixels accuracies and the average pixel accuracies in files per image to a specified path
void writePixelAccTxt(const std::vector<double> foreground_accuracy,const std::vector<double> background_accuracy,
    const std::vector<double> average_accuracy,const std::string path)
{
    // Initialization of the stream and the final path of the .txt file
    int flag = 0;
    std::fstream file;
    std::string final_path;
    // Check if the path is consistent otherwise do nothing
    checkPath(flag,path,final_path);
    if(flag == 0)
    {
        for(int i=0;i<foreground_accuracy.size();i++)
        {
            // derive the current image name
            std::string file_path_name;
            fileName(i,final_path,file_path_name);
            // Open the file in w mode            
            file.open(file_path_name, std::ios_base::out);
            if (!file.is_open()) 
            {
                std::cout << "failed to open " << file_path_name << '\n';
            }else
            {
                file <<"                                        Pixel Accuracy :                                      "<< std::endl;
                file <<""<< std::endl;
                file <<"Foreground Pixel Accuracy         Background Pixel Accuracy             Average Pixel Accuracy"<< std::endl;
                file <<""<< std::endl;
                file << "       "+ std::to_string(foreground_accuracy[i]) +"                           "+ std::to_string(background_accuracy[i]) +"                         "+std::to_string(average_accuracy[i])<< std::endl;
                file.close();
            }
        }
    }
}
// retrieveBoxes reads the .txt files where are stored the labels and puts them into a vector of vector of Boxes
// if the files failed to open return -1 otherwise 0
int retrieveBoxes(size_t count,std::vector<cv::String> file_name,std::vector<std::vector<cv::Rect>> &boxes)
{
    // Check if there are labels.txt or exit
	if(count == 0)
	{
		std::cout << "No Image Files in the folder chosen." << std::endl;
		return -1;
	}
    for(int i=0;i<file_name.size();i++)
    {
        std::fstream file;
        file.open(file_name[i],std::ios::in);
        if (!file.is_open()) 
        {
            std::cout << "failed to open " << file_name[i] << std::endl;
            return -1;
        }else
        {
            std::vector<cv::Rect> file_boxes;
            std::string tp;
            int coord = 0;
            int j = 0;
            std::string str_coord;
            while(getline(file, tp))
            {
                cv::Rect box;
                std::stringstream X(tp); 
                // Conversion from string to int
                X >> box.x >> box.y >> box.width >> box.height;
                file_boxes.push_back(box);
            }
            file.close();
            // include the boxes of the current file in the vector of boxes
            boxes.push_back(file_boxes);
        }
    }
    return 0;
}
// boundBoxAccuracy takes as input the vectors of ground truths and derived labels and save the results of the accuracies of the single box in a vector od vector of double 
// and the average accuracy of an image in a vector of double
void boundBoxAccuracy(std::vector<std::vector<cv::Rect>> true_box,std::vector<std::vector<cv::Rect>> derived_box,
    std::vector<std::vector<double>> &iou_single_accuracy,std::vector<double> &iou_accuracy, std::vector<cv::Mat> &correct_mask)
{
    
    for(int i = 0;i<true_box.size();i++)
    {
        int corr_size = true_box[i].size();
        // Vector for store the unidentified objects idexes from the True labels;
        std::vector<int> k;
        // Vector for store the False Positives objects idexes from the derived labels;
        std::vector<int> d;
        // Number of False Positives or Unidentified objects in the labels only if != from 0 
        int diff_size = 0;
        findUnmatchedIndex(diff_size,true_box[i],derived_box[i],k,d);
        std::vector<double> single_box_accuracy;
        double sum_accuracies = 0;
        double average_accuracies =0;
        int flag = 0;
        for(int j=0;j<true_box[i].size();j++)
        {
            // Check if exist unmatching or False Positives , track the index and erace it from the vector of labels
            if(diff_size != 0)
            {
                if(k.size()>0)
                {
                    for(int f=0;f<k.size();f++)
                    {
                        flag = 0;
                        if(j == k[f])
                        {
                            flag = 1;
                            break;
                        }
                    }
                }else if(d.size()>0)
                {
                    for(int f=0;f<d.size();f++)
                    {
                        flag = 0;
                        if(j == d[f])
                        {
                            flag = 1;
                            break;
                        }
                    }
                }
            }
            if(flag == 1 && k.size()>0)
            {
                single_box_accuracy.push_back(0);
                true_box[i].erase(true_box[i].begin()+j);
                std::cout << "True Box Size : " <<true_box[i].size() << std::endl;
            }else if(flag == 1 && d.size()>0)
            {
                derived_box[i].erase(derived_box[i].begin()+j);
            }
            // Compute the IoU between ground truth and derived labels
            double accuracy=0;
            cv::Mat partial_box_area_corr = cv::Mat::zeros(cv::Size(correct_mask[i].size()),correct_mask[i].type());
            // Compute the mask with the area of the bounding box of the true label
            makeMaskBoundingBox(correct_mask[i],partial_box_area_corr,true_box[i][j]);
            cv::Mat partial_box_area_der = cv::Mat::zeros(cv::Size(correct_mask[i].size()),correct_mask[i].type());
            // Compute the mask with the area of the bounding box of the true label
            makeMaskBoundingBox(correct_mask[i],partial_box_area_der,derived_box[i][j]); 
            intersectOverUnionSegm(partial_box_area_corr,partial_box_area_der,accuracy);
            std::cout << "Image n : "<< (i+1) << std::endl;
            std::cout << "Bound Box n "<< (j+1) << " accuracy : " << accuracy << std::endl; 
            single_box_accuracy.push_back(accuracy);
            sum_accuracies +=accuracy;
        }
        // Compute the average accuraces of all the boxes
        average_accuracies = sum_accuracies/corr_size;
        std::cout << "Average Accuracy : " << average_accuracies << std::endl;
        iou_accuracy.push_back(average_accuracies);
        iou_single_accuracy.push_back(single_box_accuracy);
    }
}
// makeMaskBoundingBox creates the box mask in order to performs IoU
void makeMaskBoundingBox(const cv::Mat &img,cv::Mat &partial_box_area,cv::Rect box)
{
    // Create a mask with only the area of the bounding boxes
    partial_box_area = cv::Mat::zeros(cv::Size(img.size()),img.type());
    cv::rectangle(partial_box_area,box,cv::Scalar(255,255,255),-1);
}
// finUnmatchedIndex controls if the derived labels has False Positives or lacks of unmatched objects with the respect to the ground truth 
void findUnmatchedIndex(int &diff_size,std::vector<cv::Rect> true_box,std::vector<cv::Rect> derived_box,std::vector<int> &k,std::vector<int> &di)
{
    int size_true = true_box.size();
    int size_der = derived_box.size();
    // Check if there is an unmatched box
    if(size_true != size_der)
    {
       diff_size = (size_true - size_der);
       // Multiple False Positives
       if(diff_size < 0)
       {
            std::cout << "Multiple False Positives " << std::endl;
            for(int i =0;i<size_der;i++)
            {
                // If the differeces of the current x coordinate of derived boxes  with the ground truth ones is larger or smaller than 200
                // save the index on d array for ignoring them during the accuracy test
                int box_der_x_cord = derived_box[i].x;
                int num_coord_max_dist = 1;
                for(int indx=0;indx<true_box.size();indx++)
                {
                    // initialize the counter for which it tails the number of elements that differents most from that value
                    // if it is equal to the size of the truth boxes array insert the coordinate to the unmatched array of index d
                    int box_corr_x_cord = true_box[indx].x;
                    int distance_coord = box_der_x_cord - box_corr_x_cord;
                    int min_dist_count = 0;
                    // If exist 1 coordinate that is closes reset the maximum distance's counter
                    if(distance_coord >= -50 && distance_coord <= 50)
                    {
                        num_coord_max_dist = 1;
                    }
                    if(distance_coord < (-200) || distance_coord > 200)
                    {
                        num_coord_max_dist +=1;
                    }
                    // If at the ends the maximum distance's counter is equal to the number of boxes in the derived box vector add the index
                    // of the current Box analyzed
                    if(num_coord_max_dist == derived_box.size() && indx == true_box.size()-1)
                    {
                        di.push_back(i);
                    }
                }
            }
        // the number of False Positive 
        diff_size = di.size();    
       }else
       {
            // Miss Positives
            //std:: cout << "Missed Positives " << std::endl;
            for(int i =0;i<size_true;i++)
            {
                // If the differeces of the current x coordinate of ground truth  with the derived ones is larger or smaller than 200
                // save the index on k array for put a 0 in the accuracy test
                int box_corr_x_cord = true_box[i].x;
                int num_coord_max_dist = 1;
                for(int indx=0;indx<derived_box.size();indx++)
                {
                    // initialize the counter for which it tails the number of elements that differents most from that value
                    // if it is equal to the size of the derived boxes insert the coordinate to the unmatched array of index k
                    int box_der_x_cord = derived_box[indx].x;
                    int distance_coord = box_corr_x_cord - box_der_x_cord;
                    int min_dist_count = 0;
                    // If exist 1 coordinate that is closes reset the maximum distance's counter
                    if(distance_coord >= -50 && distance_coord <= 50)
                    {
                        num_coord_max_dist = 1;
                    }
                    if(distance_coord < (-200) || distance_coord > 200)
                    {
                        num_coord_max_dist +=1;
                    }
                    // If at the ends the maximum distance's counter is equal to the number of boxes in the derived box vector add the index
                    // of the current Box analyzed
                    if(num_coord_max_dist == derived_box.size() && indx == derived_box.size()-1)
                    {
                        k.push_back(i);
                    }
                }
                
            }
            // number of Unmatched elements
            diff_size = k.size();
        }
    }else
    {
        // all the boxes are matched
        diff_size = 0;
    }
}
// comp defines the ordering rules of the boxes in the vector
static bool comp(const cv::Rect& a, const cv::Rect& b)
{
    // boxes ordered by x coordinate in ascending order
    return a.x < b.x;
}
// swapArrayBoxPos swap a box from an initial position to the final one in the vector of boxes of an image specified in the input
void swapArrayBoxPos(std::vector<std::vector<cv::Rect>> &sorted_box,int img_pos,int box_pos_in,int box_pos_fin)
{
    // Swap a box from an initial position to the final position given
    cv::Rect tmp_box;
    tmp_box = sorted_box[img_pos][box_pos_fin];
    sorted_box[img_pos][box_pos_fin] = sorted_box[img_pos][box_pos_in];
    sorted_box[img_pos][box_pos_in] = tmp_box;
}
// sortVectorOfBox takes the current vector containing the vectors of bounding boxes of each image and sort them in ascending order from the Box's x coordinate
void sortVectOfBox(std::vector<std::vector<cv::Rect>> &sorted_box)
{
    for(int i=0;i<sorted_box.size();i++)
    {
        std::sort(sorted_box[i].begin(),sorted_box[i].end(),comp);
    }
}
// boundBoxAccuracy takes as input the vectors of ground truths and derived labels and save the results of the accuracies of the single box in a vector od vector of double 
// and the average accuracy of an image in a vector of double
void writeBoxAccTxt(const std::vector<std::vector<double>> &iou_single_accuracy,
    const std::vector<double> &iou_accuracy,const std::string path)
{
    // Initialization of the stream and the final path of the .txt file
    int flag = 0;
    std::fstream file;
    std::string final_path;
    // Check if the path is consistent otherwise do nothing
    checkPath(flag,path,final_path);
    if(flag == 0)
    {
        for(int i=0;i<iou_accuracy.size();i++)
        {
            // derive the current image name
            std::string file_path_name;
            fileName(i,final_path,file_path_name);
            // Open the file in w mode            
            file.open(file_path_name, std::ios_base::out);
            if (!file.is_open()) 
            {
                std::cout << "failed to open " << file_path_name << '\n';
            }else
            {
                file <<"                                        Bounding Boxes Accuracy image "+std::to_string(i+1)+":                                      "<< std::endl;
                file <<""<< std::endl;
                file <<"Average Accuracy : "<< std::to_string(iou_accuracy[i]) << std::endl;
                file <<""<< std::endl;
                file <<"Accuracy per box"<< std::endl;
                for(int j=0;j<iou_single_accuracy[i].size();j++)
                {
                    file <<"Box "+std::to_string(j+1)+" Accuracy : "<< std::to_string(iou_single_accuracy[i][j]) << std::endl;
                    file <<""<< std::endl;
                }
                file.close();
            }
        }
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
// fileName creates the final name of the file and merge it with the path given the path to store a file and the current image number 
void fileName(int num,std::string final_path, std::string &file_path_name)
{
    // derive the current image name
    std::string current_img = "0";
    // if the image is less than the 10th write the current indx with a 0 before
    // otherwise write just the current indx 
    if(num<9)
    {
        current_img = '0'+std::to_string(num+1);
    }else{
        current_img = std::to_string(num+1);
    }
    // Complete the final name of the .txt file
    file_path_name = final_path+current_img+"_accuracy.txt";   
}
