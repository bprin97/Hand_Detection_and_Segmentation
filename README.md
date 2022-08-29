# Hand_Detection_and_Segmentation

Human hands play a very important role when people interact with each other or with objects and tools. Therefore, the reliable detection of human hands from images is a very attractive result for applications of human-robot interaction, gesture recognition or human activity analysis.

The goal of this project is to develop a system capable of 

1. detecting human hands in an input image and 
2. segmenting human hands in the image from the background. 

The appearance of hands can change sensibly from image to image, depending on the viewpoint, and can be influenced by many factors such as skin tones or the presence of clothes (e.g., shirt, rings) and objects.

# Our Solution in Summary :

As described above the task is splitted into half :

1. Detection of hands : data preprocessing + CNN Yolov5
2. Segmentation of hands : 
   1. Color Images (RBG) :
      1. Background Removal through Graph Cut triggered by bounding boxes derived in the previous step
      2. The resulting image then is converted from RBG to YCrCb and is performed a threshold : 0≤𝑌≤255 , 135≤𝐶𝑟≤180 and 85≤𝐶𝑏≤135
      3. Refined the outcomes with and Open Morphological operation with a rectangular structuring element.
   2. Gray Images :<br>
      1. Convertion from an image of 3 Channels to 1 Channel , OpenCV read every image as 3 Channels as default if you did not use the specific option however from the fact that the application must run in every cases we left 3 Channels lecture as default.
      2. Intensify the contrast of the 1 Channel image with Histogram Equalization
      3. Binary | OTSU's thresholding
      4. 2 iteration of Erosion Operation and 1 iteration of Dilatation Operation to recover the areas of the subjetc's shape 
      5. Extract the area of interest (ROI) using the bounding boxes found previously
      6. Canny method to extract edges from the image
      7. 1 Iteration of Dilatation for enclose the edges found
      8. findContours from the resulting image
      9. approximate the Polygons by Convex Hull approach
      10. filtering from the Polygons found by chosing the one that is not less than (Bounding box Area)/6
3. (Additional Feature) :
   1. Save Output images in .jpg format
   2. Write Bounding Box Coordinates in a .txt file for each images
4. Accuracy Metric :
   1. Hand's Detection metric : Intersection over Union (IoU) for each bounding boxes and average :
      1. Read both the Ground truth bounding boxes coordinates and the derived ones.
      2. Sorting the Coordinates of the bounding boxes in the ascending order by the x coordinate
      3. Find Indeces in case of False Positives or Unmatched Hands :
         1. Unmatches Case :
             1. pick the current x coordinate of the ith ground truth bounding box of the current image under study 
             2. check the distance from all the x coordinates of the corresponding derived coordinate of the image under study
             3. if the distance is too large for all the list of the x coordinates of the derived ones then save it into a vector
             4. otherwise continue the cicle until no one coordinate is left.
         2. False Positive Case :
             1. pick the current x coordinate of the ith derived bounding box of the current image under study 
             2. check the distance from all the x coordinates of the corresponding ground truth coordinate of the image under study
             3. if the distance is too large for all the list of the x coordinates of the ground truth ones then save it into a vector
             4. otherwise continue the cicle until no one coordinate is left.
       4. All the previous steps are mandatory to compute the metrics, in fact it is needed to jump false positives (Excees of object with the respect of the ground truth) or set directly to 0 the current bounding box metric in case of unmatching (Shortage of object wtr of the ground truth) and so continue to the next bounding box evaluation.
       5. Compute the intersection and the Union area of the 2 Bounding Boxes and then the IoU% = ((Intersection)/(Union))*100
       6. IoU_average = (sum_of_IoU%_of_each_boxes_of_the_current_image)/(number_of_bounding_boxes_of_the_current_image)
   2. Hand's Segmentation metric : Pixel's Accuracy for both classes and average.
       1. given the ith image with the derived and the ground truth mask compute the intersection and the union area of the foreground and f_pixel_accuracy% = ((Intersection)/(Union))*100
       2. perform the not_bitwise to invert the pixel values (black --> white || white --> black) this is the bacground mask
       3. compute the intersection and the union area of the background and b_pixel_accuracy% = ((Intersection)/(Union))*100
       4. pixel_accuracy_average% = (f_pixel_accuracy% + b_pixel_accuracy%)/2
   3. Write the Bounding Boxes Accuracies found in a .txt file for each image
   4. Write the Pixel Accuracies found in a .txt file for each image

# Test Set :

A collection of 20 images from EgoHand with Bounding Boxes and Masks ground truth

# Results :

The outcomes are stored in the related folder , with both derived mask and coordinates. In a nutshell the results are good with an accuracy of 85% (Segmentation) and 80% (Detection) for most of the images.

# Computational time :

From the project requirement the solution asked must not be end-to-end , so the usage of a Deep Learning solution for all the task was not allowed. This is why the consideration of the identification with a CNN was more appealing than its usage with the segmentation, in fact usually it has higher rate of correctness and precision in that task and more approximative for segmentation in which then have to be refined always by Graph Cut method in which its initialization with the mask derived by the CNN.
On Overall :
1. Detection : 10 minutes more or less
2. Segmentation : this depends on the device in use, if it is performed in the cloud it takes just few minutes : 3 minutes at images, however with a common computer it can run also for an hour for all the images in the worst case.

# Running the Experiment :
In order to run the application do :
1. Install the OpenCV library and CMake and configure your system to makes them ready for use
2. Open the Terminal and go to the project folder
3. create a build empty folder and move the terminal to build
4. here run the following terminal command : cmake ..
5. run the following command : make
6. then move in the folder Debug where the executable file named as 'CVProject' or 'CVProject.exe'
7. run :  .\CVProject.exe -TestImgPath="your_path_to_the_test_dataset" -model="your_path_to_the_CNN_model" -DetImgSave="your_path_where_save_the_images_with_the_boxes" -BoxCoordSave="your_path_where_to_save_the_txt_files_with_the_derived_boxes" -SegImgSave="path_to_save_the_segmented_images"
-MaskImgSave="path_to_save_the_mask_derived"
