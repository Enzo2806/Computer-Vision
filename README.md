# Computer-Vision
Computer Vision Assignments


## Assignment 1: Image Filtering and Edge Detection

The first assignment involved various image processing tasks using Python, primarily focusing on image filtering and edge detection.

### Key Actions
1. **Image Acquisition:** Captured two images of a household object from slightly different viewpoints using a camera.
2. **Grayscale Conversion:** Converted the captured images to grayscale using the Python formula to weight RGB values.
3. **Image Resizing:** Adjusted the image dimensions so the largest dimension was 256 pixels, using scikit-image's resize function.
4. **Gaussian Smoothing:** Applied Gaussian smoothing to the images with both 5x5 and 15x15 pixel kernels to reduce noise and detail.
5. **Gradient Computation:** Calculated x and y image gradients using Sobel filters on the smoothed images.
6. **Edge Magnitude and Orientation:** Computed the edge gradient magnitude and orientation from the gradients, visualized with an RGB colormap.
![A1-output](https://github.com/user-attachments/assets/7cfcdd26-e70e-4489-9a1e-0c9657db3983)

## Assignment 2: Feature Matching, Image Stitching (PANORAMA!)

The second assignment focused on advanced image processing tasks such as feature matching and image stitching using Python.

### Key Actions
1. **Harris Corner Detection:**
   - Computed image derivatives using Sobel filters.
   - Applied Gaussian filtering to smooth the derivatives.
   - Calculated the corner response function and performed non-maximum suppression to identify potential corner points.
   - Detected and marked corners by applying a threshold and overlaying detected corners on the original images.
   - Experimented with different threshold values.

2. **SIFT Features:**
   - **SIFT Keypoint Matching Between Two Images:**
     - Computed SIFT keypoints and descriptors.
     - Matched keypoints using a brute-force method and sorted matches by distance.
     - Displayed the top ten matches and plotted matching distances for the top 100 matches.
   - **Scale Invariance:**
     - Computed SIFT keypoints for scaled versions of an image and matched keypoints between the original and scaled images.
     - Analyzed how scaling affects matching distances.
   - **Rotation Invariance:**
     - Rotated an image by various angles and computed SIFT keypoints for each.
     - Matched keypoints between the original and rotated images and discussed the trends observed.

3. **Image Stitching:**
   - Detected and matched keypoints between different views of the same scene.
   - Estimated and applied homography using RANSAC to align and transform images.
   - Stitched images together by averaging pixel intensity in the overlap regions and displayed the final stitched images.
![A2-output](https://github.com/user-attachments/assets/2c90a38c-86bd-4ff5-88b9-d0d38df817a9)
   
## Assignment 3: Classifiers, Object Detection

The third assignment focused on machine learning classifiers and object detection using Python, specifically utilizing the MNIST dataset and YOLOv5 for object detection.

### Key Actions
1. **MNIST Classification:**   
   - **Support Vector Machines (SVM):**
     - Trained a linear SVM and reported its accuracy and confusion matrix.
     - Explored non-linear SVMs with polynomial and RBF kernels, conducted a grid search for the polynomial SVM, and compared the performances.
   - **Random Forest Classifiers:**
     - Trained a Random Forest classifier with different numbers of trees and evaluated the impact of max depth on performance.
     - Identified the optimal number of trees for the best performance.
   - **Comparison of Classifiers:**
     - Compared the best-performing Random Forest classifier with the best-performing SVM, analyzing their accuracy and difficulty in classifying specific digits.
     - Evaluated and reported performance on the test set using both classifiers.

2. **YOLO Object Detection:**
   - **Image Acquisition:**
     - Captured an image of a Montreal street scene using DSLR. 
   - **Object Detection:**
     - Utilized pre-trained YOLOv5 weights to detect objects in the captured image.
     - Implemented custom functions to draw bounding boxes and labels on detected objects.
     - Experimented with different object detection confidence thresholds and analyzed the impact on detected object counts.
![A3-output](https://github.com/user-attachments/assets/422d0cb2-4ade-43e6-9c8d-4b7551a5d2ad)

## Assignment 4: Image Segmentation

The fourth assignment was focused on implementing and evaluating a semantic segmentation model using the Oxford-IIIT Pets Dataset, with the goal of gaining practical experience in deep learning techniques for image segmentation.

### Key Actions
1. **Data Preparation:**
   - Downloaded and extracted the Oxford-IIIT Pet Dataset.
   - Preprocessed the data by resizing images, normalizing pixel values, and splitting the dataset into training and validation subsets.

2. **Baseline Method:**
   - Implemented Gaussian mixture with k=3.

3. **Model Implementation:**
   - Implemented U-Net and applied optional data augmentations like rotations and flips to improve model generalization.
   - Trained the model on the training set and monitored training and validation losses, as well as model accuracy.

4. **Model Evaluation:**
   - Evaluated the model performance using the Dice Score on the validation set.
   - Visualized results by displaying images from the validation set alongside the predicted and ground truth masks.
![A4-output](https://github.com/user-attachments/assets/7caed433-18eb-4c1d-a88a-646566d0cbd7)
