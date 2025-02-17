
# Project Title

## Part 1

### Steps Involved

1. **Preprocessing (`pre_process`)**  
   - Converts the input image to grayscale.  
   - Scales the image for consistency.  
   - Applies Gaussian blur and adaptive thresholding for better edge detection.

2. **Edge Detection (`edge`)**  
   - Finds contours in the thresholded image.  
   - Calculates circularity and area to identify coin-like shapes.  
   - Draws detected coin edges on the original image.

3. **Region Based Segmentation (`segment_coins`)**  
   - Creates a mask using detected coin contours.  
   - Extracts and isolates the coin regions from the original image.

4. **Extracting Individual Coins (`extract_each_coin`)**  
   - Detects the minimum enclosing circle for each coin.  
   - Uses bitwise operations to isolate and crop each coin from the image.

5. **Counting Coins (`count_coin`)**  
   - Computes the total number of detected coins based on the segmented results.

### How to Run
1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Go to `part1` folder:

```bash
cd part1
```

3. Run the script using:

   ```bash
   python a.py <input_image_path> <output_dir>
   ```

4. The script will process the input image and create output directories.

### Output Folder Contents
The Ouput Folder contains :
- Edges_on_image.jpg – Image with detected coin edges outlined.
- coin_segmented.jpg – Segmented image showing extracted coins.
- coinX.jpg – Individual images of each detected coin.

___
## Panorama Stitching


### Steps Involved

1. **Keypoint Detection and Feature Extraction**(`sift_detect_descriptor`) 
   - Detects key points and computes descriptors using the SIFT algorithm.
   - Returns keypoints as a NumPy array and corresponding feature descriptors.

2. **Matching Keypoints Between Images** (`interest_point_matcher`)  
   - Matches keypoints between two images using BFMatcher and Lowe’s ratio test.

3. **Homography Estimation for Image Alignment** (`interest_point_matcher`) 
   - Computes a transformation matrix (homography) to align images based on matched key points.  
   - Uses RANSAC to eliminate outliers and improve alignment quality.

4. **Image Warping and Blending** (`stitch`)  
   - Warps one image to align with the next based on the computed transformation.  
   - Blends images smoothly to reduce visible seams.

5. **Black Border Cropping**   (`crop_black_region`)  
   - Removes unnecessary black regions caused by perspective transformation.  
   - Extracts only the region containing meaningful image content.

### How to Run
1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Go to `part2` folder:

```bash
cd part2
```

3. Run the script using:
   ```bash
    python b.py <input_directory> <output_directory>
   ```

### Output Folder Contents
- `stitched_X.jpg`: Visualization of matched key points between the panorama of images [1...X] and image X+1
- `panorama.jpg`: The final stitched panorama image.

___
Further details on the methodology, output images, and analysis are discussed in the [report](https://github.com/VR-2025/VR_Assignment1_AdityaPriyadarshi_IMT2022075/blob/main/report.pdf).
Github - [ap5967ap](https://github.com/VR-2025/VR_Assignment1_AdityaPriyadarshi_IMT2022075/)
