import numpy as np
import imutils
import cv2
import os
from tqdm import tqdm
import argparse

# Function to detect keypoints and compute descriptors using SIFT
def sift_detect_descriptor(image): 
    descriptor=cv2.SIFT_create() #SIFT
    kps,features=descriptor.detectAndCompute(image,None)
    kps=np.float32([kp.pt for kp in kps]) #keypoints to float32 array
    return (kps,features)

# Function to match keypoints between two images using BFMatcher and ratio test
def interest_point_matcher(interestA,interestB,xA,xB,ratio,re_proj):
    matcher=cv2.BFMatcher()
    rawMatches=matcher.knnMatch(xA,xB,2) #KNN matching
    matches=[]
    for m in rawMatches:
        if len(m) == 2 and m[0].distance<m[1].distance*ratio: #Lowe's ratio test
            matches.append((m[0].trainIdx,m[0].queryIdx))
    if len(matches)>4: #Ensure enough matches exist to compute homography
        ptsA=np.float32([interestA[i] for (_,i) in matches])
        ptsB=np.float32([interestB[i] for (i,_) in matches])
        H,status=cv2.findHomography(ptsA,ptsB,cv2.RANSAC,re_proj) #homography matrix
        return (matches,H,status)
    return None

#Function to visualize matches between keypoints of two images
def viz_matches(imageA,imageB,interestA,interestB,matches,status):
    hA,wA=imageA.shape[:2]
    hB,wB=imageB.shape[:2]
    viz=np.zeros((max(hA,hB),wA+wB,3),dtype="uint8")# Create an empty canvas for visualization
    viz[0:hA,0:wA]=imageA
    viz[0:hB,wA:]=imageB
    for((trainIdx,queryIdx),s) in zip(matches,status):
        if s == 1:# If the match is good, draw a line
            ptA=(int(interestA[queryIdx][0]),int(interestA[queryIdx][1]))
            ptB=(int(interestB[trainIdx][0])+wA,int(interestB[trainIdx][1]))
            cv2.line(viz,ptA,ptB,(0,255,0),1)
    return viz

# Function to crop the extra black regions from the stitched panorama
def crop_black_region(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY) # Convert to binary image
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h=cv2.boundingRect(contours[0]) # Find bounding box of the content area
        return image[y:y+h-1,x:x+w-1] # the cropped img
    return image

# Function to perform image stitching
def stitch(images,ratio=0.75,re_proj=5.0,show_overlay=False):
    imageB,imageA=images
    interestA,xA=sift_detect_descriptor(imageA)
    interestB,xB=sift_detect_descriptor(imageB)
    M=interest_point_matcher(interestA,interestB,xA,xB,ratio,re_proj)
    if M is None:
        print("Not enough matches found.")
        return None
    matches,H,status=M
    pano_img=cv2.warpPerspective(imageA,H,(imageA.shape[1]+imageB.shape[1],imageA.shape[0]))# Warp imageA
    pano_img[0:imageB.shape[0],0:imageB.shape[1]]=imageB # Overlay imageB on top
    pano_img = crop_black_region(pano_img)
    if show_overlay:
        visualization=viz_matches(imageA,imageB,interestA,interestB,matches,status)
        return (pano_img,visualization)
    return pano_img

#Function to display an image for a given time in seconds
def show(img,time=3,msg="Image"):
    cv2.imshow(msg,img)
    cv2.waitKey(int(time*1000))
    cv2.destroyAllWindows()

def main(input_dir,output_dir):
    img_path=[]
    for i in os.listdir(input_dir):
        img_path.append(os.path.join(input_dir,i))
    assert len(img_path)>0,"No image found in input folder"
    img_path.sort(key=lambda x:int(os.path.splitext(os.path.basename(x))[0]))
    if os.path.exists(output_dir):
        pass
    else:
        os.makedirs(output_dir)

    left_img=cv2.imread(img_path[0])
    left_img=imutils.resize(left_img,width=600)
    for i in tqdm(range(1,len(img_path))):
        right_img=cv2.imread(img_path[i])
        right_img=imutils.resize(right_img,width=600)
        pano_img=stitch([left_img,right_img],show_overlay=True)
        if pano_img is not None:
            left_img,viz=pano_img
            cv2.imwrite(os.path.join(output_dir,f"stiched_{i}.jpg"),viz)
    cv2.imwrite(os.path.join(output_dir,"panorama.jpg"),left_img)

parser=argparse.ArgumentParser(description="Process input panorama.")
parser.add_argument("input_dir",help="Name of the input directory")
parser.add_argument("output_dir",help="Name of the output directory")
args=parser.parse_args()
main(args.input_dir,args.output_dir)
