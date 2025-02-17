import cv2
import numpy as np
import os
import argparse

#Function to display an image for a given time in seconds
def show(img,time=3,msg="Image"):
    cv2.imshow(msg,img)
    cv2.waitKey(int(time*1000))
    cv2.destroyAllWindows()

#Function to preprocess the image by converting it to grayscale, resizing, and thresholding
def pre_process(path):
    image=cv2.imread(path)  #Read the image
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Convert to grayscale
    scale_factor=700/max(image.shape[:2]) #scale factor for a better logic
    image=cv2.resize(image,(0,0),fx=scale_factor,fy=scale_factor)
    gray=cv2.resize(gray,(0,0),fx=scale_factor,fy=scale_factor)
    blurred=cv2.GaussianBlur(gray,(5,5),0)  #Gaussian blur
    thresh=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) #adaptive thresholding edge detection
    return (image,thresh,scale_factor)

#Function to detect circular edges (coins) in the image
def edge(image,thresh,scale_factor,output_dir):
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #contours found
    circle_coins=[]
    for cnt in contours:
        perimeter=cv2.arcLength(cnt,True) #contour perimeter
        area=cv2.contourArea(cnt) #contour area
        if perimeter:
            circularity=4*np.pi*(area/(perimeter**2)) #circularity - so only circular coins will be detected  
            if 0.7 < circularity < 1.2 and area > 500*(scale_factor**2): # size conditions
                circle_coins.append(cnt)
    cv2.drawContours(image,circle_coins,-1,(0,255,0),2)
    cv2.imwrite(os.path.join(output_dir,f"Edges_on_image.jpg"),image)
    return circle_coins

# Function to segment out the coins from the image using a mask
def segment_coins(image,thresh,circle_coins,output_dir):
    mask=np.zeros_like(thresh) #empty mask
    cv2.drawContours(mask,circle_coins,-1,255,thickness=cv2.FILLED) #Draw filled contours of coins on mask
    segment=cv2.bitwise_and(image,image,mask=mask)
    bg=np.zeros_like(image) #black bg 
    bg[mask==255]=segment[mask==255] #overlay segmented coins on black bg
    cv2.imwrite(os.path.join(output_dir,f"coin_segmented.jpg"),bg)

# Function to extract and save each detected coin individually
def extract_each_coin(image,circle_coins,output_dir):
    segmented_coins=[]
    for cnt in circle_coins:
        (x,y),radius=cv2.minEnclosingCircle(cnt) #minimum enclosing circle
        center=(int(x),int(y)) #center of circle
        radius=int(radius) #radius
        mask=np.zeros_like(image,dtype=np.uint8)
        cv2.circle(mask,center,radius,(255,255,255),-1) 
        coin_segment=cv2.bitwise_and(image,mask) ## mask applied to extract the coin
        x1,y1,x2,y2=center[0]-radius,center[1]-radius,center[0]+radius,center[1]+radius #Crop area for circular coin region
        coin_segment=coin_segment[y1:y2,x1:x2] #coins area cropped 
        segmented_coins.append(coin_segment)
    for i,coin in enumerate(segmented_coins):
        #show(coin,2,f'{i+1}th segment coin')
        cv2.imwrite(os.path.join(output_dir,f"coin{i+1}.jpg"),coin)
    return segmented_coins

# Function to count the number of detected coins
def count_coin(segmented_coins):
    return len(segmented_coins) 

# pipeline for one image
def main(input_path,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    image,thresh,scale_factor=pre_process(input_path) #Preprocess
    circle_coins=edge(image,thresh,scale_factor,output_path) #detect circular edges - coins
    segment_coins(image,thresh,circle_coins,output_path) #segment detected coin
    segmented_coins=extract_each_coin(image,circle_coins,output_path) #extract each coin
    str="coins"
    if count_coin(segmented_coins)==1:str=str[:-1]
    print(f'There are {count_coin(segmented_coins)} {str} in the image.') 
    
parser=argparse.ArgumentParser(description="Process input coins.")
parser.add_argument("input_file",help="Name of the input image")
parser.add_argument("output_dir",help="Name of the output dir")
args=parser.parse_args()
main(args.input_file,args.output_dir)