import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology

import matplotlib as plt



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=200, thresh_max=250):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to uint8, then apply threshold to get binary image

        ## TODO

        kernel = (5, 5)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(gray,kernel,0)
        gX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
        gY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
        combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
        abs_grad = cv2.convertScaleAbs(combined) # back to uint8
        # apply threshold
        binary_output = cv2.inRange(abs_grad, thresh_min, thresh_max)

        ####
        
        # cv2.imshow('image', img)
        # cv2.imshow('binary', binary_output)
        # cv2.waitKey(0)
        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        # h_thresh=(100, 255)
        # s_thresh=(100, 255)
        # v_thresh=(100, 255)

        h_thresh = 0 # for now
        HLS_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        
        line_output = cv2.inRange(HLS_img, np.array([0, 150, 0]), np.array([255, 255, 255])) # works for getting lines
        yellow_output = cv2.inRange(HLS_img, np.array([20, 50, 0]), np.array([30, 255, 255]))

        # binary_output = np.array([])

        result = cv2.bitwise_or(line_output, yellow_output)
        
        # cv2.imshow('yellow_output', result)
        # cv2.imshow('line_output', img)

        # cv2.waitKey(0)

        return result


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO

        ColorOutput = self.color_thresh(img)
        # SobelOutput = self.gradient_thresh(img)

        ####

        # binaryImage = np.zeros_like(SobelOutput)
        #binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1

        binaryImage = cv2.bitwise_or(ColorOutput, ColorOutput)

        # Remove noise from binary image
        # binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=10,connectivity=2)

        # binaryImage = np.uint8(binaryImage) * 255
        
        # cv2.imshow("before remove obj", binaryImage)
        # cv2.waitKey(0)
        

        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO
        # play with these
        maxWidth = 500
        maxHeight = 600

        # Source points (coordinates in the input image) definitely adjust these to get the right valuu
        # when testing
        # trapezoid shape for original
        pts1 = np.float32([[0, 260],  # Top-left
                        [640, 260],   # Top-right
                        [0, 400],     # Bottom-left
                        [640, 400]])  # Bottom-right

        # Destination points (coordinates in the output/bird's-eye view image)
        # rectangle shape for warped
        pts2 = np.float32([[100, 0], # Top-left
                        [300, 0],    # Top-right
                        [100, 600],  # Bottom-left
                        [300, 600]]) # Bottom-right

        
        # Apply Perspective Transform Algorithm
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        Minv = np.linalg.inv(M)
        ####

        # cv2.imshow("warped", warped_img)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        return warped_img, M, Minv


    def detection(self, img):
        # cv2.imshow("original", img)
        # cv2.waitKey(0)
        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
