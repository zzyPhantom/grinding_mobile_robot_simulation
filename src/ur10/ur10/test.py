#!/usr/bin/env python

import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import sys
 
 
desired_aruco_dictionary = "DICT_7X7_1000"
 
# The different ArUco dictionaries built into the OpenCV library. 
ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

camera_matrix = np.array([[1000, 0, 320],
                          [0, 1000, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)

def draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    """ 在图像上绘制坐标轴 """
    # 生成坐标轴的三维点
    axis = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    
    # 投影坐标轴点到图像平面
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # 绘制坐标轴
    image = cv2.line(image, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # X轴 - 红色
    image = cv2.line(image, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)  # Y轴 - 绿色
    image = cv2.line(image, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)  # Z轴 - 蓝色
    
    return image

def main():
    """
    Main method of the program.
    """

    this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()

    tag = np.zeros((300, 300, 1), dtype="uint8")

    cv2.aruco.drawMarker(this_aruco_dictionary, 77, 300, tag, 1)

    tag = cv2.imread('tag.png')
  
    # Detect ArUco markers in the video frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
      tag, this_aruco_dictionary, parameters=this_aruco_parameters)
       
    # Check that at least one ArUco marker was detected
    # if len(corners) > 0:
    #     # Flatten the ArUco IDs list
    #     ids = ids.flatten()
        
    #     # Loop over the detected ArUco corners
    #     for (marker_corner, marker_id) in zip(corners, ids):
        
    #         # Extract the marker corners
    #         corners = marker_corner.reshape((4, 2))
    #         (top_left, top_right, bottom_right, bottom_left) = corners
            
    #         # Convert the (x,y) coordinate pairs to integers
    #         top_right = (int(top_right[0]), int(top_right[1]))
    #         bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
    #         bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
    #         top_left = (int(top_left[0]), int(top_left[1]))
            
    #         # Draw the bounding box of the ArUco detection
    #         cv2.line(tag, top_left, top_right, (0, 255, 0), 2)
    #         cv2.line(tag, top_right, bottom_right, (0, 255, 0), 2)
    #         cv2.line(tag, bottom_right, bottom_left, (0, 255, 0), 2)
    #         cv2.line(tag, bottom_left, top_left, (0, 255, 0), 2)
            
    #         # Calculate and draw the center of the ArUco marker
    #         center_x = int((top_left[0] + bottom_right[0]) / 2.0)
    #         center_y = int((top_left[1] + bottom_right[1]) / 2.0)
    #         cv2.circle(tag, (center_x, center_y), 4, (0, 0, 255), -1)
            
    #         # Draw the ArUco marker ID on the video frame
    #         # The ID is always located at the top_left of the ArUco marker
    #         cv2.putText(tag, str(marker_id), 
    #         (top_left[0], top_left[1] - 15),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5, (0, 255, 0), 2)
        
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

    for i in range(len(ids)):
        cv2.aruco.drawDetectedMarkers(tag, corners, ids)
        tag = draw_axes(tag, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
  
    # Display the resulting frame
    cv2.imshow('frame',tag)
          
    # If "q" is pressed on the keyboard, 
    # exit this loop
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
if __name__ == '__main__':
  main()