import numpy as np
import cv2
from matplotlib import pyplot as plt

def show_image (src, title = "image"):
  cv2.imshow(title, src)
  cv2.waitKey(0)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    print "end"
  cv2.destroyAllWindows()

img = cv2.imread('water_coins.jpg')


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
# Only available for OpenCV 3.0
# ret, markers = cv2.connectedComponents(sure_fg)


cont_img = sure_fg.copy()
contours, hierachy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
  # area = cv2.contourArea(cnt)
  # if area < 2000 or area > 4000:
  #   continue
  # if len(cnt) < 5:
  #   continue
  ellipse = cv2.fitEllipse(cnt)
  cv2.ellipse(img, ellipse, (0, 255, 0), 2)

# Add one to all labels so that sure background is not 0, but 1
# markers = markers+1

# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0

# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]

cv2.imwrite('watershed_output.png', img)

show_image(img)