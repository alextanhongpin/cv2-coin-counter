import numpy as np
import cv2


# Load an color image in grayscale
img = cv2.imread('coins.jpg',0)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('gray.png',gray)
gray_blur = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imwrite('gray_blur.png',gray_blur)

thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
cv2.imwrite('adaptive_threshold_image.png', thresh)

# Threshold occ


kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
cv2.imwrite('morphology_closing.png', closing)

cont_img = closing.copy()
contours, hierachy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
  # area = cv2.contourArea(cnt)
  # if area < 2000 or area > 4000:
  #   continue
  # if len(cnt) < 5:
  #   continue
  ellipse = cv2.fitEllipse(cnt)
  cv2.ellipse(img, ellipse, (0, 255, 0), 2)

cv2.imwrite('final.png', img)
def show_image (src, title = "image"):
  cv2.imshow(title, src)
  cv2.waitKey(0)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    print "end"
  cv2.destroyAllWindows()


show_image(thresh)
# Watershed