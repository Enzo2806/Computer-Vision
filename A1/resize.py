# IMport IMG1.jpg and resize it to w1024xh768 pixels and resave it as IMG1_resized.jpg
import cv2

img = cv2.imread('IMG2.jpg')
resized = cv2.resize(img, (768, 1024))
cv2.imwrite('IMG2_resized.jpg', resized)
