import cv2

img2 = cv2.imread('test.jpg')
cv2.imshow('es', img2)
img2 = cv2.putText(img2, 'Replaying...', 
                            (10,10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.2, 
                            (255, 255, 255), 
                            2, 
                            cv2.LINE_AA)

cv2.imshow('es', img2)
cv2.waitKey(-1)