import cv2
import numpy as np

# Initializing the VideoCapture Object to access webcam
cap = cv2.VideoCapture(0)

# Initially record the background
for i in range(30):
    ret,background = cap.read()

# Flip the background
background = np.flip(background,axis=1)

while(1):
    # Capture Frames
    ret, frame = cap.read()
    img = np.flip(frame,axis=1)

    # Convert the image to HSV color space for mask extraction
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # GaussianBlur to reduce Gaussian noise
    blurred = cv2.GaussianBlur(hsv, (35, 35),0)
    
    # Initializing the upper and lower hsv values of the color 'Orange' ( Cloak Color )
    lower = np.array([2,170,0])
    upper = np.array([28,255,255])
    mask = cv2.inRange(hsv,lower,upper)

    # Enhancing the Mask
    mask = cv2.erode(mask,np.ones((7,7),np.uint8))
    mask = cv2.dilate(mask,np.ones((19,19),np.uint8))
    
    # Replace the Mask's white region pixels with that of the background image pixels
    img[np.where(mask==255)] = background[np.where(mask==255)]

    # Show Image
    cv2.imshow('MAGIC',img)
    
    if cv2.waitKey(1) == 27:
        break
        
cv2.destroyAllWindows()
cap.release()
