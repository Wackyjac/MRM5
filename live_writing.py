import cv2
import numpy as np
import mnistinference
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


loadFromSys = True

if loadFromSys:
	hsv_value = np.load('hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 12800)
cap.set(4,720)

kernel = np.ones((5, 5), np.int8)

canvas = np.zeros((720, 1280, 3))

x1 = 0
y1 = 0

noise_thresh = 600

while True:
	_, frame = cap.read()
	#frame = cv2.flip(frame, 1)

	if canvas is not None:
		canvas = np.zeros_like(frame)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if loadFromSys:
		lower_range = hsv_value[0]
		upper_range = hsv_value[1]
	
	mask = cv2.inRange(hsv, lower_range, upper_range)
	mask = cv2.erode(mask, kernel, iterations = 1)
	mask = cv2.dilate(mask, kernel, iterations = 3)
	
	contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if contours  and cv2.contourArea(max(contours, key = cv2.contourArea)) > noise_thresh:
		c = max(contours, key = cv2.contourArea)
		x2, y2, w, h = cv2.boundingRect(c)
		padding = 30
		x2 -= padding
		y2 -= padding
		w += 2 * padding
		h += 2 * padding
		x2 = max(0, x2)
		y2=max(0,y2)

		if x1 == 0 and y1 == 0:
			x1,y1 = x2,y2
		else:
			canvas = cv2.rectangle(canvas, (x2,y2), (x2+w,y2+h), [0,255,0], 4)

		x1,y1 = x2,y2
		frame = cv2.add(canvas, frame)

		stacked = np.hstack((canvas, frame))
		cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx = 0.6, fy = 0.6))
		bb=frame[y2:y2+h,x2:x2+w]
		bbox = cv2.inRange(bb, lower_range, upper_range)
		cv2.imshow(" ",bbox)
		
	
	else:
		x1,y1 = 0, 0
	

	
	if cv2.waitKey(1) == 10:
		break
	

	#Clear the canvas when 'c' is pressed
	if cv2.waitKey(1) & 0xFF == ord('c'):
		canvas = None

	if cv2.waitKey(1) & 0xFF == ord('p'):
		bbox_tensor=cv2.resize(bbox, (28,28))
		plt.imshow(bbox_tensor,cmap='gray')
		plt.show()
		frame_tensor = torch.from_numpy(bbox_tensor)
		gray_tensor=frame_tensor.numpy()
		plt.imshow(gray_tensor,cmap='gray')
		plt.show()
		frame_tensor=torch.Tensor.view(frame_tensor, [1,1,28,28])
		mnistinference.testlive(frame_tensor)


cv2.destroyAllWindows()
cap.release()