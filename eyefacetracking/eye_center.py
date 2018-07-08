import numpy as np
import cv2
import time
import urllib.request

left_counter = 0
right_counter = 0

# def thresholding( value ):  # function to threshold and give either left or right
# 	global left_counter
# 	global right_counter
# 	th_value=5
# 	if (value<=54):   #check the parameter is less than equal or greater than range to 
# 		left_counter=left_counter+1		#increment left counter 

# 		if (left_counter>th_value):  #if left counter is greater than threshold value 
# 			# print ('RIGHT')  #the eye is left
# 			left_counter=0   #reset the counter

# 	elif(value>=54):  # same procedure for right eye
# 		right_counter=right_counter+1

# 		if(right_counter>th_value):
# 			# print ('LEFT')
# 			right_counter=0


cap = cv2.VideoCapture(1)
while True:
	# with urllib.request.urlopen("http://10.184.57.17:4747/cam/1/frame.jpg") as url:
	# 	s = url.read()
	# 	# print (s)
	# imgNp=np.array(bytearray(s),dtype=np.uint8)
	# frame=cv2.imdecode(imgNp,-1)
	curr = time.time()
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	# frame = cv2.imread('image6.jpg')
	# ret = True
	if ret==True:
		#detect face
		frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		faces = cv2.CascadeClassifier('haarcascade_eye.xml')
		detected = faces.detectMultiScale(frame, 1.3, 5)

		#faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		#detected2 = faces.detectMultiScale(frameDBW, 1.3, 5)
		
		pupilFrame = frame
		pupilO = pupilFrame
		windowClose = np.ones((5,5),np.uint8)
		windowOpen = np.ones((1,1),np.uint8)
		windowErode = np.ones((2,2),np.uint8)

		#draw square
		if (len(detected)==2):
			x,y,w,h = detected[0]
			pupilFrame = cv2.equalizeHist(frame[(int(y)+int(h*.25)):(y+h), x:(x+w)])
			cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255),1) 
			ex,ey,ew,eh = detected[1]
			pupilFrame2 = cv2.equalizeHist(frame[(int(ey)+int(eh*.25)):(ey+eh), ex:(ex+ew)])
			cv2.rectangle(frame, (ex,ey), ((ex+ew),(ey+eh)), (0,0,255),1) 			
			pupil123 = pupilFrame
			
			ret, pupilFrame = cv2.threshold(pupilFrame,20,255,cv2.THRESH_BINARY_INV)        
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)

			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_DILATE, windowErode)
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
			# pupilFrame = cv2.dilate(pupilFrame, windowErode, iterations=1)
			ret, pupilFrame = cv2.threshold(pupilFrame,20,255,cv2.THRESH_BINARY_INV)        


			cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #set grid size
			clahe = cl1.apply(pupilFrame)  #clahe
			blur = cv2.medianBlur(clahe, 7)  #median blur
			laplacian = cv2.Laplacian(blur, cv2.CV_64F)

			circles = cv2.HoughCircles(blur ,cv2.HOUGH_GRADIENT,1,200,param1=200,param2=1,minRadius=7,maxRadius=21) #houghcircles
			
			if circles is not None: #if atleast 1 is detected
				circles = np.round(circles[0, :]).astype("int") #change float to integer
				# print ('integer',circles)
				for (ax,ay,r) in circles:
					# cv2.circle(pupilFrame, (x, y), r, (0, 0, 0), 2)
					cv2.rectangle(pupilFrame, (ax - 2, ay - 2), (ax + 2, ay + 2), (255, 255, 255), -1)
					cv2.circle(frame, (x+ax, y+int(h*.25)+ay), 3, (255, 255, 255), 2)
					#set thresholds
					# thresholding(x)

		cv2.imshow('frame', frame)
			# cv2.imshow('pupilframe', pupilFrame)
			# cv2.imshow('pupilframe2', blur)
			# cv2.imshow('pupilO',laplacian)
		print("fps:", 1/(time.time()-curr))



		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		

cv2.destroyAllWindows()