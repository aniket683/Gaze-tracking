import numpy as np
import cv2
import time
import urllib.request
from collections import deque
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
def average(q):
	ax, ay =  0, 0
	for x, y in q:
		ax+=x
		ay+=y
	return [ax//len(q), ay//len(q)]

def main():
	cap = cv2.VideoCapture(0)
	q = deque()
	l = []
	avg_x = 0
	avg_y = 0
	avg = [0, 0, 0]
	while True:
		curr = time.time()
		ret, frame = cap.read()
		frame = cv2.flip(frame, 1)
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
				ex,ey,ew,eh = detected[1]
				
				if x<ex:
					ex, ey, ew, eh = x, y, w, h
				cv2.rectangle(frame, (ex,ey), ((ex+ew),(ey+eh)), (0,0,255),1)
				pupilFrame = cv2.equalizeHist(frame[(int(ey)+int(eh*.25)):(ey+eh), ex:(ex+ew)])
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

				circles = cv2.HoughCircles(blur ,cv2.HOUGH_GRADIENT,1,200,param1=200,param2=1,minRadius=7,maxRadius=21) #houghcircles
				if circles is not None: #if atleast 1 is detected
					
					circles = np.round(circles[0, :]).astype("int") #change float to integer
					
					ax, ay, r = circles[0]
					# print(ax, ay, r)
					# print ('integer',circles)
					# for (ax,ay,r) in circles:
						# cv2.circle(pupilFrame, (x, y), r, (0, 0, 0), 2)
					cv2.rectangle(pupilFrame, (ax - 2, ay - 2), (ax + 2, ay + 2), (255, 255, 255), -1)
					q.append([ex+ax, ey+ay+int(eh*0.25)])
					if(len(q)>5):
						q.popleft()
					ex, ey = average(q)
					
					# print(type(ex), type(ey))
					cv2.circle(frame, (ex, ey), 3, (255, 255, 255), 2)
					if(len(l)<30):
						l.append([ex, ey])
					if (len(l)==30):
						l.append([ex, ey])
						avg_x, avg_y=average(l)
					if(len(l)>30):
						diff = [ex-avg_x, ey-avg_y]
						print(diff)
						if(abs(diff[0])+abs(diff[1])>100):
							l=[]
						cv2.line(frame, (ex, ey), (avg_x, avg_y), (255, 255, 255), 2)
				cv2.imshow('frame', frame)
				cv2.imshow('pupilframe', pupilFrame)
				cv2.imshow('pupilframe2', blur)
				# cv2.imshow('pupilO',laplacian)
			# print("fps:", 1/(time.time()-curr))



			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()