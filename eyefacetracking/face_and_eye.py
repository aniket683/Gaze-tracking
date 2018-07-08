import cv2
import urllib.request
import numpy as np 
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

url = "http://10.184.50.206:4747/video"
cap = cv2.VideoCapture(0)

while True:
	with urllib.request.urlopen("http://10.184.57.17:4747/cam/1/frame.jpg") as url:
		s = url.read()
		# print (s)
	imgNp=np.array(bytearray(s),dtype=np.uint8)
	img=cv2.imdecode(imgNp,-1)
	# ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print (gray)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print(len(faces))
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

	cv2.imshow('img', img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()