import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#number signifies camera
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    # img = cv2.imread('image6.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # laplacian = cv2.Laplacian(laplacian, cv2.CV_64F)
    # laplacian = cv2.Laplacian(laplacian, cv2.CV_64F)


    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        roi_gray2 = gray[ey:ey+eh, ex:ex+ew]
        roi_color2 = img[ey:ey+eh, ex:ex+ew]
        circles = cv2.HoughCircles(roi_gray2,cv2.HOUGH_GRADIENT,1,200,param1=200,param2=20,minRadius=0,maxRadius=0)
        try:
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(roi_color2,(i[0],i[1]),i[2],(255,255,255),2)
                print("drawing circle")
                # draw the center of the circle
                cv2.circle(roi_color2,(i[0],i[1]),3,(255,255,255),3)
        except Exception as e:
            pass
    cv2.imshow('img',img)
    # cv2.imshow('lap', laplacian)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()