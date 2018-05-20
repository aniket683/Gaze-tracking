import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.01,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
# ret, old_frame = cap.read()


while True:
    
    ret, old_frame = cap.read()
    gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces)>0:
        x, y, w, h = faces[0]
        # print (x, y, w, h)
        my_face = old_frame[x+(w//3):x+(2*w//3), y+(h//3):y+(2*h//3)]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.cvtColor(my_face, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(face_gray, mask = None, **feature_params)
        try:
            for a in range(len(p0)):
                p0[a][0][0] = p0[a][0][0] + x + w//3
                p0[a][0][1] = p0[a][0][1] + y + h//3
        except:
            pass
        break

while True:
    curr = time.time()
    ret, old_frame = cap.read()
    old_frame = cv2.flip(old_frame, 1)
    gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces)>0:
        x, y, w, h = faces[0]
        # print (x, y, w, h)
        my_face = old_frame[x+(w//3):x+(2*w//3), y+(h//3):y+(2*h//3)]
        cv2.rectangle(old_frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = old_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.cvtColor(my_face, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(face_gray, mask = None, **feature_params)
        try:
            for a in range(len(p0)):
                p0[a][0][0] = p0[a][0][0] + x + w//3
                p0[a][0][1] = p0[a][0][1] + y + h//3
        except:
            pass
        cv2.imshow('frame', old_frame)

    else:
        
        ret,frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        x_mean = 0
        y_mean = 0
        total = 0
        for a in range(len(p1)):
            x_mean+=p1[a][0][0]
            y_mean+=p1[a][0][1]
            total = total+1
        x_mean = int(x_mean//total)
        y_mean = int(y_mean//total)
        cv2.rectangle(frame, (x_mean-w//2,y_mean-h//2), (x_mean+w//2, y_mean+h//2), (255,0,0), 2)
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            frame = cv2.circle(frame,(a,b),2,color[i].tolist(),-1)
        # roi_gray = frame_gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        old_frame = frame
        cv2.imshow('frame',old_frame)
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    print("fps:", 1/(time.time()-curr))
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    
cv2.destroyAllWindows()
cap.release()