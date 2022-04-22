import cv2
import mediapipe as mp
import time
import numpy as np
import os
import easyocr

path = r'C:\Users\ADMIN\Documents\BTP-2022\BTP'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


## For webcam input:__init__
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
##For Video
#cap = cv2.VideoCapture("hands.mp4")
prevTime = 0
tipIds = [4, 8, 12, 16, 20]
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:             #Detection Sensitivity
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.

    image.flags.writeable = False
    results = hands.process(image)

    #print(results.multi_hand_landmarks)

    

    # Draw the hand annotations on the image.
    #image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #if results.multi_hand_landmarks:

      #for hand_landmarks in results.multi_hand_landmarks:
        #mp_drawing.draw_landmarks(
            #image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    xList = []
    yList = []
    bbox = []
    lmList = []
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            # print(id, cx, cy)
            lmList.append([id, cx, cy])
            #if results:
                    #cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)


        #xmin, xmax = min(xList), max(xList)
        #ymin, ymax = min(yList), max(yList)
        #bbox = xmin, ymin, xmax, ymax
        #if results  :
                #cv2.rectangle(image, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                #(0, 255, 0), 2)
        #print(lmList)


        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
     
        # Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
     
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

        if fingers[1] and fingers[2] == False:
          cv2.circle(image, (x1, y1), 10, (0,255,255), cv2.FILLED)
          #print("Drawing Mode")
          if xp == 0 and yp == 0:
              xp, yp = x1, y1
          cv2.line(image, (xp, yp), (x1, y1), (0,255,255), 10)
          cv2.line(imgCanvas, (xp, yp), (x1, y1), (0,255,255), 10)
          xp, yp = x1, y1      

        if fingers[0] == False:
            cv2.circle(image, (x1, y1), 10, (0,0,0), cv2.FILLED)
            if xp == 0 and yp == 0:
              xp, yp = x1, y1
            cv2.line(image, (xp, yp), (x1, y1), (0,0,0), 50)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), (0,0,0), 50)
            xp, yp = x1, y1

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    image = cv2.addWeighted(image,1,imgCanvas,1,0)
    cv2.imshow('Air-Writing', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()

cv2.imwrite(os.path.join(path, 'forOCR.jpg'), imgCanvas)

reader = easyocr.Reader(['en'],gpu = False)
image_path = r'C:\Users\ADMIN\Documents\BTP-2022\BTP\forOCR.jpg'
result = reader.readtext(image_path)

with open('text_result2.txt', mode ='w') as file:
    file.write(result[0][1])
    print('ready!')
