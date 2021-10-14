import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

#making the frame rates
pTime=0
cTime=0

while True :
    success , img= cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:                       #drawing the single hands
        for handLms in results.multi_hand_landmarks:   #drawing the hands
            for id,lm in enumerate(handLms.landmark):  # giving index nummber to all the 21 points on land
               # print(id , lm)
                h, w, c =img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)  # this will give the cx and cy position
                print(id , cx , cy)
                if id ==0:  # change any value of 0 to 21 to detect the location of the points on the hands
                    cv2.circle(img , (cx , cy) , 25, (255 , 0, 255), cv2.FILLED) # detecting the land mark 0 ;
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)  # draw the connections on the hands

    cTime = time.time()  # this will give us the current time
    fps= 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)) , (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


    cv2.imshow("image", img)
    cv2.waitKey(1)