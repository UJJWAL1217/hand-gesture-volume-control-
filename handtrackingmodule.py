import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode=False , maxHands=2 , detectionCon=0.5 , trackCon=0.5 ):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon = detectionCon
        self.trackCon=trackCon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon , self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findHands(self,img , draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
            #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:                       #drawing the single hands
            for handLms in self.results.multi_hand_landmarks:   #drawing the hands
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms , self.mpHands.HAND_CONNECTIONS)  # draw the connections on the hands

        return img

    def findPosition(self,img,handNo=0,draw= True ):
        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # giving index nummber to all the 21 points on land
                # print(id , lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # this will give the cx and cy position
                #print(id, cx, cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7 , (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector=handDetector()
    while True:
        success, img = cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])
        cTime = time.time()  # this will give us the current time
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()

