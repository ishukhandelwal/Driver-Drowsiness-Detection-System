from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image

import cv2
# Numpy for array related functions

import numpy as np
# Dlib for deep learning based Modules and face landmarkp detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
from sympy import bottom_up

def info_DDS():
    new_window = tk.Tk()
    new_window.geometry("400x400")
    new_window.resizable(False, False)
    text = Text(new_window)  
    text.insert(INSERT," Driver drowsiness detection")  
    text.insert(END, "This project is successfully compleated by Sanchit Agarkar,Ishu khandelwal,Ajinkya Mahure,Lakshya Pokharana,Harsh Makwana")  
    text.pack()  
    b1 = Button(new_window, text = "Exit",command = new_window.destroy,font=("Segoe UI", 16,'bold'))
    
    b1.place(x=150, y=350, width=75, height=30)
    label2 = Label(new_window, text ="Driver drowsiness detection")
    label2.pack(side = TOP, pady = 10)

def predictdrowsiness():

    cap = cv2.VideoCapture(0)

    #Initializing the face detector and landmark detector
    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #status marking for current state
    sleep = 0
    drowsy = 0
    active = 0
    status=""
    color=(0,0,0)

    def compute(ptA,ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def blinked(a,b,c,d,e,f):
        up = compute(b,d) + compute(c,e)
        down = compute(a,f)
        ratio = up/(2.0*down)

        #Checking if it is blinked
        if(ratio>0.25):
            return 2
        elif(ratio>0.21 and ratio<=0.25):
            return 1
        else:
            return 0


    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        face_frame = frame.copy()
        #detected face in faces array
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            #The numbers are actually the landmarks which will show eye
            left_blink = blinked(landmarks[36],landmarks[37], 
                landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42],landmarks[43], 
                landmarks[44], landmarks[47], landmarks[46], landmarks[45])
            
            #Now judge what to do for the eye blinks
            if(left_blink==0 or right_blink==0):
                sleep+=1
                drowsy=0
                active=0
                if(sleep>6):
                    status="SLEEPING !!!"
                    color = (255,0,0)

            elif(left_blink==1 or right_blink==1):
                sleep=0
                active=0
                drowsy+=1
                if(drowsy>6):
                    status="Drowsy !"
                    color = (0,0,255)

            else:
                drowsy=0
                sleep=0
                active+=1
                if(active>6):
                    status="Active :)"
                    color = (0,255,0)
            
            cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

            for n in range(0, 68):
                (x,y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Frame", frame)
        cv2.imshow("Result of detector", face_frame)
        key = cv2.waitKey(1)
        if key == 27:
            
            break

window =  Tk()
window.geometry("1000x600+120+100")
#window.resizable(True, True)
window.resizable(False, False)

label1 = Label(window, text="Driver drowsiness detection Prototype", font=("Comic Sans MS", 18, "bold"), bg="grey16", fg="white",
                       relief=SUNKEN)
label1.pack(side=TOP, fill=X)
#############################################################################################################################



# Create Canvas
canvas1 = Canvas(window, width = 200,
                 height = 200)

canvas1.pack(fill = "both", expand = True)

bg = ImageTk.PhotoImage(file="driver.jpg")

# Display image
canvas1.create_image(30, 0, image = bg, anchor = "nw")

#############################################################################################################################

statusbar = Label(window, text="By : Team Ishu ", font=("Comic Sans MS", 10, "bold"),
                  bg="grey16", fg="white", relief=SUNKEN, anchor=N)
statusbar.pack(side=BOTTOM, fill=X)
statusbar.pack(side=BOTTOM, fill=X)

btn1 = Button(window, text="Predict drowsiness",command= predictdrowsiness,font=("Segoe UI", 10,'bold'))
btn1.place(x=750, y=260, width=170, height=35)

btn2 = Button(window, text="About Us",command=info_DDS,font=("Segoe UI", 10,'bold')) 
btn2.place(x=750, y=325, width=170, height=35)

window.mainloop()