import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import csv
import os

class FacialExpressionDatasetExpander:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_snapshot = tk.Button(window, text="Capture", width=10, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.mode_var = tk.StringVar(value="training")
        self.mode_radio_training = tk.Radiobutton(window, text="Training", variable=self.mode_var, value="training")
        self.mode_radio_testing = tk.Radiobutton(window, text="Testing", variable=self.mode_var, value="testing")
        self.mode_radio_training.pack(anchor=tk.W)
        self.mode_radio_testing.pack(anchor=tk.W)

        self.emotion_var = tk.StringVar(value="neutral")
        self.emotion_menu = ttk.Combobox(window, textvariable=self.emotion_var, 
                                         values=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'gay'])
        self.emotion_menu.pack()

        self.delay = 15

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.update()

        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                
                margin = int(0.5 * w)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(frame.shape[1] - x, w + 2 * margin)
                h = min(frame.shape[0] - y, h + 2 * margin)
                
                face = frame[y:y+h, x:x+w]
                face_gray = gray[y:y+h, x:x+w]
                
                augmented_faces = self.augment_image(face_gray)
                self.save_augmented_faces(augmented_faces)
                self.show_augmented_faces(augmented_faces)

    def augment_image(self, image):
        augmented = []
        
        # Original image
        augmented.append(cv2.resize(image, (48, 48)))
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented.append(cv2.resize(flipped, (48, 48)))
        
        # Adjust contrast
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_img)
        contrasted = enhancer.enhance(1.5)  # Increase contrast by 50%
        augmented.append(cv2.resize(np.array(contrasted), (48, 48)))
        
        # Rotate
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)  # 15 degree rotation
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented.append(cv2.resize(rotated, (48, 48)))
        
        return augmented

    def save_augmented_faces(self, faces):
        emotion_dict = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6, 'gay': 7}
        emotion = emotion_dict[self.emotion_var.get()]
        usage = self.mode_var.get()
        
        with open('dataset.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for face in faces:
                pixels = ' '.join(map(str, face.flatten().tolist()))
                writer.writerow([emotion, pixels, usage])
        
        print(f"Saved {len(faces)} augmented {self.emotion_var.get()} faces in {usage} mode")

    def show_augmented_faces(self, faces):
        face_window = tk.Toplevel(self.window)
        face_window.title("Augmented Faces")
        
        for i, face in enumerate(faces):
            # Resize the image to 200x200 for display
            face_resized = cv2.resize(face, (200, 200), interpolation=cv2.INTER_NEAREST)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_photo = ImageTk.PhotoImage(face_pil)
            
            label = tk.Label(face_window, image=face_photo)
            label.image = face_photo  # Keep a reference
            label.grid(row=i//2, column=i%2, padx=10, pady=10)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                margin = int(0.5 * w)
                cv2.rectangle(frame, 
                              (max(0, x - margin), max(0, y - margin)), 
                              (min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)), 
                              (255, 0, 0), 2)
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

if __name__ == "__main__":
    FacialExpressionDatasetExpander(tk.Tk(), "Facial Expression Dataset Expander")