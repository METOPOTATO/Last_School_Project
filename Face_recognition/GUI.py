from PIL import Image, ImageTk
import tkinter as tk
import datetime
import cv2
import os
import numpy as np
from controller import Controller


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

class Application:
    def __init__(self):
        self.num = 17
        self.vs = cv2.VideoCapture(0) 
        self.current_image = None  
        self.img_frame = None

        self.root = tk.Tk() 
        self.root.title("PyImageSearch PhotoBooth")  

        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)  
        self.panel.pack(padx=10, pady=10)

        self.con = Controller()

        btn = tk.Button(self.root, text="Snapshot!", command=self.save_data)
        btn.pack(fill="both", expand=True, padx=10, pady=10)

        _,embeddings,labels = self.con.load_data()
        self.in_encoder = Normalizer(norm='l2')
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(labels)
        X_transformed = self.in_encoder.transform(embeddings)
        y_transformed = self.out_encoder.transform(labels)
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_transformed, y_transformed)

        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        self.img_frame = frame
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if ok:  
            try:
                text = ''
                face,location = self.con.extract_face(image)
                x,y,w,h = location
                mask = self.con.detect_mask(face)
                if mask == 'without_mask':
                    current_embedding = self.con.get_embedding(face)
                    current_embedding = np.expand_dims(current_embedding,axis=0)
                    current_embedding_tranformed = self.in_encoder.transform(current_embedding)
                
                    predict = self.model.predict(current_embedding_tranformed)   
                    predict_probability = self.model.predict_proba(current_embedding_tranformed)    
                    idx = predict[0]
                    probability = predict_probability[0,idx] * 100
                    
                    if probability  >50:
                        text = str(self.out_encoder.inverse_transform(predict)[0])+'-' + str(probability)
                    else:
                        text = 'unknown'
                    text = str(self.out_encoder.inverse_transform(predict)[0])+'-' + str(probability)
                else:
                    text = mask
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
            except:
                print('error')
                
            self.current_image = Image.fromarray(image)  
            imgtk = ImageTk.PhotoImage(image=self.current_image)  
            self.panel.imgtk = imgtk 
            self.panel.config(image=imgtk)  

    
        self.root.after(30, self.video_loop)  

    def take_snapshot(self):
        image = cv2.cvtColor(self.img_frame,cv2.COLOR_BGR2RGB)
        cv2.imwrite('imgs.png',image)

    def save_data(self):
        image = self.img_frame
        face,_ = self.con.extract_face(image)
        embedding = self.con.get_embedding(face)
        str_embedding = np.array_str(embedding)
        data = (str(self.num),'bim',[str_embedding])
        con.insert_person(data)
        self.num+=1

    def destructor(self):
        self.root.destroy()
        self.vs.release()  
        cv2.destroyAllWindows()  


if __name__ == "__main__":
    pba = Application()
    pba.root.mainloop()