from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import datetime
import cv2
import os
import numpy as np
from controller import Controller

from tkinter import Tk,Frame, Button,Label,Entry,Text,StringVar

import pyttsx3

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

class MainPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller

        self.time_count = 0
        
        self.current_image = None  
        self.img_frame = None

        controller.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = Label(self)  
        self.panel.pack(padx=10, pady=10)
        self.panel1 = Label(self)  
        self.panel1.pack(padx=10, pady=10)
        self.con = Controller()

        self.employee_id = StringVar()
        lbl_employee_id = Label(self.panel1, text = 'ID', font = ('calibre',10,'bold')) 
        self.textField_id = Entry(self.panel1,textvariable=self.employee_id,font=('calibre',10,'bold'))
        self.employee_name = StringVar()
        lbl_employee_name = Label(self.panel1, text = 'Name', font = ('calibre',10,'bold')) 
        self.textField_name = Entry(self.panel1,textvariable=self.employee_name,font=('calibre',10,'bold'))

        self.textField_id.config(state='disabled')
        self.textField_name.config(state='disabled')

        self.load_data()
  
        self.CBB = ttk.Combobox(self.panel1,values=self.list_id,)
        self.CBB.current(0)
        self.CBB.bind('<<ComboboxSelected>>', self.cbb_onlected)
        self.CBB.grid(row=2)

        lbl_employee_id.grid(row=0,column=0)
        lbl_employee_name.grid(row=0,column=1)
        self.textField_id.grid(row=1,column=0)
        self.textField_name.grid(row=1,column=1)
        
        btn = Button(self, text="Save Data", command=self.save_embbeding)
        btn.pack()

        btnQuit = Button(self, text="Quit", command=self.controller.quit)
        btnQuit.pack()

        button = Button(self, text="Manage Data",command=self.close)
        button.pack()

    def run_video(self):
        self.vs = cv2.VideoCapture(0)

    def video_loop(self):
        
        ok, frame = self.vs.read()
        self.img_frame = frame
        if ok:  
            try:
                image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
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
                        name = str(self.out_encoder.inverse_transform(predict)[0])
                        text = name +'-' + str(probability) 
                        self.time_count +=1  
                        print(self.time_count)
                        if self.time_count == 6:
                            self.say(name)    
                    else:
                        self.time_count = 0
                        text = 'unknown'
                    
                else:
                    self.time_count = 0
                    text = mask
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
            except:
                pass
                
            self.current_image = Image.fromarray(image)  
            imgtk = ImageTk.PhotoImage(image=self.current_image)  
            self.panel.imgtk = imgtk 
            self.panel.config(image=imgtk)  
        self.controller.after(20, self.video_loop)

    def save_embbeding(self):
        image = self.img_frame
        face,_ = self.con.extract_face(image)
        embedding = self.con.get_embedding(face)
        str_embedding = np.array_str(embedding)
        id = self.CBB.get()
        data = (int(id),str_embedding)
        self.con.insert_image(data)
        self.CBB.configure(values=self.list_id)
        self.load_data()

    def destructor(self):
        self.controller.destroy()
        self.vs.release()  
        cv2.destroyAllWindows()  

    def close(self):
        self.vs.release()  
        self.img_frame = None
        self.panel = False
        cv2.destroyAllWindows() 
        self.controller.show_frame('PageManage')
        print(self.winfo_height())
        print(self.winfo_width())
        
    def say(self,text):
        engine = pyttsx3.init()
        engine.setProperty('volume',1.0)
        engine.setProperty('rate',100)
        engine.say('Hello Mr ' +text)
        engine.runAndWait()

    def load_data(self):
        
        id,embeddings,labels = self.con.load_data()
        ids = list(set(id))
        # print(len(ids))
        # result = self.con.get_one(int(ids[0]) )
        # employee_name = result['name']
        # num_embeded = 0
        # try:
        #     num_embeded = len(result['embeddings'])
        # except:
        #     num_embeded = 0

        # self.employee_id.set(result['employee_id'])
        # self.employee_name.set(employee_name)

        self.list_id = ids
        self.in_encoder = Normalizer(norm='l2')
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(labels)
        X_transformed = self.in_encoder.transform(embeddings)
        y_transformed = self.out_encoder.transform(labels)
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_transformed, y_transformed)

    def cbb_onlected(self,event=None):
        id = int(self.CBB.get() )
        result = self.con.get_one(id)
        employee_name = result['name']
        num_embeded = len(result['embeddings'])
        self.employee_id.set(id)
        self.employee_name.set(employee_name)
        # self.textField_name.config(text=employee_name)


class PageManage(Frame):
    def __init__(self,parent,controller):
        Frame.__init__(self, parent)
        self.controller = controller
        self.con = Controller()


        list_data = self.con.find_all()

        self.tree = ttk.Treeview(self,columns=('ID', 'Name'))

        iid = 0
        self.tree.heading('#0', text='ID')
        self.tree.heading('#1', text='Name')
        self.tree.heading('#2', text='Numbers images')
        self.tree.column("#0", width=270, minwidth=270, stretch=tk.NO)
        self.tree.column("#1", width=200, minwidth=200, stretch=tk.NO)
        self.tree.column("#2", width=200, minwidth=200, stretch=tk.NO)
        for data in list_data:
            id = data['employee_id']
            name = data['name']
            try:
                images = len(data['embeddings'])
            except:
                images = 0
            self.tree.insert('', index="end", iid =id, text=id, values=(name,images))
            iid+=1
        self.tree.pack(side=tk.TOP,fill=tk.X)

        self.panel = Label(self)  
        self.panel.pack(padx=10, pady=10)
        self.panel1 = Label(self)  
        self.panel1.pack(padx=10, pady=10)
        self.employee_id = StringVar()
        lbl_employee_id = Label(self.panel, text = 'ID', font = ('calibre',10,'bold')) 
        self.textField_id = Entry(self.panel,textvariable=self.employee_id,font=('calibre',10,'bold'))
        self.employee_name = StringVar()
        lbl_employee_name = Label(self.panel1, text = 'Name', font = ('calibre',10,'bold')) 
        self.textField_name = Entry(self.panel1,textvariable=self.employee_name,font=('calibre',10,'bold'))

        lbl_employee_id.grid(row=0,column=0)
        lbl_employee_name.grid(row=0,column=1)
        self.textField_id.grid(row=1,column=0)
        self.textField_name.grid(row=1,column=1)

        myframe = Frame(self)
        myframe.pack(fill=tk.BOTH, expand=True)

        self.btn_add = Button( text=" Add Data  ",command=self.add)
         
        self.btn_del = Button( text="Delete Data",command=self.delete)
        
        self.btn_switch = Button( text="View Camera", command=self.hide)
        

    def show(self):
        self.btn_add.place(x=450,y=500)
        self.btn_del.place(x=150,y=500)
        self.btn_switch.place(x=300,y=550)
        

    def hide(self):
        self.btn_add.place_forget()
        self.btn_del.place_forget()
        self.btn_switch.place_forget()
        self.controller.show_frame('MainPage')

    def add(self):
        id =int( self.employee_id.get())
        name = self.employee_name.get()
        embedd = None
        data = (id,name,embedd)
        self.tree.insert('', index="end", iid =id, text=id, values=(name,embedd))
        self.con.insert_person(data)

    def delete(self):
        row_id = int(self.tree.focus())
        print(row_id)
        self.tree.delete(row_id)
        self.con.delete_person(row_id)