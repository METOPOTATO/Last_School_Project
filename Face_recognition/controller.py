import numpy as np 
import keras
from keras.models import load_model
import cv2 
import sys 

from model import DB

class Controller:
    def __init__(self):
        self.model = load_model('mylib/facenet_keras.h5')
        self.cascade = cv2.CascadeClassifier('mylib/haarcascade_frontalface_default.xml')
        self.mask_model = load_model('mylib/facemask_keras.h5')
        self.db = DB()

    def extract_face(self,img,size = (160,160)):
        try:
            faces = self.cascade.detectMultiScale(img, 1.05, 5)
            x,y,w,h = faces[0]
            location = x,y,w,h
            face =img[y:y+h,x:x+w]
            face = cv2.resize(face,size)
            
        except:
            return None
        return face, location

    def get_embedding(self,face):
        face = face.astype('float32')
        mean,std =  face.mean(), face.std()
        standardized = (face - mean) / std
        samples = np.expand_dims(standardized,axis=0)
        predict = self.model.predict(samples)[0]
        return predict

    def load_data(self):
        id = []
        embeddings = []
        labels = []

        results = self.db.find_all()
        for result in results:
            for embedding in result['embeddings']:
                e = np.fromstring(embedding[1:-1],sep=' ')
                
                id.append(result['employee_id'])
                labels.append(result['name'])
                embeddings.append(e)

        id  = np.asarray(id) 
        labels = np.asarray(labels)
        embeddings = np.array(embeddings)
        return id,embeddings,labels

    def detect_mask(self,face, size = (224,224)):
        face = cv2.resize(face,size)
        face = np.expand_dims(face,axis=0)
        result = self.mask_model.predict(face)[0]
        y_int = result.argmax()
        mask = self.convert_name(y_int)
        return mask

    def insert_person(self,data):
        return self.db.insert_person(data)

    def delete_person(self,data):
        return self.db.delete_person(data)  

    def update_person(self,data):
        return self.db.update_person(data) 
    
    def find_all(self,data):
        return self.db.find_all(data)  
    
    def insert_image(self,data):
        return self.db.insert_image(data)

    def convert_name(self,y):
        if y==0:
            return "with_mask"
        else:
            return "without_mask"

if __name__ == "__main__":
    controller = Controller()

    img = cv2.imread('imgs.png')
    face,_ = controller.extract_face(img)
    embedding = controller.get_embedding(face)
    str_embedding = np.array_str(embedding)

    data = (2,'bim',[str_embedding])
   
    controller.insert_person(data)

    # id,embeddings,labels = controller.load_data()
    # for e in labels:
    #     print(e)
    #     print(e.shape)

    # print(labels.shape)