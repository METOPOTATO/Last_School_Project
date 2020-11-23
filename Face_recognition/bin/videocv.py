import numpy as np
import cv2
from controller import Controller


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

con = Controller()
cap = cv2.VideoCapture(0)

_,embeddings,labels = con.load_data()

in_encoder = Normalizer(norm='l2')
out_encoder = LabelEncoder()
out_encoder.fit(labels)

X_transformed = in_encoder.transform(embeddings)
y_transformed = out_encoder.transform(labels)

model = SVC(kernel='linear', probability=True)
model.fit(X_transformed, y_transformed)


while(True):
    _, frame = cap.read()
    try:
        image = frame
        text = ''
        face,location = con.extract_face(image)
        x,y,w,h = location
        mask = con.detect_mask(face)

        if mask == 'without_mask':
            current_embedding = con.get_embedding(face)
            current_embedding = np.expand_dims(current_embedding,axis=0)
            current_embedding_tranformed = in_encoder.transform(current_embedding)
        
            predict = model.predict(current_embedding_tranformed)   
            predict_probability = model.predict_proba(current_embedding_tranformed)    
            idx = predict[0]
            probability = predict_probability[0,idx] * 100
            
            if probability  >50:
                text = str(out_encoder.inverse_transform(predict)[0])+'-' + str(probability)
            else:
                text = 'unknown'

            text = str(out_encoder.inverse_transform(predict)[0])+'-' + str(probability)
    
        else:
            text = mask
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow('frame',image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        continue
cap.release()
cv2.destroyAllWindows()