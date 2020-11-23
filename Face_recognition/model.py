from pymongo import MongoClient 
import pymongo
import numpy as np

class DB:
    def __init__(self):
        client = MongoClient('mongodb://localhost:27017')
        self.db = client.project
        self.db.people.create_index(
            [('employee_id', pymongo.ASCENDING)], unique = True
        )
    
    def insert_person(self,data):
        document = {
        'employee_id':data[0],
        'name':data[1],
        'embeddings':data[2]
        }
        try:
            result = self.db.people.insert_one(document)
            return 'Insert successfully'
        except:
            return 'Can not insert'

    def delete_person(self,data):
        document = {
        'employee_id':data
        }
        result = self.db.people.delete_one(document)
        return result

    def update_person(self,id,data):
        result = self.db.people.update(
            {'employee_id':id},
            data,
            upsert= True)
        return result

    def find_all(self):
        result = self.db.people.find({'embeddings':{'$exists':True}})
        return result

    def insert_image(self,data):
        result = self.db.people.update_one(
            {'employee_id':data[0]},
            {'$push':{
                'embeddings':data[1]
            }}
        )
        return result

if __name__ == "__main__":
    arr1 = np.array([1,2,3,4,5,6])
    arr2 = np.array([1,2,3,4,5,6])
    arr1 = np.array_str(arr1)
    arr2 = np.array_str(arr2)
    data = (1,'linh',[arr1,arr2])
    # data = 1
    db = DB()
    # db.delete_person(data)
    db.insert_person(data)
    res = db.find_all()
    for re in res:
        print(re)