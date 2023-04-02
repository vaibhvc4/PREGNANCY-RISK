import pymongo 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def db_entry():
     y=0
     for i in  collection.find():
           y+=1
     
     pat_dict={'_id':y,'name':'','age':0,'sys':0.0,'dia':0.0,'spo2':0.0,'BMI':0.0,'hrate':0,'temp':0.0,'glevel':0} 
            
          
     print('--------------ENTER THE MEDICAL RECORD---------------') 
   
     name = input("PATIENTS NAME                 :\t") 
     pat_dict['name']=name 
     age =  input('PATIENT AGE                   :\t')
     pat_dict['age']=age
     sys =  input('SYSTOLIC BLOOD PRESSURE(mmHg) :\t') 
     pat_dict['sys']=sys 
     dia =  input('DIASTOLIC BLOOD PRESSURE(mmHg):\t') 
     pat_dict['dia']=dia
     spo2 = input('SP02 LEVELS (%)               :\t') 
     pat_dict['spo2']=spo2
     BMI =  input('BMI(kg/m^2)                   :\t') 
     pat_dict['BMI']=BMI
     hrate =input('HEART RATE(PER MIN)           :\t') 
     pat_dict['hrate']=hrate
     temp = input('TEMPERATURE(CELCIUS)          :\t') 
     pat_dict['temp']=temp
     glevel = input('GLUCOSE LEVELS(mg/dl)       :\t') 
     
     pat_dict['glevel']=glevel
     collection.insert_one(pat_dict)
     print('--------DATABASE UPDATED-----------\n\t YOUR PATIENT ID IS',pat_dict['_id']) 
     collection.find().sort('_id',1)
     
             
          
     pat_dict['_id']+=1 ;
     ml(sys,dia,hrate,glevel)
    


def delete(y):
       i = 0
       i = int(input("\n\n[1]DELETE ENTIRE DATABASE\n[2]DELETE LAST ENTRY\n"))
       if i==1:
         db.drop_collection(collection);
       elif i==2:
            y=0
            for i in collection.find():
                  y=i['_id']
                  print(y)
            tt ={'_id':y}
            collection.find_one_and_delete(tt)     

       
              

 
       

def stats():
     print("\n\t\t\t================HEALTH STATS===================\n")
     idd =int( input("ENTER THE PATIENT ID :"))
     pat1= {'_id':idd}
    
     x = collection.find_one(pat1)
     print(x['sys'])
     tri =int( input('\n\t\t\tENTER THE TRIMESTER\n[1]FIRST TRIMESTER\n[2]SECOND TRIMESTER\n[3]THIRD TRIMESTER\n'))
     print('\n\t\t\t==================DATA===================\n')
     print(x)
     if tri ==1:
           if float(x['sys'])>123.5 or float(x['sys'])<94.8:
                  print('\nCOMPLICATIONS/RISKS:\n preeclampsia, preterm birth, placental abruption, and cesarean birth.')
                  print('CAUSE:\nSYSTOLIC BLOOD PRESSURE           :UNSTABLE\n')
           else:
                  print('SYSTOLIC BLOOD PRESSURE                   :STABLE\n')
           print("\n---------------------------------------------------------------------------------\n")
           if float(x['dia'])>86.9 or float(x['dia'])<55.5:
                  print('\n\nCOMPLICATIONS/RISKS:\n preeclampsia, preterm birth, placental abruption, and cesarean birth.')
                  print('CAUSE:\nDIASTOLIC BLOOD PRESSURE          :UNSTABLE\n')
           else:
                  print('DIASTOLIC BLOOD PRESSURE                  :STABLE\n')
           print("\n---------------------------------------------------------------------------------\n")       
           if float(x['spo2'])>99.4 or float(x['spo2'])<94.3:
                  print('COMPLICATIONS:\nHYPOXEMIA(shortness of breath, headache, and confusion or restlessness)')       
                  print('CAUSE:\nSPO2 LEVELS(o2 saturation levels) :UNSTABLE\n')
           else:
                  print('SPO2 LEVELS(o2 saturation levels)         :STABLE\n')
           print("\n---------------------------------------------------------------------------------\n")       
           if float(x['temp'])>35:
                  print('\nCOMPLICATIONS:\nFEVER')
                  print('TEMPERATURE                      :UNSTABLE\n')
           else:
                  print('TEMPERATURE                       :STABLE\n')
           print("\n---------------------------------------------------------------------------------\n")       
           if float(x['glevel'])>118 or float(x['glevel'])<97:
                  print("COMPLICATIONS:\nLow glucose levels post birth,obese baby,birth defects and heart problems")
                  print("CAUSE:\nGLUCOSE LEVELS(mg/dl)             :UNSTABLE\n")
           else: 
                  print("GLUCOSE LEVELS(mg/dl)                     :STABLE ")
           print("\n---------------------------------------------------------------------------------\n")   
                  



def ml(x,y,z,w):
     mater = "D:\Maternal Health Risk Data Set.csv"
     names = ['Age','SYS-BP','DIA-BP','Blood-sugar','Temp','Heart-rate','risk-level' ]
     ds = pd.read_csv(mater, names = names)    
     le = LabelEncoder()
   
     ds.iloc[:,6] = le.fit_transform(ds.iloc[:,6].values )
     ds_n=ds.drop(labels=['Age','Temp'], axis=1)
     print(ds_n)
     ds.pivot_table('risk-level',index= 'Heart-rate').plot()
     
     #split for training and testing dataset
     x = ds_n.iloc[:,0:4].values
     y = ds_n.iloc[:,4].values
     x_train ,x_test,y_train,y_test = train_test_split(x , y ,test_size = 0.3 , random_state = 1)
     s  =StandardScaler()
     x_train = s.fit_transform(x_train)
     x_test = s.fit_transform(x_test)
     def models(x_train,y_train):
              lr = LogisticRegression() #logistic regression
              lr.fit(x_train,y_train)

              kn = KNeighborsClassifier(n_neighbors = 5 ,metric = 'minkowski',p=2)   #kneigbours algorithm
              kn.fit(x_train,y_train)
                
              svc_l = SVC(kernel = 'linear',random_state=0) # svc linear kernel
              svc_l.fit(x_train,y_train)
                
                
              svc_r = SVC(kernel = 'rbf',random_state=0)  # svc rbf kernel
              svc_r.fit(x_train,y_train)
                
                
              gaus = GaussianNB()   #gaussian algorithm
              gaus.fit(x_train,y_train)
                
                
              tree = DecisionTreeClassifier(criterion = 'entropy',random_state=0)  #decision tree
              tree.fit(x_train,y_train)
                
    
              forest =  RandomForestClassifier(n_estimators=10 , criterion = 'entropy',random_state=0) #random forest
              forest.fit(x_train,y_train)
               
              print('[0]Logistic Regression accuracy',lr.score(x_train,y_train))
              print('[1]KNeighbors accuracy',kn.score(x_train,y_train))
              print('[2]SVC_LINEAR accuracy',svc_l.score(x_train,y_train))
              print('[3]SVC_RBF accuracy',svc_r.score(x_train,y_train))
              print('[4]Gaussian accuracy',gaus.score(x_train,y_train))
              print('[5]DecisionTree accuracy',tree.score(x_train,y_train))
              print('[6]RandomForest accuracy',forest.score(x_train,y_train)) 
               
    
              return lr,kn ,svc_l,svc_r,gaus,tree,forest
    
     model = models(x_train,y_train)
     for i in range (len(model)):
          con = confusion_matrix(y_test , model[i].predict(x_test))
          sns.heatmap(con, annot=True)
          plt.tight_layout()
          plt.show()
          p,q,r,s ,t , u , v ,x,y= confusion_matrix(y_test , model[i].predict(x_test)).ravel()
          tp = p
          
          fn = q+r
          fp =s+v
          tn = t+u+x+y
          tp = t
          test_score = (tp+tn)/(tp+tn+fp+fn)
          print(con)
          print('model[{}] test accuracy : {}'.format(i,test_score))
     tree = model[5]
     importances = pd.DataFrame({'feature':ds_n.iloc[:,0:4].columns, 'importance':np.round(tree.feature_importances_,3)})
     importances = importances.sort_values('importance',ascending =False).set_index('feature')
     print(importances)
     importances.plot.bar()   
      

     #scale input
     sc = StandardScaler() 
     pat_risk = [[ x  ,    y      ,   z  ,        w ]]
     pat_risk_scale = sc.fit_transform(pat_risk)
     
     pred = model[5].predict(pat_risk_scale)
     print(pred) 
     if pred == 1:
           print('THERE IS LOW MATERNAL RISK(the prediction may not be accurate to limited info from dataset)')
     elif pred==2:
           print("THERE IS MEDIUM MATERNAL RISK(the prediction may not be accurate to limited info from dataset)")
     else:
          print("THERE IS HIGH MATERNAL RISK(the prediction may not be accurate to limited info from dataset)")      


if  __name__=="__main__":
     client = pymongo.MongoClient("mongodb://localhost:27017/")
     print(client)
     db = client['PREGNANCY']
     collection = db['patientdata'] 
     print("\t\t\tCONNECTED TO DATABASE.....")
     print('\n\n\t\t\t==========MEDIHELP==========')
     c=0
     y=0
    
     while c!=5:
           c=0
           c =int( input("\n[1]DATA ENTRY\n[2]HEALTH STATS\n[3]DISPLAY DATA\n[4]DELETE DATA\n[5]EXIT\n"))
           if c==1:
              db_entry()
           elif c==2:
             stats()
           elif c==3:
             
              u = collection.find()
              for u in collection.find():
                print(u)
              if len(list(u))==0:
                   print('\n\t\t\tEMPTY DATABSE ...')
           elif c==4:
            delete(y)
           
           elif c==5:
                exit(125)

                 


  




         