from firebase import firebase  
firebase = firebase.FirebaseApplication('https://iot-rover-69-default-rtdb.firebaseio.com/', None)  
data =  { 'Name': 'Vivek',  
          'RollNo': 1,  
          'Percentage': 76.02  
          }  
result = firebase.post('/Test/',data)  
print(result)  