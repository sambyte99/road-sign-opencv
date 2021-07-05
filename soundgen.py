from gtts import gTTS


lst=[
'Speed limit 20',
'Speed limit 30',
'Speed limit 50',
'Speed limit 60',
'Speed limit 70',
'Speed limit 80',
'End of speed limit 80',
'Speed limit 100',
'Speed limit 120',
'No passing',
'No passing for vechiles over 3.5 metric tons',
'Right-of-way at the next intersection',
'Priority road',
'Yield',
'Stop',
'No vechiles',
'Vechiles over 3.5 metric tons prohibited',
'No entry',
'General caution',
'Dangerous curve to the left',
'Dangerous curve to the right',
'Double curve',
'Bumpy road',
'Slippery road',
'Road narrows on the right',
'Road work',
'Traffic signals',
'Pedestrians',
'Children crossing',
'Bicycles crossing',
'Beware of ice/snow',
'Wild animals crossing',
'End of all speed and passing limits',
'Turn right ahead',
'Turn left ahead',
'Ahead only',
'Go straight or right',
'Go straight or left',
'Keep right',
'Keep left',
'Roundabout mandatory',
'End of no passing',
'End of no passing by vechiles over 3.5 metric tons']


for i in range(len(lst)):
    tts = gTTS(lst[i], lang='en')
    tts.save(str(i+1)+'.mp3')
    print(lst[i])
