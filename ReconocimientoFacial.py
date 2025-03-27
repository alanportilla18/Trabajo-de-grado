import cv2
import os

dataPath = './data'
imagesPaths = os.listdir(dataPath)
print('ImagesPaths= ',imagesPaths)

# face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.FisherFaceRecognizer_create()

#leyendo modelo
# face_recognizer.read('ModeloEigenFace.xml')
face_recognizer.read('modeloFisherFace.xml')

# cap = cv2.VideoCapture('tests/Video1.mp4')
# cap = cv2.VideoCapture('tests/Video2.mp4')
# cap = cv2.VideoCapture('tests/Video3.mp4')
# cap = cv2.VideoCapture('tests/Plinio.mp4')
# cap = cv2.VideoCapture('tests/Plinio2.mp4')
cap = cv2.VideoCapture('tests/Prueba.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
        rostro = rostro.reshape(1, -1) 
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result),(x, y-5), 1, 1.3, (255,255,0), 1, cv2.LINE_AA)
        '''
        #EigenFaces
        if result[1] < 5900:
            cv2.putText(frame, '{}'.format(imagesPaths[result[0]]),(x, y+15), 1, 1.3, (0,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            cv2.putText(frame,'Desconocido ',(x, y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        '''
        # '''
        #FisherFaces
        if result[1] < 500:
            cv2.putText(frame, '{}'.format(imagesPaths[result[0]]),(x, y+15), 1, 1.3, (0,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            cv2.putText(frame,'Desconocido ',(x, y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        # '''

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()