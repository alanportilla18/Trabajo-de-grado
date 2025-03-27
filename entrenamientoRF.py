import cv2
import os
import numpy as np

dataPath = 'C:/Users/porti/Documents/Reconocimiento/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas registrada: ',peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('leyendo las imagenes')

    for fileName in os.listdir(personPath):
        print('Rosotros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
    label = label + 1

# print('labels= ',labels)
# print('Numero de 0 = ',np.count_nonzero(np.array(labels)==0))
# print('Numero de 1 = ',np.count_nonzero(np.array(labels)==1))

# face_recognizer = cv2.face.EigenFaceRecognizer_create() 
face_recognizer = cv2.face.FisherFaceRecognizer_create()

print('Entrenando... ')
face_recognizer.train(facesData, np.array(labels))

# face_recognizer.write('modeloEigenFace.xml')
face_recognizer.write('modeloFisherFace.xml')

print('Modelo almacenado...')
cv2.destroyAllWindows()