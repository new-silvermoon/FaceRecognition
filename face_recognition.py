import cv2
import _csv
import numpy as np
import os


TRAIN_IMAGES_COUNT =100
FACE_LABEL = 0
CSV_DATA_PATH = 'data/face_data/'

def generate_face_data():

    face_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_eye.xml')

    camera = cv2.VideoCapture(0)
    count = 0

    while count <= TRAIN_IMAGES_COUNT:
        ret, frame = camera.read()
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame_gray,1.3,5)

        for face in faces:
            x,y,width,height = face

            cv2.rectangle(frame_gray,(x,y),(x+width,y+height),(255,0,0),2)

            data_img = cv2.resize(frame_gray[y:y+height, x:x+width], (200,200))
            cv2.imwrite('data/face_data/'+str(count)+'.pgm',data_img)
            print("Image saved")
            count+=1

        cv2.imshow("frame",frame)

        if cv2.waitKey(int(1000/12)) & 0xff == ord('q'):
            break

    csv_data = []

    for root,dir,files in os.walk('data/face_data'):
        for file in files:
            if '.pgm' in file:
                csv_data.append([CSV_DATA_PATH+file,FACE_LABEL])

    with open('data/face_data/facedata.csv','w') as file:
        writer = _csv.writer(file)
        writer.writerows(csv_data)

    camera.release()
    cv2.destroyAllWindows()

def train_face_data(size=None):

    category_val = 0
    x, y = [],[]
    train_face_data_path = 'data/face_data/facedata.csv'

    with open(train_face_data_path) as file:
        csvreader = _csv.reader(file)

        for row in csvreader:
            try:
                img = cv2.imread(row[0],cv2.IMREAD_GRAYSCALE)

                if size is not None:
                    w,h = size
                    cv2.resize(img,(w,h))

                x.append(np.asarray(img,dtype=np.uint8))
                y.append(category_val)
            except Exception as e:
                print(e)

    return x,y

def recognise_face():

    names = ['Sagar']
    x,y = train_face_data()
    y = np.asarray(y,dtype=np.int32)

    face_model = cv2.face.EigenFaceRecognizer_create()

    face_model.train(np.asarray(x),np.asarray(y))

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')

    while True:
        ret,frame = camera.read()


        faces = face_cascade.detectMultiScale(frame,1.3,5)

        for face in faces:
            x,y,width,height = face

            cv2.rectangle(frame,(x,y),(x+width,y+height),(255,0,0),2)
            img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            face_roi = img_gray[x:x+width, y:y+height]

            try:
                face_roi = cv2.resize(face_roi,(200,200),interpolation=cv2.INTER_LINEAR)
                params = face_model.predict(face_roi)
                print("Label: %s, Confidence: %.2f"%(params[0],params[1]))
                cv2.putText(frame,names[params[0]],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)

            except Exception as e:
                cv2.imshow("face", frame)
                continue

        cv2.imshow("face", frame)

        if cv2.waitKey(int(1000/12)) & 0xff == ord("q"):
            camera.release()
            cv2.destroyAllWindows()


generate_face_data()

#recognise_face()




