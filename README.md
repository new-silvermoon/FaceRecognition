# FaceRecognition
 Face recognition using OpenCV.
 
 This repo showcases the usage of EigenFaceRecognizer in OpenCV.
 
 Python package requirements:
 opencv
 opencv-contrib
 numpy
 
 Usage:
 
 1. For creating a train dataset
 
 Execute generate_face_data() in face_recognition.py
 This will capture your face via webcam and will create TRAIN_IMAGES_COUNT number of images for train dataset.
 Update FACE_LABEL variable if you are training for multiple faces.
 
 2. For recognising faces
 
 Execute recognise_face() in face_recognition.py
