import cv2

video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")
eye_eyeGlasses_Cascade = cv2.CascadeClassifier("dataset/haarcascade_eye_eyeglasses.xml")

cnt = 0
while True:
    success, img = video.read()  # read image from video
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to B&W
    faces = faceCascade.detectMultiScale(grayImg, 1.1,
                                         4)  # recognize faces using an already included haarcascade file and detectMultiscale() function
    keyPressed = cv2.waitKey(1)

    for x, y, w, h in faces:
        img_face = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0),
                                3)  # if face detected we will draw an outer boundary
        eye_eyeGlasses_Images = eye_eyeGlasses_Cascade.detectMultiScale(grayImg, 1.1, 4)
        smiles = smileCascade.detectMultiScale(grayImg, 1.8, 15)
        for x, y, w, h in smiles:
            if cnt > 30:
                break
            img_smile = cv2.rectangle(img_face, (x, y), (x + w, y + h), (100, 100, 100),
                                5)  # if smile detected we will draw an outer boundary
            print("Image smile " + str(cnt) + " Saved")
            path = r'C:\Users\rache\Desktop\SmileCapture\images\img' + str(cnt) + '.jpg'
            cv2.imwrite(path, img_smile)
            cnt += 1


        for x, y, w, h in eye_eyeGlasses_Images:
            if cnt > 30:
                break
            img_eye = cv2.rectangle(img_face, (x, y), (x + w, y + h), (100, 100, 100),
                                 5)  # if eyes detected we will draw an outer boundary
            print("Image eye " + str(cnt) + " Saved")
            path = r'C:\Users\rache\Desktop\SmileCapture\img' + str(cnt) + '.jpg'
            cv2.imwrite(path, img_eye)
            cnt += 1

    cv2.imshow('live video', img)
    if keyPressed & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
