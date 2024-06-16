import cv2

cascade = 'haarcascade_frontalface_default.xml'

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    facecascade = cv2.CascadeClassifier(cascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face = facecascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    
    cv2.imshow("FACE", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break