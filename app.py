from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('webpage.html')

@app.route('/', methods=['POST'])
def upload():
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haar_cascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haar_cascade/haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imwrite('static/detected.jpg', img)

    return render_template('webpage.html')

if __name__ == '__main__':
    app.run(debug=True)


'''
import cv2
import numpy as np
def sketch(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(5,5),900)
    edges = cv2.Canny(blur_gray,45,90)
    ret,thre = cv2.threshold(edges,70,255,cv2.THRESH_BINARY_INV)
    return thre
cam = cv2.VideoCapture(0)
while 1:
    ret,frame = cam.read()
    cv2.imshow('Live Sketch', sketch(frame))
    if cv2.waitKey(1)==27:
        break
    if cv2.waitKey(1)==13:
        cv2.imwrite('sketch.jpg',sketch(frame))
        print('Image Saved!!!')
cam.release()
cv2.destroyAllWindows()
'''