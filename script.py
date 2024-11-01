import cv2


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords


def detect(img, faceCascade, eyesCascade, mouthCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}

    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]  # (y,x)
        coords = draw_boundary(roi_img, eyesCascade, 1.1, 30, color['red'], "Eye")
        coords = draw_boundary(roi_img, mouthCascade, 1.1, 50, color['green'], "Mouth")
    return img


faceCascade = cv2.CascadeClassifier("clasiffiers/haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("clasiffiers/haarcascade_eye.xml")
mouthCascade = cv2.CascadeClassifier("clasiffiers/haarcascade_mcs_mouth.xml")

videoCapture = cv2.VideoCapture(0)

while True:
    _, img = videoCapture.read()
    img = detect(img, faceCascade, eyesCascade, mouthCascade)
    cv2.imshow("Facial Feature Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Facial Feature Recognition",
                                                                  cv2.WND_PROP_VISIBLE) < 1:
        # End loop if 'q' is pressed or if the window is closed
        break

videoCapture.release()
cv2.destroyAllWindows()
