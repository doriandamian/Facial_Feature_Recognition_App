import cv2
import sys

faceCascade = cv2.CascadeClassifier("clasiffiers/haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("clasiffiers/haarcascade_eye.xml")
mouthCascade = cv2.CascadeClassifier("clasiffiers/haarcascade_mcs_mouth.xml")

def draw_boundaries(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords_list = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords_list.append((x, y, w, h))

    return coords_list

def detect(img, faceCascade, eyesCascade, mouthCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
    face_coords = draw_boundaries(img, faceCascade, 1.1, 12, color['blue'], "Face")
    for (fx, fy, fw, fh) in face_coords:
        roi_img = img[fy:fy + fh, fx:fx + fw]
        draw_boundaries(roi_img, eyesCascade, 1.1, 22, color['red'], "Eye")
        draw_boundaries(roi_img, mouthCascade, 1.1, 60, color['green'], "Mouth")
    return img

if len(sys.argv) > 1: # Image Input
    img_path = sys.argv[1]
    img = cv2.imread(img_path)

    if img is None:
        print("Error: Unable to load image")
        sys.exit()

    img = detect(img, faceCascade, eyesCascade, mouthCascade)
    cv2.imshow("Facial Feature Recognition", img)
    cv2.waitKey(0)
else: # Camera Input
    videoCapture = cv2.VideoCapture(1)
    while True:
        _, img = videoCapture.read()
        img = detect(img, faceCascade, eyesCascade, mouthCascade)
        cv2.imshow("Facial Feature Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Facial Feature Recognition", cv2.WND_PROP_VISIBLE) < 1:
            # End loop if 'q' is pressed or if the window is closed
            break
    videoCapture.release()

cv2.destroyAllWindows()
