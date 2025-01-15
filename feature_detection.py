import cv2
import sys
import os

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
    face_coords = draw_boundaries(img, faceCascade, 1.1, 10, color['blue'], "Face")
    for (fx, fy, fw, fh) in face_coords:
        roi_img = img[fy:fy + fh, fx:fx + fw]
        draw_boundaries(roi_img, eyesCascade, 1.1, 15, color['red'], "Eye")
        draw_boundaries(roi_img, mouthCascade, 1.1, 30, color['green'], "Mouth")
    return img

if len(sys.argv) > 1:
    if sys.argv[1] == '-test' and len(sys.argv) > 2:  # Test mode
        dir_path = sys.argv[2]
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            print(f"Error: '{dir_path}' is not a valid directory")
            sys.exit()

        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        if not image_files:
            print(f"No image files found in directory '{dir_path}'")
            sys.exit()

        output_dir = os.path.join(dir_path, "output")
        os.makedirs(output_dir, exist_ok=True)

        for img_file in image_files:
            img_path = os.path.join(dir_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error: Unable to load image '{img_file}'")
                continue

            processed_img = detect(img, faceCascade, eyesCascade, mouthCascade)
            output_path = os.path.join(output_dir, f"processed_{img_file}")
            cv2.imwrite(output_path, processed_img)
            print(f"Processed and saved: {output_path}")

    else:  # Single image input
        img_path = sys.argv[1]
        img = cv2.imread(img_path)

        if img is None:
            print("Error: Unable to load image")
            sys.exit()

        img = detect(img, faceCascade, eyesCascade, mouthCascade)
        cv2.imshow("Facial Feature Recognition", img)
        cv2.waitKey(0)

else:  # Camera input
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