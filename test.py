import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} works")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Cam {i}", frame)
            cv2.waitKey(3000)
        cap.release()

cv2.destroyAllWindows()
