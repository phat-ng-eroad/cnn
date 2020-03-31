import cv2

cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,320)

while True:
    rect, frame = cap.read()
    cv2.imwrite('image.png', frame)

    cv2.imshow('frame75', frame)

    key = cv2.waitKey(1) & 0xFF



cap.release()
cv2.destroyAllWindows()