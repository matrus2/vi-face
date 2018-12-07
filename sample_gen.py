import numpy as np
import cv2
import datetime

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        st = datetime.datetime.utcnow()
        cv2.imwrite('dataset/mateusz/mateusz' + (str(st) + '.png'), frame)
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()