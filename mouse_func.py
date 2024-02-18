import cv2

def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      
      cv2.putText(frame, f'({x},{y})',(x,y),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
      cv2.circle(frame, (x,y), 3, (0,255,255), -1)

cap = cv2.VideoCapture('videos/D303erer_20240216183826.mp4')

if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()


cv2.namedWindow('Point Coordinates')

cv2.setMouseCallback('Point Coordinates', click_event)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Point Coordinates', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
