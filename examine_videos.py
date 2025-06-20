import cv2

video_path = r"C:\Users\Moorman\calcium_imaging\raw\IG_Recording_appetitive_food_test__FL_repeat_presentation_June172025\IL16-24-086\2025_06_17\12_15_11\BehaviorCam\0.avi"
cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Frame", frame)
    print(f"Frame Number: {frame_num}")  # Display frame number

    key = cv2.waitKey(0)  # Press any key to go to next frame
    if key == ord('q'):   # Press 'q' to quit
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
