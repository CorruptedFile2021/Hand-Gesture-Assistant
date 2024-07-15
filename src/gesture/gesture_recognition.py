import mediapipe as mp
import cv2
import os
import sys

webcam = cv2.VideoCapture(0)
#Model_path = f'{os.path.dirname(sys.argv[0])}\models\gesture_recognizer.task'
Model_path = f'D:\VsCodeProjects\Python\Github Projects\Hand Gesture Assistant\src\gesture\models\gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

    
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=Model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_hand_detection_confidence=0.05,
    min_hand_presence_confidence=0.10,
    min_tracking_confidence=0.1,
    num_hands=1)

with GestureRecognizer.create_from_options(options) as recognizer:
    width  = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Width = {width}")
    print(f"Height = {height}")


    frame_timestamp = 1    
    while True:
        
        ret, frame = webcam.read() 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        results = recognizer.recognize_for_video(mp_image,frame_timestamp)
        
        frame_timestamp = frame_timestamp+1
        try:
            gesture = results.gestures[0][0]
            gesture = gesture.category_name
            gesture = str(gesture).replace("_"," ").lower()

            hand = results.handedness[0][0]
            hand = hand.display_name
            gesture = str(gesture).lower()

            cv2.putText(frame,f"{gesture}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
            cv2.putText(frame,f"{hand}",(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)

        
        except Exception as e:
            print(e)
        
        cv2.imshow('Gesture Recognizer', frame) 

       
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    

    webcam.release() 
    cv2.destroyAllWindows() 
     
    