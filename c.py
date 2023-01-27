import time

import cv2
import mediapipe as mp
#from djitellopy import Tello

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#For static images:
IMAGE_FILES = []
# is_takeoff = False
# is_land = False
# tello = Tello()
# tello.connect()

with mp_hands.Hands(
    model_complexity=1,
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
time_s = time.time()
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # if results.multi_hand_landmarks and not is_takeoff:
    #     is_takeoff = True
    #     # tello.takeoff()
    #     is_land = False
    #     print("UP")
    #
    # if not results.multi_hand_landmarks and is_takeoff and not is_land:
    #     is_land = True
    #     # tello.land()
    #     is_takeoff = False
    #     print("LAN")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xf = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            yf = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            xf2 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            yf2 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            height_fix = ((xf - xf2) ** 2 + (yf - yf2) ** 2) ** 0.5
    if results.multi_hand_landmarks:
        for hand_landmarks in  results.multi_handedness:
            if hand_landmarks.classification[0].label == 'Right':
                while height_fix >= 0.29:
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                             x1 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 500
                             y1 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 500
                             x2 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * 500
                             y2 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * 500
                             if results.multi_hand_landmarks:
                                 for hand_landmarks in results.multi_handedness:
                                     if hand_landmarks.classification[0].label == 'Left':
                                        r_height = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                                        if time.time() - time_s > 5:
                                            if r_height > 75:
                                                r_height = r_height - 75
                                                if r_height < 20:
                                                    r_height = 20
                                                print('вверх')# tello.move('up', int(r_height))
                                            else:
                                                r_height = 75 - r_height
                                            if r_height < 20:
                                                r_height = 20
                                                print('вниз')#tello.move('down', int(height))
                                            time_s = time.time()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_handedness:
            if hand_landmarks.classification[0].label == 'Left':
                while height_fix >= 0.27:
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            x1l = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 500
                            y1l = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 500
                            x2l = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * 500
                            y2l = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * 500
                            for hand_landmarks in results.multi_handedness:
                                if hand_landmarks.classification[0].label == 'Right':
                                            l_height = ((x1l - x2l) ** 2 + (y1l - y2l) ** 2) ** 0.5
                                            if time.time() - time_s > 5:
                                                 if l_height > 75:
                                                     l_height = l_height - 75
                                                     if l_height < 20:
                                                         l_height = 20
                                                     print('вправо')
                                                     #tello.move('right', int(height))
                                                 else:
                                                     l_height = 75 - l_height
                                                     if l_height < 20:
                                                         l_height = 20
                                                     print('влево')
                                                    # tello.move('left', int(l_height))
                                                 time_s = time.time()
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()