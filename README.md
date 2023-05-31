# Redcat_project3
### opencv를 이용해 손 모양 인식 시키기
#### 인식 시킬 모양 - 가위, 바위, 보
##### pinky 변수는 모델이 없다는 오류가 떠서 일단 주석 처리함

    import cv2
    import mediapipe as mp
    from PIL import ImageFont, ImageDraw, Image
    import numpy as np

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
      min_detection_confidence = 0.5,
      min_tracking_confidence = 0.5) as hands:
  
      while cap.isOpened():
        success, image = cap.read()
    
        if not success:
          print("lgnoring empty camera frame.")
          continue
    
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
        image.flags.writeable = False
        results = hands.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
    
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            thumb_finger_state = 0
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height:
              if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height:
                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height:
                  thumb_finger_state = 1
              
            index_finger_state = 0
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height:
              if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y *image_height:
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height:
                  index_finger_state = 1
        
            middle_finger_state = 0
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height:
              if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y *image_height:
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height:
                  middle_finger_state = 1
              
            ring_finger_state = 0
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height:
              if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y *image_height:
                if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height:
                  ring_finger_state = 1
              
            # pinky_finger_state = 0                
            # # if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_PIP].y * image_height:                
            # #     if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_DIP].y *image_height:                         
            # if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_TIP].y * image_height:                             
            # pinky_finger_state = 1                                            
            # 
            font = ImageFont.truetype("fonts/gulim.ttc", 80)
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
        
            text = ""
            if thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1:
              text = "보"
            elif thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0:
              text = "가위"
            elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 0 and ring_finger_state == 0:
              text = "가위"
            elif thumb_finger_state == 0 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0:
              text = "바위"
          
            w, h = font.getsize(text)
        
            x = 50
            y = 50
          
            draw.rectangle((x, y, x + w, y + h), fill = 'black')
            draw.text((x, y), text, font = font, fill = (255, 255, 255))
          
            image = np.array(image)
          
            mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_style.get_default_hand_landmarks_style(),
              mp_drawing_style.get_default_hand_connections_style())
      
          
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    
    cap.release()
