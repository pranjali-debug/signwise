from collections import Counter
import copy
import csv
import itertools
import threading
import time
import base64
import os

import cv2 as cv
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template, send_from_directory

from keypoint_classifier import KeyPointClassifier

app = Flask(__name__)

# Global variables to store processing results
global_frame = None
global_string = ""
global_hand_sign = ""
global_timeleft = 0

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def process_webcam():
    global global_frame, global_string, global_hand_sign, global_timeleft
    
    cap_device = 0
    cap_width = 960
    cap_height = 540
    
    use_static_image_mode = False
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5
    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    keypoint_classifier_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ', 'not recognised']

    mode = 0
    res = []
    string = ""
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
            
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                res.append(hand_sign_id)
                if len(res) == 200:
                    if keypoint_classifier_labels[Counter(res).most_common(1)[0][0]] != 'not recognised':
                        string = string + keypoint_classifier_labels[Counter(res).most_common(1)[0][0]]    
                    print(string)
                    res = []

                global_hand_sign = keypoint_classifier_labels[hand_sign_id]
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id]
                )
        else:
            res.append(26)
            if len(res) == 200:
                string = string + keypoint_classifier_labels[Counter(res).most_common(1)[0][0]]
                print(string)
                res = []

        timeleft = len(res) / 100
        global_timeleft = timeleft
        global_string = string
        
        debug_image = draw_info(debug_image, mode, -1, string, timeleft)
        global_frame = debug_image

        # Add a small delay to reduce CPU usage
        time.sleep(0.01)


def generate_frames():
    while True:
        if global_frame is not None:
            ret, buffer = cv.imencode('.jpg', global_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    return render_template('index.html', string=global_string, active_page='home')

@app.route('/learn')
def learn():
    return render_template('learn.html', active_page='learn')

@app.route('/practice')
def practice():
    return render_template('practice.html', active_page='practice')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    return {"text": global_string}

@app.route('/image/<path:filename>')
def serve_image(filename):
    directory = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(directory, filename)


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 30):
        csv_path = 'keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
            print(number, " : ")

    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Draw connections (lines) - same as original code
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Draw points - same as original code
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    cv.putText(image, hand_sign_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_info(image, mode, number, string, timeleft):
    cv.putText(image, "Time:" + str(timeleft), (10, 70), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 4, cv.LINE_AA)
    cv.putText(image, "TEXT:" + str(string), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    # Create and start webcam processing thread
    webcam_thread = threading.Thread(target=process_webcam)
    webcam_thread.daemon = True
    webcam_thread.start()
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASL Learning Platform | Home</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent: #4895ef;
            --light: #f8f9fa;
            --dark: #212529;
            --success: #4cc9f0;
            --warning: #f72585;
            --info: #560bad;
            --transition: all 0.3s ease;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
            color: var(--dark);
            min-height: 100vh;
            padding: 0;
            margin: 0;
        }
        
        .navbar {
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .navbar-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-weight: 700;
            font-size: 1.5rem;
            color: var(--primary);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
            gap: 2rem;
        }
        
        .nav-link {
            text-decoration: none;
            color: var(--dark);
            font-weight: 500;
            transition: var(--transition);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            position: relative;
        }
        
        .nav-link:hover {
            color: var(--primary);
        }
        
        .active {
            color: var(--primary);
            background-color: rgba(67, 97, 238, 0.1);
        }
        
        .active::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 30%;
            height: 2px;
            background-color: var(--primary);
        }
        
        header {
            text-align: center;
            margin: 3rem 0;
            padding: 0 2rem;
        }
        
        h1 {
            color: var(--primary);
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            position: relative;
            display: inline-block;
        }
        
        h1::after {
            content: '';
            position: absolute;
            width: 70%;
            height: 3px;
            background: var(--accent);
            left: 15%;
            bottom: -8px;
            border-radius: 2px;
        }
        
        .subtitle {
            color: var(--secondary);
            font-size: 1.2rem;
            font-weight: 300;
            opacity: 0.9;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem 3rem;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.05);
            overflow: hidden;
            margin-bottom: 2rem;
            transition: var(--transition);
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }
        
        .main-content {
            display: flex;
            flex-wrap: wrap;
            gap: 2rem;
        }
        
        .video-section, .reference-section {
            flex: 1;
            min-width: 300px;
        }
        
        .card-header {
            background: var(--primary);
            color: white;
            padding: 1rem 1.5rem;
            font-size: 1.2rem;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .card-header i {
            font-size: 1.1rem;
        }
        
        .card-body {
            padding: 1.5rem;
        }
        
        .video-container {
            width: 100%;
            background: var(--dark);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        
        .video-container img {
            max-width: 100%;
            height: auto;
            display: block;
        }
        
        .video-overlay {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.5);
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        
        .reference-image {
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
            transition: var(--transition);
        }
        
        .reference-image:hover {
            transform: scale(1.02);
        }
        
        .reference-image img {
            max-width: 100%;
            height: auto;
            display: block;
        }
        
        .output-section {
            margin-top: 2rem;
        }
        
        #output-text {
            font-size: 2rem;
            font-weight: 600;
            color: var(--primary);
            margin-top: 1rem;
            min-height: 80px;
            padding: 1rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            border-left: 4px solid var(--accent);
            transition: var(--transition);
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .instructions {
            color: var(--info);
            margin-top: 1rem;
            font-size: 1rem;
            line-height: 1.6;
            padding: 0.5rem 0;
        }
        
        .current-sign {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 1.5rem;
            background-color: rgba(67, 97, 238, 0.05);
            padding: 1rem;
            border-radius: 8px;
        }
        
        .sign-label {
            font-weight: 600;
            color: var(--secondary);
        }
        
        .sign-value {
            font-size: 1.5rem;
            padding: 0.5rem 1rem;
            background: var(--accent);
            color: white;
            border-radius: 4px;
            transition: var(--transition);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .sign-value:empty::before {
            content: "-";
            opacity: 0.5;
        }
        
        .features {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .feature {
            flex: 1;
            min-width: 250px;
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            transition: var(--transition);
        }
        
        .feature:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        .feature i {
            font-size: 2rem;
            color: var(--primary);
            margin-bottom: 1rem;
        }
        
        .feature h3 {
            margin-bottom: 0.5rem;
            color: var(--secondary);
        }
        
        footer {
            background: white;
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            color: var(--dark);
            opacity: 0.8;
            font-size: 0.9rem;
        }
        
        .btn {
            padding: 0.8rem 1.5rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            font-size: 1rem;
            transition: var(--transition);
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            text-decoration: none;
            box-shadow: 0 4px 10px rgba(67, 97, 238, 0.3);
        }
        
        .btn:hover {
            background: var(--secondary);
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(67, 97, 238, 0.4);
        }
        
        .btn i {
            font-size: 1rem;
        }
        
        .page-transitions {
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @media (max-width: 768px) {
            .main-content {
                flex-direction: column;
            }
            
            .video-section, .reference-section {
                width: 100%;
            }
            
            .navbar-container {
                flex-direction: column;
                gap: 1rem;
            }
            
            .nav-links {
                width: 100%;
                justify-content: center;
                flex-wrap: wrap;
                gap: 1rem;
            }
            
            .card-header {
                flex-direction: column;
                gap: 0.5rem;
                text-align: center;
            }
            
            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-container">
            <a href="/" class="logo">
                <i class="fas fa-american-sign-language-interpreting"></i>
                ASL Learning
            </a>
            <ul class="nav-links">
                <li><a href="/" class="nav-link active">Home</a></li>
                <li><a href="/learn" class="nav-link">Learn</a></li>
                <li><a href="/practice" class="nav-link">Practice</a></li>
                <li><a href="/about" class="nav-link">About</a></li>
            </ul>
        </div>
    </nav>

    <div class="container page-transitions">
        <header>
            <h1>ASL Recognition & Learning Platform</h1>
            <p class="subtitle">Practice American Sign Language with real-time feedback</p>
        </header>
        
        <div class="main-content">
            <div class="video-section">
                <div class="card">
                    <div class="card-header">
                        <span>Your Signs (Webcam)</span>
                        <i class="fas fa-video"></i>
                    </div>
                    <div class="card-body">
                        <div class="video-container">
                            <img src="{{ url_for('video_feed') }}" alt="Webcam Feed">
                            <div class="video-overlay">Live</div>
                        </div>
                        <div class="current-sign">
                            <span class="sign-label">Current Sign:</span>
                            <span class="sign-value" id="current-sign"></span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="reference-section">
                <div class="card">
                    <div class="card-header">
                        <span>ASL Reference Guide</span>
                        <i class="fas fa-book-open"></i>
                    </div>
                    <div class="card-body">
                        <div class="reference-image">
                            <img src="{{ url_for('serve_image', filename='image.png') }}" alt="ASL Reference Chart">
                        </div>
                        <p class="instructions">
                            <i class="fas fa-info-circle"></i> Use this reference chart to practice your ASL signs. Position your hand in front of the camera and try to match the signs shown above.
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="output-section">
            <div class="card">
                <div class="card-header">
                    <span>Recognized Text</span>
                    <i class="fas fa-font"></i>
                </div>
                <div class="card-body">
                    <div id="output-text">{{ string }}</div>
                </div>
            </div>
        </div>
        
        <div class="features">
            <div class="feature">
                <i class="fas fa-graduation-cap"></i>
                <h3>Learn</h3>
                <p>Access comprehensive ASL tutorials designed for all levels.</p>
                <a href="/learn" class="btn" style="margin-top: 1rem;">
                    Start Learning
                    <i class="fas fa-arrow-right"></i>
                </a>
            </div>
            
            <div class="feature">
                <i class="fas fa-hand-paper"></i>
                <h3>Practice</h3>
                <p>Enhance your skills with interactive practice exercises.</p>
                <a href="/practice" class="btn" style="margin-top: 1rem;">
                    Practice Now
                    <i class="fas fa-arrow-right"></i>
                </a>
            </div>
            
            <div class="feature">
                <i class="fas fa-info-circle"></i>
                <h3>About</h3>
                <p>Learn more about our ASL Recognition platform and technology.</p>
                <a href="/about" class="btn" style="margin-top: 1rem;">
                    Read More
                    <i class="fas fa-arrow-right"></i>
                </a>
            </div>
        </div>
    </div>
    
    <footer>
        <p>&copy; 2023 ASL Recognition Platform | Created to help people learn American Sign Language</p>
    </footer>

    <script>
        // Function to update the text periodically
        function updateText() {
            fetch('/get_text')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('output-text').textContent = data.text || "-";
                    document.getElementById('current-sign').textContent = "{{ global_hand_sign }}" || "-";
                })
                .catch(error => console.error('Error fetching text:', error));
        }
        
        // Update every second
        setInterval(updateText, 1000);

        // Active navigation highlight
        document.addEventListener('DOMContentLoaded', () => {
            const currentLocation = window.location.pathname;
            const navLinks = document.querySelectorAll('.nav-link');
            navLinks.forEach(link => {
                if (link.getAttribute('href') === currentLocation) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        });
    </script>
</body>
</html>
        ''')
    
    # Create learn.html template
    with open('templates/learn.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASL Learning Platform | Learn</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent: #4895ef;
            --light: #f8f9fa;
            --dark: #212529;
            --success: #4cc9f0;
            --warning: #f72585;
            --info: #560bad;
            --transition: all 0.3s ease;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
            color: var(--dark);
            min-height: 100vh;
            padding: 0;
            margin: 0;
        }
        
        .navbar {
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .navbar-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-weight: 700;
            font-size: 1.5rem;
            color: var(--primary);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
            gap: 2rem;
        }
        
        .nav-link {
            text-decoration: none;
            color: var(--dark);
            font-weight: 500;
            transition: var(--transition);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            position: relative;
        }
        
        .nav-link:hover {
            color: var(--primary);
        }
        
        .active {
            color: var(--primary);
            background-color: rgba(67, 97, 238, 0.1);
        }
        
        .active::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 30%;
            height: 2px;
            background-color: var(--primary);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        h1 {
            color: var(--primary);
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--secondary);
            opacity: 0.8;
        }
        
        .page-transitions {
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            overflow: hidden;
            margin-bottom: 2rem;
            transition: var(--transition);
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }
        
        .card-header {
            background: var(--primary);
            color: white;
            padding: 1rem 1.5rem;
            font-size: 1.2rem;
            font-weight: 500;
        }
        
        .card-body {
            padding: 1.5rem;
        }
        
        .lessons-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .lesson-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            transition: var(--transition);
        }
        
        .lesson-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }
        
        .lesson-image {
            width: 100%;
            height: 180px;
            background-color: #f0f0f0;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 3rem;
            color: var(--primary);
        }
        
        .lesson-content {
            padding: 1.5rem;
        }
        
        .lesson-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 0.5rem;
        }
        
        .lesson-description {
            color: #555;
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }
        
        .btn {
            padding: 0.8rem 1.5rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            font-size: 1rem;
            transition: var(--transition);
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            text-decoration: none;
        }
        
        .btn:hover {
            background: var(--secondary);
            transform: translateY(-2px);
        }
        
        .btn-sm {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
        
        footer {
            background: white;
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            color: var(--dark);
            opacity: 0.8;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .navbar-container {
                flex-direction: column;
                gap: 1rem;
            }
            
            .nav-links {
                width: 100%;
                justify-content: center;
                flex-wrap: wrap;
                gap: 1rem;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .lessons-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-container">
            <a href="/" class="logo">
                <i class="fas fa-american-sign-language-interpreting"></i>
                ASL Learning
            </a>
            <ul class="nav-links">
                <li><a href="/" class="nav-link">Home</a></li>
                <li><a href="/learn" class="nav-link active">Learn</a></li>
                <li><a href="/practice" class="nav-link">Practice</a></li>
                <li><a href="/about" class="nav-link">About</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container page-transitions">
        <header>
            <h1>Learn American Sign Language</h1>
            <p class="subtitle">Comprehensive tutorials for learning ASL</p>
        </header>
        
        <div class="card">
            <div class="card-header">Get Started with ASL</div>
            <div class="card-body">
                <p>American Sign Language (ASL) is a complete, natural language that has the same linguistic properties as spoken languages, with grammar that differs from English. ASL is expressed by movements of the hands and face.</p>
                <p style="margin-top: 1rem;">Start with our beginner lessons below and gradually build your skills.</p>
            </div>
        </div>
        
        <div class="lessons-grid">
            <div class="lesson-card">
                <div class="lesson-image">A</div>
                <div class="lesson-content">
                    <h3 class="lesson-title">Alphabet Basics (A-G)</h3>
                    <p class="lesson-description">Learn how to sign the first seven letters of the alphabet. Perfect for beginners!</p>
                    <a href="#" class="btn btn-sm">View Lesson <i class="fas fa-arrow-right"></i></a>
                </div>
            </div>
            
            <div class="lesson-card">
                <div class="lesson-image">H</div>
                <div class="lesson-content">
                    <h3 class="lesson-title">Alphabet Continued (H-N)</h3>
                    <p class="lesson-description">Continue your alphabet journey with the next seven letters.</p>
                    <a href="#" class="btn btn-sm">View Lesson <i class="fas fa-arrow-right"></i></a>
                </div>
            </div>
            
            <div class="lesson-card">
                <div class="lesson-image">O</div>
                <div class="lesson-content">
                    <h3 class="lesson-title">Completing the Alphabet (O-Z)</h3>
                    <p class="lesson-description">Finish learning the ASL alphabet with the remaining letters.</p>
                    <a href="#" class="btn btn-sm">View Lesson <i class="fas fa-arrow-right"></i></a>
                </div>
            </div>
            
            <div class="lesson-card">
                <div class="lesson-image"><i class="fas fa-hand-pointer"></i></div>
                <div class="lesson-content">
                    <h3 class="lesson-title">Common Greetings</h3>
                    <p class="lesson-description">Learn how to introduce yourself and greet others in ASL.</p>
                    <a href="#" class="btn btn-sm">View Lesson <i class="fas fa-arrow-right"></i></a>
                </div>
            </div>
            
            <div class="lesson-card">
                <div class="lesson-image"><i class="fas fa-question-circle"></i></div>
                <div class="lesson-content">
                    <h3 class="lesson-title">Asking Questions</h3>
                    <p class="lesson-description">Learn how to form questions and express curiosity in ASL.</p>
                    <a href="#" class="btn btn-sm">View Lesson <i class="fas fa-arrow-right"></i></a>
                </div>
            </div>
            
            <div class="lesson-card">
                <div class="lesson-image"><i class="fas fa-comments"></i></div>
                <div class="lesson-content">
                    <h3 class="lesson-title">Simple Conversations</h3>
                    <p class="lesson-description">Put your skills to the test with guided conversation practice.</p>
                    <a href="#" class="btn btn-sm">View Lesson <i class="fas fa-arrow-right"></i></a>
                </div>
            </div>
        </div>
    </div>
    
    <footer>
        <p>&copy; 2023 ASL Recognition Platform | Created to help people learn American Sign Language</p>
    </footer>

    <script>
        // Active navigation highlight
        document.addEventListener('DOMContentLoaded', () => {
            const currentLocation = window.location.pathname;
            const navLinks = document.querySelectorAll('.nav-link');
            navLinks.forEach(link => {
                if (link.getAttribute('href') === currentLocation) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        });
    </script>
</body>
</html>
        ''')
    
    # Create practice.html template
    with open('templates/practice.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASL Learning Platform | Practice</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent: #4895ef;
            --light: #f8f9fa;
            --dark: #212529;
            --success: #4cc9f0;
            --warning: #f72585;
            --info: #560bad;
            --transition: all 0.3s ease;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
            color: var(--dark);
            min-height: 100vh;
            padding: 0;
            margin: 0;
        }
        
        .navbar {
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .navbar-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-weight: 700;
            font-size: 1.5rem;
            color: var(--primary);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
            gap: 2rem;
        }
        
        .nav-link {
            text-decoration: none;
            color: var(--dark);
            font-weight: 500;
            transition: var(--transition);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            position: relative;
        }
        
        .nav-link:hover {
            color: var(--primary);
        }
        
        .active {
            color: var(--primary);
            background-color: rgba(67, 97, 238, 0.1);
        }
        
        .active::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 30%;
            height: 2px;
            background-color: var(--primary);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        h1 {
            color: var(--primary);
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--secondary);
            opacity: 0.8;
        }
        
        .page-transitions {
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            overflow: hidden;
            margin-bottom: 2rem;
            transition: var(--transition);
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }
        
        .card-header {
            background: var(--primary);
            color: white;
            padding: 1rem 1.5rem;
            font-size: 1.2rem;
            font-weight: 500;
        }
        
        .card-body {
            padding: 1.5rem;
        }
        
        .practice-options {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .practice-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            transition: var(--transition);
            border: 1px solid rgba(0,0,0,0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        .practice-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }
        
        .practice-banner {
            background-color: #f8f9fa;
            padding: 2rem;
            text-align: center;
            color: var(--primary);
            font-size: 2rem;
        }
        
        .practice-content {
            padding: 1.5rem;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }
        
        .practice-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 0.5rem;
        }
        
        .practice-description {
            color: #555;
            margin-bottom: 1.5rem;
            line-height: 1.5;
            flex-grow: 1;
        }
        
        .difficulty {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .difficulty-label {
            font-weight: 500;
            margin-right: 0.5rem;
        }
        
        .difficulty-dots {
            display: flex;
            gap: 0.3rem;
        }
        
        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #ddd;
        }
        
        .dot.active {
            background-color: var(--primary);
        }
        
        .btn {
            padding: 0.8rem 1.5rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            font-size: 1rem;
            transition: var(--transition);
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            text-decoration: none;
        }
        
        .btn:hover {
            background: var(--secondary);
            transform: translateY(-2px);
        }
        
        footer {
            background: white;
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            color: var(--dark);
            opacity: 0.8;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .navbar-container {
                flex-direction: column;
                gap: 1rem;
            }
            
            .nav-links {
                width: 100%;
                justify-content: center;
                flex-wrap: wrap;
                gap: 1rem;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .practice-options {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-container">
            <a href="/" class="logo">
                <i class="fas fa-american-sign-language-interpreting"></i>
                ASL Learning
            </a>
            <ul class="nav-links">
                <li><a href="/" class="nav-link">Home</a></li>
                <li><a href="/learn" class="nav-link">Learn</a></li>
                <li><a href="/practice" class="nav-link active">Practice</a></li>
                <li><a href="/about" class="nav-link">About</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container page-transitions">
        <header>
            <h1>Practice Your ASL Skills</h1>
            <p class="subtitle">Test and improve your sign language abilities with interactive exercises</p>
        </header>
        
        <div class="card">
            <div class="card-header">Choose Your Practice Mode</div>
            <div class="card-body">
                <p>Select from various practice modes below. Each mode focuses on different aspects of American Sign Language to help you master specific skills.</p>
                <p style="margin-top: 1rem;">Start with beginner exercises and progress as you become more comfortable with signing.</p>
            </div>
        </div>
        
        <div class="practice-options">
            <div class="practice-card">
                <div class="practice-banner">
                    <i class="fas fa-font"></i>
                </div>
                <div class="practice-content">
                    <h3 class="practice-title">Alphabet Recognition</h3>
                    <div class="difficulty">
                        <span class="difficulty-label">Difficulty:</span>
                        <div class="difficulty-dots">
                            <span class="dot active"></span>
                            <span class="dot"></span>
                            <span class="dot"></span>
                        </div>
                    </div>
                    <p class="practice-description">Practice recognizing and signing the ASL alphabet. Great for beginners!</p>
                    <a href="/" class="btn">Start Practice <i class="fas fa-play"></i></a>
                </div>
            </div>
            
            <div class="practice-card">
                <div class="practice-banner">
                    <i class="fas fa-comment-dots"></i>
                </div>
                <div class="practice-content">
                    <h3 class="practice-title">Simple Words</h3>
                    <div class="difficulty">
                        <span class="difficulty-label">Difficulty:</span>
                        <div class="difficulty-dots">
                            <span class="dot active"></span>
                            <span class="dot active"></span>
                            <span class="dot"></span>
                        </div>
                    </div>
                    <p class="practice-description">Learn and practice common words and short phrases in ASL.</p>
                    <a href="/" class="btn">Start Practice <i class="fas fa-play"></i></a>
                </div>
            </div>
            
            <div class="practice-card">
                <div class="practice-banner">
                    <i class="fas fa-comments"></i>
                </div>
                <div class="practice-content">
                    <h3 class="practice-title">Conversations</h3>
                    <div class="difficulty">
                        <span class="difficulty-label">Difficulty:</span>
                        <div class="difficulty-dots">
                            <span class="dot active"></span>
                            <span class="dot active"></span>
                            <span class="dot active"></span>
                        </div>
                    </div>
                    <p class="practice-description">Practice full conversations and improve your fluency in ASL.</p>
                    <a href="/" class="btn">Start Practice <i class="fas fa-play"></i></a>
                </div>
            </div>
            
            <div class="practice-card">
                <div class="practice-banner">
                    <i class="fas fa-gamepad"></i>
                </div>
                <div class="practice-content">
                    <h3 class="practice-title">ASL Games</h3>
                    <div class="difficulty">
                        <span class="difficulty-label">Difficulty:</span>
                        <div class="difficulty-dots">
                            <span class="dot active"></span>
                            <span class="dot active"></span>
                            <span class="dot"></span>
                        </div>
                    </div>
                    <p class="practice-description">Fun games to help you practice ASL in an engaging way.</p>
                    <a href="/" class="btn">Start Practice <i class="fas fa-play"></i></a>
                </div>
            </div>
        </div>
    </div>
    
    <footer>
        <p>&copy; 2023 ASL Recognition Platform | Created to help people learn American Sign Language</p>
    </footer>

    <script>
        // Active navigation highlight
        document.addEventListener('DOMContentLoaded', () => {
            const currentLocation = window.location.pathname;
            const navLinks = document.querySelectorAll('.nav-link');
            navLinks.forEach(link => {
                if (link.getAttribute('href') === currentLocation) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        });
    </script>
</body>
</html>
        ''')
    
    # Create about.html template
    with open('templates/about.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASL Learning Platform | About</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent: #4895ef;
            --light: #f8f9fa;
            --dark: #212529;
            --success: #4cc9f0;
            --warning: #f72585;
            --info: #560bad;
            --transition: all 0.3s ease;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, var(--light) 0%, #e9ecef 100%);
            color: var(--dark);
            min-height: 100vh;
            padding: 0;
            margin: 0;
        }
        
        .navbar {
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .navbar-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-weight: 700;
            font-size: 1.5rem;
            color: var(--primary);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
            gap: 2rem;
        }
        
        .nav-link {
            text-decoration: none;
            color: var(--dark);
            font-weight: 500;
            transition: var(--transition);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            position: relative;
        }
        
        .nav-link:hover {
            color: var(--primary);
        }
        
        .active {
            color: var(--primary);
            background-color: rgba(67, 97, 238, 0.1);
        }
        
        .active::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 30%;
            height: 2px;
            background-color: var(--primary);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        h1 {
            color: var(--primary);
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--secondary);
            opacity: 0.8;
        }
        
        .page-transitions {
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            overflow: hidden;
            margin-bottom: 2rem;
            transition: var(--transition);
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }
        
        .card-header {
            background: var(--primary);
            color: white;
            padding: 1rem 1.5rem;
            font-size: 1.2rem;
            font-weight: 500;
        }
        
        .card-body {
            padding: 1.5rem;
        }
        
        .about-section {
            line-height: 1.8;
        }
        
        .about-section p {
            margin-bottom: 1.5rem;
        }
        
        .about-section h2 {
            color: var(--primary);
            margin: 2rem 0 1rem;
        }
        
        .tech-stack {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .tech-badge {
            background: #f0f0f0;
            border-radius: 50px;
            padding: 0.5rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 500;
            color: var(--dark);
        }
        
        footer {
            background: white;
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            color: var(--dark);
            opacity: 0.8;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .navbar-container {
                flex-direction: column;
                gap: 1rem;
            }
            
            .nav-links {
                width: 100%;
                justify-content: center;
                flex-wrap: wrap;
                gap: 1rem;
            }
            
            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-container">
            <a href="/" class="logo">
                <i class="fas fa-american-sign-language-interpreting"></i>
                ASL Learning
            </a>
            <ul class="nav-links">
                <li><a href="/" class="nav-link">Home</a></li>
                <li><a href="/learn" class="nav-link">Learn</a></li>
                <li><a href="/practice" class="nav-link">Practice</a></li>
                <li><a href="/about" class="nav-link active">About</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container page-transitions">
        <header>
            <h1>About ASL Recognition Platform</h1>
            <p class="subtitle">Learn about our project and technology</p>
        </header>
        
        <div class="card">
            <div class="card-header">Our Mission</div>
            <div class="card-body about-section">
                <p>The ASL Recognition Platform was created with the mission to make American Sign Language learning accessible to everyone. By leveraging modern technology, we aim to bridge communication gaps and promote inclusivity.</p>
                
                <p>This tool uses computer vision and machine learning to recognize hand gestures in real-time, providing immediate feedback to learners. Our goal is to help users develop ASL skills in an interactive and engaging way.</p>
                
                <h2>Technology Stack</h2>
                <p>Our platform is built using cutting-edge technologies:</p>
                
                <div class="tech-stack">
                    <div class="tech-badge">
                        <i class="fab fa-python"></i>
                        Python
                    </div>
                    <div class="tech-badge">
                        <i class="fas fa-flask"></i>
                        Flask
                    </div>
                    <div class="tech-badge">
                        <i class="fas fa-camera"></i>
                        OpenCV
                    </div>
                    <div class="tech-badge">
                        <i class="fas fa-brain"></i>
                        MediaPipe
                    </div>
                    <div class="tech-badge">
                        <i class="fas fa-code"></i>
                        JavaScript
                    </div>
                    <div class="tech-badge">
                        <i class="fab fa-html5"></i>
                        HTML5/CSS3
                    </div>
                </div>
                
                <h2>How It Works</h2>
                <p>The ASL Recognition system uses your webcam to capture hand movements. MediaPipe's hand tracking technology identifies key points on your hand, which are then processed by our machine learning classifier to recognize ASL signs. The recognized signs are displayed as text, allowing you to practice and learn in real-time.</p>
            </div>
        </div>
    </div>
    
    <footer>
        <p>&copy; 2023 ASL Recognition Platform | Created to help people learn American Sign Language</p>
    </footer>

    <script>
        // Active navigation highlight
        document.addEventListener('DOMContentLoaded', () => {
            const currentLocation = window.location.pathname;
            const navLinks = document.querySelectorAll('.nav-link');
            navLinks.forEach(link => {
                if (link.getAttribute('href') === currentLocation) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        });
    </script>
</body>
</html>
        ''')
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)