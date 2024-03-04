import cv2
import numpy as np

def player_detection_brazil(frame, players_brazil_detected):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([5, 25, 75])
    upper_yellow = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 300:  # Adjust area threshold
            player_id = (x, y, w, h)
            if player_id not in players_brazil_detected:
                player_boxes.append(player_id)
    return player_boxes

def player_detection_russia(frame, players_russia_detected):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([173, 152, 149])
    upper_red = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 500:  # Adjust area threshold
            player_id = (x, y, w, h)
            if player_id not in players_russia_detected:
                player_boxes.append(player_id)
    return player_boxes

def detect_ball(frame, prev_ball_position):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_ball_color = np.array([10, 80, 80])
    upper_ball_color = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_ball_color, upper_ball_color)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_position = None
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            ball_position = center
            break
    if ball_position and prev_ball_position is not None:
        cv2.line(frame, prev_ball_position, ball_position, (0, 255, 0), 2)
    prev_ball_position = ball_position
    return ball_position, prev_ball_position

video_path = "volleyball_match.mp4"
cap = cv2.VideoCapture(video_path)

player_color_brazil = (0, 255, 255)
player_color_russia = (0, 0, 255)

prev_ball_position = None
team1_count = 0
team2_count = 0

players_brazil_detected = set()
players_russia_detected = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    
    ball_position, prev_ball_position = detect_ball(frame_resized, prev_ball_position)

    players_brazil = player_detection_brazil(frame_resized, players_brazil_detected)
    for player_bbox in players_brazil:
        x, y, w, h = player_bbox
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), player_color_brazil, 2)
        team1_count += 1
        players_brazil_detected.add(player_bbox)

    players_russia = player_detection_russia(frame_resized, players_russia_detected)
    for player_bbox in players_russia:
        x, y, w, h = player_bbox
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), player_color_russia, 2)
        team2_count += 1
        players_russia_detected.add(player_bbox)

    # Display player counts on the video screen
    cv2.putText(frame_resized, f"Team 1 (Brazil) count: {team1_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame_resized, f"Team 2 (Russia) count: {team2_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Volleyball Match", frame_resized)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


