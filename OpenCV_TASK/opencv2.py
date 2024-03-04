import cv2
import numpy as np

# Define default HSV color ranges
lower_yellow_default = np.array([10, 80, 80])
upper_yellow_default = np.array([38, 225, 255])

lower_red_default = np.array([173, 152, 149])
upper_red_default = np.array([179, 255, 255])

# Create a window to display trackbars
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)

# Create trackbars for adjusting HSV color ranges
def nothing(x):
    pass

# For Brazil team (yellow jerseys)
cv2.createTrackbar('Lower H', 'Trackbars', lower_yellow_default[0], 179, nothing)
cv2.createTrackbar('Lower S', 'Trackbars', lower_yellow_default[1], 255, nothing)
cv2.createTrackbar('Lower V', 'Trackbars', lower_yellow_default[2], 255, nothing)
cv2.createTrackbar('Upper H', 'Trackbars', upper_yellow_default[0], 179, nothing)
cv2.createTrackbar('Upper S', 'Trackbars', upper_yellow_default[1], 255, nothing)
cv2.createTrackbar('Upper V', 'Trackbars', upper_yellow_default[2], 255, nothing)

# For Russia team (red jerseys)
cv2.createTrackbar('Lower H (Russia)', 'Trackbars', lower_red_default[0], 179, nothing)
cv2.createTrackbar('Lower S (Russia)', 'Trackbars', lower_red_default[1], 255, nothing)
cv2.createTrackbar('Lower V (Russia)', 'Trackbars', lower_red_default[2], 255, nothing)
cv2.createTrackbar('Upper H (Russia)', 'Trackbars', upper_red_default[0], 179, nothing)
cv2.createTrackbar('Upper S (Russia)', 'Trackbars', upper_red_default[1], 255, nothing)
cv2.createTrackbar('Upper V (Russia)', 'Trackbars', upper_red_default[2], 255, nothing)

def player_detection_brazil(frame, lower_yellow, upper_yellow):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only yellow regions
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    player_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 350:  # Adjust area threshold
            player_boxes.append((x, y, w, h))

    return player_boxes

def player_detection_russia(frame, lower_red, upper_red):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only red regions
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    player_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 500:  # Adjust area threshold
            player_boxes.append((x, y, w, h))

    return player_boxes

def detect_ball(frame, lower_ball_color, upper_ball_color):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only regions similar to the ball color
    mask = cv2.inRange(hsv, lower_ball_color, upper_ball_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    ball_position = None
    for contour in contours:
        if cv2.contourArea(contour) > 20:  # Adjust area threshold
            # Compute the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate the centroid of the contour
            ball_position = (int(x + w / 2), int(y + h / 2))
            break  # Assuming only the largest contour corresponds to the ball

    return ball_position

# Load the image
image_path = "ball.jpg"
frame = cv2.imread(image_path)

while True:
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Get trackbar positions
    lower_yellow = np.array([cv2.getTrackbarPos('Lower H', 'Trackbars'),
                             cv2.getTrackbarPos('Lower S', 'Trackbars'),
                             cv2.getTrackbarPos('Lower V', 'Trackbars')])
    upper_yellow = np.array([cv2.getTrackbarPos('Upper H', 'Trackbars'),
                             cv2.getTrackbarPos('Upper S', 'Trackbars'),
                             cv2.getTrackbarPos('Upper V', 'Trackbars')])
    
    lower_red = np.array([cv2.getTrackbarPos('Lower H (Russia)', 'Trackbars'),
                          cv2.getTrackbarPos('Lower S (Russia)', 'Trackbars'),
                          cv2.getTrackbarPos('Lower V (Russia)', 'Trackbars')])
    upper_red = np.array([cv2.getTrackbarPos('Upper H (Russia)', 'Trackbars'),
                          cv2.getTrackbarPos('Upper S (Russia)', 'Trackbars'),
                          cv2.getTrackbarPos('Upper V (Russia)', 'Trackbars')])

    # Detect the ball
    ball_position = detect_ball(frame_resized, lower_yellow, upper_yellow)

    # Detect players from the Brazil team
    players_brazil = player_detection_brazil(frame_resized, lower_yellow, upper_yellow)

    # Detect players from the Russia team
    players_russia = player_detection_russia(frame_resized, lower_red, upper_red)

    # Draw rectangles around players and ball
    frame_resized_with_boxes = frame_resized.copy()  # Make a copy to draw on
    
    for player_bbox in players_brazil:
        x, y, w, h = player_bbox
        cv2.rectangle(frame_resized_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow color

    for player_bbox in players_russia:
        x, y, w, h = player_bbox
        cv2.rectangle(frame_resized_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color

    if ball_position:
        cv2.circle(frame_resized_with_boxes, ball_position, 5, (0, 255, 255), -1)  # Yellow color

    cv2.imshow("Volleyball Match", frame_resized_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
