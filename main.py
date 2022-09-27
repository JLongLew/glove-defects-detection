# Import libraries
import cv2
import numpy as np

# Fixed size for all the displayed screen
fixed_size = (540, 380)

# Accept for input
cap = cv2.VideoCapture('videos/general_sample1.mp4')

# Keep looping until the end of the video
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the video
    frame = cv2.resize(frame, fixed_size, fx=0, fy=0,
                       interpolation=cv2.INTER_CUBIC)
    # Convert the video to HSV format
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the original video frame
    cv2.imshow('Original video', frame)

    # Mask for detecting glove
    lower = np.array([85, 80, 97])
    upper = np.array([106, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Find Contours
    # findContours alters the image to show only the glove (blue color)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Detect the defect within the glove
    internal_cnt = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] >= 0]

    # Find glove
    if len(contours) > 0:
        blue_area = max(contours, key=cv2.contourArea)
        (xg, yg, wg, hg) = cv2.boundingRect(blue_area)

        # Draw rectangle for glove
        cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (255, 0, 0), 1)

        # Label the glove
        frame = cv2.putText(frame, 'Glove', (xg, yg - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Find defect
        if len(internal_cnt) > 0:
            for i in internal_cnt:

                # Check defect size
                area = cv2.contourArea(i)
                if area > 80:
                    (xd, yd, wd, hd) = cv2.boundingRect(i)

                    # Draw rectangle for defect
                    cv2.rectangle(frame, (xd, yd), (xd + wd, yd + hd), (0, 0, 255), 1)

                    # Crop the image for identifying dirt
                    crop_image = frame[yd:yd + hd, xd:xd + wd]

                    # Check whether the cropped defect is dirt or not
                    # Mask for detecting dirty defects
                    lower_dirt = np.array([104, 68, 0])
                    upper_dirt = np.array([117, 255, 90])
                    mask_dirt = cv2.inRange(crop_image, lower_dirt, upper_dirt)
                    contours_dirt = cv2.findContours(mask_dirt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                    # Label the defect
                    if len(contours_dirt) > 0:
                        # Defect Type: Dirt
                        frame = cv2.putText(frame, 'Dirt', (xd, yd - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    elif area > 900:
                        # Defect Type: Tearing
                        frame = cv2.putText(frame, 'Tearing', (xd, yd - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        # Defect Type: Hole
                        frame = cv2.putText(frame, 'Hole', (xd, yd - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Display the glove mask video
                cv2.imshow('Mask', mask)
                # Display the output video
                cv2.imshow('Output', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture object
cap.release()
# Close all the current windows
cv2.destroyAllWindows()
