import KalmanFilter
import objTracking
import Detector
import cv2

kalman = KalmanFilter.KalmanFilter(
    dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1
    )
prev_cord = []
cap = cv2.VideoCapture('2D_Kalman-Filter_TP1/video/randomball.avi')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('kalman_output.avi', fourcc, fps, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
        
    centers = Detector.detect(frame)
    
    if len(centers) > 0:
        center = centers[0]
        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0,255,0), -1)
        
        kalman.predict()
        kalman.update(center)
        
        predicted_x = int(kalman.x_k[0, 0])
        predicted_y = int(kalman.x_k[1, 0])
        cv2.rectangle(frame, (predicted_x-5, predicted_y-5), (predicted_x+5, predicted_y+5), (255,0,0), 2)

        estimated_x = int(kalman.x_k[0, 0])
        estimated_y = int(kalman.x_k[1, 0])
        cv2.rectangle(frame, (estimated_x-10, estimated_y-10), (estimated_x+10, estimated_y+10), (0,0,255), 2)
        
        prev_cord.append((estimated_x, estimated_y))
        
        if len(prev_cord) > 100:
            prev_cord.pop(0)

        for i in range(len(prev_cord)-1, 0, -1):
            alpha = i / 20
            overlay = frame.copy()
            cv2.line(overlay, prev_cord[i-1], prev_cord[i], (0, 0, 255), 2)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    out.write(frame)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as kalman_output.avi")