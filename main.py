import cv2
import numpy as np

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('add.mp4')
fps = cam.get(cv2.CAP_PROP_FPS)
tracker = cv2.TrackerKCF_create()
tracked_cars = []

ret, frame = cam.read()
# frame = cv2.resize(frame, (1020, 500))
prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(frame)
mask[...,1] = 255
id = 1
while True:
    ret, frame2 = cam.read()
    # frame2 = cv2.resize(frame, (1020, 500))
    if not ret:
        break

    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow)
    
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # print(flow)
    mgtude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    print(mgtude)
    threshold = 10
    msk2 = mgtude > threshold

    contours, _ = cv2.findContours(msk2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print("HII", x,y,w,h)
        new_car = True
        for car in tracked_cars:
            if (x < car['x'] + car['w'] and x + w > car['x'] and y < car['y'] + car['h'] and y + h > car['y']):
                new_car = False
                success, bbox = car['tracker'].update(frame2)
                if success and bbox[2] > 0 and bbox[3] > 0:
                    avg_speed = np.mean(mgtude[y:y + h, x:x + w])
                    car['speed'] = avg_speed
                    cv2.rectangle(frame2, (int(bbox[0]), int(bbox[1])),(int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame2, f" Speed: {int(avg_speed)}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                break
        if new_car:
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame2, (x, y, w, h))
            tracked_cars.append({'tracker': tracker, 'x': x, 'y': y, 'w': w, 'h': h, 'speed': 0})

    # heatmap = cv2.applyColorMap(np.uint8(255 * magnitude / np.max(magnitude)), cv2.COLORMAP_HOT)
    # congestion_threshold = 30
    # congested_mask = magnitude > congestion_threshold
    # contourscong, _ = cv2.findContours(congested_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contourscong:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     print("Conf", x,y,w,h)
    #     cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(frame2, f"high congestion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Lane change
    for car in tracked_cars:
        if 'prev_x' in car:
            print("OK")
            delta_x = car['x'] - car['prev_x']
            print(delta_x)
            lane_change_threshold = 10
            if abs(delta_x) > lane_change_threshold:
                print("LANE change")
                cv2.putText(frame2, "Lane Change", (int(car['x']), int(car['y']) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        car['prev_x'] = car['x']


    total_avg_speed = np.mean(mgtude[msk2])
    print("Speed", total_avg_speed)
    mask[...,0] = angle * 180 /np.pi/2
    mask[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    # high_traffic = cv2.bitwise_and(frame2, frame2, mask=high_traffic_mask)
    frame2 = cv2.add(rgb, frame2)
    cv2.putText(frame2, f"Average Traffic speed: {total_avg_speed}", (501, 54), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
    # overlay = cv2.addWeighted(frame2, 0.7, heatmap, 0.3, 0)
    cv2.imshow("flow", frame2)
    # cv2.imshow("heatmap", overlay)
    # cv2.imshow("traffic", high_traffic)
    prvs = next_frame

    if cv2.waitKey(int(1000 / fps))&0xFF==ord('q'):
    # if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

