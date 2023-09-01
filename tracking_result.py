import cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Video  path for experiment
    for i in range(0, 56):
        # import our video
        path = 'fish_video/'+str(i)+'.mov' 

        # loading a YOLO model
        model = YOLO('best.pt')

        # -------------------------------------------------------
        # Reading video with cv2
        video = cv2.VideoCapture(path)
        frames_list = []

        total = 0

        # Original information of video
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = video.get(cv2.CAP_PROP_FPS)
        print('[INFO] - Original Dim: ', (width, height))

        # -------------------------------------------------------
        # Video output
        video_name = 'result_video/'+str(i)+'.mov'
        output_path = video_name

        output_video = cv2.VideoWriter(output_path,
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps, (width, height))

        # -------------------------------------------------------
        # Executing Recognition

        first_appear, last_pos = {}, {}
        center = int(width / 2)
        for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
            total = 0
            # reading frame from video
            tf, frame = video.read()
            if not tf:
                break

            # tracking
            results = model.track(frame, persist=True)
            # if there is no fish in the frame then it will pass this frame
            if results[0].boxes.id == None:
                continue
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy().astype(float)

            cv2.line(frame, (center, 0), (center, height), (255, 255, 0), 3)

            for box, id, conf in zip(boxes, ids, confs):
                if float(conf) < 0.3:
                    continue
                xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                center_x, center_y = int((xmax + xmin) / 2), int((ymax + ymin) / 2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)  # box
                cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)  # center of box

                cv2.putText(img=frame, text='#' + str(id) + ' fish - ' + str(np.round(conf, 2)),
                            org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),
                            thickness=2)
                total += 1

                if first_appear.get(id) is None:
                    first_appear[id] = center_x
                else:
                    last_pos[id] = center_x

            # The number of the fishes
            cv2.putText(img=frame, text=f'TOTAL:{total}',
                        org=(10, int(height * 1 / 8)), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1.5, color=(0, 0, 0), thickness=2)
            
            # Saving frames in a list
            cv2.imshow('detect_result', frame)
            cv2.waitKey(1)
            frames_list.append(frame)
            # saving transformed frames in a output video formaat
            output_video.write(frame)

        right2left, left2right = 0, 0
        r2l_record, l2r_record = [], []
        for key in last_pos:
            last_p = last_pos[key]
            first_p = first_appear[key]
            if first_p < center < last_p:
                left2right += 1
                l2r_record.append(key)
            elif first_p > center > last_p:
                right2left += 1
                r2l_record.append(key)

        print('left to right: ', left2right)
        print(l2r_record)
        print('right to left: ', right2left)
        print(r2l_record)

        # Releasing the video
        output_video.release()

