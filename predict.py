from ultralytics import YOLO
import os
import numpy as np
import cv2
import argparse
import json

ROOT_DIR = os.path.abspath("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (218, 170, 34), # Gold
    (0, 128, 128),  # Dark Cyan
    (176, 224, 230),# Light Blue
    (189, 183, 117),# Chartreuse
    (128, 128, 0),  # Olive
    (255, 192, 203),# Pink
    (0, 255, 0),    # Lime
    (255, 140, 0),  # Dark Orange
    (0, 0, 128),    # Navy Blue
    (255, 69, 0),   # Red-Orange
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (128, 0, 0),    # Brown
    (255, 255, 255),# White
    (192, 192, 192),# Light Gray
    (0, 0, 0),      # Black
    (70, 130, 180), # Steel Blue
    (255, 99, 71),  # Tomato
    (0, 128, 255),  # Royal Blue
    (255, 20, 147), # Deep Pink
    (255, 215, 0),  # Gold
    (0, 255, 128),  # Spring Green
    (139, 69, 19),  # Saddle Brown
    (205, 92, 92)   # Indian Red
]

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def solve_image(mode, image):
    result = model(image)
    boxes = result[0].boxes
    keypoints = result[0].keypoints

    xy = keypoints.xy.cpu().numpy()
    xy = [[[idx, float(kp[0]), float(kp[1])] for idx, kp in enumerate(kps) if not (kp[0] == 0 and kp[1] == 0)] for kps in xy]

    xyxy = boxes.xyxy.cpu().numpy().astype(float).tolist()

    score = boxes.conf.cpu().numpy().astype(float).tolist()

    cls_id = boxes.cls.cpu().numpy()
    names = result[0].names
    cls_name = [str(names[idx]) for idx in cls_id]

    return {
            'keypoints': xy,
            'bbox': xyxy,
            'score': score,
            'category': cls_name
        }
        
def solve_folder(source, render):
    for file in os.listdir(source):
        file_path = os.path.join(source, file)
        if file_path.endswith(('mp4', 'avi', 'jpg', 'png')):
            cap = cv2.VideoCapture(file_path) if file_path.endswith(('mp4', 'avi')) else None
            image = cv2.imread(file_path) if not cap else None

            rets = []
            if cap is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)

                if args.render:
                    video_path = os.path.join(OUTPUT_DIR, f'{file}-keypoints.mp4')
                    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                cnt = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    ret = solve_image(model, frame)
                    rets.append(ret)

                    if args.render:
                        for box in ret['bbox']:
                            cv2.rectangle(frame, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), 2)
                        for kps in ret['keypoints']:
                            for kp in kps:
                                cv2.circle(frame, tuple(map(int, kp[1:])), 3, colors[int(kp[0]) % len(colors)], -1)
                        out.write(frame)

                    print(f'Frame {cnt} finished')
                    cnt += 1

                json_path = os.path.join(OUTPUT_DIR, f'{file}-keypoints.json')
                with open(json_path, 'w') as f:
                    json.dump(rets, f)
                print(f'Save predictions to {json_path}')
                if args.render:
                    out.release()
                    print(f'Save rendered video to {video_path}')

            if image is not None:
                ret = solve_image(model, image)
                rets.append(ret)

                json_path = os.path.join(OUTPUT_DIR, f'{file}-keypoints.json')
                with open(json_path, 'w') as f:
                    json.dump(rets, f)
                print(f'Save predictions to {json_path}')

                if args.render:
                    for box in ret['bbox']:
                        cv2.rectangle(image, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), 2)
                    for kps in ret['keypoints']:
                        for kp in kps:
                            cv2.circle(image, tuple(map(int, kp[1:])), 3, colors[int(kp[0]) % len(colors)], -1)

                    img_path = os.path.join(OUTPUT_DIR, f'{file}-keypoints.jpg')
                    cv2.imwrite(img_path, image)
                    print(f'Save rendered image to {img_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Path of the source (can be image or video or folder)')
    parser.add_argument('--model', help='Path of model weights', required=True)
    parser.add_argument("--render", help='Save the detections to another image or video', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print('Model path not exist!')
        exit(0)
    # Load a model
    model = YOLO(args.model) # pretrained model

    # Load source
    source = args.source
    cap = None
    image = None
    
    if os.path.isdir(source):
        solve_folder(source, args.render)
    elif os.path.exists(source):
        if source.endswith(('mp4', 'avi')):
            cap = cv2.VideoCapture(source)
        elif source.endswith(('jpg', 'png', 'bmp')):
            image = cv2.imread(source)
        else:
            print('Bad file format.')
            exit(0)
    else:
        print('Source path not exists.')
        exit(0)

    rets = []
    if cap is not None:
        ext = os.path.splitext(os.path.basename(source))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)

        if args.render:
            video_path = os.path.join(OUTPUT_DIR,'{0}-keypoints.mp4'.format(ext[0]))
            out = cv2.VideoWriter(video_path, fourcc, fps, (width,height))

        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            ret = solve_image(model, frame)
            rets.append(ret)

            if args.render:
                for box in ret['bbox']:
                    cv2.rectangle(frame, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), 2)
                for kps in ret['keypoints']:
                    for kp in kps:
                        cv2.circle(frame, tuple(map(int, kp[1:])), 3, colors[int(kp[0])][::-1], -1)
                out.write(frame)

            print('Frame {} finished'.format(cnt))
            cnt += 1

        json_path = os.path.join(OUTPUT_DIR, f'{file}-keypoints.json')
        with open(json_path, 'w') as f:
            json.dump(rets, f)
        print(f'Save predictions to {json_path}')
        if args.render:
            out.release()
            print('Save rendered video to {}'.format(video_path))
    
    if image is not None:
        ext = os.path.splitext(os.path.basename(source))
    
        ret = solve_image(model, image)
        rets.append(ret)

        json_path = os.path.join(OUTPUT_DIR, f'{file}-keypoints.json')
        with open(json_path, 'w') as f:
            json.dump(rets, f)
        print(f'Save predictions to {json_path}')

        if args.render:
            for box in ret['bbox']:
                cv2.rectangle(image, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), 2)
            for kps in ret['keypoints']:
                for kp in kps:
                    if int(kp[0]) in [24, 25, 26, 27, 28, 29, 30, 31]:
                        cv2.circle(image, tuple(map(int, kp[1:])), 3, colors[int(kp[0])][::-1], -1)

            img_path = os.path.join(OUTPUT_DIR,'{0}-keypoints{1}'.format(ext[0], ext[1]))
            cv2.imwrite(img_path, image)
            print('Save rendered image to {}'.format(img_path))
