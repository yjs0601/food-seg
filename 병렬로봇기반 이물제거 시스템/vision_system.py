import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import serial
import time
import os
from ultralytics import YOLO

# ─────────────────────────────────────────
# 설정 (환경변수 or 기본값)
# ─────────────────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH",    "weights/best.pt")
TRACKER_PATH  = os.getenv("TRACKER_PATH",  "config/bytetrack.yaml")
COM_F103      = os.getenv("COM_F103",      "COM3")
COM_F429      = os.getenv("COM_F429",      "COM9")
WEBCAM_INDEX  = int(os.getenv("WEBCAM_INDEX", "2"))
SERVER_PORT   = int(os.getenv("SERVER_PORT",  "9999"))

# ─────────────────────────────────────────
# 모델 및 클래스 설정
# ─────────────────────────────────────────
model = YOLO(MODEL_PATH)
class_names = ['chips', 'trash']
colors = {'trash': (0, 0, 255), 'chips': (0, 0, 255)}
PIXEL_TO_MM = 200.0 / 216.54

# ─────────────────────────────────────────
# 상태 변수
# ─────────────────────────────────────────
current_zone = 'A'
current_id   = None
entered_ids  = []
used_ids     = set()
zone_coords  = []
saved_info   = {}


def next_zone(c):
    return chr((ord(c) - 65 + 1) % 26 + 65)


def send_labeled_jpeg(sock, label: bytes, image):
    result, encimg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not result:
        return False
    data = encimg.tobytes()
    try:
        sock.sendall(label)
        sock.sendall(struct.pack(">I", len(data)))
        sock.sendall(data)
        time.sleep(0.01)
        return True
    except Exception:
        return False


def get_color_name_from_bgr(bgr):
    b, g, r = bgr
    max_val = max(r, g, b)
    if max_val < 60:
        return "black"
    if r == max_val:
        return "red"
    elif g == max_val:
        return "green"
    elif b == max_val:
        return "blue"
    return "unknown"


def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline


def initialize_webcam():
    cam = cv2.VideoCapture(WEBCAM_INDEX)
    if not cam.isOpened():
        raise RuntimeError(f"웹캠({WEBCAM_INDEX})을 열 수 없습니다.")
    return cam


def apply_yolo_segmentation(roi_image):
    global saved_info
    results = model.track(
        roi_image, conf=0.4, iou=0.4,
        persist=True, tracker=TRACKER_PATH, verbose=False
    )[0]

    annotated = roi_image.copy()
    roi_h, roi_w = roi_image.shape[:2]
    centers = {}

    if results.boxes is not None and results.boxes.id is not None:
        boxes   = results.boxes.xywh.cpu().numpy()
        scores  = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        ids     = results.boxes.id.cpu().numpy().astype(int)

        for box, score, cls_id, tid in zip(boxes, scores, classes, ids):
            print(f"🧾 감지됨 ID {tid}, class: {class_names[cls_id]}, score: {score:.2f}")

            if class_names[cls_id] != 'trash':
                print(f"❌ ID {tid}는 trash가 아니므로 제외됨")
                continue
            if score < 0.05:
                print(f"❌ ID {tid}는 score {score:.2f} < 0.05로 제외됨")
                continue

            x, y, w, h = box
            cx = int(x)
            cy = int(y)
            dx_mm  = (cx - roi_w // 2) * PIXEL_TO_MM
            dy_mm  = (cy - roi_h // 2) * PIXEL_TO_MM
            robot_x = -dy_mm
            robot_y = -dx_mm

            if robot_x > 90:
                print(f"❌ ID {tid}는 로봇 좌표 기준 x={robot_x:.2f}mm → 제외됨")
                continue

            # 중복 제거: 이미 등록된 center들과 너무 가까우면 무시
            is_duplicate = any(
                ((robot_x - ox) ** 2 + (robot_y - oy) ** 2) ** 0.5 < 10
                for ox, oy in centers.values()
            )
            if is_duplicate:
                continue

            centers[tid] = (robot_x, robot_y)

            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            masked_roi  = cv2.bitwise_and(roi_image, roi_image, mask=mask)
            mean_color  = cv2.mean(masked_roi, mask)
            color_name  = get_color_name_from_bgr(mean_color[:3])

            saved_info[tid] = {
                'width':  int(w),
                'height': int(h),
                'color':  color_name,
                'conf':   score,
            }

            color = colors[class_names[cls_id]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.putText(annotated, class_names[cls_id], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return annotated, centers


def run_server_loop():
    global current_zone, current_id, entered_ids, used_ids, zone_coords, saved_info

    pipeline = initialize_realsense()
    cam      = initialize_webcam()
    CX, CY   = 320, 240
    HALF_SIZE   = 0.1
    fixed_depth = None
    lost_counter = 0

    ser_103 = serial.Serial(COM_F103, 9600, timeout=1)
    ser_429 = serial.Serial(COM_F429, 9600, timeout=1)
    time.sleep(2)

    while True:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', SERVER_PORT))
        server.listen(1)
        print("🔌 MFC 연결 대기 중...")
        sock, addr = server.accept()
        print(f"✅ 연결됨: {addr}")

        try:
            while True:
                frames      = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                ret, webcam_frame = cam.read()
                if not ret:
                    continue
                webcam_frame = cv2.flip(webcam_frame, 0)

                color_image  = np.asanyarray(color_frame.get_data())
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                if fixed_depth is None:
                    fixed_depth = depth_frame.get_distance(CX, CY)
                    if fixed_depth == 0:
                        continue

                center_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [CX, CY], fixed_depth)
                roi_3d_points = [
                    [center_point[0] - HALF_SIZE, center_point[1] - HALF_SIZE, center_point[2]],
                    [center_point[0] + HALF_SIZE, center_point[1] - HALF_SIZE, center_point[2]],
                    [center_point[0] + HALF_SIZE, center_point[1] + HALF_SIZE, center_point[2]],
                    [center_point[0] - HALF_SIZE, center_point[1] + HALF_SIZE, center_point[2]],
                ]
                roi_pixel_points = [
                    tuple(map(int, rs.rs2_project_point_to_pixel(depth_intrin, pt)))
                    for pt in roi_3d_points
                ]
                xs, ys = zip(*roi_pixel_points)
                x_min, x_max = max(min(xs), 0), min(max(xs), 639)
                y_min, y_max = max(min(ys), 0), min(max(ys), 479)
                roi_image = color_image[y_min:y_max, x_min:x_max].copy()

                roi_segmented, centers = apply_yolo_segmentation(roi_image)
                current_ids   = set(centers.keys())
                available_ids = current_ids - used_ids

                if current_id is None and available_ids:
                    current_id  = list(available_ids)[0]
                    entered_ids = [current_id]
                    zone_coords = [centers[current_id]]
                    while len(zone_coords) < 3:
                        zone_coords.append(("n", "n"))

                    msg = f"{current_zone};" + "|".join(
                        f"{x/1000:.4f},{y/1000:.4f}" if isinstance(x, float) else "n,n"
                        for x, y in zone_coords
                    )
                    ser_103.write(current_zone.encode())
                    ser_429.write((msg + '\n').encode())
                    print(f"🟢 기준 ID {current_id} 진입 → 구역 {current_zone} 시작")
                    print(f"📨 F103 전송: {current_zone}")
                    print(f"📤 F429 전송: {msg}")
                    lost_counter = 0

                elif current_id:
                    new_ids = (current_ids - set(entered_ids)) - used_ids
                    if new_ids:
                        all_valid_ids = list({current_id} | {
                            tid for tid in current_ids if tid not in used_ids
                        })[:3]
                        entered_ids = all_valid_ids
                        zone_coords = [centers[tid] for tid in entered_ids if tid in centers]
                        while len(zone_coords) < 3:
                            zone_coords.append(("n", "n"))

                        msg = f"{current_zone};" + "|".join(
                            f"{x/1000:.4f},{y/1000:.4f}" if isinstance(x, float) else "n,n"
                            for x, y in zone_coords
                        )
                        ser_103.write(current_zone.encode())
                        ser_429.write((msg + '\n').encode())
                        print(f"📨 F103 전송 (신규 trash 진입): {current_zone}")
                        print(f"📤 F429 전송 (업데이트): {msg}")

                    if current_id not in current_ids:
                        lost_counter += 1
                        print(f"⚠️ 기준 ID {current_id} 감지되지 않음 (시도 {lost_counter}/5)")

                        if zone_coords and isinstance(zone_coords[0][0], float):
                            old_x, old_y = zone_coords[0]
                            recovered = False
                            for tid, (x, y) in centers.items():
                                if tid in used_ids or tid == current_id:
                                    continue
                                if abs(x - old_x) < 5.0:
                                    print(f"✅ 기준 ID {current_id} 회복됨 (ID {tid}와 위치 유사)")
                                    lost_counter = 0
                                    recovered = True
                                    break
                            if not recovered and lost_counter < 5:
                                continue

                        if lost_counter >= 5:
                            print(f"⛔ 기준 ID {current_id} 완전 이탈 → 구역 종료")
                            for i, tid in enumerate(entered_ids):
                                if (i < len(zone_coords)
                                        and isinstance(zone_coords[i][0], float)
                                        and tid in saved_info):
                                    x, y = zone_coords[i]
                                    info = saved_info[tid]
                                    mfc_info = (
                                        f"({int(x)},{int(y)})"
                                        f"|{info.get('width', 0)}"
                                        f"|{info.get('height', 0)}"
                                        f"|{info.get('color', 'unknown')}"
                                        f"|{info.get('conf', 0.0):.2f}"
                                    )
                                    try:
                                        sock.sendall(b'DATA')
                                        sock.sendall(struct.pack(">I", len(mfc_info.encode())))
                                        sock.sendall(mfc_info.encode())
                                        print(f"📤 MFC 전송: {mfc_info}")
                                    except Exception as e:
                                        print(f"❌ MFC 전송 실패: {e}")

                            used_ids |= set(entered_ids)
                            current_id = None
                            entered_ids.clear()
                            zone_coords.clear()
                            current_zone = next_zone(current_zone)
                            lost_counter = 0
                    else:
                        lost_counter = 0

                if not send_labeled_jpeg(sock, b'WEBC', webcam_frame):
                    continue
                if not send_labeled_jpeg(sock, b'YOLO', roi_segmented):
                    continue

        finally:
            sock.close()
            server.close()
            ser_103.close()
            ser_429.close()
            cam.release()
            pipeline.stop()


if __name__ == "__main__":
    run_server_loop()
