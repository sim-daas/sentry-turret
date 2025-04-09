import cv2
from ultralytics import YOLO
import serial
import time
import math
from collections import defaultdict
import numpy as np


def get_face_boxes(frm, mdl, trk=False):
    if trk:
        res = mdl.track(frm, persist=True)
    else:
        res = mdl(frm)
    
    bxs = []
    ids = []
    
    for r in res:
        if r.boxes:
            for i, bx in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, bx.xyxy[0])
                cf = float(bx.conf[0])
                cl = int(bx.cls[0])
                lb = r.names[cl]
                
                tid = None
                if hasattr(bx, 'id') and bx.id is not None:
                    tid = int(bx.id[0])
                    ids.append(tid)
                
                bxs.append({
                    'coords': (x1, y1, x2, y2),
                    'conf': cf,
                    'cls': cl,
                    'label': lb,
                    'tid': tid
                })
    
    return bxs, ids


def select_face_to_track(frm, mdl):
    print("Select a face/object to track. Press SPACE to confirm selection or ESC to track first detected face.")
    
    bxs, _ = get_face_boxes(frm, mdl)
    if not bxs:
        print("No objects detected in first frame. Will track first object when detected.")
        return None
    
    sel_frm = frm.copy()
    sel_idx = 0
    
    def update_selection():
        tmp_frm = frm.copy()
        for i, bx in enumerate(bxs):
            x1, y1, x2, y2 = bx['coords']
            clr = (0, 0, 255) if i == sel_idx else (0, 255, 0)
            thk = 3 if i == sel_idx else 2
            cv2.rectangle(tmp_frm, (x1, y1), (x2, y2), clr, thk)
            
            lbl = f"{i+1}: {bx['label']} {bx['conf']:.2f}"
            cv2.putText(tmp_frm, lbl, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        
        inst = "Use arrow keys to select, SPACE to confirm, ESC to auto-select"
        cv2.putText(tmp_frm, inst, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return tmp_frm
    
    sel_frm = update_selection()
    cv2.imshow("Select Object to Track", sel_frm)
    
    while True:
        key = cv2.waitKey(0)
        
        if key == 27:
            sel_obj = bxs[0] if bxs else None
            break
            
        elif key == 32:
            sel_obj = bxs[sel_idx] if bxs else None
            break
            
        elif key == 83 or key == 100:
            sel_idx = (sel_idx + 1) % len(bxs)
            sel_frm = update_selection()
            cv2.imshow("Select Object to Track", sel_frm)
            
        elif key == 81 or key == 97:
            sel_idx = (sel_idx - 1) % len(bxs)
            sel_frm = update_selection()
            cv2.imshow("Select Object to Track", sel_frm)
    
    cv2.destroyWindow("Select Object to Track")
    
    if sel_obj:
        print(f"Selected object: {sel_obj['label']} (class {sel_obj['cls']})")
    
    return sel_obj


def logic(bxs, sel_obj, ids, prev_ang=None):
    global xth, yth
    
    if prev_ang is None:
        prev_ang = [90, 90]
    
    tgt_bx = None
    
    if sel_obj and sel_obj.get('tid') is not None:
        for bx in bxs:
            if bx.get('tid') == sel_obj['tid']:
                tgt_bx = bx
                break
    
    if not tgt_bx and bxs:
        tgt_bx = bxs[0]
    
    if not tgt_bx:
        return prev_ang
    
    if sel_obj:
        sel_obj.update(tgt_bx)
    
    x1, y1, x2, y2 = tgt_bx['coords']
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    if cx >= 320:
        cx -= 320
        ang = math.atan2(xth*cx, 320)
        np_ang = 90 + math.degrees(ang)
    else:
        cx = 320 - cx
        ang = math.atan2(xth*cx, 320)
        np_ang = 90 - math.degrees(ang)
    
    if cy >= 240:
        cy -= 240
        ang = math.atan2(yth*cy, 240)
        nt_ang = 90 + math.degrees(ang)
    else:
        cy = 240 - cy
        ang = math.atan2(yth*cy, 240)
        nt_ang = 90 - math.degrees(ang)
    
    np_ang = max(10, min(170, np_ang))
    nt_ang = max(10, min(170, nt_ang))
    
    sf = 0.3
    
    np_ang = prev_ang[0] + sf * (np_ang - prev_ang[0])
    nt_ang = prev_ang[1] + sf * (nt_ang - prev_ang[1])
    
    return [np_ang, nt_ang]


def draw_boxes(frm, bxs, hist):
    for bx in bxs:
        x1, y1, x2, y2 = bx['coords']
        lbl = f"{bx['label']} {bx['conf']:.2f}"
        
        clr = (0, 255, 0)
        if bx.get('tid') is not None:
            tid = bx['tid']
            c_id = tid * 5 % 256
            clr = (c_id, 255, 255 - c_id)
            
            lbl = f"ID:{tid} " + lbl
            
            if tid in hist and len(hist[tid]) > 1:
                pts = np.array(hist[tid], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frm, [pts], isClosed=False, color=(230, 230, 230), thickness=2)
        
        cv2.rectangle(frm, (x1, y1), (x2, y2), clr, 2)
        cv2.putText(frm, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, clr, 2)
    
    return frm


def setup_servo_connection(port='/dev/ttyACM0', baud=9600):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"Connected to Arduino on {port}")
        time.sleep(0.5)
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port:{e}")
        print("Make sure Arduino is connected and the port is correct.")
        return None


def send_servo_command(ser, angs):
    if not ser or not ser.is_open:
        print("Serial connection not available")
        return False
        
    try:
        p_ang, t_ang = angs
        
        if 10 <= p_ang <= 170 and 10 <= t_ang <= 170:
            cmd = f"{int(p_ang)},{int(t_ang)}\n" 
            ser.write(cmd.encode())
            time.sleep(0.001)
            return True
        else:
            print(f"Invalid angle values: Pan={p_ang}, Tilt={t_ang}. Must be between 10 and 170.")
            return False
    except Exception as e:
        print(f"Error sending command: {e}")
        return False


def main():
    global xth, yth
    
    xth = math.tan(math.radians(30))
    yth = math.tan(40)
    
    mdl = YOLO("yolov11n-face.pt")
    
    lst_cmd_t = 0
    cmd_intv = 0.001
    prev_ang = [90, 90]
    
    trk_hist = defaultdict(lambda: [])
    max_hist = 50
    
    fps = 0
    frm_cnt = 0
    start_t = time.time()
    fps_upd_intv = 0.5
    
    flip_frm = False
    
    srv_conn = setup_servo_connection()
    
    cap = cv2.VideoCapture('/dev/video0')
    ret, frst_frm = cap.read()
    if not ret:
        print("Failed to capture first frame from camera")
        return
    
    if flip_frm:
        frst_frm = cv2.flip(frst_frm, 1)
    
    sel_obj = select_face_to_track(frst_frm, mdl)
    
    trk_en = True
    
    while cap.isOpened():
        ret, frm = cap.read() 
        if not ret:
            break
        
        if flip_frm:
            frm = cv2.flip(frm, 1)
        
        frm_cnt += 1
        elap_t = time.time() - start_t
        
        if elap_t > fps_upd_intv:
            fps = frm_cnt / elap_t
            frm_cnt = 0
            start_t = time.time()
        
        bxs, ids = get_face_boxes(frm, mdl, trk=trk_en)
        
        for bx in bxs:
            if bx.get('tid') is not None:
                tid = bx['tid']
                x1, y1, x2, y2 = bx['coords']
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                trk_hist[tid].append((cx, cy))
                if len(trk_hist[tid]) > max_hist:
                    trk_hist[tid].pop(0)
        
        frm = draw_boxes(frm, bxs, trk_hist)
        
        cur_t = time.time()
        if bxs and (cur_t - lst_cmd_t) >= cmd_intv:
            angs = logic(bxs, sel_obj, ids, prev_ang)
            
            print(f"Sending angles: Pan={angs[0]:.1f}, Tilt={angs[1]:.1f}")
            send_servo_command(srv_conn, angs)
            prev_ang = angs
            lst_cmd_t = cur_t
        
        status = "Tracking: ON" if trk_en else "Tracking: OFF"
        cv2.putText(frm, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
        
        fps_txt = f"FPS: {fps:.1f}"
        fps_sz = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        fps_x = frm.shape[1] - fps_sz[0] - 10
        cv2.putText(frm, fps_txt, (fps_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2)
        
        cv2.imshow("Object Tracking", frm)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            trk_en = not trk_en
            print(f"Tracking {'enabled' if trk_en else 'disabled'}")
        elif key == ord('r'):
            sel_obj = None
            print("Reset selected object - will track first detected object")
        elif key == ord('f'):
            flip_frm = not flip_frm
            print(f"Frame flipping {'enabled' if flip_frm else 'disabled'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if srv_conn and srv_conn.is_open:
        srv_conn.close()
        print("Serial connection closed.")


if __name__ == "__main__":
    main()
