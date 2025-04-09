import cv2
import time
from ultralytics import YOLO
import numpy as np
import psutil
import os

def main():
    mdl = YOLO("yolov11n-face.pt")
    
    mdl.conf = 0.5
    mdl.iou = 0.45
    mdl.max_det = 20
    
    thr_cnt = os.cpu_count()
    if thr_cnt:
        try:
            import torch
            torch.set_num_threads(thr_cnt)
            print(f"Torch using {thr_cnt} threads")
        except ImportError:
            pass
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    cv2.namedWindow("YOLOv11 Face Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv11 Face Detection", 1280, 720)
    
    fps = 0
    frm_cnt = 0
    st_tm = time.time()
    fps_updint = 0.5
    fps_hist = []
    
    scl_fac = 1.0
    
    cpu_pct = 0
    lst_resc = 0
    resc_int = 1.0
    
    print("Starting detection loop with performance monitoring...")
    print("Press 'q' to quit, '+'/'-' to adjust processing scale")
    
    while cap.isOpened():
        cur_tm = time.time()
        if cur_tm - lst_resc >= resc_int:
            cpu_pct = psutil.cpu_percent()
            lst_resc = cur_tm
        
        t1 = time.time()
        suc, frm = cap.read()
        cap_tm = (time.time() - t1) * 1000
        
        if not suc:
            print("Failed to read from camera")
            break
        
        frm_cnt += 1
        elp_tm = cur_tm - st_tm
        
        orig_h, orig_w = frm.shape[:2]
        disp_frm = frm.copy()
        
        if scl_fac != 1.0:
            h, w = frm.shape[:2]
            proc_w, proc_h = int(w * scl_fac), int(h * scl_fac)
            proc_frm = cv2.resize(frm, (proc_w, proc_h))
        else:
            proc_frm = frm
        
        st_inf = time.time()
        res = mdl(proc_frm)
        inf_tm = (time.time() - st_inf) * 1000
        
        if elp_tm > fps_updint:
            fps = frm_cnt / elp_tm
            frm_cnt = 0
            st_tm = cur_tm
        
        for r in res:
            bxs = r.boxes
            for bx in bxs:
                x1, y1, x2, y2 = map(int, bx.xyxy[0])
                if scl_fac != 1.0:
                    x1 = int(x1 / scl_fac)
                    y1 = int(y1 / scl_fac)
                    x2 = int(x2 / scl_fac)
                    y2 = int(y2 / scl_fac)
                
                cf = float(bx.conf[0])
                cl = int(bx.cls[0])
                lb = r.names[cl]
                
                cv2.rectangle(disp_frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp_frm, f"{lb} {cf:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
        
        fps_txt = f"FPS: {fps:.1f}"
        inf_txt = f"Inference: {inf_tm:.1f}ms"
        scl_txt = f"Scale: {scl_fac:.2f}x"
        
        cap_txt = f"Capture: {cap_tm:.1f}ms"
        cpu_txt = f"CPU: {cpu_pct}%"
        
        if elp_tm > fps_updint:
            fps_hist.append(fps)
            if len(fps_hist) > 60:
                fps_hist.pop(0)
            
            if len(fps_hist) > 5:
                fps_std = np.std(fps_hist[-5:])
                fps_var_txt = f"FPS Var: {fps_std:.1f}"
                cv2.putText(disp_frm, fps_var_txt, (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(disp_frm, fps_txt, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp_frm, inf_txt, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp_frm, cap_txt, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp_frm, cpu_txt, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp_frm, scl_txt, (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("YOLOv11 Face Detection", disp_frm)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            scl_fac = min(1.0, scl_fac + 0.1)
            print(f"Scale factor: {scl_fac:.1f}")
        elif key == ord('-') or key == ord('_'):
            scl_fac = max(0.1, scl_fac - 0.1)
            print(f"Scale factor: {scl_fac:.1f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection finished")

if __name__ == "__main__":
    main()
