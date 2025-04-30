import cv2
import os
import face_recognition
import numpy as np
import pickle
import time

def main():
    db_dir = "face_db"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    emb_file = os.path.join(db_dir, "embeddings.pkl")
    known_embs = {}
    
    if os.path.exists(emb_file):
        with open(emb_file, "rb") as f:
            known_embs = pickle.load(f)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        name = input("Enter person's name (or 'q' to quit): ")
        if name.lower() == 'q':
            break
        
        print(f"Recording face for {name}. Position face in frame and press SPACE")
        imgs = []
        
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            
            face_locs = face_recognition.face_locations(frm)
            
            for (t, r, b, l) in face_locs:
                cv2.rectangle(frm, (l, t), (r, b), (0, 255, 0), 2)
            
            cv2.putText(frm, f"Press SPACE to capture, ESC to finish for {name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Record Face", frm)
            
            k = cv2.waitKey(1) & 0xFF
            if k == 32 and len(face_locs) == 1:  # SPACE and exactly one face
                print("Face captured!")
                imgs.append(frm)
                time.sleep(0.5)  # Prevent duplicate captures
            elif k == 27:  # ESC
                break
        
        if imgs:
            all_embs = []
            for idx, img in enumerate(imgs):
                # Check if face_encodings finds a face before accessing index 0
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    emb = encodings[0]
                    all_embs.append(emb)
                else:
                    print(f"Warning: Could not generate embedding for image {idx+1}. Skipping.")
            
            # Only proceed if we successfully generated embeddings
            if all_embs:
                avg_emb = np.mean(all_embs, axis=0)
                
                if name in known_embs:
                    known_embs[name].append(avg_emb)
                else:
                    known_embs[name] = [avg_emb]
                
                with open(emb_file, "wb") as f:
                    pickle.dump(known_embs, f)
                
                print(f"Saved {len(all_embs)} valid embeddings for {name}")
            else:
                print(f"No valid embeddings generated for {name}. Nothing saved.")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Face recording complete")

if __name__ == "__main__":
    main()
