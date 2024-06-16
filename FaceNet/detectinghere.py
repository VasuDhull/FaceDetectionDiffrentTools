from facenet_pytorch import MTCNN
import cv2
from PIL import Image, ImageDraw
import numpy as np
device = 'cpu'

mtcnn = MTCNN(keep_all=True, device=device)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    boxes, _ = mtcnn.detect(frame_pil)

    frame_draw = frame_pil.copy()
    draw = ImageDraw.Draw(frame_draw)
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    
    frame_tracked = cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR)
    
    cv2.imshow("FACES", frame_tracked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

