from facenet_pytorch import MTCNN
import cv2
from PIL import Image, ImageDraw
import numpy as np

device = 'cpu'

mtcnn = MTCNN(keep_all=True, device=device)

image_path = '0a8b70166bc59ee8.jpg' 
frame = cv2.imread(image_path)

if frame is None:
    print("Failed to load image")
else:
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
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

