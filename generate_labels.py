import cv2 
import os
from detector import Detection, center_box
import pandas as pd

de = Detection()
path = "D:/User/data/image/Val"
labels = "D:/User/data/label/Val"
list_files = os.listdir(path)
for image in list_files:
    path_image = os.path.join(path,image)
    name_image = os.path.join(labels,image).replace('.png','.txt')


    image = cv2.imread(path_image)
    wt, hg = image.shape[1], image.shape[0]
    cls, box, image = de.detector_image(image)
    x,y,w,h = center_box(box[0],image)
    # print(cls[0],x,y,w,h)

    df = pd.DataFrame([[cls[0], x,y,w,h]])

    df.to_csv(name_image, sep=' ', index=False, header=False)
