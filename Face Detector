from tensorflow.keras.models import load_model
m1=load_model("face.h5")
import cv2
import numpy as np
import matplotlib.pyplot as plt
im1=cv2.imread("me.png")

def predict(__):
    vl= m1.predict(np.array(cv2.resize(__,(64,64))).reshape((1,64,64,3)))
    if vl[0][0]>vl[0][1]:
        return "FACE",vl
    else:
        return "NOT FACE",vl
plt.imshow(im1)
plt.show()
print(predict(im1))

