import cv2
import matplotlib.pyplot as plt


def plot_show(im: cv2.typing.MatLike):
    plt.imshow(im)
    plt.show()
    plt.axis('off')