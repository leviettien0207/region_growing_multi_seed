import numpy as np
from PIL import Image as PILImage


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    palette = [0] * (num_cls * 3)
    palette[0:3] = (128, 64, 128)       # 0: 'road'
    palette[3:6] = (244, 35,232)        # 1 'sidewalk'
    palette[6:9] = (70, 70, 70)         # 2''building'
    palette[9:12] = (102,102,156)       # 3 wall
    palette[12:15] =  (190,153,153)     # 4 fence
    palette[15:18] = (153,153,153)      # 5 pole
    palette[18:21] = (250,170, 30)      # 6 'traffic light'
    palette[21:24] = (220,220, 0)       # 7 'traffic sign'
    palette[24:27] = (107,142, 35)      # 8 'vegetation'
    palette[27:30] = (152,251,152)      # 9 'terrain'
    palette[30:33] = ( 70,130,180)      # 10 sky
    palette[33:36] = (220, 20, 60)      # 11 person
    palette[36:39] = (255, 0, 0)        # 12 rider
    palette[39:42] = (0, 0, 142)        # 13 car
    palette[42:45] = (0, 0, 70)         # 14 truck
    palette[45:48] = (0, 60,100)        # 15 bus
    palette[48:51] = (0, 80,100)        # 16 train
    palette[51:54] = (0, 0,230)         # 17 'motorcycle'
    palette[54:57] = (119, 11, 32)      # 18 'bicycle'
    palette[57:60] = (105, 105, 105)
    return palette


def put_cityscapes_color_to_img(img):
    """
    put cityscapes color to the input gray image
    :param img: gray img
    :return: 3 channel RGB color image
    """
    palette = get_palette(20)

    image = PILImage.fromarray(img)
    # image = image.convert('P')
    # print(image.mode)
    image.putpalette(palette)
    image.show()
    # exit()

    return image
