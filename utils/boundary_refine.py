import numpy as np
import cv2
from put_color import put_cityscapes_color_to_img


def find_max_color_local(image, index_list):
    """
    :param image: gray image
    :param index_list: the region that need to count the colors
    :return: the colors account for most biggest region
    """
    color_count = np.zeros([20])
    for point in index_list:
        x, y = index_list[point]
        pixel_color = image[x, y]
        color_count[pixel_color] += 1

    color_count.sort()
    return color_count[0]


def fill_area_color(mask, index_list, color):
    """
    fill the area with the given color
    :param mask: the output mask that contains the refined area
    :param index_list: the area need to be refined (the destination area)
    :param color: the color to fill the area
    :return: the mask
    """
    for point in index_list:
        x, y = index_list[point]
        mask[x, y] = color

    return mask


def boundary_refine(image, area):
    """
    find the most color and fill the area with this color
    :param image: the origin feature
    :param area:
        the area that need to be filled with color
        found by the region grow algorithm on boundary image
    :return: the new mask
    """
    mask = np.zeros(image.shape, dtype=np.uint8)
    # print(mask)
    color = find_max_color_local(image, area)
    mask = fill_area_color(mask, area, color)

    return mask


if __name__ == '__main__':
    gray_feature = cv2.imread("../aachen_000000_000019_gtFine_labelIds.png", 0)

    index = np.random.randint(0, 66, (3000, 2))

    origin_color_feature = put_cityscapes_color_to_img(gray_feature)


    refined_img = boundary_refine(gray_feature, index)
    print(refined_img)


    refined_img_color = put_cityscapes_color_to_img(refined_img)

    # show_img = np.concatenate([origin_color_feature, refined_img], axis=1)
