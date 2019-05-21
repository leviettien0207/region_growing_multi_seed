import numpy as np
import cv2


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


# get the difference of gray value in image
def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


# get four or eight neighbors
def selectConnects(neighbor_num):
    if neighbor_num == 8:  # eight neighbors
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    elif neighbor_num == 4:  # four neighbors
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    else:
        raise ValueError("The neighbor_num should be 4 or 8")
    return connects


def regionGrow(img, mask, seed, thresh, neighbor_num=8, label=1):
    """ single seed region grow algorithm """
    height, weight = img.shape

    connects = selectConnects(neighbor_num)

    seedList = []
    seedList.append(seed)

    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        mask[currentPoint.x, currentPoint.y] = label
        for i in range(neighbor_num):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight or mask[tmpX, tmpY] != 0:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and mask[tmpX, tmpY] == 0:
                mask[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return mask


def find_undetermined(mask):
    zero_pos = np.where(mask == 0)
    # print(zero_pos)
    if len(zero_pos[0]) == 0:
        return None
    x = zero_pos[0][0]
    y = zero_pos[1][0]
    # print(x)
    # print(y)
    return Point(x, y)


def img_region_grow(img, label_in):
    """ gray image region grow algorithm """
    mask = np.zeros(img.shape)
    thresh = 100
    label = label_in

    while True:
        seed = find_undetermined(mask)
        if seed is not None:
            mask = regionGrow(img, mask, seed, thresh, neighbor_num=8, label=label)
            # np.set_printoptions(threshold=np.inf)

            label += 10
            # cv2.imshow(' ', mask)
            # cv2.waitKey(0)
        else:
            print("Process Done!")
            break


if __name__ == '__main__':
    img = cv2.imread('../aachen_000000_000019_gtFine_edgemap.png', 0)  # read in gray image mode
    img_region_grow(img, 1)

