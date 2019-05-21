# region_growing_multi_seed  全图区域生长算法
PS中的魔棒工具, 用来方便的抠图

![](https://helpx.adobe.com/content/dam/help/en/photoshop/using/making-quick-selections/_jcr_content/main-pars/image/v2_select-subject-in-action.jpg)
###### *This image is borrowed from the official adobe website.*

Region growing algorithm is the magic wand tool in Photoshop. 

Multi seed version of region growing algorithm to segment each part of the boundary image, and give each part a different label.

## Pre-requirement
+ numpy
+ opencv-python

## How to use:
open region_grow.py, change the path to image in the bottom main function.

```
python region_grow.py
```
And then, different segmentation part will have different labels.

## Reference:
+ [https://blog.csdn.net/qq_38784098/article/details/82143117](https://blog.csdn.net/qq_38784098/article/details/82143117)
