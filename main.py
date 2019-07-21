import sys
import cv2
import paintImage

# BRUSH_HEAD_RADII = [1, 2 ,3, 5]#, 8, 13, 21, 34, 55, 89, 144]
# BRUSH_HEAD_RADII = [5, 3, 2, 1]  # , 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
# BRUSH_HEAD_RADII = [1, 2, 3, 2, 1]  # , 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
BRUSH_HEAD_RADII = [1, 2, 3, 5, 8, 13, 21, 34]


def runPaint(image, brush_radii, paintedImageName='paintedImage.png'):
    cv2.imwrite(paintedImageName, paintImage.paint(image, brush_radii))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        image = cv2.imread(sys.argv[1])
        print'Image name is: ', sys.argv[1]
        runPaint(image, BRUSH_HEAD_RADII, sys.argv[2])
    elif len(sys.argv) == 2:
        image = cv2.imread(sys.argv[1])
        print'Image name is: ', sys.argv[1]
        runPaint(image, BRUSH_HEAD_RADII)
