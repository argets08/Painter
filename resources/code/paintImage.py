import numpy as np
import cv2

BLUR_FACTOR = 2
GRID_SIZE_FG = 3
T = 50  # Threshold
MIN_STROKE_LENGTH = 1
MAX_STROKE_LENGTH = 13
CURVATURE_FILTER = 0.3



def paint(image_array, brush_head_radii):
    # https://mrl.nyu.edu/publications/painterly98/hertzmann-siggraph98.pdf
    # https://stackoverflow.com/questions/9404967/taking-the-floor-of-a-float

    # image_array = np.array(image)
    # canvas = np.zeros(image_array.size)
    canvas = np.zeros(image_array.shape)
    print image_array.shape, 'image shape'
    canvas.fill(-1)

    for radius in brush_head_radii:
        # https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        # https://stackoverflow.com/questions/27527440/opencv-error-assertion-failed-ksize-width-for-gaussianblur
        # http://answers.opencv.org/question/34314/opencv-error-failed-assertion/
        blur = BLUR_FACTOR * radius
        blur_val = int(blur)
        if blur_val % 2 == 0:
            blur_val += 1
        referenceImage = cv2.GaussianBlur(image_array, (blur_val, blur_val), 0.4,0)
        # referenceImage = cv2.GaussianBlur(image_array, (blur_val, blur_val), 0, 0)
        paintLayer(canvas, referenceImage, radius)
    return canvas


def paintLayer(canvas, referenceImage, Ri):
    # https://mrl.nyu.edu/publications/painterly98/hertzmann-siggraph98.pdf

    # a new set of strokes, initially empty
    S = []
    # create a pointwise difference image
    img_diff = difference(canvas, referenceImage)
    img_diff = np.squeeze(img_diff)
    step_size = int(GRID_SIZE_FG * Ri)

    for x in xrange(0, canvas.shape[1], step_size):
        for y in xrange(0, canvas.shape[0], step_size):
            # sum the error near (x,y)
            x_region_min = max(int(x - (step_size / 2)), 0)
            if (x_region_min > canvas.shape[1]):
                x_region_min = canvas.shape[1] - 1
            x_region_max = min(int(x + (step_size / 2)), canvas.shape[1] - 1)

            # print x, 'x'
            # print float(step_size) / 2, "step size by 2 -x"
            # print x_region_min, 'x_region_min'
            # print x_region_max, 'x_region_max'
            # print (canvas.shape[1] - 1), 'Canvas size x-1-43'
            # x_region_max = max(int(x + canvas.shape[0] - 1), 0)

            y_region_min = max(int(y - (step_size / 2)), 0)

            if (y_region_min > canvas.shape[0]):
                y_region_min = canvas.shape[0] - 1
            y_region_max = min(int(y + (step_size / 2)), canvas.shape[0] - 1)

            # y_region_max = max(int(y + canvas.shape[0] - 1), 0)
            # print y, 'y'
            # print y_region_min, 'y_region_min'
            # print y_region_max, 'y_region_max'
            # print (canvas.shape[0] - 1), 'Canvas size y-0-45'
            # print img_diff[[291, 0], [305, 6]], "error condition "
            # M = img_diff[y: y + step_size, x: x + step_size]

            # if x_region_min >= canvas.shape[1]:
            #     x_region_min = canvas.shape[1] - 1
            #     print 'xmin hit the edge'
            #
            # if x_region_max >= canvas.shape[1]:
            #     x_region_max = canvas.shape[1] - 1
            #     print 'xmax hit the edge'
            #
            # if y_region_min >= canvas.shape[0]:
            #     y_region_min = canvas.shape[0] - 1
            #     print 'ymin hit the edge'
            #
            # if y_region_max >= canvas.shape[0]:
            #     y_region_max = canvas.shape[0] - 1
            #     print 'ymax hit the edge'

            # M_coordinates = np.ix_([x_region_min, x_region_max], [y_region_min, y_region_max])
            # M_coordinates = np.ix_([x_region_min, y_region_min], [x_region_max, y_region_max])
            # M_coordinates = np.ix_([y_region_min, x_region_min], [y_region_max, x_region_max])
            M_coordinates = np.ix_([y_region_min, y_region_max], [x_region_min, x_region_max])
            # print M_coordinates, 'Region Coordinates at 47'
            # print x_region_min, 'x_region_min'
            # print x_region_max, 'x_region_max'
            # print y_region_min, 'y_region_min'
            # print y_region_max, 'y_region_max'
            # print len(M_coordinates), 'len of m-cord'
            M = img_diff[M_coordinates]

            # M = M_coordinates
            # print np.sum(M), 'M sum  before squeeze'
            # M = np.squeeze(M,axis=2)
            # # print np.sum(M), 'M sum'
            areaError = np.sum(M) / (step_size ** 2)
            if (areaError > T):
                # M = M.reshape(step_size, step_size)
                # areaError_y = np.argmax(M)
                # areaError_x = (x + (areaError_y % step_size), y + (areaError_y // step_size))
                (areaError_y, areaError_x) = np.unravel_index(M.argmax(), M.shape)
                areaError_y += int(y - (float(step_size) / 2))
                areaError_x += int(x - (float(step_size) / 2))

                S.append((areaError_x, areaError_y, Ri, referenceImage, canvas))
                # S.append(makeSplineStroke(areaError_x, areaError_y, Ri, referenceImage, canvas))

    np.random.shuffle(S)
    for arx, ary, rad, ref_im, can in S:
        makeSplineStroke(arx, ary, rad, ref_im, can)

        # for t in S:
        #     a = t[1]
        #     b = t[0]
        #     c = referenceImage[t[1], t[0], :]
        #     d = int(t[1])
        #     color = (referenceImage[int(t[1]), int(t[0]), :])
        #     print color, 'color before int list '
        #     colorList = color.astype(int).tolist()
        #     print colorList, 'color when int list'
        #     colorsq = np.squeeze(colorList)
        #     print colorsq, 'color when squeezed'
        #
        #     strokeColor = np.squeeze((referenceImage[t[1], t[0], :]).astype(int).tolist())
        #     cv2.circle(canvas, (int(t[0]), int(t[1])), Ri, int(strokeColor), -1)


def difference(canvas, referenceImage):
    # https://mrl.nyu.edu/publications/painterly98/hertzmann-siggraph98.pdf
    # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.clip.html
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.split.html
    # https://stackoverflow.com/questions/5446522/data-type-not-understood
    # https://stackoverflow.com/questions/4971368/numpy-array-conversion-to-pairs
    r1, g1, b1 = np.split(canvas, 3, axis=2)
    r2, g2, b2 = np.split(referenceImage, 3, axis=2)

    # rdiff = np.power((r1 - r2), 2)
    # gdiff = np.power((g1 - g2), 2)
    # bdiff = np.power((b1 - b2), 2)

    rdiff = (r1 - r2) ** 2
    gdiff = (g1 - g2) ** 2
    bdiff = (b1 - b2) ** 2

    sqrootVal = np.sqrt(rdiff + gdiff + bdiff)
    # clippedVal = np.clip(sqrootVal, 0.0, 255.0)
    # squuezedArr = np.squeeze(clippedVal)
    return sqrootVal
    # return squuezedArr
    # return np.squeeze(np.clip(np.sqrt((rdiff + gdiff + bdiff)), 0.0, 255.0))


def makeSplineStroke(x0, y0, R, refImage, canvas):
    # https://mrl.nyu.edu/publications/painterly98/hertzmann-siggraph98.pdf
    # Midterm project - cv2.solbel
    # OpenCV Tutorial - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/
    # py_gradients/py_gradients.html

    # adding the stroked here because issues with calculating stroke color
    S = []
    strokeColor = np.squeeze(refImage[y0, x0, :]).astype(int).tolist()
    # cv2.circle(canvas, (int(x), int(y)), R, strokeColor, -1)
    # print 'making a stroke'
    x, y = x0, y0
    lastDx, lastDy = 0, 0
    # OpenCV Tutorial - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/
    # py_gradients/py_gradients.html
    Gx = cv2.Sobel(refImage, cv2.CV_64F, 1, 0).mean(axis=2)
    Gy = cv2.Sobel(refImage, cv2.CV_64F, 0, 1).mean(axis=2)
    K = [(x0, y0)]
    cv2.circle(canvas, (int(x), int(y)), R, strokeColor, -1)
    for i in xrange(MAX_STROKE_LENGTH):
        # print x, 'is value for x - 98'
        # print y, 'is value for y - 99'
        if (i > MIN_STROKE_LENGTH) and (
                    np.abs(
                        np.sum(np.squeeze(refImage[int(y), int(x), :]).astype(int) - np.squeeze(
                            canvas[int(y), int(x), :]).astype(int))) <
                    np.abs(np.sum(np.squeeze(refImage[int(y), int(x), :]).astype(int) - strokeColor))):
            return K
        if np.sqrt(((Gy[int(y)][int(x)] ** 2) + (Gx[int(y)][int(x)] ** 2))) == 0:
            return K
        gx, gy = Gx[int(y)][int(x)], Gy[int(y)][int(x)]
        dx, dy = -gy, gx

        if (lastDx * dx) + (lastDy * dy) < 0:
            dx, dy = -dx, -dy

        dx = (CURVATURE_FILTER * dx) + ((1 - CURVATURE_FILTER) * (lastDx))
        dy = (CURVATURE_FILTER * dy) + ((1 - CURVATURE_FILTER) * (lastDy))

        dx /= np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        dy /= np.sqrt(np.power(dx, 2) + np.power(dy, 2))

        x += R * dx
        y += R * dy

        if canvas.shape[1] > 255:
            x = max(min(x, canvas.shape[1]-1), 0)
        else:
            x = max(min(x, 254), 0)
        if canvas.shape[0] > 255:
            y = max(min(y, canvas.shape[0]-1), 0)
        else:
            y = max(min(y, 254), 0)

        lastDx, lastDy = dx, dy

        cv2.circle(canvas, (int(x), int(y)), R, strokeColor, -1)

        # K.append((x, y))

        # return K
