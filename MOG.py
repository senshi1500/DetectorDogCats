import numpy as np
import cv2 as cv
from Predic import prediccion

cap = cv.VideoCapture('vtest.avi')
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
Kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
c = 1
while (1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    ImgOpen = cv.morphologyEx(fgmask, cv.MORPH_OPEN, Kernel)
    ImgEro = cv.morphologyEx(ImgOpen, cv.MORPH_ERODE, Kernel)
    ImgDil = cv.morphologyEx(ImgEro, cv.MORPH_DILATE, Kernel)
    ImgDil = cv.morphologyEx(ImgDil, cv.MORPH_DILATE, Kernel)
    labels = cv.connectedComponentsWithStats(ImgDil, 4, cv.CV_32S)

    (totalLabels, label_ids, values, centroid) = labels

    for i in range(1, totalLabels):
        area = values[i, cv.CC_STAT_AREA]
        if (area > 700) and (area < 3000):
            new_img = fgmask.copy()
            x1 = values[i, cv.CC_STAT_LEFT]
            y1 = values[i, cv.CC_STAT_TOP]
            w = values[i, cv.CC_STAT_WIDTH]
            h = values[i, cv.CC_STAT_HEIGHT]

            # Coordinate of the bounding box
            pt1 = (x1, y1)
            pt2 = (x1 + w, y1 + h)
            (X, Y) = centroid[i]
            # print(i)
            # Bounding boxes for each component
            cv.rectangle(frame, pt1, pt2, (0, 255, 0), 3)
            cv.circle(frame, (int(X), int(Y)),
                      4, (0, 0, 255), -1)
            # TODO Buscar un vvideo para provarlo
            # TODO BUscar formas de halerarlo especailmente la etapa de prediccion
            ImgPredict = frame[y1: y1 + h, x1: x1 + w]
            print(len(ImgPredict))
            Etiqueta = prediccion(ImgPredict)
            frame = cv.putText(frame, Etiqueta, (x1, y1 + h + 20), cv.FONT_HERSHEY_SIMPLEX , 1,
                               (0, 0, 255), 1, cv.LINE_AA)


    cv.imshow('frame', frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
