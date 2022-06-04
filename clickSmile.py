import cv2 as cv
from datetime import datetime as dt

####################################################################################################################
cascade = cv.CascadeClassifier('resources/cascades/haarcascade_eye.xml')
path = 'resources/img/smiles0.jpg'
sF = 1.1
mN = 15
t = 1
cam = 0
brightness = 200
lf = True
status = {
    0: False,
    1: False,
    2: False
}
####################################################################################################################


def empty(a):
    pass


def initTrackbars():
    cv.namedWindow('trackbars')
    cv.createTrackbar('sF', 'trackbars', 11, 200, empty)
    cv.createTrackbar('mN', 'trackbars', 15, 100, empty)
    cv.createTrackbar('t', 'trackbars', 1, 1000, empty)


def readTracbars():
    global sF, mN, t
    sF = cv.getTrackbarPos('sF', 'trackbars') / 10
    mN = cv.getTrackbarPos('mN', 'trackbars')
    t = cv.getTrackbarPos('t', 'trackbars')


def updateStatus(detect):
    status[len(detect)] = True


def checkStatus(boxes):
    if status[0] or status[1]:
        return True
    else:
        return False


def detectSmile(img):
    detect = cascade.detectMultiScale(img, scaleFactor=sF, minNeighbors=mN)
    updateStatus(detect)
    return detect


def draw(img, boxes):
    result = img.copy()
    for (x, y, w, h) in boxes:
        cv.rectangle(result, (x, y), (x+w, y+h), (0, 0, 0), 2)
    return result


def liveFeed():
    initTrackbars()
    cap = cv.VideoCapture(cam, cv.CAP_DSHOW)
    cap.set(10, brightness)
    while True:
        readTracbars()
        status, img = cap.read()
        if status:
            boxes = detectSmile(img)
            f = checkStatus(boxes)
            result = draw(img, boxes)
            cv.imshow('img', result)
            if f:
                print('CAPTURED')
                cv.imwrite(f'resources/saves/{str(dt.now().second)}.jpg', img)
            if cv.waitKey(1) > 0:
                break
    cv.imshow('Result', img)
    cv.waitKey(0)


def staticFeed():
    initTrackbars()
    img = cv.imread(path)
    while True:
        readTracbars()
        boxes = detectSmile(img)
        f = checkStatus(boxes)
        result = draw(img, boxes)
        cv.imshow('img', result)
        if f:
            print('CAPTURED')
            cv.imwrite('resources/saves/ti0.jpg', img)
        if cv.waitKey(1) > 0:
            break


def printdt():
    print(str(dt.now().year))


if __name__ == '__main__':
    if lf:
        liveFeed()
    else:
        staticFeed()
    # printdt()

