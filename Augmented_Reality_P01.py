import cv2

img = cv2.imread('GeoHoosh.jpg')
video = cv2.VideoCapture('ZAYN.mp4')
web = cv2.VideoCapture(0)

while (web.isOpened()):
    ret1, frame1 = web.read()
    ret2, frame2 = video.read()

    hI2, wI2, cI2 = img.shape
    print(hI2, wI2, cI2)
    cv2.imshow('WebStream', frame1)
    cv2.imshow('Video', frame2)
    cv2.imshow('Logo', img)
    cv2.waitKey(2)

