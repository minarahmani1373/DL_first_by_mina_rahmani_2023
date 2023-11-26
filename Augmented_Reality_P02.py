import cv2

logo = cv2.imread('GeoHoosh.jpg')
video = cv2.VideoCapture('ZAYN.mp4')
web = cv2.VideoCapture(0)

hI2, wI2, cI2 = logo.shape
ret1, frame1 = video.read()
frame1 = cv2.resize(frame1, (wI2, hI2))

orb = cv2.ORB_create(nfeatures = 500)
# kp = orb.detect(logo, None)
# kp, dec = orb.compute(logo, kp)
kp1, dec1 = orb.detectAndCompute(logo, None)
img2 = cv2.drawKeypoints(logo, kp1, None, (0, 0, 255))
# cv2.imshow('logo', img2)
# cv2.waitKey(0)

while (web.isOpened()):
    ret2, frame2 = web.read()
    kp2, dec2 = orb.detectAndCompute(frame2, None)
    frame2 = cv2.drawKeypoints(frame2, kp2, None, (0, 0, 255))
    cv2.imshow('Web', frame2)
    cv2.waitKey(2)

web.release()
cv2.destroyAllWindows()
