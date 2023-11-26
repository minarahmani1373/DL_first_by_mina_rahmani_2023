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
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) # sorted
bf = cv2.BFMatcher(cv2.NORM_HAMMING) # Ratio Test

while (web.isOpened()):
    ret2, frame2 = web.read()
    kp2, dec2 = orb.detectAndCompute(frame2, None)
    # frame2 = cv2.drawKeypoints(frame2, kp2, None, (0, 0, 255))
    if dec2 is not None:
        # Sorted
        # matches = bf.match(dec1, dec2)
        # matches = sorted(matches, key = lambda x: x.distance)
        # img3 = cv2.drawMatches(logo, kp1, frame2, kp2, matches[:15], flags=2, outImg=None)
        # cv2.imshow('Matching', img3)

        # Ratio Test
        matches = bf.knnMatch(dec1, dec2, k = 2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n. distance:
                good.append([m])
        img3 = cv2.drawMatchesKnn(logo, kp1, frame2, kp2, good, flags=2, outImg=None)
        cv2.imshow('Matching', img3)

        if cv2.waitKey(2) % 0xFF == ord('q'):
            break


web.release()
cv2.destroyAllWindows()
