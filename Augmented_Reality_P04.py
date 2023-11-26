import cv2
import numpy as np

MIN_MATCH_COUNT = 20
web = cv2.VideoCapture(0)
logo = cv2.imread('GeoHoosh.jpg')
video = cv2.VideoCapture('ZAYN.mp4')

hI2, wI2, CI2 = logo.shape
ret1, frame1 = video.read()
frame1 = cv2.resize(frame1, (wI2, hI2))

orb = cv2.ORB_create(nfeatures = 500)
# kp = orb.detect(logo, None)
# kp, des = orb.compute(logo, kp)
kp, des = orb.detectAndCompute(logo, None)
img2 = cv2.drawKeypoints(logo, kp, None, (0, 0, 255))

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Sorted
bf = cv2.BFMatcher(cv2.NORM_HAMMING) # Ratio Test


while (web.isOpened()):
    ret2, frame2 = web.read()
    kp2, des2 = orb.detectAndCompute(frame2, None)
    # frame2 = cv2.drawKeypoints(frame2, kp2, None, (0, 0, 255))

    if des2 is not None:
        # Method 1 ######## Sorted
        # matches = bf.match(des, des2)
        # matches = sorted(matches, key=lambda x: x.distance)
        # img3 = cv2.drawMatches(logo, kp, frame2, kp2, matches[:10], flags=2, outImg=None)

        # Method 2 ######## Ratio Test
        matches = bf.knnMatch(des, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                good.append([m])
        img3 = cv2.drawMatchesKnn(logo, kp, frame2, kp2, good, flags=2, outImg=None)
    #################################################################################
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp[m.queryIdx].pt for [m] in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for [m] in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            print(M)
        cv2.imshow('Matching', img3)


    # cv2.imshow('WebStream', frame2)
    # cv2.imshow('Matching', img3)
    # cv2.imshow('Logo', img2)

    if cv2.waitKey(2) % 0xFF == ord('q'):
        break

web.release()
cv2.destroyAllWindows()