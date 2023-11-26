import cv2
import numpy as np

MIN_MATCH_COUNT = 20
frmNumber = 0
objDetected = False

web = cv2.VideoCapture(0)
logo = cv2.imread('GeoHoosh.jpg')
logo = cv2.resize(logo, (0,0), fx=0.25, fy=0.25)
video = cv2.VideoCapture('ZAYN.mp4')
# video.set(1, 900)
hI, wI, cI = logo.shape
ret1, frame1 = video.read()
frame1 = cv2.resize(frame1, (wI, hI))

orb = cv2.ORB_create(nfeatures = 500)
# kp = orb.detect(logo, None)
# kp, des = orb.compute(logo, kp)
kp, des = orb.detectAndCompute(logo, None)
img2 = cv2.drawKeypoints(logo, kp, None, (0, 0, 255))

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Sorted
bf = cv2.BFMatcher(cv2.NORM_HAMMING) # Ratio Test


while (web.isOpened()):
    ret2, frame2 = web.read()
    frame2_ar = frame2.copy()

    if objDetected == True:
        if frmNumber == video.get(cv2.CAP_PROP_FRAME_COUNT):
            frmNumber = 0
        ret1, frame1 = video.read()
        frame1 = cv2.resize(frame1, (wI, hI))
    else:
        video.set(cv2.CAP_PROP_POS_FRAMES,0)
        frmNumber = 0

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
            objDetected = True
            src_pts = np.float32([kp[m.queryIdx].pt for [m] in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for [m] in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # print(M)
            src_bnd_pts = np.float32([[0, 0], [0, hI], [wI, hI], [wI, 0]]).reshape(-1, 1, 2)
            dst_bnd_pts = cv2.perspectiveTransform(src_bnd_pts, M)
            cv2.polylines(frame2, [np.int32(dst_bnd_pts)], True, (0, 0, 255), thickness= 3)
            videoWarped = cv2.warpPerspective(frame1, M, (frame2.shape[1], frame2.shape[0]))
            maskWin = np.zeros((frame2.shape[0], frame2.shape[1]), np.uint8)
            cv2.fillPoly(maskWin, [np.int32(dst_bnd_pts)], (255,255,255))
            maskWinInv = cv2.bitwise_not(maskWin)
            frame2_ar = cv2.bitwise_and(frame2_ar, frame2_ar, mask=maskWinInv)
            frame2_ar = cv2.bitwise_or(videoWarped, frame2_ar)
            cv2.imshow('Video Warp', videoWarped)
            cv2.imshow('Mask Window', frame2_ar)

        else:
            objDetected = False
            frame2_ar = frame2
            cv2.imshow('Mask Window', frame2_ar)

        # cv2.imshow('Matching', img3)

    cv2.imshow('WebStream', frame2)
    # cv2.imshow('Video', frame1)
    # cv2.imshow('Logo', img2)
    if cv2.waitKey(2) % 0xFF == ord('q'):
        break
    frmNumber += 1
web.release()
cv2.destroyAllWindows()