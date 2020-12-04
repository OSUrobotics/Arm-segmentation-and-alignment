import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    frame = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), markerCorners, markerIds)
    plt.figure()
    plt.imshow(frame_markers)
    print(markerIds)
    #for i in range(len(markerIds)):

    #    c = markerCorners[i][0]
    #    plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(markerIds[i]))
    #plt.legend()
    plt.show()
    print("done")

    markerLength = 0.036
    dist_coeffs= np.load('dist_mtx2.npy')
    camera_matrix = np.load('camera_mtx2.npy')


    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners[2], markerLength, camera_matrix, dist_coeffs)
    rvec2, tvec2, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners[3], markerLength, camera_matrix, dist_coeffs)
    rvec3, tvec3, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners[0], markerLength, camera_matrix, dist_coeffs)

    print(tvec, tvec2, tvec3)
    print(rvec, rvec2, rvec3)

