import os
import cv2
# import random
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
from mpl_toolkits.mplot3d import Axes3D 

plt.ion()

def preprocess_data(img_path, name, LUT):
	im = cv2.imread(os.path.join(img_path, name), 0)
	BGR_im = cv2.cvtColor(im, cv2.COLOR_BayerGR2BGR)
	un_im = UndistortImage(BGR_im, LUT)
	un_im = un_im[200:650, :]

	# cv2.imshow("image", un_im)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	# h, w = im.shape
	# dim = (int(0.6*w), int(0.6*h))
	# un_im = cv2.resize(un_im, dim)

	return un_im

def get_feature_matches(img1, img2):
	# Initiate STAR detector
	orb = cv2.ORB_create()

	img3 = deepcopy(img2)

	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# # Match descriptors.
	# matches = bf.match(des1,des2)
	
	# FLANN_INDEX_LSH = 6

	# index_params= dict(algorithm = FLANN_INDEX_LSH,
	# 				table_number = 6, # 12
	# 				key_size = 12,     # 20
	# 				multi_probe_level = 1) #2

	# search_params = dict(checks=50)   # or pass empty dictionary

	# flann = cv2.FlannBasedMatcher(index_params,search_params)

	# matches1 = flann.knnMatch(des1,des2,k=2)

	# # bf = cv2.BFMatcher()
	# # matches1 = bf.knnMatch(des1,des2, k=2)

	# # Apply ratio test
	# good = []
	# for mt in matches1:
	# 	if len(mt) == 2:
	# 		m = mt[0]
	# 		n = mt[1]
	# 	else:
	# 		continue

	# 	if m.distance < n.distance:
	# 		good.append(m)

	# matches = good

	bf = cv2.BFMatcher()
	matches1 = bf.knnMatch(des1,des2, k=2)

	# Apply ratio test
	good = []
	for m,n in matches1:
	    if m.distance < 0.5*n.distance:
	        good.append(m)

	matches = good

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	# if len(matches) > 100:
	# 	matches = matches[0:100]

	if len(matches) <= 8:
		print("Very few matches found. Aborting!!!")
		exit()

	# Draw matches
	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, img3, flags=2)
	# cv2.imshow("features", img3)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	return kp1, kp2, matches

def get_point(kp1, kp2, match):
	x1, y1 = kp1[match.queryIdx].pt
	x2, y2 = kp2[match.trainIdx].pt

	return [x1, y1, x2, y2]

def get_F_util(points):
	A = []
	for pt in points:
		x1, y1, x2, y2 = pt
		A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

	A = np.asarray(A)

	U, S, Vh = np.linalg.svd(A)
	# print(S.shape)

	f = Vh[-1,:]
	F = np.reshape(f, (3,3)).transpose()

	# Correct rank of F
	U, S, Vh = np.linalg.svd(F)
	S_ = np.eye(3)
	for i in range(3):
		S_[i, i] = S[i]
	S_[2, 2] = 0

	F = np.matmul(U, np.matmul(S_, Vh))

	# print(np.linalg.matrix_rank(F))

	return F

def get_inliers(kp1, kp2, matches, F):
	inliers = []
	err_thresh = 0.01

	for m in matches:
		p = get_point(kp1, kp2, m)
		x1 = np.asarray([p[0], p[1], 1])
		x2 = np.asarray([p[2], p[3], 1])

		e = np.matmul(x1, np.matmul(F, x2.T))
		if abs(e) < err_thresh:
			inliers.append(m)

	inliers = np.asarray(inliers)

	return inliers 



def get_F(kp1, kp2, matches):
	max_in = 0
	max_inliers = []
	F_ = []

	for i in range(1000):
		idx = np.random.choice(len(matches), 8, replace=False)
		points = []
		for i in idx:
			pt = get_point(kp1, kp2, matches[i])
			# print(pt)
			points.append(pt)

		points = np.asarray(points)
		# print(points)

		F = get_F_util(points)
		inliers = get_inliers(kp1, kp2, matches, F)

		# print("xxxxxxxxxxxxxx", inliers.shape)
		
		if inliers.shape[0] > max_in:
			max_inliers = inliers
			max_in = inliers.shape[0]
			F_ = F

	print("max inliers", max_in)

	points = []
	# Recompute F with all the inliers
	for m in max_inliers:
		pt = get_point(kp1, kp2, m)
		# print(pt)
		points.append(pt)

	points = np.asarray(points)
	F = get_F_util(points)

	return F, max_inliers


def findEssentialMatrix(F, K):
	E = np.matmul(K.T, np.matmul(F, K))
	U, S, Vh = np.linalg.svd(E)
	E = np.matmul(U, np.matmul(np.identity(3), Vh))

	# Correct rank of E
	U, S, Vh = np.linalg.svd(E)
	S_ = np.eye(3)
	for i in range(3):
		S_[i, i] = 1
	S_[2, 2] = 0

	E = np.matmul(U, np.matmul(S_, Vh))

	return E


def findCameraPose(E):
	U, D, Vt = np.linalg.svd(E)
	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
	C1 = U[:, 2]
	R1 = np.matmul(U, np.matmul(W, Vt))
	C2 = -C1
	R2 = R1
	C3 = C1
	R3 = np.matmul(U, np.matmul(W.T, Vt))
	C4 = C2
	R4 = R3

	if np.linalg.det(R1) < 0:
		R1 = -R1
		C1 = -C1

	if np.linalg.det(R2) < 0:
		R2 = -R2
		C2 = -C2

	if np.linalg.det(R3) < 0:
		R3 = -R3
		C3 = -C3

	if np.linalg.det(R4) < 0:
		R4 = -R4
		C4 = -C4	

	
	# R2 = -R2
	# C2 = -C2
	# R3 = -R3
	# C3 = -C3
	# R4 = -R4
	# C4 = -C4

	# print(np.linalg.det(R1))
	# print(np.linalg.det(R2))
	# print(np.linalg.det(R3))
	# print(np.linalg.det(R4))

	return [[C1, R1], [C2, R2], [C3, R3], [C4, R4]]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def main():
	cur_path = os.path.dirname(os.path.abspath(__file__))
	img_path = os.path.join(cur_path, 'stereo/centre')

	count = 0

	# Read camera parametres
	fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('./model')
	
	K = np.zeros((3, 3))
	K[0][0] = fx
	K[0][2] = cx
	K[1][1] = fy
	K[1][2] = cy
	K[2][2] = 1
	print(K)

	prev_frame = []
	T_prev = np.eye(4)
	R_prev = np.eye(3)
	t_prev = np.asarray([[0], [0], [0]])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# ax = fig.add_subplot(111, projection='3d')
	# ax.set_xlim(-100,100)
	# ax.set_ylim(-100,100)
	# ax.set_zlim(-100,100)
	xs = []
	ys = []
	zs = []
	for name in sorted(os.listdir(img_path)):
		print(count)
		frame = preprocess_data(img_path, name, LUT)
		cv2.imshow("frame", frame)
		cv2.waitKey(1)
		if count < 20:
			prev_frame = frame
			count += 1
			continue

		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		kp1, kp2, matches = get_feature_matches(prev_frame, frame)

		# F, inliers = get_F(kp1, kp2, matches)

		pts1 = []
		pts2 = []
		for m in matches:
			pts1.append(kp1[m.queryIdx].pt)
			pts2.append(kp2[m.trainIdx].pt)

		pts1 = np.int32(pts1)
		pts2 = np.int32(pts2)

		# F = F/F[2,2]

		# F_cv, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)

		# # We select only inlier points
		# pts1 = pts1[mask.ravel()==1]
		# pts2 = pts2[mask.ravel()==1]

		# print(F)
		# print(F_cv)

		# img1 = deepcopy(prev_frame)
		# img2 = deepcopy(frame)

		# # Find epilines corresponding to points in right image (second image) and
		# # drawing its lines on left image
		# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
		# lines1 = lines1.reshape(-1,3)
		# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
		# # Find epilines corresponding to points in left image (first image) and
		# # drawing its lines on right image
		# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
		# lines2 = lines2.reshape(-1,3)
		# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
		# cv2.imshow("prev", img3)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()
		# cv2.imshow("cur", img5)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()
		# # plt.subplot(121),plt.imshow(img5)
		# # plt.subplot(122),plt.imshow(img3)
		# # plt.show()
		
		# E = findEssentialMatrix(F, K)

		E_cv, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC)

		# print(E)
		# print(E_cv)
		
		# camera_poses = findCameraPose(E)

		# print(camera_poses)

		# points1 = []
		# points2 = []
		# for m in matches:
		# 	points1.append(kp1[m.queryIdx].pt)
		# 	points2.append(kp2[m.trainIdx].pt)

		# points1 = np.int32(points1)
		# points2 = np.int32(points2)

		num, R, t, mask = cv2.recoverPose(E_cv, pts1, pts2, K)

		# print("R:: ", R)
		# print("t:: ", t)
		# print("R_prev:: ", R_prev)
		# print("t_prev:: ", t_prev)

		count += 1 

		# R = np.matmul(R, R_prev)
		# t = np.matmul(R, t_prev) + t
		# print("R:: ", R)
		# print("t:: ", t)

		T = np.hstack((R, -t))
		T = np.vstack((T, np.asarray([0, 0, 0, 1])))

		
		T = np.matmul(T_prev, T)
		print(T)
		X = T[:,3]
		print(X)
		# P = np.matmul(K, T)
		# print(P)

		# X = np.matmul(T, np.asarray([0, 0, 0, 1]).T)

		# print(X)

		xs.append(X[0])
		ys.append(X[1])
		zs.append(X[2])

		# R_prev = R
		# t_prev = t

		T_prev = T
		prev_frame = frame

		# ax.plot([X[0]],[X[1]], [X[2]],'o') 

		ax.scatter(X[0], X[2], s=1, marker='o', color='b')
		plt.pause(0.01)

		plt.savefig('Output/frame%03d.png' % count)

		# if count == 1000:
		# 	break
		# images.append(un_im)

	# print(xs)

	# ax.scatter(xs, ys, zs, marker='o')
	plt.pause(10)

	plt.show()





if __name__ == '__main__':
	main()
