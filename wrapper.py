import os
import cv2
# import random
import numpy as np
from copy import deepcopy
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

def preprocess_data(img_path, name, LUT):
	im = cv2.imread(os.path.join(img_path, name), 0)
	BGR_im = cv2.cvtColor(im, cv2.COLOR_BayerGR2BGR)
	un_im = UndistortImage(BGR_im, LUT)
	h, w = im.shape
	dim = (int(0.6*w), int(0.6*h))
	un_im = cv2.resize(un_im, dim)

	return un_im

def get_feature_matches(img1, img2):
	# Initiate STAR detector
	orb = cv2.ORB_create()

	# # find the keypoints with ORB
	# kp = orb.detect(frame, None)
	# print(len(kp))

	# # compute the descriptors with ORB
	# kp, des = orb.compute(frame, kp)

	img3 = deepcopy(img2)

	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	if len(matches) <= 8:
		print("Very few matches found. Aborting!!!")
		exit()

	# Draw first 10 matches.
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, img3, flags=2)



	# img = deepcopy(frame)
	# # draw only keypoints location,not size and orientation
	# img = cv2.drawKeypoints(frame, kp, img, color=(0,255,0), flags=0)
	cv2.imshow("features", img3)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

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
	# print(F)
	# print(np.linalg.matrix_rank(F))

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
	err_thresh = 0.05

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

	for i in range(100):
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
		
		if inliers.shape[0] > max_in:
			max_inliers = inliers
			max_in = inliers.shape[0]
			F_ = F

	print("max inliers", max_inliers.shape)

	points = []
	# Recompute F with all the inliers
	for m in max_inliers:
		pt = get_point(kp1, kp2, m)
		# print(pt)
		points.append(pt)

	points = np.asarray(points)
	F = get_F_util(points)

	return F


def main():
	cur_path = os.path.dirname(os.path.abspath(__file__))
	img_path = os.path.join(cur_path, 'stereo/centre')

	count = 0

	# Read camera parametres
	fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('./model') 

	prev_frame = []
	for name in sorted(os.listdir(img_path)):
		print(count)
		frame = preprocess_data(img_path, name, LUT)
		if count == 0:
			prev_frame = frame
			count += 1
			continue

		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		kp1, kp2, matches = get_feature_matches(prev_frame, frame)

		F = get_F(kp1, kp2, matches)

		count += 1 

		if count == 6:
			break
		# images.append(un_im)





if __name__ == '__main__':
	main()
