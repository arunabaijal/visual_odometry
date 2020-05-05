 
import os
import cv2
import numpy as np
from copy import deepcopy
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math

def preprocess_data(img_path, name, LUT):
	im = cv2.imread(os.path.join(img_path, name), 0)
	BGR_im = cv2.cvtColor(im, cv2.COLOR_BayerGR2BGR)
	un_im = UndistortImage(BGR_im, LUT)
	un_im = cv2.cvtColor(un_im, cv2.COLOR_BGR2GRAY)
	un_im = un_im[200:650,:]

	return un_im            

def get_feature_matches(img1, img2):

	sift = cv2.xfeatures2d.SIFT_create() 

    # find the keypoints and descriptors with SIFT in current as well as next frame
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)
	
	# Ratio test as per Lowe's paper
	good = []
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.5*n.distance:
			good.append(m)

	return kp1, kp2, good

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

	U, S, Vh = np.linalg.svd(A, full_matrices = True)

	f = Vh[-1,:]
	F = np.reshape(f, (3,3)).transpose()

	# Correct rank of F
	U, S, Vh = np.linalg.svd(F)
	S_ = np.eye(3)
	for i in range(3):
		S_[i, i] = S[i]
	S_[2, 2] = 0

	F = np.matmul(U, np.matmul(S_, Vh))

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
	np.random.seed(30)

	for i in range(50):
		idx = np.random.choice(len(matches), 8, replace=False)
		points = []
		for i in idx:
			pt = get_point(kp1, kp2, matches[i])
			points.append(pt)

		points = np.asarray(points)

		F = get_F_util(points)
		inliers = get_inliers(kp1, kp2, matches, F)
		
		if inliers.shape[0] > max_in:
			max_inliers = inliers
			max_in = inliers.shape[0]
			F_ = F

	points = []

	# Recompute F with all the inliers
	for m in max_inliers:
		pt = get_point(kp1, kp2, m)
		points.append(pt)

	points = np.asarray(points)
	F = get_F_util(points)

	return F, points


def findEssentialMatrix(F, K):
	E = np.matmul(K.T, np.matmul(F, K))

	# Correct rank of E
	U, S, Vh = np.linalg.svd(E, full_matrices = True)
	S_ = np.eye(3)
	S_[2, 2] = 0

	E = np.matmul(U, np.matmul(S_, Vh))

	return E


def findCameraPose(E):
	U, D, Vt = np.linalg.svd(E, full_matrices = True)
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
	
	C1 = C1.reshape(-1,1)
	C2 = C2.reshape(-1,1)
	C3 = C3.reshape(-1,1)
	C4 = C4.reshape(-1,1)

	poses = []
	P1 = np.concatenate((R1,C1),axis = 1)
	poses.append(P1)
	P2 = np.concatenate((R2,C2),axis = 1)
	poses.append(P2)
	P3 = np.concatenate((R3,C3),axis = 1)
	poses.append(P3)
	P4 = np.concatenate((R4,C4),axis = 1)
	poses.append(P4)

	poses = np.asarray(poses)
	return poses

def cross(u):
	a1=u[0][0]
	a2=u[1][0]
	a3=u[2][0]
	return np.array([[0,-a3,a2],[a3,0,-a1],[-a2,a1,0]])

def linear_triangulation(K, P1, P2, points):
	pt1 = np.asarray([points[0], points[1],1]).reshape(-1,1)
	pt2 = np.asarray([points[2], points[3],1]).reshape(-1,1)

	cross_1 = cross(pt1)
	cross_2 = cross(pt2)

	pose1 = np.matmul(cross_1, P1)
	pose2 = np.matmul(cross_2, P2)

	A = np.vstack((pose1, pose2))
	U,S,Vh = np.linalg.svd(A, full_matrices = True)
	X = Vh[-1,:]
	X = X/X[3]
	return X

def euler_angles(R) :
 
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	singular = sy < 1e-6
	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
 
	return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])


def get_correct_pose(points, camera_poses, K):
	P1 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0]])

	pts_dict = {}

	for i in range(4):
		X = []
		for j in range(points.shape[0]):
			pt = linear_triangulation(K,P1,camera_poses[i],points[j])
			X.append(pt)

		pts_dict.update({i:X})
	
	c = 0
	flag = 0

	for i in range(4):
		P = camera_poses[i]
		
		proj = find_projection_matrix(K, P)
		res = least_squares(nonlinearerror, x0=np.asarray(pts_dict[i]).flatten(), method="dogbox", args=(proj, points),
							max_nfev=8000)
		pts_list = np.reshape(res.x, np.asarray(pts_dict[i]).shape)
		
		angles = euler_angles(P[:,:3])
		if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50:
			flag = 1
			r3 = P[2,0:3]
			r3 = np.reshape(r3,(1,3))

			C = P[:,3]
			C = np.reshape(C,(3,1))

			pts_list = np.asarray(pts_dict[i])
			pts_list = pts_list[:,0:3].T

			Z = np.matmul(r3,np.subtract(pts_list,C))
			_,pos = np.where(Z>0)

			if c < pos.shape[0]:
				final_pose = P
				c = pos.shape[0]
	
	if flag == 1:
		# print("flaaaaaaaag", flag)
		return final_pose, flag

	else: 
		# print("flaaaaaaaaaag", flag)
		return P1, flag 


def main():
	fig = plt.figure()
	ax = fig.add_subplot(111)

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
	
	# initialize previous frame values
	T_prev = np.eye(4)
	prev_frame = []
	prev_pose = np.array([[1, 0, 0, 0],
				  		[0, 1, 0, 0],
				  		[0, 0, 1, 0]], dtype = np.float32)

	for name in sorted(os.listdir(img_path)):
		print("frame: " + str(count))
		frame = preprocess_data(img_path, name, LUT)
		
		# skip first 17 farmes (over-exposure)
		if count <= 18:
			prev_frame = frame
			count += 1
			continue

		# Get feature matches
		kp1, kp2, matches = get_feature_matches(prev_frame, frame)

		# Calculate Fundamental Matrix
		F, points = get_F(kp1, kp2, matches)
		
		# Calculate Essential Matrix
		E = findEssentialMatrix(F, K)
		
		# Get all possible camera pose
		camera_poses = findCameraPose(E)

		# Get correct pose form possible four poses
		final_pose, flag = get_correct_pose(points,camera_poses,K)

		if flag == 0:
			final_pose = prev_pose
		else:
			final_pose = final_pose

		print("final_pose", final_pose)

		if final_pose[2, 3] > 0:
			final_pose[:, 3] = -final_pose[:, 3]

		R = final_pose[:,0:3].reshape(3,3)
		t = final_pose[:,3].reshape(3,1)

		T = np.hstack((R, t))
		T = np.vstack((T, np.asarray([0, 0, 0, 1])))
		
		T = np.matmul(T_prev, T)
		X = T[:,3]

		T_prev = T
		prev_frame = frame
		prev_pose = final_pose

		ax.scatter(X[0], -X[2], s=1, marker='o', color='b')
		plt.pause(0.01)
		# plt.savefig('Output/frame%03d.png' % count)

		# if count == 3:
		# 	print("why you no breakkkkkkkkkkk")
		# 	break

		count += 1 

	plt.pause(10)

	plt.show()

def find_projection_matrix(K, final_pose):
	return np.matmul(K, final_pose)


def nonlinearerror(X, P, points):
	X = X.reshape((len(points), 4))
	X_homo = X.T
	# X = X[:, :3].T
	s = np.square(np.asarray(points[:, 0]) - (np.matmul(P[0, :], X_homo) / np.matmul(P[2, :], X_homo))) + np.square(
		np.asarray(points[:, 1]) - (np.matmul(P[1, :], X_homo) / np.matmul(P[2, :], X_homo)))
	s = s + np.square(np.asarray(points[:, 2]) - (np.matmul(P[0, :], X_homo) / np.matmul(P[2, :], X_homo))) + np.square(
		np.asarray(points[:, 3]) - (np.matmul(P[1, :], X_homo) / np.matmul(P[2, :], X_homo)))
	
	return np.asarray(s, dtype=np.float32)

if __name__ == '__main__':
	main()