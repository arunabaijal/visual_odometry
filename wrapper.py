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

	# print("max inliers", max_inliers.shape)

	points = []
	# Recompute F with all the inliers
	for m in max_inliers:
		pt = get_point(kp1, kp2, m)
		# print(pt)
		points.append(pt)

	points = np.asarray(points)
	F = get_F_util(points)

	return F,points


def findEssentialMatrix(F, K):
	E = np.matmul(K.T, np.matmul(F, K))
	U, S, Vh = np.linalg.svd(E)
	E = np.matmul(U, np.matmul(np.identity(3), Vh))

	# Correct rank of E
	U, S, Vh = np.linalg.svd(E)
	S_ = np.eye(3)
	for i in range(3):
		S_[i, i] = S[i]
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

	# print(R1)
	# print(C1)
	# print(P1)
	poses = np.asarray(poses)
	return poses

	# return [[C1, R1], [C2, R2], [C3, R3], [C4, R4]]
def cross(u):
    a1=u[0][0]
    a2=u[1][0]
    a3=u[2][0]
    # print(a)
    return np.array([[0,-a3,a2],[a3,0,-a1],[-a2,a1,0]])

def linear_triangulation(K,P1,P2,points):
	# print(points[0])
	pt1 = np.asarray([points[0], points[1],1]).reshape(-1,1)
	pt2 = np.asarray([points[2], points[3],1]).reshape(-1,1)
	# print(pt1)

	p1 = np.matmul(np.linalg.inv(K),pt1)
	p2 = np.matmul(np.linalg.inv(K),pt2)

	cross_1 = cross(p1)
	cross_2 = cross(p2)

	P1 = np.concatenate((P1[:,:3], -np.matmul(P1[:,:3],P1[:,3].reshape(-1,1))),axis=1)
	P2 = np.concatenate((P2[:,:3], -np.matmul(P2[:,:3],P2[:,3].reshape(-1,1))),axis=1)

	r1 = np.matmul(cross_1,P1)
	r2 = np.matmul(cross_2,P2)
	# print(pose1)
	# print(pose2)

	A = np.concatenate((r1,r2),axis=0)
	# print(A.shape)
	U,S,Vh = np.linalg.svd(A)
	X = Vh[-1,:]
	# print(X)
	X = X/X[3]
	return X


def get_correct_pose(points,camera_poses,K):
	P1 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0]])
	# print(P0)
	pts_dict = {}
	# print(points)
	# print(points.shape)
	# print(camera_poses[0].shape)

	for i in range(4):
		X = []
		for j in range(points.shape[0]):
			pt = linear_triangulation(K,P1,camera_poses[i],points[j])
			X.append(pt)

		# print(len(X))
		pts_dict.update({i:X})
	# print(len(pts_dict))

	# print(len(pts_dict[0]))
	max = 0

	for i in range(4):
		print(i)
		P = camera_poses[i]

		# print(len(pts_dict[i]))
		# print("P", P)
		r3 = P[2,0:3]
		r3 = np.reshape(r3,(1,3))
		# print("r3", r3.shape)

		C = P[:,3]
		C = np.reshape(C,(3,1))
		# print(C.shape)

		pts_list = np.asarray(pts_dict[i])
		# print(pts_list.shape)
		pts_list = pts_list[:,0:3].T
		# print(pts_list.shape)

		Z = np.matmul(r3,np.subtract(pts_list,C))
		# print("Z", Z.shape)
		Z = Z>0
		_,pos = np.where(Z==True)
		print("Z", pos.shape[0], i)
		if max < pos.shape[0]:
			final_pose = P
			max = pos.shape[0]

	return final_pose




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

	prev_frame = []
	for name in sorted(os.listdir(img_path)):
		print("frame", count)
		frame = preprocess_data(img_path, name, LUT)
		if count == 0:
			prev_frame = frame
			count += 1
			continue

		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		kp1, kp2, matches = get_feature_matches(prev_frame, frame)

		F,points = get_F(kp1, kp2, matches)
		
		E = findEssentialMatrix(F, K)
		
		camera_poses = findCameraPose(E)

		final_pose = get_correct_pose(points,camera_poses,K)

		print("final_pose", final_pose)
		
		count += 1 

		if count == 6:
			break

		# images.append(un_im)





if __name__ == '__main__':
	main()
