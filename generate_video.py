import cv2
import glob
import matplotlib.pyplot as plt

# count = 100
# img_paths = glob.glob("rename_input/*")
# for name in sorted(img_paths):
# 	print(name)
# 	img = cv2.imread(name)
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	plt.imsave('Output_renamed/frame%04d.png' % count,img)
# 	count = count + 1



vidWriter = cv2.VideoWriter("./Result_inbuilt.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (640,480))
img_paths = glob.glob("Output_in_backup/*")
for name in sorted(img_paths):
	print(name)
	image = cv2.imread(name)
	vidWriter.write(image)
vidWriter.release()