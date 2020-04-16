import glob
import cv2
import os

size = 224

cnt = 0
for f in glob.glob('./cropped_images_yolo/*.jpg'):
	img_name = os.path.basename(f)

	im = cv2.imread(f)
	im = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC)
	cv2.imwrite('./cropped_resized_images_yolo/'+img_name, im) 
	cnt += 1
	print('finish ',cnt)

# INTER_NEAREST	最近邻插值
# INTER_LINEAR	双线性插值（默认设置）
# INTER_AREA	使用像素区域关系进行重采样。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。 但是当图像缩放时，它类似于INTER_NEAREST方法。
# INTER_CUBIC	4x4像素邻域的双三次插值
# INTER_LANCZOS4 8x8像素邻域的Lanczos插值
