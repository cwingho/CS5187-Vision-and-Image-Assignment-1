import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import cv2
import glob
import os
import numpy as np

is_use_gpu = 0
gpu_id = '0'
max_gpu_mem = 1

############################## tensorflow ##################################
if is_use_gpu:
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
	device = '/device:GPU:0'
else:
	device = "/cpu:0"

g = tf.Graph()
tf.reset_default_graph()
with g.as_default():
	with tf.device(device):
		a = tf.placeholder(tf.float32, [None], name="vec_a")
		b = tf.placeholder(tf.float32, [None,None], name="vec_b")
		normal_a = tf.nn.l2_normalize(a,0)		
		normal_b = tf.nn.l2_normalize(b,1)
		cosSim = tf.reduce_sum(tf.multiply(normal_a, normal_b),axis=1)
		initializer = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = max_gpu_mem
config.gpu_options.allow_growth=True
config.allow_soft_placement = True
config.log_device_placement=True
sess = tf.Session(graph=g, config=config)
tf.keras.backend.set_session(sess)
sess.run(initializer)
################################# VGG #####################################
# include_top = False: not to add dense layer
# model = VGG16(weights='imagenet', include_top=False) 
trained_model = load_model('./save/best_model_cpu.h5')
out_layer = trained_model.get_layer('flatten')
model = Model(inputs=trained_model.input, outputs=out_layer.output)
model.summary()
###########################################################################

def sortKeyFunc(s):
    return int(os.path.basename(s).split('_')[0])

def loadQueryFeature():
	# queries, load once
	images = list()
	file_list = glob.glob('./crop_resize_queries/*.jpg')
	file_list.sort(key=sortKeyFunc)
	for f in file_list:
		img_name = os.path.basename(f)
		img_name = img_name.split('.')[0]
		im = cv2.imread(f)
		images.append(im)
	images = np.asarray(images)
	# images = preprocess_input(images)
	images = images/255
	queries_feature = model.predict(images)
	queries_feature = queries_feature.reshape((len(images),-1))

	return queries_feature

def loadImageFeature():
	images = list()
	for f in sorted(glob.glob('./crop_resize_images/*.jpg')):
		im = cv2.imread(f)
		im = np.expand_dims(im, axis=0)
		# im = preprocess_input(im)
		im = im/255
		im = model.predict(im)
		im = im.reshape(-1,)
		yield im

def main():
	global a,b
	queries_feature = loadQueryFeature()
	feed = {a:'',b:queries_feature}
	sim_list = list()
	cnt = 0

	for image_feature in loadImageFeature():
		feed[a] = image_feature
		sim = sess.run(cosSim, feed)
		sim_list.append(sim)
		cnt += 1
		
		if cnt % 2000 == 0:
			np.save('./save/yolo_sim_cpu_255.npy', np.asarray(sim_list))

		print('finish: ',cnt)
		
	np.save('./save/yolo_sim_cpu_255.npy', np.asarray(sim_list))
	
main()











