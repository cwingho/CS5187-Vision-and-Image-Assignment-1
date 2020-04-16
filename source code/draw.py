import numpy as np
import glob
import os
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def sortKeyFunc(s):
    return int(os.path.basename(s).split('_')[0])

def handleMultiQueries(sim):
	images = list()
	file_list = glob.glob('./cropped_queries/*.jpg')
	file_list.sort(key=sortKeyFunc)

	# start from index 0
	cnt = 0
	f_list = list()
	for idx, f in enumerate(file_list):
		img_id = f.split('/')[-1].split('_')[0]
		f_list.append(int(img_id))

	f_list = np.asarray(f_list)
	num, idx, cnt = np.unique(f_list, return_index=True, return_counts=True, axis=0)

	n_img = sim.shape[0]
	n_query = len(num)

	_sim = np.zeros((n_img,n_query))

	for _num,_idx,_cnt in zip(num,idx,cnt):
		if _cnt == 1:
			_sim[:,_num-1] = sim[:,_idx]
			continue

		dup = sim[:,_idx:_idx+_cnt]
		dup = np.max(dup,axis=1)
		_sim[:,_num-1] = dup
	return _sim

def handleMultiImages(sim):
	images = list()
	file_list = glob.glob('./cropped_images_yolo/*.jpg')
	file_list.sort(key=sortKeyFunc)

	# start from index 0
	cnt = 0
	f_list = list()
	for idx, f in enumerate(file_list):
		img_id = f.split('/')[-1].split('_')[0]
		f_list.append(int(img_id))

	f_list = np.asarray(f_list)
	num, idx, cnt = np.unique(f_list, return_index=True, return_counts=True, axis=0)

	n_img = len(num)
	n_query = sim.shape[1]

	_sim = np.zeros((n_img,n_query))

	for _num,_idx,_cnt in zip(num,idx,cnt):
		if _cnt == 1:
			_sim[_num-1] = sim[_idx]
			continue

		dup = sim[_idx:_idx+_cnt]
		dup = np.max(dup,axis=0)
		_sim[_num-1] = dup
		
	return _sim

sim = np.load('./save/similaroty_matrix.npy')
sim = handleMultiQueries(sim)
sim = handleMultiImages(sim)

# (28493,50)->(50, 28493)
sim = sim.T

# sort by similarity in descending order
rank = np.argsort(sim,axis=-1)
for idx, row in enumerate(rank):
	rank[idx] = row[::-1]

# extract top 10 images for queries 1 to 5
rank = rank[:5,:10]
rank = rank.flatten()

max_pad = 5
for idx in rank:
	_idx = str(idx+1)
	padding = max_pad - len(_idx)
	name = ''.join(['0' for i in range(padding)])+_idx
	im = cv2.imread('./Images/'+name+'.jpg')
	w = im.shape[1]
	h = im.shape[0]

	with open('./Images_json/'+name+'.json') as json_f:
		coord = json.load(json_f)
		confidence = 0
		topleft_x = 0
		topleft_y = 0
		bottomright_x = 0
		bottomright_y = 0
		has_box = False
		for row in coord:
			_confidence = row['confidence']

			if _confidence > 0.4:
				topleft_x = row['topleft']['x']
				topleft_y = row['topleft']['y']
				bottomright_x = row['bottomright']['x']
				bottomright_y = row['bottomright']['y']
				cv2.rectangle(im, (topleft_x, topleft_y), (bottomright_x, bottomright_y), (0,0,255), 20)
				has_box = True

		if not has_box:
			cv2.rectangle(im, (0, 0), (w, h), (0,0,255), 20)

		cv2.imwrite('./result/'+name+'.jpg', im) 

# define figure size
figsize = (5,11)
plt.figure(figsize=figsize)
cnt = 0

# index of queries
query_idxs = [1,12,23,34,45,56]

# draw figure
for idx in rank:
	qid = cnt+1
	if qid in query_idxs:
		query_idx = str(qid%10)
		name = '0'+query_idx
		im = mpimg.imread('./Queries/'+name+'.jpg')
		plt.subplot(figsize[0],figsize[1],qid)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(im)
		cnt += 1
		qid = cnt+1

	_idx = str(idx+1)
	padding = max_pad - len(_idx)
	name = ''.join(['0' for i in range(padding)])+_idx
	im = mpimg.imread('./result/'+name+'.jpg')

	plt.subplot(figsize[0],figsize[1],qid)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(im)
	cnt += 1

# plt.show()
plt.savefig("./result.png")




