import numpy as np
import glob
import os

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

with open('./result/rankList.txt','w') as f:
	cnt = 1
	for row in rank:
		content = " ".join([str(i) for i in row])
		f.write('Q{}: {}\n'.format(cnt,content))
		cnt += 1




