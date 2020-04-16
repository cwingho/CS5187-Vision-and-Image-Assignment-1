from PIL import Image
import os
import glob
import json


for f in glob.glob('./Images/*.jpg'):
	img_name = os.path.basename(f)
	img_no = img_name.split('.')[0]

	# image instance
	im = Image.open(f)

	with open('./Images_json/'+img_no+'.json') as json_f:
		rows = json.load(json_f)
		cnt = 0
		has_obj = False

		cnt = 0
		for row in rows:
			confidence = row['confidence']
			
			topleft_x = row['topleft']['x']
			topleft_y = row['topleft']['y']
			bottomright_x = row['bottomright']['x']
			bottomright_y = row['bottomright']['y']

			region = im.crop((topleft_x,topleft_y,bottomright_x,bottomright_y))
			region.save('./cropped_images_yolo/{}_{}.jpg'.format(img_no,cnt))
			has_obj = True
			cnt += 1

		# if no bounding box, then output origin image
		if not has_obj:
			im.save('./crop_images_yolo/{}_0.jpg'.format(img_no))

	print('finished ',img_no)
