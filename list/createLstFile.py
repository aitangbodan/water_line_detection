import glob
import random
import os

img_dirs = glob.glob('../../wsoc11/images/*.jpg')
img_dirs.extend(glob.glob('../../wsoc11/images/*.png'))
random.shuffle(img_dirs)
n_data = len(img_dirs)

train = []
val = []
for i, img_dir in enumerate(img_dirs):
    _, img_name = os.path.split(img_dir)
    final_img_dir = os.path.join('images', img_name) # 'wsoc11', 
    seg_dir = final_img_dir.replace('images', 'labels')
    seg_dir = seg_dir.replace('jpg', 'png')
    
    if i < n_data * 0.98:
        train.append(f'{final_img_dir}\t{seg_dir}\n')
    else: 
        val.append(f'{final_img_dir}\t{seg_dir}\n')  
        
with open('train.lst', 'w') as f:
    f.writelines(train)
with open('test.lst', 'w') as f:
    f.writelines(val)