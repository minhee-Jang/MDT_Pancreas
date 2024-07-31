import numpy as np
import PIL.Image as Image
import cv2
import os
image= np.load('/home/aix24705/data/MDT/output_mdt_xl2_eval/samples_6000x512x512x3.npz')
output_dir = '/home/aix24705/data/MDT/31000_sampling_images/'
#mu=image['arr_0']

array1=image['arr_0']
array2=image['arr_1']

#print(array1.shape)
#print(array2)
#os.mkdir('./results_image')

os.makedirs(os.path.join(output_dir, 'BENIGN'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'MALIGNANT'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'NORMAL'), exist_ok=True)

folders = {
    0: './31000_sampling_images/BENIGN',
    1: './31000_sampling_images/MALIGNANT',
    2: './31000_sampling_images/NORMAL'
}

for i in range(len(array1)):
    label = array2[i]
    folder = folders[label]
    img_path = os.path.join(folder, f'sample{i}_label{label}.png')
    cv2.imwrite(img_path, array1[i])



# import numpy as np
# import PIL.Image as Image
# import cv2
# import os
# image= np.load('/home/aix24705/data/MDT/output_mdt_xl2_eval/samples_5000x512x512x3_random.npz')
# #mu=image['arr_0']

# array1=image['arr_0']
# array2=image['arr_1']

# #print(array1.shape)
# #print(array2)
# #os.mkdir('./results_image')

# for i in range(5000): 
#     img=cv2.imwrite('./results_image_310000/sample{}_label{}.png'.format(i, array2[i]),array1[i])