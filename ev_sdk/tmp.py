# d = {'airplane': 0, 'banana': 1, 'baseball': 2, 'bicycle': 3, 'bird': 4, 'book': 5, 'bulldozer': 6, 'cake': 7, 'camel': 8, 'camera': 9, 'cannon': 10, 'car': 11, 'cat': 12, 'chair': 13, 'computer': 14, 'cookie': 15, 'crown': 16, 'dog': 17, 'ear': 18, 'eye': 19, 'fish': 20, 'flower': 21, 'hand': 22, 'hat': 23, 'horse': 24, 'keyboard': 25, 'key': 26, 'knife': 27, 'ladder': 28, 'monkey': 29, 'mouse': 30, 'nose': 31}

# dd = {}
# for k, v in d.items():
#     dd[v] = k

# print(dd)

import os
import cv2

# root = '/home/data/7'
# save = '/home/data/112'
raw = '/usr/local/ev_sdk/raw'

ff = open('/usr/local/ev_sdk/file_list.txt', 'w')

for i in os.listdir(raw):
    if i.endswith('.raw'):
        path = os.path.join(raw, i)
        ff.write('{}\n'.format(path))
        print(path)

ff.close()