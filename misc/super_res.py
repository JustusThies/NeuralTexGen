import numpy as np
from PIL import Image
from ISR.models import RDN

#rdn = RDN(weights='psnr-small')
rdn = RDN(weights='noise-cancel')

def upsample(filename_in, filename_out):
    img = Image.open(filename_in)
    print(img)
    lr_img = np.array(img)
    sr_img = rdn.predict(lr_img)
    #sr_img = rdn.predict(lr_img, by_patch_of_size=50)
    res = Image.fromarray(sr_img)
    res.save(filename_out)


img_list = [83, 
177,
181,
185,
207,
289,
314,
317,
366,
367,
368,
369,
370,
371,
381,
387,
389,
392,
393,
394,
396,
400,
430,
542,
543,
544,
545,
636,
637,
638,
640,
643,
646,
667,
669,
670,
750,
751,
753,
775,
776,
802,
829,
843,
845,
848,
850,
851,
854,
856,
889,
890,
891,
892,
914,
916,
918,
922,
1002,
1005,
1098,
1104,
1105,
1106,
1108,
1109,
1128,
1129,
1130,
1133,
1213,
1216,
1217,
1240,
1290,
1294,
1295,
1302,
1309,
1310,
1314,
1315,
1316,
1319,
1320,
1352,
1375,
1379,
1381]

for id in img_list:
    upsample('datasets/room1_us/frames/ZMojNkEp431/color/' + str(id) + '.jpg', 'datasets/room1_us/images/' + str(id) + '.jpg')