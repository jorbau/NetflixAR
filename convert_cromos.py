from PIL import Image
import os
orig_path = ('Input/')
dest_path = ('Output/')
images_name = os.listdir('Input')
im_done = []
for im_name in images_name:
    ind = im_name.index('.')
    if im_name[ind-1] == '1' or im_name[:ind] in im_done:
        continue

    im_done.append(im_name[:ind])
    im = Image.open(orig_path + im_name)
    name = im_name[:ind+1] + 'png'
    im.save(dest_path + name, "png")


    
