import json
import glob
from PIL import Image,ImageDraw
import os
import numpy as np
lbl_dir_gen = "./lbl/"

pil_img = Image.open('../2007_000033.png')
palette = pil_img.getpalette()

path=('./*.json')
fileList=glob.glob(path)

for i,f in enumerate(fileList):
    file = open(f, 'r')
    print(file)
    jsonData = json.load(file)
    h=jsonData['imageHeight']
    w=jsonData['imageWidth']

    im = Image.new('P', (w, h))
    draw = ImageDraw.Draw(im)
    for t in range(len(jsonData['shapes'])):
        points2=jsonData['shapes'][t]['points']
        array=[]
        for j,p in enumerate(points2):
            array.append(p[0])
            array.append(p[1])
        
        draw.polygon(array, fill=1, outline=255)#fill=21
     
     
    basename=os.path.basename(f)
    root_ext_pair = os.path.splitext(basename)
    file.close()
    im.putpalette(palette)
    im.save(lbl_dir_gen + root_ext_pair[0] + ".png", quality = 100)

