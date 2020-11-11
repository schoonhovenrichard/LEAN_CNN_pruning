import imageio
import numpy as np
import tifffile
from pathlib import Path

n = 256
nit = 24
np.random.seed(12345)
sz = n//8

def generate():
    im = np.zeros((n,n),dtype=np.float32)
    l = np.zeros((n,n),dtype=np.uint8)

    tmpl = np.zeros((4,sz,sz),dtype=np.float32)
    tmpl[0] = 1
    tmpl[1] = 1
    tmpl[1][sz//4:-sz//4,sz//4:-sz//4]=0
    xx,yy = np.mgrid[-1:1:1j*sz,-1:1:1j*sz]
    tmpl[2] = xx**2+yy**2<1
    tmpl[3] = xx**2+yy**2<1
    tmpl[3][xx**2+yy**2<0.25]=0

    i = 0
    tp = 0
    while i<nit:
        found=False
        while found==False:
            x, y = (np.random.random(2)*(n-sz)).astype(np.int)
            if l[x:x+sz,y:y+sz].max()==0:
                found=True
        vl = np.random.random()*0.8+0.2
        im[x:x+sz,y:y+sz] = tmpl[tp]*vl
        l[x:x+sz,y:y+sz] = tmpl[2*(tp//2)]*(tp+1)
        tp+=1
        if tp==4:
            tp=0
        i+=1

    imn = im+np.random.normal(size=im.shape)
    return imn, im, l

pth = Path('data/train')
pth.mkdir(parents=True, exist_ok=True)
(pth / 'noisy').mkdir(parents=True, exist_ok=True)
(pth / 'noiseless').mkdir(parents=True, exist_ok=True)
(pth / 'label').mkdir(parents=True, exist_ok=True)
for i in range(500):
    imn, im, l = generate()
    imageio.imsave(pth / 'noisy' / '{:05d}.tiff'.format(i), imn.astype(np.float32))
    imageio.imsave(pth / 'noiseless' / '{:05d}.tiff'.format(i), im.astype(np.float32))
    imageio.imsave(pth / 'label' / '{:05d}.tiff'.format(i), l.astype(np.uint8))
    
pth = Path('data/val')
pth.mkdir(parents=True, exist_ok=True)
(pth / 'noisy').mkdir(parents=True, exist_ok=True)
(pth / 'noiseless').mkdir(parents=True, exist_ok=True)
(pth / 'label').mkdir(parents=True, exist_ok=True)
for i in range(100):
    imn, im, l = generate()
    imageio.imsave(pth / 'noisy' / '{:05d}.tiff'.format(i), imn.astype(np.float32))
    imageio.imsave(pth / 'noiseless' / '{:05d}.tiff'.format(i), im.astype(np.float32))
    imageio.imsave(pth / 'label' / '{:05d}.tiff'.format(i), l.astype(np.uint8))
    
pth = Path('data/test')
pth.mkdir(parents=True, exist_ok=True)
(pth / 'noisy').mkdir(parents=True, exist_ok=True)
(pth / 'noiseless').mkdir(parents=True, exist_ok=True)
(pth / 'label').mkdir(parents=True, exist_ok=True)
for i in range(50):
    imn, im, l = generate()
    imageio.imsave(pth / 'noisy' / '{:05d}.tiff'.format(i), imn.astype(np.float32))
    imageio.imsave(pth / 'noiseless' / '{:05d}.tiff'.format(i), im.astype(np.float32))
    imageio.imsave(pth / 'label' / '{:05d}.tiff'.format(i), l.astype(np.uint8))   
