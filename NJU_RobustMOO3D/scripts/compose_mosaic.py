import numpy as np, matplotlib.image as mpimg

def stitch_3x3(img_paths, out_path):
    imgs=[mpimg.imread(str(p)) for p in img_paths]
    h = min(im.shape[0] for im in imgs); w = min(im.shape[1] for im in imgs)
    proc=[]
    for im in imgs:
        if im.ndim==2: im = np.stack([im]*3,axis=-1)
        proc.append(im[:h,:w,:3])
    row1=np.concatenate(proc[0:3], axis=1)
    row2=np.concatenate(proc[3:6], axis=1)
    row3=np.concatenate(proc[6:9], axis=1)
    mosaic=np.concatenate([row1,row2,row3], axis=0)
    mpimg.imsave(out_path, mosaic)
