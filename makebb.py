import numpy as np

def makeBB(img, scale=1):
    if len(np.unique(img)) == 1:
        return None
    
    y, x = np.nonzero(img)
    if y.size == 0:
        return (0, 0, 0, 0)
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    w = xmax - xmin
    h = ymax - ymin

    if scale != 1:
        scale = 1 + (scale - 1)/2
        xmid = (xmin + w)/2.
        ymid = (ymin + h)/2.
        xmin = int(max(xmid*2 - w*scale, 0))
        ymin = int(max(ymid*2 - h*scale, 0))
        xmax = int(min((xmid + w*scale/2.), img.shape[1]))
        ymax = int(min((ymid + h*scale/2.), img.shape[0]))
        scale = 1 + (scale - 1)*2
        w = int(w*(scale))
        h = int(h*(scale))

    return (xmin, ymin, w, h)
