import os,sys
import h5py
import numpy as np
try:
    from scipy.misc import imread
except ImportError as ie:
    print ie
    from scipy import ndimage
    imread = ndimage.imread

from matplotlib import pyplot as plt

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')

   
fn = sys.argv[1]
fn_root,fn_ext = os.path.splitext(fn)
h5fn = '%s.hdf5'%fn_root
h5 = h5py.File(h5fn,'r')

im = imread(fn)

fig = plt.figure()
#ax = fig.add_subplot(111)
ax = plt.axes([0,0,1,1])

ax.imshow(im)
plt.autoscale(False)
try:
    sb_endpoints = h5['scalebar']['endpoints'][:]
    xs = sb_endpoints[:,0]
    ys = sb_endpoints[:,1]
    ax.plot(xs,ys,'g-',lw=2,alpha=0.7)
except:
    pass


tags = sys.argv[2:]

if len(tags):
    if tags[0]=='all':
        tags = [x for x in h5.keys() if x not in ['scalebar','test']]


colors = 'rgbkcym'
leg_handles = []
leg_labels = []
for idx,tag in enumerate(tags):
    color = colors[idx%len(colors)]
    x1s = h5[tag]['x1']
    x2s = h5[tag]['x2']
    y1s = h5[tag]['y1']
    y2s = h5[tag]['y2']
    measurements_m = h5[tag]['measurements_m'][:]
    print '%s:\t\t%0.2e um (min),%0.2e um (max), %0.2e um (mean)'%(tag,np.min(measurements_m)*1e6,np.max(measurements_m)*1e6,np.mean(measurements_m)*1e6)

    leg_handle_added = False
    
    for x1,x2,y1,y2 in zip(x1s,x2s,y1s,y2s):
        ph, = ax.plot([x1,x2],[y1,y2],'%s-'%color,lw=2,alpha=0.5)
        if not leg_handle_added:
            leg_handles.append(ph)
            leg_labels.append(tag)
            leg_handle_added = True

if len(tags)>1:
    ax.legend(leg_handles,leg_labels,prop=fontP)


plt.xticks([])
plt.yticks([])

out_fn = fn_root + '_' + '_'.join(tags) + '_annotated.png'

plt.savefig(out_fn,dpi=100)

plt.show()
