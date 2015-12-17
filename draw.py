import os,sys
import h5py
import numpy as np
import colorsys
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


hues = np.arange(len(tags))/float(len(tags))
saturations = np.ones(hues.shape)
values = np.ones(hues.shape)*0.67
hsv = np.array([hues,saturations,values]).T
rgb = np.zeros(hsv.shape)
for idx,row in enumerate(hsv):
    rgb[idx,:] = colorsys.hsv_to_rgb(*row)


leg_handles = []
leg_labels = []
label_lines = True
for idx,tag in enumerate(tags):
    #color = rgb[idx%len(rgb)]
    x1s = h5[tag]['x1']
    x2s = h5[tag]['x2']
    y1s = h5[tag]['y1']
    y2s = h5[tag]['y2']
    color = colorsys.hsv_to_rgb(np.mean(list(y1s)+list(y2s))/float(im.shape[0]),1.0,0.5)
    measurements_m = h5[tag]['measurements_m'][:]
    print '%s:\t\t%0.2e um (min),%0.2e um (max), %0.2e um (mean)'%(tag,np.min(measurements_m)*1e6,np.max(measurements_m)*1e6,np.mean(measurements_m)*1e6)

    leg_handle_added = False
    
    for x1,x2,y1,y2,m_m in zip(x1s,x2s,y1s,y2s,measurements_m):
        ph, = ax.plot([x1,x2],[y1,y2],color=color,linestyle='-',lw=3,alpha=0.75)
        if label_lines:
            theta = -np.arctan((y2-y1)/(x2-x1))*180.0/np.pi
            xt = (x1+x2)/2.0
            yt = (y1+y2)/2.0
            th = ax.text(xt,yt,'%0.1f $\mu m$'%(m_m*1e6),ha='center',va='top',rotation=theta,fontsize=6,color=color)
        if not leg_handle_added:
            leg_handles.append(ph)
            leg_labels.append(tag)
            leg_handle_added = True

if len(tags)>1:
    ax.legend(leg_handles,leg_labels,prop=fontP)


plt.xticks([])
plt.yticks([])

if not os.path.exists('./png'):
    os.makedirs('./png')

out_fn = os.path.join('./png',fn_root + '_' + '_'.join(tags) + '_annotated.png')

plt.savefig(out_fn,dpi=100)

plt.show()
