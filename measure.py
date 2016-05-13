import os,sys
import numpy as np
try:
    from scipy.misc import imread
except ImportError as ie:
    print ie
    from scipy import ndimage
    imread = ndimage.imread

from matplotlib import pyplot as plt
import h5py

fn = sys.argv[1]
fn_root,fn_ext = os.path.splitext(fn)
h5fn = '%s.hdf5'%fn_root
h5 = h5py.File(h5fn,'a')

h5.require_group('scalebar')


# set up a group in the h5 file for tagged objects
try:
    tag = sys.argv[2]
except IndexError:
    default_tag = 'test'
    tag = raw_input('Please enter a tag for the objects to be selected [%s]: '%default_tag)
    if len(tag)==0:
        tag = default_tag
h5.require_group(tag)


im = imread(fn)

fig = plt.figure()
ax = fig.add_subplot(111)

sbp1 = None
p1 = None
measurements_pixels = []
measurements_m = []
x1 = []
x2 = []
y1 = []
y2 = []

try:
    m_per_pixel = h5['scalebar']['m_per_pixel'][0]
    print 'Retrieved pixel scale of %0.1e m/px from h5 file.'%m_per_pixel
    endpoints = h5['scalebar']['endpoints'][:]
    sbp1 = endpoints[0]
    sbp2 = endpoints[1]
    plt.plot([sbp1[0],sbp2[0]],[sbp1[1],sbp2[1]],'g-',lw=2,alpha=0.5)
except Exception as e:
    print e
    m_per_pixel = 1.0

def scalebar_click(event):
    global sbp1
    global m_per_pixel
    if sbp1 is None:
        sbp1 = (event.xdata,event.ydata)
    else:
        sbp2 = (event.xdata,event.ydata)
        d = np.sqrt((sbp1[0]-sbp2[0])**2+(sbp1[1]-sbp2[1])**2)
        physical_length = float(raw_input('Physical length in meters: '))
        m_per_pixel = physical_length/float(d)
        plt.plot([sbp1[0],sbp2[0]],[sbp1[1],sbp2[1]],'g-',lw=2,alpha=0.5)
        plt.autoscale(False)
        plt.draw()
        h5['scalebar'].create_dataset('endpoints',data=np.array([sbp1,sbp2]))
        h5['scalebar'].create_dataset('length_px',data=np.array([d]))
        h5['scalebar'].create_dataset('length_m',data=np.array([physical_length]))
        h5['scalebar'].create_dataset('m_per_pixel',data=np.array([m_per_pixel]))
        print h5['scalebar'].keys()
        sbp1 = None
    
def add_measurement(p1,p2):
    x1.append(p1[0])
    x2.append(p2[0])
    y1.append(p1[1])
    y2.append(p2[1])
    d_pixels = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    d_m = d_pixels*m_per_pixel
    measurements_pixels.append(d_pixels)
    measurements_m.append(d_m)
    
    plt.plot([p1[0],p2[0]],[p1[1],p2[1]],'r-',lw=2,alpha=0.5)
    plt.autoscale(False)
    plt.draw()

def report():
    print '%0.1e meters/pixel'%m_per_pixel,measurements_m


#def draw():
#    plt.cla()
    
    
def onclick(event):
    global p1
#    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f, key=%s'%(
#        event.button, event.x, event.y, event.xdata, event.ydata, event.key)
    if event.key=='shift':
        scalebar_click(event)
    else:
        if event.key is None:
            p1 = (event.xdata,event.ydata)
        if event.key=='control':
            p2 = (event.xdata,event.ydata)
            add_measurement(p1,p2)
            p1 = None

cid = fig.canvas.mpl_connect('button_press_event', onclick)

ax.imshow(im)
plt.title('shift-click ends of scalebar, then enter length; click/ctrl-click object edges after')
plt.show()

print h5['scalebar'].keys()


h5[tag].create_dataset('x1',data=x1)
h5[tag].create_dataset('x2',data=x2)
h5[tag].create_dataset('y1',data=y1)
h5[tag].create_dataset('y2',data=y2)
h5[tag].create_dataset('measurements_pixels',data=measurements_pixels)
h5[tag].create_dataset('measurements_m',data=measurements_m)

print h5['scalebar'].keys()

print h5[tag].keys()

h5.close()
