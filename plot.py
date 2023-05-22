import numpy as np
import matplotlib.pyplot as plt
from mandel_calc import calc_dim, run_mandel

# Set global variables, x1, x2, y1, y2 to originial dims
x1=-2.0
x2=1.0
y1=-1.0
y2=1.0

def draw_image_func(mat, x1, x2, y1, y2, cmap='inferno_r', powern=0.5):

    mat = np.power(mat, powern) #Smoothing

    def onevent(event, mat):
        global x1, x2, y1, y2
        
        if event.xdata != None and event.ydata != None:
            x1, x2, y1, y2 = calc_dim(event.xdata, event.ydata, x1, x2, y1, y2)
            
            gimage = run_mandel(x1, x2, y1, y2)
            mat = gimage
            imbrot.set_data(mat)
            imbrot.autoscale()
            imbrot.figure.canvas.draw_idle()

    # Plot figure
    plt.figure()            
    ax = plt.gca()
    imbrot = ax.imshow(mat, cmap=cmap, origin='lower')
     
    cid = imbrot.figure.canvas.mpl_connect('button_press_event', lambda event: onevent(event, mat))

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.show()