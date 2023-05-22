import numpy as np
from timeit import default_timer as timer
from numba import cuda
import numpy as np
import math


def mandel(x, y, max_iters):
  """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
        # Smooth iteration count
        temp = i + 1 - math.log(math.log(abs(z*z+c)))/math.log(2)
        return temp  


  return max_iters


mandel_gpu = cuda.jit('int32(float64, float64, float64)', device=True)(mandel)


@cuda.jit('(float64, float64, float64, float64, uint8[:,:], uint32)')
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
  """
  GPU function
  """
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX, startY = cuda.grid(2)
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandel_gpu(real, imag, iters)
      

def run_mandel(min_x, max_x, min_y, max_y):
  """
  Helper function to set up and run functions for calculating image matrix
  """
  gimage = np.zeros((1334,2000), dtype = np.uint8) #Pre-determined image size 1334x2000
  blockdim = (32, 8)
  griddim = (32,16)
  iter_lvl = 1000

  start = timer()
  d_image = cuda.to_device(gimage)
  mandel_kernel[griddim, blockdim](min_x, max_x, min_y, max_y, d_image, iter_lvl) 
  gimage = d_image.copy_to_host()
  dt = timer() - start
  print("Mandelbrot created on GPU in %f s" % dt)

  return gimage


def calc_dim(xpos,ypos,xmin, xmax, ymin, ymax):
    """
    Function to calculate the area in a zoomed image, base on where you click.
    """
    ypixels, xpixels = 1334, 2000 # Pre determined x, y size
    halfY, halfX = ypixels/2, xpixels/2
    
    # Step size
    pixel_size_x = (xmax - xmin) / xpixels
    pixel_size_y = (ymax - ymin) / ypixels
    
    # image lower left = (0,0), initial min_x, min_y = (-2, -1)
    xcenter = xmin + xpos*pixel_size_x
    ycenter = ymin + ypos*pixel_size_y

    #Calculate x, y bounadries for new image
    min_x= xcenter-halfX/10*pixel_size_x
    max_x = xcenter+halfX/10*pixel_size_x
    min_y= ycenter-halfY/10*pixel_size_y
    max_y = ycenter+halfY/10*pixel_size_y
    
    return min_x, max_x, min_y, max_y