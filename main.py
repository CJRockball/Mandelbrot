from mandel_calc import run_mandel
from plot import draw_image_func

min_x, max_x, min_y, max_y = -2.0, 1.0, -1.0, 1.0
gimage = run_mandel(min_x, max_x, min_y, max_y)
draw_image_func(gimage, min_x, max_x, min_y, max_y)



