import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from skimage.measure import find_contours
from skimage.morphology import binary_opening, binary_closing
from skimage.morphology import binary_dilation, binary_erosion

from . import data_utils

tau = 2 * np.pi


def resize_and_recenter(points, num_points=None, recenter=False):
  # Interpolate the curve to make it have less number of points
  if num_points is not None:
    points = np.interp(x=np.linspace(0,1,num_points),xp=np.linspace(0,1,len(points)),fp=points)
  else:
    points = np.array(points)  
  if recenter:  
    points -= points.mean()  # Make it centered at origin (optional).
  return points
  
def mandelbrot(size, iters, num_points=1001):
  x, y = np.linspace(-2, 1, size), np.linspace(-1.5, 1.5, size)
  x, y = np.meshgrid(x, y)

  z0 = x + 1j * y
  z = z0.copy()
  for _ in tqdm(range(iters)):
    z[z<=2] = z[z<=2]**2 + z0[z<=2]

  img = np.zeros_like(z0, dtype=bool)
  img[z<=2] = True

  # Can optionally apply dilation/closing to not miss out points on boundary.
  img = binary_dilation(img)

  # Boundary is the contour with max length.
  contours = find_contours(img)
  main_contour = max(contours, key=len)

  points = main_contour[:,1] +  1j*main_contour[:,0]
  points.real = points.real*(3/size) - 2
  points.imag = points.imag*(3/size) - 1.5

  # Interpolate the curve to make it have less number of points
  points = resize_and_recenter(points, num_points)
  return points

def fermat_spiral(time_range, num_points=1001):
  t = np.linspace(*time_range, num_points)
  s, t = np.sign(t), abs(t)
  points = s*np.sqrt(t)*np.exp(1j*t)
  return points

def euler_spiral(num_steps=720, dth=tau/360, num_points=1001):
  th_values = np.arange(num_steps + 1)
  th_values = (th_values*(th_values+1) / 2) * dth
  pos_values = np.exp(1j * th_values)
  points = np.cumsum(pos_values)
  points = resize_and_recenter(points, num_points)
  return points

def polyskelion(Nsp=3, Nwh=4, a=1, dt=0.1, num_points=1001):
  '''Nsp: Number of spirals.
     Nwh: Number of whirls in each spiral.'''
  points = []
  dt = 0.1

  for n in range(Nsp):
    r, t = 0, 0
    point0 = a * np.exp(1j * tau * n / Nsp)
    t1 = tau * Nwh - tau / (2*Nsp) + tau / 4
    t2 = t1 + tau / Nsp
    c = (4 * a * np.sin(tau / (2*Nsp))) / (tau * (1 + 4 * Nwh))

    while t < t2:
      t += dt
      r = c * t
      point = point0 + r * np.exp(1j * (t + tau * n / Nsp))
      points.append(point)

      if t <= t1:
        point = point0 + r * np.exp(1j * (t + (tau/2) + tau * n / Nsp))
        points.append(point)

  points = np.array(points)
  points = data_utils.sort_points(points)
  points = resize_and_recenter(points, num_points)
  return points

def butterfly(num_points=1001):
  t = np.linspace(0,6*tau,num_points)
  d = np.exp(np.cos(t)) - 2*np.cos(4*t) - (np.sin(t/12))**5
  points = d*1j*np.exp(-1j*t)
  return points

def lorenz(a=10,b=28,c=8/3, dt=0.01, iters=5000, num_points=1001):
  x,y,z = 0.01,0,0
  points = [(x,y,z)]
  for _ in range(iters):
    dx, dy, dz = a*(y-x), x*(b-z) - y, x*y - c*z
    x += dx*dt; y += dy*dt; z += dz*dt
    points.append((x,y,z))
  points = np.array([x+1j*z for x,y,z in points])
  points = resize_and_recenter(points, num_points)
  return points


# Some simple functions. #
def r(angle, a=1,b=1,m=16,n1=0.5,n2=0.5,n3=16):
  x = abs(np.cos(m*angle/4) / a) ** n2
  y = abs(np.sin(m*angle/4) / b) ** n3
  return abs(x + y) ** (-1 / n1)
  
def superformula(angle=(0, tau), a=1,b=1,m=16,n1=0.5,n2=0.5,n3=16, num_points=1001):
  angle = np.linspace(*angle, num_points)
  radii = np.array(list(map(lambda angle: r(angle,a,b,m,n1,n2,n3), angle)))
  points = radii * np.exp(1j * angle)
  return points

# Can be approximated with very few coeffs. 
def lissajous_curve(a=5,b=5,delta=tau/4,A=1,B=1,num_points=1001):
  t = np.linspace(0,tau, num_points)
  x = A * np.sin(a*t + delta)
  y = B * np.sin(b*t)
  return x + 1j*y
# Simple functions end. #


def L_system_generate(rules, start, iters):
  state = start
  for _ in range(iters):
    new_state = ''
    for var in state:
      if var in rules:
        new_state += rules[var]
      else:
        new_state += var
    state = new_state
  return state

def L_system_evaluate(instructions, angle=tau/4):
  point, theta = 0 + 0j, 0
  points = [point]
  for instruction in instructions:
    if instruction == 'F' or instruction == 'G':
      point += np.exp(1j*theta)
      points.append(point)
    elif instruction == '+':
      theta += angle
    elif instruction == '-':
      theta -= angle
  return np.array(points)

def peano_curve(iters=3, num_points=729):  # Disconnected.
  instructions = L_system_generate(rules={'L': 'LFRFL-F-RFLFR+F+LFRFL', 'R': 'RFLFR+F+LFRFL-F-RFLFR'},
                                   start='L', iters=iters)
  
  points = L_system_evaluate(instructions)
  points = resize_and_recenter(points, num_points)
  return points
  
def peano_curve2(iters=3, num_points=730):  # Disconnected.
  instructions = L_system_generate(rules={'F': 'FF+F+F+FF+F+F-F'},
                                   start='F', iters=iters)
  
  points = L_system_evaluate(instructions)
  points = resize_and_recenter(points, num_points)
  return points

def hilbert_curve(iters=5, num_points=1024):  # Disconnected.
  instructions = L_system_generate(rules={'A': '+BF-AFA-FB+', 'B': '-AF+BFB+FA-'},
                                   start='A', iters=iters)
  
  points = L_system_evaluate(instructions)
  points = resize_and_recenter(points, num_points)
  return points

def moore_curve(iters=4, num_points=1024):
  instructions = L_system_generate(rules={'L': '-RF+LFL+FR-', 'R': '+LF-RFR-FL+'},
                                   start='LFL+F+LFL', iters=iters)
  
  points = L_system_evaluate(instructions)
  points = resize_and_recenter(points, num_points)
  return points

def koch_snowflake(iters=5, num_points=1537):
  instructions = L_system_generate(rules={'F':'F+F--F+F'},
                                   start='F--F--F', iters=iters)
  points = L_system_evaluate(instructions=instructions, angle=tau/6)
  points = resize_and_recenter(points, num_points)
  return points

def sierpinski_triangle(iters=6, num_points=2188):
  instructions = L_system_generate(rules={'F': 'FF', 'X': '--FXF++FXF++FXF--'},
                                 start='FXF--FF--FF', iters=iters)
  points = L_system_evaluate(instructions=instructions, angle=tau/6)
  points = -resize_and_recenter(points, num_points)
  return points

def sierpinski_triangle_2(iters=6, num_points=2188):
  instructions = L_system_generate(rules={'F': 'F-G+F+G-F', 'G': 'GG'},
                                 start='F-G-G', iters=iters)
  points = L_system_evaluate(instructions=instructions, angle=tau/3)
  points = -resize_and_recenter(points, num_points)
  return points

def sierpinski_arrowhead(iters=7, num_points=2188):
  instructions = L_system_generate(rules={'X': 'YF+XF+Y', 'Y': 'XF-YF-X'},
                                 start='XF', iters=iters)
  points = L_system_evaluate(instructions=instructions, angle=tau/6)
  points *= np.exp(1j*tau/(2+iters%2)) # Make it always facing upwards.
  points = resize_and_recenter(points, num_points)
  return points  

def sierpinski_square(iters=4, num_points=1365):
  instructions = L_system_generate(rules={'X': 'XF-F+F-XF+F+XF-F+F-X'},
                                 start='F+XF+F+XF', iters=iters)
  points = L_system_evaluate(instructions=instructions)
  points *= np.exp(1j*tau/8)
  points = resize_and_recenter(points, num_points)
  return points

def box(iters=4, num_points=2501):
  instructions = L_system_generate(rules={'F': 'F-F+F+F-F'},
                                 start='F-F-F-F', iters=iters)
  points = L_system_evaluate(instructions=instructions)
  points = resize_and_recenter(points, num_points)
  return points

def twin_dragon_curve(iters=9, num_points=1025):
  instructions = L_system_generate(rules={'X': 'X+YF', 'Y': 'FX-Y'},
                                 start='FX+FX+', iters=iters)
  points = L_system_evaluate(instructions=instructions)
  points = resize_and_recenter(points, num_points)
  points = (points*1j).conjugate()
  return points

def levy(iters=10, num_points=1025):  # Disconnected.
  instructions = L_system_generate(rules={'F': '+F--F+'},
                                 start='F', iters=iters)
  points = L_system_evaluate(instructions=instructions, angle=tau/8)
  points = -resize_and_recenter(points, num_points)
  return points

def gosper(iters=4, num_points=2402):  # Disconnected.
  instructions = L_system_generate(rules={'F':'F-G--G+F++FF+G-', 'G': '+F-GG--G-F++F+G'},
                                 start='F', iters=iters)
  points = L_system_evaluate(instructions=instructions, angle=tau/6)
  points = resize_and_recenter(points, num_points)
  return points
