import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# For animation.
from celluloid import Camera

tau = 2 * np.pi
plt.style.use('dark_background')



def plot_complex(z, **kwargs):
  plt.plot(z.real, z.imag, **kwargs)

def double_plot(points, points_appr, combined=False):

  if combined:
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_complex(points, alpha=0.5)
    plot_complex(points_appr, alpha=0.5)

  else:
    fig, axs = plt.subplots(1,2, figsize=(16, 8))

    plt.sca(axs[0])
    plot_complex(points)

    plt.sca(axs[1])
    plot_complex(points_appr)


def add_arrows(points):

  diff_points = np.diff(points)
  arrow_lens = abs(diff_points) / 10
  angles = np.angle(diff_points)
  angles -= tau / 2

  arrowed_points = np.zeros(5*len(points) - 4, dtype=complex)
  arrowed_points[0] = points[0]

  points = points[1:]
  arrowed_points[1::5] = points
  arrowed_points[2::5] = points + arrow_lens * np.exp(1j*(angles - tau/8))
  arrowed_points[3::5] = points
  arrowed_points[4::5] = points + arrow_lens * np.exp(1j*(angles + tau/8))
  arrowed_points[5::5] = points
  return arrowed_points




def evolution_animate(points_apprs, errs=None, info=1):
  if errs is None and info >= 3:
    raise ValueError("errs argument is required when info >= 3")

  fig, ax = plt.subplots(figsize=(6,6))  # Using default figsize.
  camera = Camera(fig)
  if not info >= 3:
    plt.axis('off')
  
  #Looping the data and capturing frame at each iteration
  for i, points_appr in enumerate(tqdm(points_apprs, desc='generating evolution animation')):
    plot_complex(points_appr, c='y')
    if info >= 1:
      plt.text(0.5, 1.05, f'coeffs: {i+1}', transform = ax.transAxes)
    if info >= 2:  
      plt.text(0.25, 1.05, f'MAE: {errs[i]:.3f}', transform = ax.transAxes)

    camera.snap()
  plt.close(fig)  
    
  anim = camera.animate()
  return anim


def epicycles_animate(centers_time, radii_time, circle_points_time, info=1):
  fig, ax = plt.subplots(figsize=(6,6))  # Using default figsize.
  camera = Camera(fig)
  if not info >= 2:
    plt.axis('off')

  n_samples = centers_time.shape[0]
  #Looping the data and capturing frame at each iteration
  for i in tqdm(range(n_samples), desc='generating epicycles animation'):
    centers, radii = centers_time[i], radii_time[i]
    # Plot lines with arrows.
    plot_complex(add_arrows(centers), c='w')
    # Plot circles.
    plot_complex(circle_points_time[i].T, c='w', alpha=0.3)
    # Plot curve drawn so far.
    plot_complex(centers_time[:i+1, -1], c='y')

    if info >= 1:
      plt.text(0.4, 1.05, f'time: {i / (n_samples-1):.3f}', transform = ax.transAxes)

    camera.snap()
  plt.close(fig)

  anim = camera.animate()
  return anim
