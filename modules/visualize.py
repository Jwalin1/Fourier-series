import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection

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


def get_arrows(lines):
  lines_diffs = np.diff(lines).squeeze()
  arrow_lens = abs(lines_diffs) / 10
  angles = np.angle(lines_diffs)

  arrows = np.zeros((len(lines), 3), dtype=complex)
  arrows[:,0] = lines[:,1] - arrow_lens * np.exp(1j*(angles - tau/8))
  arrows[:,1] = lines[:,1]
  arrows[:,2] = lines[:,1] - arrow_lens * np.exp(1j*(angles + tau/8))
  return arrows

def get_orientations(points_t0, points_t1):
  differences_t0 = np.diff(points_t0)
  differences_t1 = np.diff(points_t1)
  angles_t0 = np.angle(differences_t0)
  angles_t1 = np.angle(differences_t1)
  orientations = np.sign(angles_t1 - angles_t0)
  return orientations

def plot_colored_lines(lines):
  plot_complex(lines[::2].T, color='red')
  plot_complex(lines[1::2].T, color='blue')

def plot_circles(centers, radii):
  patches = [plt.Circle((center.real, center.imag), radius)
                    for center, radius in zip(centers, radii)]
  coll_circles = PatchCollection(patches, alpha=0.3, color='white', facecolor='None')
  plt.gca().add_collection(coll_circles)



def evolution_animate(points_apprs, show_stats=True):
  fig, ax = plt.subplots(figsize=(6,6))  # Using default figsize.
  camera = Camera(fig)
  plt.axis('off')
  
  #Looping the data and capturing frame at each iteration
  for i, points_appr in enumerate(tqdm(points_apprs, desc='generating evolution animation')):
    plot_complex(points_appr, c='y')
    if show_stats:
      plt.text(0.5, 1.05, f'coeffs: {i+1}', transform = ax.transAxes)
    camera.snap()
  plt.close(fig)  
    
  anim = camera.animate()
  return anim


def epicycles_animate(centers_time, detail=7, show_stats=True):
  '''
  detail =  1: white lines
  detail =  3: arrowed white lines
  detail =  7: arrowed white lines with circles
  detail = 10: colored (blue & red) lines
  detail = 13: colored (blue & red) lines with arrows
  detail = 17: colored (blue & red) lines with circles
  '''
  fig, ax = plt.subplots(figsize=(6,6))  # Using default figsize.
  camera = Camera(fig)
  plt.axis('off')

  n_samples, n_centers = centers_time.shape
  theta = np.linspace(0, tau, 36)
  orientations = get_orientations(centers_time[0], centers_time[1])
  colors = ['r' if orientation==1 else 'b'for orientation in orientations]
  #Looping the data and capturing frame at each iteration
  for i, centers in enumerate(tqdm(centers_time, desc='generating epicycles animation')):
    radii = abs(np.diff(centers))
    lines = np.column_stack((centers[:-1], centers[1:]))
    # circles = centers[:-1][:,None] + radii[:,None] * np.exp(1j * theta.T)

    if detail >= 1 and detail < 10:
      plot_complex(centers, color='white')
      if detail >= 3:
        arrows = get_arrows(lines)
        plot_complex(arrows.T, color='white')
      if detail >= 7:
        plot_circles(centers, radii)

    elif detail >= 10 and detail < 20:
      # Problem with z order of lines.
      # plot_colored_lines(lines)
      
      # Handles z order properly.
      coll_lines = LineCollection(np.dstack([lines.real, lines.imag]), colors=colors)
      ax.add_collection(coll_lines)
      if detail >= 13:
        arrows = get_arrows(lines)
        # plot_colored_lines(arrows)
        coll_arrows = LineCollection(np.dstack([arrows.real, arrows.imag]), colors=colors)
        ax.add_collection(coll_arrows)
      if detail >= 17:
        plot_circles(centers, radii)

    # Plot curve drawn so far.
    plot_complex(centers_time[:i+1, -1], c='y')
    if show_stats:  # Display the elapsed time
      plt.text(0.4, 1.05, f'time: {i / (n_samples-1):.3f}', transform = ax.transAxes)

    camera.snap()
  plt.close(fig)

  anim = camera.animate()
  return anim