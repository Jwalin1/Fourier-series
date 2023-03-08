from typing import OrderedDict
import numpy as np
from tqdm.auto import tqdm
from fourier_core import fourier_series

tau = 2 * np.pi



def get_apprs(pts, coeffs, order):
  pts_apprs = np.zeros((order, len(pts)), dtype=complex)
  for order in tqdm(range(order), desc='calculating approximations'):
    # coeffs = [c0, c1, c-1, c2, c-2, ...]
    pts_apprs[order] = fourier_series(coeffs[:1 + 2*order], len(pts))

  errs = abs(pts - pts_apprs).mean(axis=1)  
  return pts_apprs, errs



def get_epicycles(coeffs, time):
  centers = np.zeros(len(coeffs), dtype=complex)

  indices = np.arange(len(coeffs))  #  0,  1,  2,  3,  4, ...
  signs = 2*(indices % 2) - 1       # -1,  1, -1,  1, -1, ...
  magnitudes = (indices + 1) // 2   #  0,  1,  1,  2,  2, ...
  sequence = signs * magnitudes     #  0,  1, -1,  2, -2, ...

  series = coeffs * np.exp(1j * tau * sequence * time)
  centers = series.cumsum(0)

  # Both give same result.
  # radii = abs(np.diff(centers))
  radii = abs(coeffs[sequence[1:]])
  return centers, radii


def get_epicycle_data(coeffs, n_samples, order=None):
  # Generate data to animate.
  n_centers = 1 + 2*order if order else len(coeffs)  # number of coeffs to sum

  time_pts = np.linspace(0, 1, n_samples)
  centers_time = np.zeros((n_samples, n_centers), dtype=complex)
  radii_time = np.zeros((n_samples, n_centers-1), dtype=float)

  circle_pts = 36
  angle = np.linspace(0, tau, circle_pts)
  circle_pts_time = np.zeros((n_samples, n_centers-1, circle_pts), dtype=complex)

  for i, time_pt in enumerate(tqdm(time_pts, desc='calculating centers and radii')):
    centers, radii = get_epicycles(coeffs[:n_centers], time_pt)
    circle_pts = centers[:-1,None] + radii[:,None] * np.exp(1j * angle)[:,None].T
    centers_time[i], radii_time[i] = centers, radii
    circle_pts_time[i] = circle_pts

  return centers_time, radii_time, circle_pts_time