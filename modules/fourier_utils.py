from typing import OrderedDict
import numpy as np
from tqdm.auto import tqdm
from fourier_core import fourier_series

tau = 2 * np.pi



def get_apprs(points, coeffs):
  num_coeffs = len(coeffs)
  points_apprs = np.zeros((num_coeffs, len(points)), dtype=complex)
  for n in tqdm(range(num_coeffs), desc='calculating approximations'):
    # coeffs = [c0, c1, c-1, c2, c-2, ...]
    points_apprs[n] = fourier_series(coeffs[:n], len(points))

  errs = abs(points - points_apprs).mean(axis=1)  
  return points_apprs, errs



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


def get_epicycle_data(coeffs, num_samples):
  # Generate data to animate.
  num_centers = num_coeffs = len(coeffs)

  time_points = np.linspace(0, 1, num_samples)
  centers_time = np.zeros((num_samples, num_centers), dtype=complex)
  radii_time = np.zeros((num_samples, num_centers-1), dtype=float)

  circle_points = 36
  angle = np.linspace(0, tau, circle_points)
  circle_points_time = np.zeros((num_samples, num_centers-1, circle_points), dtype=complex)

  for i, time_pt in enumerate(tqdm(time_points, desc='calculating centers and radii')):
    centers, radii = get_epicycles(coeffs, time_pt)
    circle_points = centers[:-1,None] + radii[:,None] * np.exp(1j * angle)[:,None].T
    centers_time[i], radii_time[i] = centers, radii
    circle_points_time[i] = circle_points

  return centers_time, radii_time, circle_points_time