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
  return points_apprs



def get_circle_centers(coeffs, num_samples):
  # Generate data to animate.
  num_centers = num_coeffs = len(coeffs)

  time_points = np.linspace(0, 1, num_samples)
  centers_time = np.zeros((num_samples, num_centers), dtype=complex)

  indices = np.arange(len(coeffs))  #  0,  1,  2,  3,  4, ...
  signs = 2*(indices % 2) - 1       # -1,  1, -1,  1, -1, ...
  magnitudes = (indices + 1) // 2   #  0,  1,  1,  2,  2, ...
  sequence = signs * magnitudes     #  0,  1, -1,  2, -2, ...

  for i, time in enumerate(tqdm(time_points, desc='calculating centers and radii')):
    series = coeffs * np.exp(1j * tau * sequence * time)
    centers = series.cumsum(0)
    centers_time[i] = centers

  return centers_time