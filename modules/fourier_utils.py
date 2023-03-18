import numpy as np
from tqdm.auto import tqdm
from fourier_core import fourier_series

tau = 2 * np.pi



def get_apprs(points, coeffs):
  num_coeffs = len(coeffs)
  points_apprs = np.zeros((num_coeffs, len(points)), dtype=complex)
  for n in tqdm(range(1, 1 + num_coeffs), desc='calculating approximations'):
    # coeffs = [c0, c1, c-1, c2, c-2, ...]
    coeffs_n = dict(list(coeffs.items())[:n])
    points_apprs[n-1] = fourier_series(coeffs_n, len(points))
  return points_apprs



def get_circle_centers(coeffs, num_samples, sort=False):
  # Generate data to animate.
  num_centers = len(coeffs)

  time_points = np.linspace(0, 1, num_samples)
  centers_time = np.zeros((num_samples, num_centers), dtype=complex)

  sequence, coeffs = map(np.array, zip(*coeffs.items()))

  if sort:  # Sort all except first elem
    sort_indices = np.argsort(abs(coeffs[1:]))[::-1]
    coeffs = np.append(coeffs[0], coeffs[1:][sort_indices])
    sequence =  np.append(0, sequence[1:][sort_indices])

  for i, time in enumerate(tqdm(time_points, desc='calculating centers and radii')):
    vectors = coeffs * np.exp(1j * tau * sequence * time)
    centers = vectors.cumsum(0)
    centers_time[i] = centers

  return centers_time