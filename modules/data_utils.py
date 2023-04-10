import numpy as np
from tqdm.auto import tqdm

# For parsing svg.
from svg.path import parse_path
from xml.dom import minidom



def parse_svg(svg_path, n_points=1000):
  doc = minidom.parse(svg_path)
  path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]

  n_paths = len(path_strings)
  n_ts = int(n_points/n_paths)

  points = []
  for path_indx, path_string in enumerate(tqdm(path_strings, desc='paths')):
    path_data = parse_path(path_string)

    t_start = path_indx / n_paths
    t_end = (path_indx+1) / n_paths
    points_t =  np.linspace(0, 1, n_ts)
    
    for time in tqdm(points_t, desc='time'):
      points.append(path_data.point(time))

  # Flip the imaginary component
  points = np.array(points).conj()
  return points


def sort_points(points, start_index=0):
  current_point = points[start_index]
  sorted_points = np.zeros_like(points)
  sorted_points[0] = current_point
  points = np.delete(points, start_index)

  for i in range(1,len(points)):
    # Find the index of the nearest point
    distances = abs(points - current_point)
    index = np.argmin(distances)
    current_point = points[index]

    sorted_points[i] = current_point
    points = np.delete(points, index)

  return sorted_points
