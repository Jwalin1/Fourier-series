import os
import argparse
from matplotlib import animation

from modules.data_utils import parse_svg
from fourier_core import get_coeffs, get_fourier_approximation

from modules.fourier_utils import get_apprs
from modules.visualize import evolution_animate

from modules.fourier_utils import get_epicycle_data
from modules.visualize import add_arrows, epicycles_animate



def main(svg_path, order=100):
    # Your main code here
    print(f"SVG file path: {svg_path}")
    print(f"Order of terms used for approximation: {order}")

    base = os.path.basename(svg_path)
    file_name = os.path.splitext(base)[0]

    print('\nReading svg')
    points = parse_svg(svg_path)

    print('\nEvolution animation')
    coeffs = get_coeffs(points, order)
    x_apprs, errs = get_apprs(points, coeffs, order)
    anim = evolution_animate(x_apprs, errs)
    anim.save(f"animations/{file_name}_evolution.gif", writer=animation.PillowWriter(fps=10))


    print('\nEpicycles animation')
    n_samples = len(points)  # Number of time points to sample.
    centers_time, radii_time, circle_pts_time = get_epicycle_data(coeffs, n_samples, order)
    anim = epicycles_animate(centers_time, radii_time, circle_pts_time)
    anim.save(f"animations/{file_name}_epicycles.gif", writer=animation.PillowWriter(fps=10))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_path", help="Path to the SVG file")
    parser.add_argument("-o", "--order", type=int, default=100,
                        help="Order of terms used for approximation (default: 1)")
    args = parser.parse_args()

    main(args.svg_path, args.order)