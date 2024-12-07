# AreaVolume

A Python project for generating and visualizing three-dimensional rotational volumes derived from mathematical function intersections.

## Features

- 3D visualization of function intersections
- Rotation of intersection regions to create volumes
- Custom axes with tick marks and labels
- Interactive viewing with Mayavi
- Support for multiple mathematical functions

## Dependencies

- Python 3.9+
- NumPy
- SciPy
- Mayavi
- VTK

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/AreaVolume.git
```

2. Create and activate a conda environment:
```bash
conda create -n Volume python=3.9
conda activate Volume
```

3. Install dependencies:
```bash
conda install numpy scipy mayavi
```

## Usage

Run the Mayavi visualization:
```bash
python volumetric_plots_mayavi.py
```

## Controls

- Left mouse button: Rotate view
- Middle mouse button: Zoom
- Right mouse button: Pan
