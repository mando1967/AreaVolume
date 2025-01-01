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

## Coordinate System Transformations

The application implements three key coordinate system transformations:

1. **Multiple Coordinate Systems in Single Scene**: 
   - Creates three separate coordinate systems within one Mayavi scene
   - Uses fixed offsets (`PLOT_SPACING`) to position each plot
   - Enables simultaneous visualization of disk method for both functions and washer method

2. **Normalization Transformation**:
   - Normalizes all three plots to fit within a consistent region
   - Uses scale factors based on `NORM_SCALE` constant
   - Maintains consistent proportions and spacing regardless of function values
   - Ensures proper visual comparison between different methods

3. **Zoom Transformation**:
   - Applies a linear transformation when zoom is enabled
   - Makes x and y ranges visually equal for detailed examination
   - Maintains proper proportions while providing enhanced view of intersection region
