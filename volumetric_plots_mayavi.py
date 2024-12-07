import numpy as np
from scipy.optimize import fsolve
from mayavi import mlab

def f1(x):
    """First function: parabola"""
    return x**2

def f2(x):
    """Second function: line"""
    return 2*x + 1

def find_intersection():
    """Find intersection points of f1 and f2"""
    def equation(x):
        return f1(x) - f2(x)
    
    # Find two intersection points
    x1 = fsolve(equation, -1)[0]
    x2 = fsolve(equation, 2)[0]
    return x1, x2

def create_rotation_points(x, y1, y2, num_points=50):
    """Create points for rotation"""
    theta = np.linspace(0, np.pi, num_points)  # 180-degree rotation
    r1 = y1  # radius for upper surface
    r2 = y2  # radius for lower surface
    
    # Create meshgrid for x and theta
    X, T = np.meshgrid(x, theta)
    
    # Calculate points for upper surface
    Y1 = np.outer(np.cos(theta), r1)
    Z1 = np.outer(np.sin(theta), r1)
    
    # Calculate points for lower surface
    Y2 = np.outer(np.cos(theta), r2)
    Z2 = np.outer(np.sin(theta), r2)
    
    return X, Y1, Z1, Y2, Z2

def plot_volumetric_figure():
    # Create figure with white background
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))
    
    # Generate points
    x = np.linspace(-2, 3, 100)
    y1 = f1(x)
    y2 = f2(x)
    
    # Find intersection points
    x1, x2 = find_intersection()
    x_fill = np.linspace(x1, x2, 50)
    y1_fill = f1(x_fill)
    y2_fill = f2(x_fill)
    
    # Create points for rotation around x-axis
    X, Y1, Z1, Y2, Z2 = create_rotation_points(x_fill, y1_fill, y2_fill)
    
    # Plot original functions
    mlab.plot3d(x, y1, np.zeros_like(x), color=(0, 0, 1), tube_radius=0.02, line_width=2)
    mlab.plot3d(x, y2, np.zeros_like(x), color=(1, 0, 0), tube_radius=0.02, line_width=2)
    
    # Plot rotated surfaces around x-axis
    surf1 = mlab.mesh(X, Y1, Z1, color=(0, 1, 0), opacity=0.3)
    surf2 = mlab.mesh(X, Y2, Z2, color=(0, 1, 0), opacity=0.3)
    
    # Plot vertical lines connecting surfaces
    for i in range(0, len(x_fill), 5):
        mlab.plot3d([x_fill[i], x_fill[i]], 
                   [Y1[0,i], Y2[0,i]], 
                   [Z1[0,i], Z2[0,i]], 
                   color=(0, 1, 0), opacity=0.2, tube_radius=0.01)
    
    # Add end caps
    theta = np.linspace(0, 2*np.pi, 20)
    for i in [0, -1]:
        r = np.linspace(y2_fill[i], y1_fill[i], 20)
        R, T = np.meshgrid(r, theta)
        X_end = np.full_like(R, x_fill[i])
        Y_end = R * np.cos(T)
        Z_end = R * np.sin(T)
        mlab.mesh(X_end, Y_end, Z_end, color=(0, 1, 0), opacity=0.3)
    
    # Create custom axes
    # Generate points for axes (35% longer)
    axis_points = np.linspace(-2.7, 4, 50)  # Extended range
    zeros = np.zeros_like(axis_points)
    
    # Draw X axis (red)
    mlab.plot3d(axis_points, zeros, zeros, color=(1, 0, 0), tube_radius=0.01, line_width=2)
    # X-axis label along the axis
    mlab.text3d(4.2, 0, 0, 'x', color=(1, 0, 0), scale=0.3)
    # X-axis ticks and labels
    for x in np.arange(-2, 3.1, 1):
        # Tick mark
        mlab.plot3d([x, x], [0, -0.1], [0, 0], color=(1, 0, 0), tube_radius=0.005)
        # Label
        if x != 0:  # Skip 0 to avoid cluttering the origin
            mlab.text3d(x, -0.3, 0, f'{int(x)}', color=(1, 0, 0), scale=0.2)
    
    # Draw Y axis (black)
    mlab.plot3d(zeros, axis_points, zeros, color=(0, 0, 0), tube_radius=0.01, line_width=2)
    # Y-axis label along the axis
    mlab.text3d(0, 4.2, 0, 'y', color=(0, 0, 0), scale=0.3)
    # Y-axis ticks and labels
    for y in np.arange(-2, 5.1, 1):
        # Tick mark
        mlab.plot3d([0, -0.1], [y, y], [0, 0], color=(0, 0, 0), tube_radius=0.005)
        # Label
        if y != 0:  # Skip 0 to avoid cluttering the origin
            mlab.text3d(-0.3, y, 0, f'{int(y)}', color=(0, 0, 0), scale=0.2)
    
    # Draw Z axis (blue)
    mlab.plot3d(zeros, zeros, axis_points, color=(0, 0, 1), tube_radius=0.01, line_width=2)
    # Z-axis label along the axis
    mlab.text3d(0, 0, 4.2, 'z', color=(0, 0, 1), scale=0.3)
    # Z-axis ticks and labels
    for z in np.arange(-2, 3.1, 1):
        # Tick mark
        mlab.plot3d([0, -0.1], [0, 0], [z, z], color=(0, 0, 1), tube_radius=0.005)
        # Label
        if z != 0:  # Skip 0 to avoid cluttering the origin
            mlab.text3d(-0.3, 0, z, f'{int(z)}', color=(0, 0, 1), scale=0.2)
    
    # Add grid planes with darker color
    grid_color = (0.2, 0.2, 0.2)
    
    # Add complete bounding box
    mlab.outline(color=(0, 0, 0), line_width=1.0)
    
    # XY plane (z=0) - main grid plane
    x_grid, y_grid = np.mgrid[-2:3:20j, -2:5:20j]
    z_grid = np.zeros_like(x_grid)
    xy_grid = mlab.surf(x_grid, y_grid, z_grid, color=grid_color, opacity=0.15)
    
    # Set the view to match the example
    mlab.view(azimuth=45, elevation=45, distance='auto', roll=0)
    
    # Show the plot
    mlab.show()

if __name__ == "__main__":
    plot_volumetric_figure()
