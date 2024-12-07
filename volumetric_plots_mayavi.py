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

def create_rotation_points(x, y1, y2):
    """Create points for rotation around x-axis"""
    # Create meshgrid for rotation
    theta = np.linspace(0, 2*np.pi, 50)
    X, Theta = np.meshgrid(x, theta)
    
    # Calculate rotated points for first function
    Y1 = y1[np.newaxis, :] * np.cos(Theta)
    # Flip the Z coordinate to rotate in opposite direction
    Z1 = -y1[np.newaxis, :] * np.sin(Theta)
    
    # Calculate rotated points for second function
    Y2 = y2[np.newaxis, :] * np.cos(Theta)
    # Flip the Z coordinate to rotate in opposite direction
    Z2 = -y2[np.newaxis, :] * np.sin(Theta)
    
    return X, Y1, Z1, Y2, Z2

def create_disk_or_washer(x_pos, f1_val, f2_val=None, method='washer', num_points=100, thickness=0.2, offset_disks=False):
    """
    Create a disk or washer at specified x position.
    Args:
        x_pos: x-coordinate where to place the disk/washer
        f1_val: Value of outer function at x_pos
        f2_val: Value of inner function at x_pos (for washer method)
        method: 'disk', 'washer', or 'separate_disks'
        num_points: Number of points for circle discretization
        thickness: Thickness of the disk/washer along x-axis
        offset_disks: If True, offset the disks in x-axis for better visibility
    """
    theta = np.linspace(0, 2*np.pi, num_points)
    x_left = x_pos - thickness/2
    x_right = x_pos + thickness/2
    
    if method == 'separate_disks':
        # Create two separate disks with different colors
        # First function disk (blue)
        r1 = np.abs(f1_val)
        create_single_disk(x_left, x_right, r1, theta, color=(0, 0, 1), opacity=0.3)
        
        # Second function disk (red)
        r2 = np.abs(f2_val)
        if offset_disks:
            # Offset the second disk slightly to make both visible
            offset = thickness * 1.2
            x_left_2 = x_left + offset
            x_right_2 = x_right + offset
        else:
            # Keep disks on same plane
            x_left_2 = x_left
            x_right_2 = x_right
        create_single_disk(x_left_2, x_right_2, r2, theta, color=(1, 0, 0), opacity=0.3)
        
    elif method == 'disk':
        r = np.abs(f1_val)
        create_single_disk(x_left, x_right, r, theta, color=(0.7, 0.7, 0), opacity=0.3)
    
    else:  # washer
        # Create washer (two concentric circles)
        r_outer = np.abs(f1_val)
        r_inner = np.abs(f2_val) if f2_val is not None else 0
        
        # Create circular edges
        for x in [x_left, x_right]:
            # Outer circle
            x_circle = x * np.ones_like(theta)
            y_outer = r_outer * np.cos(theta)
            z_outer = r_outer * np.sin(theta)
            mlab.plot3d(x_circle, y_outer, z_outer, color=(0.7, 0.7, 0), tube_radius=0.01, opacity=0.7)
            
            # Inner circle
            y_inner = r_inner * np.cos(theta)
            z_inner = r_inner * np.sin(theta)
            mlab.plot3d(x_circle, y_inner, z_inner, color=(0.7, 0.7, 0), tube_radius=0.01, opacity=0.7)
        
        # Fill washer with triangular mesh (both faces and sides)
        # Front and back faces
        theta_mesh, r_mesh = np.meshgrid(np.linspace(0, 2*np.pi, 20), 
                                       np.linspace(r_inner, r_outer, 20))
        for x in [x_left, x_right]:
            x_mesh = x * np.ones_like(theta_mesh)
            y_mesh = r_mesh * np.cos(theta_mesh)
            z_mesh = r_mesh * np.sin(theta_mesh)
            mlab.mesh(x_mesh, y_mesh, z_mesh, color=(0.7, 0.7, 0), opacity=0.3)
        
        # Outer side surface
        theta_mesh, x_mesh = np.meshgrid(np.linspace(0, 2*np.pi, 20), [x_left, x_right])
        y_mesh = r_outer * np.cos(theta_mesh)
        z_mesh = r_outer * np.sin(theta_mesh)
        mlab.mesh(x_mesh, y_mesh, z_mesh, color=(0.7, 0.7, 0), opacity=0.3)
        
        # Inner side surface
        y_mesh = r_inner * np.cos(theta_mesh)
        z_mesh = r_inner * np.sin(theta_mesh)
        mlab.mesh(x_mesh, y_mesh, z_mesh, color=(0.7, 0.7, 0), opacity=0.3)

def create_single_disk(x_left, x_right, radius, theta, color, opacity):
    """Helper function to create a single disk"""
    # Create circular edges
    for x in [x_left, x_right]:
        x_circle = x * np.ones_like(theta)
        y_circle = radius * np.cos(theta)
        z_circle = radius * np.sin(theta)
        mlab.plot3d(x_circle, y_circle, z_circle, color=color, tube_radius=0.01, opacity=opacity+0.4)
    
    # Fill disk with triangular mesh (both faces and side)
    # Front and back faces
    theta_mesh, r_mesh = np.meshgrid(np.linspace(0, 2*np.pi, 20), np.linspace(0, radius, 20))
    for x in [x_left, x_right]:
        x_mesh = x * np.ones_like(theta_mesh)
        y_mesh = r_mesh * np.cos(theta_mesh)
        z_mesh = r_mesh * np.sin(theta_mesh)
        mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=opacity)
    
    # Side surface
    theta_mesh, x_mesh = np.meshgrid(np.linspace(0, 2*np.pi, 20), [x_left, x_right])
    y_mesh = radius * np.cos(theta_mesh)
    z_mesh = radius * np.sin(theta_mesh)
    mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=opacity)

def plot_rotated_curves(x, y1, z1, y2, z2, angle_interval=15):
    """
    Plot the function curves at different rotation angles
    Args:
        x, y1, z1: Points for first function
        x, y2, z2: Points for second function
        angle_interval: Angle interval in degrees between curves
    """
    angles = np.arange(0, 360, angle_interval)
    for angle in angles:
        # Convert angle to radians
        theta = np.radians(angle)
        # Rotation matrix (flipped sign for z rotation)
        rot_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        
        # Rotate points for first function
        rotated_points1 = np.dot(rot_matrix, np.vstack((y1, z1)))
        y1_rot, z1_rot = rotated_points1
        mlab.plot3d(x, y1_rot, z1_rot, color=(0, 0, 1), tube_radius=0.005, opacity=0.5)
        
        # Rotate points for second function
        rotated_points2 = np.dot(rot_matrix, np.vstack((y2, z2)))
        y2_rot, z2_rot = rotated_points2
        mlab.plot3d(x, y2_rot, z2_rot, color=(1, 0, 0), tube_radius=0.005, opacity=0.5)

def plot_single_function(f, x_range, show_disk_pos, color, name):
    """
    Plot a single function with its volumetric rotation
    Args:
        f: Function to plot
        x_range: (x_min, x_max) tuple for plotting
        show_disk_pos: x position for disk
        color: Color tuple (r,g,b) for the function
        name: Name for the axis label
    """
    x_plot = np.linspace(x_range[0], x_range[1], 100)
    y_plot = f(x_plot)
    
    # Create points for rotation around x-axis
    theta = np.linspace(0, 2*np.pi, 50)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = np.zeros_like(X)
    
    # Calculate axis ranges
    y_range = max(y_plot) - min(y_plot)
    y_extension_pos = y_range * 0.25
    y_extension_neg = y_range * 0.25
    y_min_ext = min(y_plot) - y_extension_neg
    y_max_ext = max(y_plot) + y_extension_pos
    
    z_max = max(abs(max(y_plot)), abs(min(y_plot)))
    z_extension = z_max * 0.15
    z_min, z_max = -(z_max + z_extension), (z_max + z_extension)
    
    # Create axes
    axis_points_x = np.linspace(x_range[0], x_range[1], 50)
    axis_points_y = np.linspace(y_min_ext, y_max_ext, 50)
    axis_points_z = np.linspace(z_min, z_max, 50)
    zeros = np.zeros_like(axis_points_x)
    
    # Draw axes
    mlab.plot3d(axis_points_x, zeros, zeros, color=(1, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(x_range[1] + 0.2, 0, 0, 'x', color=(1, 0, 0), scale=0.3)
    mlab.plot3d(zeros, axis_points_y, zeros, color=(0, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(0, y_max_ext + 0.2, 0, name, color=color, scale=0.3)
    mlab.plot3d(zeros, zeros, axis_points_z, color=(0, 0, 1), tube_radius=0.01, line_width=2)
    mlab.text3d(0, 0, z_max + 0.2, 'z', color=(0, 0, 1), scale=0.3)
    
    # Create rotational surface
    for angle in range(0, 360, 15):
        theta = np.radians(angle)
        Y_rot = y_plot * np.cos(theta)
        Z_rot = y_plot * np.sin(theta)
        mlab.mesh(X[0:1,:], Y_rot.reshape(1,-1), Z_rot.reshape(1,-1), color=color, opacity=0.1)
    
    # Plot base curve
    mlab.plot3d(x_plot, y_plot, np.zeros_like(x_plot), color=color, tube_radius=0.04, line_width=2)
    
    # Show disk if position specified
    if show_disk_pos is not None:
        y_val = f(show_disk_pos)
        create_disk_or_washer(show_disk_pos, y_val, 0, method='disk', thickness=0.2, offset_disks=False)
    
    # Plot rotated curves
    angle_interval = 15
    for angle in range(0, 360, angle_interval):
        theta = np.radians(angle)
        y_rot = y_plot * np.cos(theta)
        z_rot = y_plot * np.sin(theta)
        mlab.plot3d(x_plot, y_rot, z_rot, color=color, tube_radius=0.005)


def plot_volumetric_figure(method='washer', show_disk_pos=None, offset_disks=False, show_curves=True, 
                         angle_interval=15, base_curve_radius=0.04, x_offset=0):
    """
    Plot the volumetric figure with optional disk/washer visualization
    Args:
        method: 'disk', 'washer', or 'separate_disks'
        show_disk_pos: x position where to show the disk/washer, or None to hide
        offset_disks: If True, offset the disks in x-axis for better visibility
        show_curves: If True, show the rotated function curves
        angle_interval: Angle interval in degrees between curves
        base_curve_radius: Thickness of the curves in the x/y plane
        x_offset: Offset in x direction for positioning
    """
    # Create figure with white background
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))
    
    # Find intersection points
    x1, x2 = find_intersection()
    
    # Calculate points around intersection
    x_margin = (x2 - x1) * 0.25  # 25% margin
    x_plot = np.linspace(x1 - x_margin, x2 + x_margin, 100)
    y1_plot = f1(x_plot)
    y2_plot = f2(x_plot)
    
    # Points for rotation volume
    x_fill = np.linspace(x1, x2, 100)
    y1_fill = f1(x_fill)
    y2_fill = f2(x_fill)
    
    # Create points for rotation around x-axis
    X, Y1, Z1, Y2, Z2 = create_rotation_points(x_fill, y1_fill, y2_fill)
    
    # Calculate axis ranges with 25% extension
    x_range = x2 - x1
    y_values = np.concatenate([y1_plot, y2_plot])
    y_min, y_max = y_values.min(), y_values.max()
    y_range = y_max - y_min
    
    # Calculate extensions
    x_extension = x_range * 0.25
    y_extension_pos = y_range * 0.25  # 25% extension for positive direction
    y_extension_neg = y_range * 0.25  # 25% extension for negative direction
    
    # Apply extensions to ranges
    x_min, x_max = x1 - x_extension, x2 + x_extension
    y_min_ext = y_min - y_extension_neg
    y_max_ext = y_max + y_extension_pos
    
    # For z-axis, use a shorter range than y
    z_max = max(abs(y_max), abs(y_min))  # Use original y values, not extended
    z_extension = z_max * 0.15  # 15% extension for z-axis
    z_min, z_max = -(z_max + z_extension), (z_max + z_extension)
    
    # Create custom axes with extended ranges
    axis_points_x = np.linspace(x_min, x_max, 50) + x_offset
    axis_points_y = np.linspace(y_min_ext, y_max_ext, 50)
    axis_points_z = np.linspace(z_min, z_max, 50)
    zeros = np.zeros_like(axis_points_x)
    
    # Draw X axis (red)
    mlab.plot3d(axis_points_x, zeros, zeros, color=(1, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(x_max + x_offset + 0.2, 0, 0, 'x', color=(1, 0, 0), scale=0.3)
    # X-axis ticks and labels
    for x in np.arange(int(x_min), int(x_max) + 1):
        if x != 0:  # Skip 0 to avoid cluttering the origin
            mlab.plot3d([x + x_offset, x + x_offset], [0, -0.1], [0, 0], color=(1, 0, 0), tube_radius=0.005)
            mlab.text3d(x + x_offset, -0.3, 0, f'{int(x)}', color=(1, 0, 0), scale=0.2)
    
    # Draw Y axis (black)
    mlab.plot3d([x_offset]*len(axis_points_y), axis_points_y, zeros[0:len(axis_points_y)], color=(0, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(x_offset, y_max_ext + 0.2, 0, 'y', color=(0, 0, 0), scale=0.3)
    # Y-axis ticks and labels
    for y in np.arange(int(y_min_ext), int(y_max_ext) + 1):
        if y != 0:  # Skip 0 to avoid cluttering the origin
            mlab.plot3d([x_offset - 0.1, x_offset], [y, y], [0, 0], color=(0, 0, 0), tube_radius=0.005)
            mlab.text3d(x_offset - 0.3, y, 0, f'{int(y)}', color=(0, 0, 0), scale=0.2)
    
    # Draw Z axis (blue)
    mlab.plot3d([x_offset]*len(axis_points_z), zeros[0:len(axis_points_z)], axis_points_z, color=(0, 0, 1), tube_radius=0.01, line_width=2)
    mlab.text3d(x_offset, 0, z_max + 0.2, 'z', color=(0, 0, 1), scale=0.3)
    # Z-axis ticks only (no labels)
    for z in np.arange(int(z_min), int(z_max) + 1):
        if z != 0:  # Skip 0 to avoid cluttering the origin
            mlab.plot3d([x_offset - 0.1, x_offset], [0, 0], [z, z], color=(0, 0, 1), tube_radius=0.005)
    
    # Plot surfaces
    surf1 = mlab.mesh(X + x_offset, Y1, Z1, color=(0, 0, 1), opacity=0.1)
    surf2 = mlab.mesh(X + x_offset, Y2, Z2, color=(1, 0, 0), opacity=0.1)
    
    # Plot the base curves in the x/y plane
    mlab.plot3d(x_plot + x_offset, y1_plot, np.zeros_like(x_plot), color=(0, 0, 1), tube_radius=base_curve_radius, line_width=2)
    mlab.plot3d(x_plot + x_offset, y2_plot, np.zeros_like(x_plot), color=(1, 0, 0), tube_radius=base_curve_radius, line_width=2)
    
    # Plot rotated curves if requested
    if show_curves:
        plot_rotated_curves(X[0,:], Y1[0,:], Z1[0,:], Y2[0,:], Z2[0,:], angle_interval)
    
    # Add grid planes with darker color
    grid_color = (0.2, 0.2, 0.2)
    
    # Create grid on XY plane (z=0)
    x_grid, y_grid = np.mgrid[x_min:x_max:20j, y_min_ext:y_max_ext:20j]
    z_grid = np.zeros_like(x_grid)
    mlab.mesh(x_grid + x_offset, y_grid, z_grid, color=grid_color, opacity=0.1)
    
    # Show disk/washer if position is specified
    if show_disk_pos is not None:
        f1_val = f1(show_disk_pos)
        f2_val = f2(show_disk_pos)
        # Create disk or washer
        create_disk_or_washer(show_disk_pos + x_offset, f1_val, f2_val, method=method, thickness=0.2, offset_disks=offset_disks)
    
    # Set the view to face x/y plane and rotate 90 degrees to the right
    mlab.view(azimuth=90, elevation=90, distance='auto', roll=0)
    
    # Show the plot
    mlab.show()

def plot_volumetric_figures():
    """Create three plots showing individual functions and combined volume"""
    # Find intersection points
    x1, x2 = find_intersection()
    x_range = x2 - x1
    x_extension = x_range * 0.25
    plot_x_range = (x1 - x_extension, x2 + x_extension)
    
    # Create a single figure with white background
    #fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))
    
    # Plot first function
    #plot_single_function(f1, plot_x_range, 1.5, (0, 0, 1), 'f1')
    #mlab.view(azimuth=90, elevation=90, distance='auto', roll=0)
    #mlab.show()
    
    # Clear the previous figure
    #mlab.close(all=True)
    
    # Create new figure for second function
    #fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))
    #plot_single_function(f2, plot_x_range, 1.5, (1, 0, 0), 'f2')
    #mlab.view(azimuth=90, elevation=90, distance='auto', roll=0)
    #mlab.show()
    
    # Clear the previous figure
    #mlab.close(all=True)
    
    # Create new figure for combined volume
    #fig = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 800))
    plot_volumetric_figure(method='washer', show_disk_pos=1.5, offset_disks=False, show_curves=True)
    mlab.show()
    
if __name__ == '__main__':
    plot_volumetric_figures()
