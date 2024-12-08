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

def get_global_ranges(x_range):
    """Calculate global ranges for consistent axes across all plots"""
    x_plot = np.linspace(x_range[0], x_range[1], 100)
    y1_plot = f1(x_plot)
    y2_plot = f2(x_plot)
    
    # Calculate global y range
    y_min = min(min(y1_plot), min(y2_plot))
    y_max = max(max(y1_plot), max(y2_plot))
    y_range = y_max - y_min
    y_extension = y_range * 0.25
    
    # Global ranges
    y_min_ext = y_min - y_extension
    y_max_ext = y_max + y_extension
    
    # Z range should match Y range for proper circular rotation
    z_max = max(abs(y_max_ext), abs(y_min_ext))
    z_min = -z_max
    
    return y_min_ext, y_max_ext, z_min, z_max

def plot_single_function(f, x_range, show_disk_pos, color, name, x_offset=0, global_ranges=None):
    """
    Plot a single function with its volumetric rotation
    Args:
        f: Function to plot
        x_range: (x_min, x_max) tuple for plotting
        show_disk_pos: x position for disk
        color: Color tuple (r,g,b) for the function
        name: Name for the axis label
        x_offset: Offset in x direction for positioning
        global_ranges: Tuple of (y_min, y_max, z_min, z_max) for consistent axes
    """
    x_plot = np.linspace(x_range[0], x_range[1], 100)
    y_plot = f(x_plot)
    
    # Use global ranges if provided
    if global_ranges:
        y_min_ext, y_max_ext, z_min, z_max = global_ranges
    else:
        # Calculate local ranges (fallback)
        y_range = max(y_plot) - min(y_plot)
        y_extension = y_range * 0.25
        y_min_ext = min(y_plot) - y_extension
        y_max_ext = max(y_plot) + y_extension
        z_max = max(abs(y_max_ext), abs(y_min_ext))
        z_min = -z_max
    
    # Create points for the surface - only back half
    theta = np.linspace(np.pi, 2*np.pi, 25)  # Half the points for back half rotation
    X, Theta = np.meshgrid(x_plot, theta)
    Y = np.outer(np.cos(theta), y_plot)
    Z = np.outer(np.sin(theta), y_plot)
    
    # Create axes
    axis_points_x = np.linspace(x_range[0], x_range[1], 50)
    axis_points_y = np.linspace(y_min_ext, y_max_ext, 50)
    axis_points_z = np.linspace(z_min, z_max, 50)
    zeros = np.zeros_like(axis_points_x)
    
    # Draw axes
    mlab.plot3d(axis_points_x + x_offset, zeros, zeros, color=(1, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(x_range[1] + x_offset + 0.2, 0, 0, 'x', color=(1, 0, 0), scale=0.3)
    mlab.plot3d([x_offset]*len(axis_points_y), axis_points_y, zeros[0:len(axis_points_y)], color=(0, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(x_offset, y_max_ext + 0.2, 0, name, color=color, scale=0.3)
    mlab.plot3d([x_offset]*len(axis_points_z), zeros[0:len(axis_points_z)], axis_points_z, color=(0, 0, 1), tube_radius=0.01, line_width=2)
    mlab.text3d(x_offset, 0, z_max + 0.2, 'z', color=(0, 0, 1), scale=0.3)
    
    # Add ticks and labels
    for x in np.arange(int(x_range[0]), int(x_range[1]) + 1):
        if x != 0:
            mlab.plot3d([x + x_offset, x + x_offset], [0, -0.1], [0, 0], color=(1, 0, 0), tube_radius=0.005)
            mlab.text3d(x + x_offset, -0.3, 0, f'{int(x)}', color=(1, 0, 0), scale=0.2)
    
    for y in np.arange(int(y_min_ext), int(y_max_ext) + 1):
        if y != 0:
            mlab.plot3d([x_offset - 0.1, x_offset], [y, y], [0, 0], color=(0, 0, 0), tube_radius=0.005)
            mlab.text3d(x_offset - 0.3, y, 0, f'{int(y)}', color=(0, 0, 0), scale=0.2)
    
    for z in np.arange(int(z_min), int(z_max) + 1):
        if z != 0:
            mlab.plot3d([x_offset - 0.1, x_offset], [0, 0], [z, z], color=(0, 0, 1), tube_radius=0.005)
    
    # Plot the volume surface
    mlab.mesh(X + x_offset, Y, Z, color=color, opacity=0.4)
    
    # Plot intersection with x/y plane
    mlab.plot3d(x_plot + x_offset, y_plot, np.zeros_like(x_plot), color=color, tube_radius=0.04, line_width=2)
    
    # Plot base curve
    mlab.plot3d(x_plot + x_offset, y_plot, np.zeros_like(x_plot), color=color, tube_radius=0.04, line_width=2)
    
    # Show disk if position specified
    if show_disk_pos is not None:
        y_val = f(show_disk_pos)
        create_disk_or_washer(show_disk_pos + x_offset, y_val, 0, method='disk', thickness=0.4, offset_disks=False)
    
    # Add grid planes
    grid_color = (0.2, 0.2, 0.2)
    x_grid, y_grid = np.mgrid[x_range[0]:x_range[1]:20j, y_min_ext:y_max_ext:20j]
    z_grid = np.zeros_like(x_grid)
    mlab.mesh(x_grid + x_offset, y_grid, z_grid, color=grid_color, opacity=0.05)

def plot_volumetric_figure(method='washer', show_disk_pos=None, offset_disks=False, show_curves=True, 
                         angle_interval=15, base_curve_radius=0.08, x_offset=0):
    """
    Plot the volumetric figure using either washer or shell method
    Args:
        method: 'disk', 'washer', or 'separate_disks'
        show_disk_pos: x position for disk
        offset_disks: If True, offset the disks in x-axis for better visibility
        show_curves: If True, show the rotated function curves
        angle_interval: Angle interval in degrees between curves
        base_curve_radius: Thickness of the curves in the x/y plane
        x_offset: Offset in x direction for positioning
    """
    # Find intersection points
    x1, x2 = find_intersection()
    x_range = x2 - x1
    x_extension = x_range * 0.35  # Increased from 0.25 to 0.35 (10% more)
    x_min = x1 - x_extension
    x_max = x2 + x_extension
    
    # Create points for plotting
    x_plot = np.linspace(x_min, x_max, 100)
    y1_plot = f1(x_plot)
    y2_plot = f2(x_plot)
    
    # Calculate axis ranges
    y_min = min(min(y1_plot), min(y2_plot))
    y_max = max(max(y1_plot), max(y2_plot))
    y_range = y_max - y_min
    y_extension = y_range * 0.25
    y_min_ext = y_min - y_extension
    y_max_ext = y_max + y_extension
    
    z_max = max(abs(y_max_ext), abs(y_min_ext))
    z_min = -z_max
    
    # Create points for the volume surface
    x_fill = np.linspace(x1, x2, 100)
    y1_fill = f1(x_fill)
    y2_fill = f2(x_fill)
    
    # Create rotation points
    X, Y1, Z1, Y2, Z2 = create_rotation_points(x_fill, y1_fill, y2_fill)
    
    # Create axes
    axis_points_x = np.linspace(x_min, x_max, 50)
    axis_points_y = np.linspace(y_min_ext, y_max_ext, 50)
    axis_points_z = np.linspace(z_min, z_max, 50)
    zeros = np.zeros_like(axis_points_x)
    
    # Draw axes
    mlab.plot3d(axis_points_x + x_offset, zeros, zeros, color=(1, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(x_max + x_offset + 0.2, 0, 0, 'x', color=(1, 0, 0), scale=0.3)
    mlab.plot3d([x_offset]*len(axis_points_y), axis_points_y, zeros[0:len(axis_points_y)], color=(0, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(x_offset, y_max_ext + 0.2, 0, 'y', color=(0, 0, 0), scale=0.3)
    mlab.plot3d([x_offset]*len(axis_points_z), zeros[0:len(axis_points_z)], axis_points_z, color=(0, 0, 1), tube_radius=0.01, line_width=2)
    mlab.text3d(x_offset, 0, z_max + 0.2, 'z', color=(0, 0, 1), scale=0.3)
    
    # Add ticks and labels
    for x in np.arange(int(x_min), int(x_max) + 1):
        if x != 0:
            mlab.plot3d([x + x_offset, x + x_offset], [0, -0.1], [0, 0], color=(1, 0, 0), tube_radius=0.005)
            mlab.text3d(x + x_offset, -0.3, 0, f'{int(x)}', color=(1, 0, 0), scale=0.2)
    
    for y in np.arange(int(y_min_ext), int(y_max_ext) + 1):
        if y != 0:
            mlab.plot3d([x_offset - 0.1, x_offset], [y, y], [0, 0], color=(0, 0, 0), tube_radius=0.005)
            mlab.text3d(x_offset - 0.3, y, 0, f'{int(y)}', color=(0, 0, 0), scale=0.2)
    
    for z in np.arange(int(z_min), int(z_max) + 1):
        if z != 0:
            mlab.plot3d([x_offset - 0.1, x_offset], [0, 0], [z, z], color=(0, 0, 1), tube_radius=0.005)
            mlab.text3d(x_offset - 0.3, 0, z, f'{int(z)}', color=(0, 0, 1), scale=0.2)
    
    # Set volume color
    volume_color = (0.7, 0.7, 0)  # Yellow-gold color
    
    # For the bounded region, use the difference between the two functions
    if method == 'washer':
        # Calculate where f2 > f1
        mask = y2_fill > y1_fill
        
        if np.any(mask):
            # Get x values and function values only where f2 > f1
            x_masked = x_fill[mask]
            y1_masked = y1_fill[mask]
            y2_masked = y2_fill[mask]
            
            # Create points for the surface - only back half for visibility
            theta = np.linspace(np.pi, 2*np.pi, 25)  # Half the points for back half rotation
            
            # Create meshgrids for outer surface (f2)
            X_outer, Theta_outer = np.meshgrid(x_masked, theta)
            Y_outer = np.outer(np.cos(theta), y2_masked)
            Z_outer = np.outer(np.sin(theta), y2_masked)
            
            # Create meshgrids for inner surface (f1)
            X_inner, Theta_inner = np.meshgrid(x_masked, theta)
            Y_inner = np.outer(np.cos(theta), y1_masked)
            Z_inner = np.outer(np.sin(theta), y1_masked)
            
            # Plot the outer and inner surfaces
            mlab.mesh(X_outer + x_offset, Y_outer, Z_outer, color=volume_color, opacity=0.7)
            mlab.mesh(X_inner + x_offset, Y_inner, Z_inner, color=volume_color, opacity=0.7)
            
            # Create end caps at the start and end x positions
            for x in [x_masked[0], x_masked[-1]]:
                # Get the y values at this x position
                idx = np.where(x_masked == x)[0][0]
                r_outer = y2_masked[idx]
                r_inner = y1_masked[idx]
                
                # Create circular end cap
                theta_cap = np.linspace(0, 2*np.pi, 50)
                r_points = np.linspace(r_inner, r_outer, 10)
                theta_mesh, r_mesh = np.meshgrid(theta_cap, r_points)
                
                x_mesh = x * np.ones_like(theta_mesh)
                y_mesh = r_mesh * np.cos(theta_mesh)
                z_mesh = r_mesh * np.sin(theta_mesh)
                
                mlab.mesh(x_mesh + x_offset, y_mesh, z_mesh, color=volume_color, opacity=0.7)
            
            # Add darker region at intersection with x/y plane
            # Create filled cross-section
            theta_cross = np.linspace(0, 2*np.pi, 50)
            x_cross = np.repeat(x_masked[:, np.newaxis], len(theta_cross), axis=1)
            r_outer_cross = np.abs(y2_masked[:, np.newaxis]) * np.ones_like(theta_cross)
            r_inner_cross = np.abs(y1_masked[:, np.newaxis]) * np.ones_like(theta_cross)
            
            # Create points for outer and inner boundaries
            y_outer_cross = r_outer_cross * np.cos(theta_cross)
            y_inner_cross = r_inner_cross * np.cos(theta_cross)
            z_cross = np.zeros_like(x_cross)
            
            # Plot filled cross-sections
            mlab.mesh(x_cross + x_offset, y_outer_cross, z_cross, color=volume_color, opacity=0.8)
            mlab.mesh(x_cross + x_offset, y_inner_cross, z_cross, color=volume_color, opacity=0.8)
    
    else:
        mlab.mesh(X + x_offset, Y1, Z1, color=volume_color, opacity=0.4)
        mlab.mesh(X + x_offset, Y2, Z2, color=volume_color, opacity=0.4)
    
    # Plot the base curves in their original colors with doubled thickness
    mlab.plot3d(x_plot + x_offset, y1_plot, np.zeros_like(x_plot), color=(0, 0, 1), tube_radius=base_curve_radius, line_width=4)
    mlab.plot3d(x_plot + x_offset, y2_plot, np.zeros_like(x_plot), color=(1, 0, 0), tube_radius=base_curve_radius, line_width=4)
    
    # Add rotated curves every 15 degrees
    if method == 'washer':
        angles = np.arange(15, 360, angle_interval)  # Every 15 degrees
        for angle in angles:
            theta = np.radians(angle)
            # Rotate f1
            y_rot = y1_plot * np.cos(theta)
            z_rot = y1_plot * np.sin(theta)
            mlab.plot3d(x_plot + x_offset, y_rot, z_rot, color=(0, 0, 1), tube_radius=base_curve_radius/4, line_width=1)
            
            # Rotate f2
            y_rot = y2_plot * np.cos(theta)
            z_rot = y2_plot * np.sin(theta)
            mlab.plot3d(x_plot + x_offset, y_rot, z_rot, color=(1, 0, 0), tube_radius=base_curve_radius/4, line_width=1)
    
    # Show disk/washer if position is specified
    if show_disk_pos is not None:
        y1_val = f1(show_disk_pos)
        y2_val = f2(show_disk_pos)
        if method == 'washer':
            create_disk_or_washer(show_disk_pos + x_offset, y1_val, y2_val, method=method, thickness=0.4, offset_disks=offset_disks)
        else:
            create_disk_or_washer(show_disk_pos + x_offset, y1_val, 0, method='disk', thickness=0.4, offset_disks=offset_disks)
    
    # Add grid planes
    grid_color = (0.2, 0.2, 0.2)
    x_grid, y_grid = np.mgrid[x_min:x_max:20j, y_min_ext:y_max_ext:20j]
    z_grid = np.zeros_like(x_grid)
    mlab.mesh(x_grid + x_offset, y_grid, z_grid, color=grid_color, opacity=0.05)

def plot_volumetric_figures():
    """Create three plots showing individual functions and combined volume"""
    # Find intersection points
    x1, x2 = find_intersection()
    x_range = x2 - x1
    x_extension = x_range * 0.35  # Increased from 0.25 to 0.35 (10% more)
    plot_x_range = (x1 - x_extension, x2 + x_extension)
    
    # Calculate global ranges for consistent axes
    global_ranges = get_global_ranges(plot_x_range)
    
    # Calculate offset based on x range
    plot_width = plot_x_range[1] - plot_x_range[0]
    offset = plot_width * 4.0  # Increased from 3.0 to 4.0 for more spacing
    
    # Create a single figure with white background
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1800, 800))
    
    # Determine function order based on maximum values
    x_test = np.linspace(plot_x_range[0], plot_x_range[1], 100)
    f1_max = max(f1(x_test))
    f2_max = max(f2(x_test))
    
    # First function should be the one with higher values
    if f1_max > f2_max:
        first_func, second_func = f1, f2
        first_color, second_color = (0, 0, 1), (1, 0, 0)
        first_name, second_name = 'f1', 'f2'
    else:
        first_func, second_func = f2, f1
        first_color, second_color = (1, 0, 0), (0, 0, 1)
        first_name, second_name = 'f2', 'f1'
    
    # Plot functions in determined order with global ranges
    plot_single_function(first_func, plot_x_range, 1.5, first_color, first_name, x_offset=-offset, global_ranges=global_ranges)
    plot_single_function(second_func, plot_x_range, 1.5, second_color, second_name, x_offset=0, global_ranges=global_ranges)
    
    # Plot combined volume (rightmost)
    plot_volumetric_figure(method='washer', show_disk_pos=1.5, offset_disks=False, show_curves=True, x_offset=offset)
    
    # Set initial view to match the image
    mlab.view(azimuth=45, elevation=45, distance='auto', roll=0)
    mlab.show()

if __name__ == '__main__':
    plot_volumetric_figures()
