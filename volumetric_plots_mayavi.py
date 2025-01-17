import numpy as np
from mayavi import mlab
from scipy.optimize import fsolve
from tvtk.api import tvtk

equation1 = 'x**2'
equation2 = '2*x + 1'

def f1(x):
    """First function: parabola"""
    return eval(equation1) 
def f2(x):
    """Second function: line"""
    return eval(equation2)

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

def create_disk_or_washer(x_pos, f1_val, f2_val=None, method='washer', num_points=100, thickness=0.2, offset_disks=False, volume_color=(0, 1, 0), full_rotation=False):
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
        volume_color: Color of the volume
    """
    if full_rotation:
        theta = np.linspace(0, 2*np.pi, num_points)
    else:
        theta = np.linspace(np.pi, 2*np.pi+0.05, num_points)
        
    x_left = x_pos - thickness/2
    x_right = x_pos + thickness/2
    
    if method == 'separate_disks':
        # Create two separate disks with different colors
        # First function disk (blue)
        r1 = np.abs(f1_val)
        create_single_disk(x_left, x_right, r1, theta, color=volume_color, opacity=1.0)
        
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
        create_single_disk(x_left_2, x_right_2, r2, theta, color=volume_color, opacity=1.0)
        
    elif method == 'disk':
        r = np.abs(f1_val)
        create_single_disk(x_left, x_right, r, theta, color=volume_color, opacity=1.0)
    
    else:  # washer
        # Create washer (two concentric circles)
        r_outer = max(np.abs(f1_val), np.abs(f2_val))
        r_inner = min(np.abs(f1_val), np.abs(f2_val))
        
        # Create circular edges
        for x in [x_left, x_right]:
            # Outer circle
            x_circle = x * np.ones_like(theta)
            y_outer = r_outer * np.cos(theta)
            z_outer = r_outer * np.sin(theta)
            mlab.plot3d(x_circle, y_outer, z_outer, color=volume_color, tube_radius=0.01, opacity=1.0)
            
            # Inner circle
            y_inner = r_inner * np.cos(theta)
            z_inner = r_inner * np.sin(theta)
            mlab.plot3d(x_circle, y_inner, z_inner, color=volume_color, tube_radius=0.01, opacity=1.0)
        
        # Fill washer with triangular mesh (both faces and sides)
        # Front and back faces
        theta_mesh, r_mesh = np.meshgrid(theta, 
                                       np.linspace(r_inner, r_outer, num_points))
        for x in [x_left, x_right]:
            x_mesh = x * np.ones_like(theta_mesh)
            y_mesh = r_mesh * np.cos(theta_mesh)
            z_mesh = r_mesh * np.sin(theta_mesh)
            washer = mlab.mesh(x_mesh, y_mesh, z_mesh, color=volume_color, opacity=1.0)
        
        # Outer side surface
        theta_mesh, x_mesh = np.meshgrid(theta, [x_left, x_right])
        y_mesh = r_outer * np.cos(theta_mesh)
        z_mesh = r_outer * np.sin(theta_mesh)
        outer = mlab.mesh(x_mesh, y_mesh, z_mesh, color=volume_color, opacity=1.0)
        
        # Inner side surface
        y_mesh = r_inner * np.cos(theta_mesh)
        z_mesh = r_inner * np.sin(theta_mesh)
        inner = mlab.mesh(x_mesh, y_mesh, z_mesh, color=volume_color, opacity=1.0)

def create_single_disk(x_left, x_right, radius, theta, color, opacity):
    """Helper function to create a single disk"""
    # Create circular edges
    
    for x in [x_left, x_right]:
        x_circle = x * np.ones_like(theta)
        y_circle = radius * np.cos(theta)
        z_circle = radius * np.sin(theta)
        mlab.plot3d(x_circle, y_circle, z_circle, color=(0,0,0), tube_radius=0.01, opacity=opacity)
        xvert = np.zeros_like(theta) + x
        mlab.plot3d(xvert, np.linspace(-radius,radius,len(xvert)), np.zeros_like(theta), color=(0,0,0), tube_radius=0.05, opacity=1.0)   
        
    # Fill disk with triangular mesh (both faces and side)
    # Front and back faces
    theta_mesh, r_mesh = np.meshgrid(theta, np.linspace(0, radius, 20))
    for x in [x_left, x_right]:
        x_mesh = x * np.ones_like(theta_mesh)
        y_mesh = r_mesh * np.cos(theta_mesh)
        z_mesh = r_mesh * np.sin(theta_mesh)
        disk = mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=opacity)
    
    # Side surface
    theta_mesh, x_mesh = np.meshgrid(theta, [x_left, x_right])
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

def plot_single_function(f, x_range, show_disk_pos, color, name, x_offset=0, global_ranges=None, all_disks = True, disk_count = 10):
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
    
    x_intersect1, x_intersect2 = find_intersection()
    y_intersect1, y_intersect2 = f1(x_intersect1), f2(x_intersect2)     
    intersect_range = y_intersect2 - y_intersect1
        
    x_plot = np.linspace(x_intersect1 - intersect_range * 0.1, x_intersect2 + intersect_range * 0.1, 100)
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
        
    # Create axes
    axis_points_x = np.linspace(x_range[0], x_range[1], 50)
    axis_points_y = np.linspace(y_min_ext, y_max_ext, 50)
    axis_points_z = np.linspace(z_min, z_max, 50)
    zeros = np.zeros_like(axis_points_x)
    
    # Draw axes
    mlab.plot3d(axis_points_x + x_offset, zeros, zeros, color=(1, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(x_range[1] + x_offset + 0.2, 0, 0, 'x', color=(1, 0, 0), scale=0.3)
    mlab.plot3d([x_offset]*len(axis_points_y), axis_points_y, zeros[0:len(axis_points_y)], color=(0, 0, 0), tube_radius=0.01, line_width=2)
    mlab.text3d(x_offset, y_max_ext + 0.2, 0, name, color=color, scale=0.2)
    mlab.plot3d([x_offset]*len(axis_points_z), zeros[0:len(axis_points_z)], axis_points_z, color=(0, 0, 1), tube_radius=0.01, line_width=2)
    mlab.text3d(x_offset, 0, z_max + 0.2, 'z', color=(0, 0, 1), scale=0.3)
    
   # Draw lines at y-intersection points     
    mlab.plot3d(axis_points_x + x_offset, zeros + y_intersect1, zeros, color=(0, 0, 0), tube_radius=0.02, line_width=4)
    mlab.plot3d(axis_points_x + x_offset, zeros + y_intersect2, zeros, color=(0, 0, 0), tube_radius=0.02, line_width=4)         
      
    # Draw lines at x-intersection points            
    mlab.plot3d([x_intersect1+x_offset]*len(axis_points_y), axis_points_y, zeros[0:len(axis_points_y)], color=(0, 0, 0), tube_radius=0.02, line_width=4) 
    mlab.plot3d([x_intersect2+x_offset]*len(axis_points_y), axis_points_y, zeros[0:len(axis_points_y)], color=(0, 0, 0), tube_radius=0.02, line_width=4)    
    
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
    
    # Plot the volume surface - only back
    theta_back = np.linspace(np.pi, 2*np.pi, 25)  # Back half (π to 2π)    
    X_back, Theta = np.meshgrid(x_plot, theta_back)
    Y_back = np.outer(np.cos(theta_back), y_plot)
    Z_back = np.outer(np.sin(theta_back), y_plot)
    
    volume = mlab.mesh(X_back + x_offset, Y_back, Z_back, color=color, opacity=0.5)
    
     # Plot the volume surface - only front   
    theta_front = np.linspace(0, np.pi, 25)  # Front half (0 to π)        
    X_front, Theta = np.meshgrid(x_plot, theta_front)
    Y_front = np.outer(np.cos(theta_front), y_plot)
    Z_front = np.outer(np.sin(theta_front), y_plot)        
    
    volume = mlab.mesh(X_front + x_offset, Y_front, Z_front, color=color, opacity=0.2)    
    
    # Plot intersection with x/y plane
    # Use the same x,y coordinates as the volume surface to show true intersection
    X_intersection = X_front + x_offset
    Y_intersection = Y_front
    Z_intersection = np.zeros_like(X_front)  # Set z=0 for intersection plane
    mlab.mesh(X_intersection, Y_intersection, Z_intersection, color=color, opacity=1.0)
    
    # Plot base curve
    mlab.plot3d(x_plot + x_offset, y_plot, np.zeros_like(x_plot), color=color, tube_radius=0.08, line_width=4)
    
    # Create custom axes for the third plot
    if x_offset > 0:
        # Draw axes lines only
        axis_points_x = np.linspace(x_range[0], x_range[1], 50)
        axis_points_y = np.linspace(y_min_ext, y_max_ext, 50)
        axis_points_z = np.linspace(z_min, z_max, 50)
        zeros = np.zeros_like(axis_points_x)
        
        # Draw x-axis
        mlab.plot3d(axis_points_x + x_offset, zeros, zeros, color=(1, 0, 0), tube_radius=0.01, line_width=2)
        mlab.text3d(x_range[1] + x_offset + 0.2, 0, 0, 'x', color=(1, 0, 0), scale=0.3)
        
        # Draw y-axis
        mlab.plot3d([x_offset]*len(axis_points_y), axis_points_y, zeros[0:len(axis_points_y)], color=(0, 0, 0), tube_radius=0.01, line_width=2)
        mlab.text3d(x_offset, y_max_ext + 0.2, 0, 'y', color=(0, 0, 0), scale=0.3)
        
        # Draw z-axis (no labels)
        mlab.plot3d([x_offset]*len(axis_points_z), zeros[0:len(axis_points_z)], axis_points_z, color=(0, 0, 1), tube_radius=0.01, line_width=2)
        
        # Add ticks for x and y axes only
        for x in np.arange(int(x_range[0]), int(x_range[1]) + 1):
            if x != 0:
                mlab.plot3d([x + x_offset, x + x_offset], [0, -0.1], [0, 0], color=(1, 0, 0), tube_radius=0.005)
                mlab.text3d(x + x_offset, -0.3, 0, f'{int(x)}', color=(1, 0, 0), scale=0.2)
        
        for y in np.arange(int(y_min_ext), int(y_max_ext) + 1):
            if y != 0:
                mlab.plot3d([x_offset - 0.1, x_offset], [y, y], [0, 0], color=(0, 0, 0), tube_radius=0.005)
                mlab.text3d(x_offset - 0.3, y, 0, f'{int(y)}', color=(0, 0, 0), scale=0.2)
        
        # Add tick marks for z-axis (no labels)
        for z in np.arange(int(z_min), int(z_max) + 1):
            if z != 0:
                mlab.plot3d([x_offset - 0.1, x_offset], [0, 0], [z, z], color=(0, 0, 1), tube_radius=0.005)
    
    # Show disk if position specified
    if show_disk_pos is not None or all_disks:        
        if all_disks:
            disk_width = (x_intersect2-x_intersect1) / disk_count / 2
            for i in range(disk_count+1):
                show_disk_pos = x_intersect1 + i * (disk_width*2)
                y_val = f(show_disk_pos)
                create_disk_or_washer(show_disk_pos + x_offset, y_val, 0, method='disk', thickness=disk_width, offset_disks=True, volume_color=color)
        else:
            y_val = f(show_disk_pos)
            disk_width = (x_intersect2-x_intersect1)*0.15
            create_disk_or_washer(show_disk_pos + x_offset, y_val, 0, method='disk', thickness=disk_width, offset_disks=True, volume_color=color)
    
    # Add grid planes
    grid_color = (0.2, 0.2, 0.2)
    x_grid, y_grid = np.mgrid[x_range[0]:x_range[1]:20j, y_min_ext:y_max_ext:20j]
    z_grid = np.zeros_like(x_grid)
    mlab.mesh(x_grid + x_offset, y_grid, z_grid, color=grid_color, opacity=0.05)

def plot_volumetric_figure(show_disk_pos=None, offset_disks=False, show_curves=True, 
                         angle_interval=15, base_curve_radius=0.08, x_offset=0, all_disks = True, disk_count = 10):
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
    x_intersect1, x_intersect2 = find_intersection()
    y_intersect1, y_intersect2 = f1(x_intersect1), f2(x_intersect2)
    
    x_range = x_intersect2 - x_intersect1
    x_extension = x_range * 0.35  # Increased from 0.25 to 0.35 (10% more)
    x_min = x_intersect1 - x_extension
    x_max = x_intersect2 + x_extension
    
    # Create points for plotting
    x_plot = np.linspace(x_min, x_max, 100)
    y1_plot = f1(x_plot)
    y2_plot = f2(x_plot)
    
    # Determine which function is greater at each point
    greater_mask = y2_plot > y1_plot
    
    # Create arrays for the greater and lesser functions
    y_greater = np.where(greater_mask, y2_plot, y1_plot)
    y_lesser = np.where(greater_mask, y1_plot, y2_plot)
    
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
    x_fill = np.linspace(x_intersect1, x_intersect2, 100)
    y1_fill = f1(x_fill)  
    y2_fill = f2(x_fill)  
    
    # For washer method around x-axis, we want where f2 > f1
    greater_mask = y2_fill > y1_fill
    
    # Create arrays for outer and inner functions
    y_outer = y2_fill  # f2 is always outer for washer method
    y_inner = y1_fill  # f1 is always inner for washer method
    
    # Create rotation points using the original functions for the basic shape
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
         
    # Draw lines at y-intersection points     
    mlab.plot3d(axis_points_x + x_offset, zeros + y_intersect1, zeros, color=(0, 0, 0), tube_radius=0.02, line_width=4)
    mlab.plot3d(axis_points_x + x_offset, zeros + y_intersect2, zeros, color=(0, 0, 0), tube_radius=0.02, line_width=4)         
      
    # Draw lines at x-intersection points            
    mlab.plot3d([x_intersect1+x_offset]*len(axis_points_y), axis_points_y, zeros[0:len(axis_points_y)], color=(0, 0, 0), tube_radius=0.02, line_width=4) 
    mlab.plot3d([x_intersect2+x_offset]*len(axis_points_y), axis_points_y, zeros[0:len(axis_points_y)], color=(0, 0, 0), tube_radius=0.02, line_width=4)
    
    # Add ticks and labels for x and y only
    for x in np.arange(int(x_min), int(x_max) + 1):
        if x != 0:
            mlab.plot3d([x + x_offset, x + x_offset], [0, -0.1], [0, 0], color=(1, 0, 0), tube_radius=0.005)
            mlab.text3d(x + x_offset, -0.3, 0, f'{int(x)}', color=(1, 0, 0), scale=0.2)
    
    for y in np.arange(int(y_min_ext), int(y_max_ext) + 1):
        if y != 0:
            mlab.plot3d([x_offset - 0.1, x_offset], [y, y], [0, 0], color=(0, 0, 0), tube_radius=0.005)
            mlab.text3d(x_offset - 0.3, y, 0, f'{int(y)}', color=(0, 0, 0), scale=0.2)
    
    # Add tick marks for z-axis but no labels
    for z in np.arange(int(z_min), int(z_max) + 1):
        if z != 0:
            mlab.plot3d([x_offset - 0.1, x_offset], [0, 0], [z, z], color=(0, 0, 1), tube_radius=0.005)
    
    # Set volume color
    volume_color = (0, 1, 0)  # Green color
    
    # Draw washer
    # Find where f2 > f1 for the volume
    mask = y_outer > y_inner
    x_masked = x_fill[mask]
    y_outer_masked = y_outer[mask]  # f2 values (outer radius)
    y_inner_masked = y_inner[mask]  # f1 values (inner radius)
    
    # Create points for both front and back halves
    theta_back = np.linspace(np.pi, 2*np.pi, 25)  # Back half (π to 2π)
    theta_front = np.linspace(0, np.pi, 25)  # Front half (0 to π)
    theta_slice1 = np.linspace(0, 0.1, 10)       # First thin slice near θ=0
    theta_slice2 = np.linspace(3.04, 3.14, 10)   # Second thin slice near θ=π
    
    # Create meshgrids for both slices
    r_points = np.linspace(0, 1, 10)  # Radial points from 0 to 1
    theta_r1, r_r1 = np.meshgrid(theta_slice1, r_points)
    theta_r2, r_r2 = np.meshgrid(theta_slice2, r_points)
    
    # Plot both thin slices
    dark_color = tuple(c * 0.5 for c in volume_color)  # 50% darker
    
    # Plot first slice (θ=0 to 0.1)
    for i in range(len(x_masked)):
        x = x_masked[i]
        r_outer = y_outer_masked[i]
        r_inner = y_inner_masked[i]
        
        # Scale the radial points between inner and outer radius
        r_scaled = r_inner + r_r1 * (r_outer - r_inner)
        
        # Convert to Cartesian coordinates
        x_slice = x * np.ones_like(theta_r1)
        y_slice = r_scaled * np.cos(theta_r1)
        z_slice = r_scaled * np.sin(theta_r1)
        
        # Plot the solid slice
        mlab.mesh(x_slice + x_offset, y_slice, z_slice, color=dark_color, opacity=1.0)
    
    # Plot second slice (θ=3.04 to π)
    for i in range(len(x_masked)):
        x = x_masked[i]
        r_outer = y_outer_masked[i]
        r_inner = y_inner_masked[i]
        
        # Scale the radial points between inner and outer radius
        r_scaled = r_inner + r_r2 * (r_outer - r_inner)
        
        # Convert to Cartesian coordinates
        x_slice = x * np.ones_like(theta_r2)
        y_slice = r_scaled * np.cos(theta_r2)
        z_slice = r_scaled * np.sin(theta_r2)
        
        # Plot the solid slice
        mlab.mesh(x_slice + x_offset, y_slice, z_slice, color=dark_color, opacity=1.0)
    
    # Plot back half of volume(more solid)   
    X_back, _ = np.meshgrid(x_masked, theta_back)
    Y_back = np.outer(np.cos(theta_back), y_outer_masked)
    Z_back = np.outer(np.sin(theta_back), y_outer_masked)
    vol_back_outer = mlab.mesh(X_back + x_offset, Y_back, Z_back, color=volume_color, opacity=0.5)
    
    # Plot front half of volume(very transparent)
    X_front, _ = np.meshgrid(x_masked, theta_front)
    Y_front = np.outer(np.cos(theta_front), y_outer_masked)
    Z_front = np.outer(np.sin(theta_front), y_outer_masked)
    vol_front_outer = mlab.mesh(X_front + x_offset, Y_front, Z_front, color=volume_color, opacity=0.2)

    # Plot top interface between volume and x/y plane  
    X_back, _ = np.meshgrid(x_masked, theta_slice1)
    Y_back = np.outer(np.cos(theta_slice1), y_outer_masked)
    Z_back = np.outer(np.sin(theta_slice1), y_outer_masked)
    vol_back_outer = mlab.mesh(X_back + x_offset, Y_back, Z_back, color=volume_color, opacity=1)
    
    # Plot bottom interface between volume and x/y plane
    X_front, _ = np.meshgrid(x_masked, theta_slice2)
    Y_front = np.outer(np.cos(theta_slice2), y_outer_masked)
    Z_front = np.outer(np.sin(theta_slice2), y_outer_masked)
    vol_front_outer = mlab.mesh(X_front + x_offset, Y_front, Z_front, color=volume_color, opacity=1)
    
    # Plot the base curves in their original colors with doubled thickness
    intersect_area = (x_intersect2 - x_intersect1) * (y_intersect2 - y_intersect1)
    mlab.plot3d(x_plot + x_offset, y1_plot, np.zeros_like(x_plot), color=(0, 0, 1), tube_radius=intersect_area*0.01, line_width=intersect_area*0.01)
    mlab.plot3d(x_plot + x_offset, y2_plot, np.zeros_like(x_plot), color=(1, 0, 0), tube_radius=intersect_area*0.01, line_width=intersect_area*0.01)

    # Show disk if position specified
    if show_disk_pos is not None or all_disks:       
        if all_disks:
            disk_width_proportion = 1.0
            disk_width = ((x_intersect2-x_intersect1) / disk_count) * disk_width_proportion
            for i in range(disk_count+1):
                show_disk_pos = x_intersect1 + i * (disk_width / disk_width_proportion)
                y_val1 = f1(show_disk_pos)
                y_val2 = f2(show_disk_pos)
                create_disk_or_washer(show_disk_pos + x_offset, y_val1, y_val2, method='washer', thickness=disk_width, offset_disks=True, volume_color=volume_color)
        else:
            y_val1 = f1(show_disk_pos)
            y_val2 = f2(show_disk_pos)
            disk_width = (x_intersect2-x_intersect1)*0.15
            create_disk_or_washer(show_disk_pos + x_offset, y_val1, y_val2, method='washer', thickness=disk_width, offset_disks=True, volume_color=volume_color)
    
    # Add grid planes
    grid_color = (0.2, 0.2, 0.2)
    x_grid, y_grid = np.mgrid[x_min:x_max:20j, y_min_ext:y_max_ext:20j]
    z_grid = np.zeros_like(x_grid)
    mlab.mesh(x_grid + x_offset, y_grid, z_grid, color=grid_color, opacity=0.05)

def plot_volumetric_figures():
    """Create three plots showing individual functions and combined volume"""
    # Find intersection points
    x_intersect1, x_intersect2 = find_intersection()
    x_range = x_intersect2 - x_intersect1
    x_extension = x_range * 0.35
    plot_x_range = (x_intersect1 - x_extension, x_intersect2 + x_extension)
    
    # Calculate global ranges for consistent axes
    global_ranges = get_global_ranges(plot_x_range)
    
    # Calculate offset based on x range
    plot_width = plot_x_range[1] - plot_x_range[0]
    offset = plot_width * 2.0
    
    # Create a single figure with white background
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1800, 800))
    
    # Determine which function is higher based on area
    higher_f, lower_f, higher_color, lower_color, equation1, equation2 = determine_higher_function(f1, f2, plot_x_range, criteria='area')
    
    disk_washer_position =  (x_intersect1+x_intersect2) / 2
    disk_count = 5
    all_disks = True
    
    # Plot higher function first (leftmost)
    plot_single_function(higher_f, plot_x_range, disk_washer_position, higher_color, 'y1 = ' + equation1.lower(), x_offset=-offset, global_ranges=global_ranges, all_disks=all_disks, disk_count=disk_count)
    
    # Plot lower function second (middle)
    plot_single_function(lower_f, plot_x_range, disk_washer_position, lower_color, 'y2 = ' + equation2.lower( ), x_offset=0, global_ranges=global_ranges, all_disks=all_disks, disk_count=disk_count)
    
    # Plot combined volume (rightmost)
    plot_volumetric_figure(show_disk_pos=disk_washer_position, offset_disks=False, show_curves=False, x_offset=offset, all_disks=all_disks, disk_count=disk_count)
    
    # Set initial view to match the image
    mlab.view(azimuth=45, elevation=45, distance='auto', roll=0)
    
    mlab.show()

def determine_higher_function(f1, f2, x_range, criteria='area'):
    """
    Determine which function is higher based on different criteria
    Args:
        f1, f2: The two functions to compare
        x_range: (x_min, x_max) tuple for analysis
        criteria: 'area' (total area), 'max_y' (maximum y value), 
                 'avg_y' (average y value), or 'x_point' (value at specific x)
    Returns:
        (higher_f, lower_f, higher_color, lower_color): Tuple containing the ordered functions and their colors
    """
    x = np.linspace(x_range[0], x_range[1], 1000)
    y1 = f1(x)
    y2 = f2(x)
    
    if criteria == 'area':
        # Compare total area under each curve
        area1 = np.trapezoid(np.abs(y1), x)
        area2 = np.trapezoid(np.abs(y2), x)
        is_f1_higher = area1 > area2
    elif criteria == 'max_y':
        # Compare maximum y values
        is_f1_higher = np.max(np.abs(y1)) > np.max(np.abs(y2))
    elif criteria == 'avg_y':
        # Compare average y values
        is_f1_higher = np.mean(np.abs(y1)) > np.mean(np.abs(y2))
    else:  # 'x_point'
        # Compare values at the midpoint
        mid_x = (x_range[0] + x_range[1]) / 2
        is_f1_higher = abs(f1(mid_x)) > abs(f2(mid_x))
    
    if is_f1_higher:
        return f1, f2, (0, 0, 1), (1, 0, 0),equation1,equation2 # f1 higher (blue), f2 lower (red)
    else:
        return f2, f1, (1, 0, 0), (0, 0, 1),equation2,equation1  # f2 higher (red), f1 lower (blue)

if __name__ == '__main__':
    plot_volumetric_figures()
