import os
os.environ['ETS_TOOLKIT'] = 'qt'
os.environ['QT_API'] = 'pyqt5'

import sys
import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
from scipy.integrate import quad

# Qt imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QComboBox, QCheckBox, QPushButton,
    QStatusBar, QSplitter, QTabWidget, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# Mayavi and Traits imports
from traits.api import HasTraits, Instance, on_trait_change, Int, String, Float, Bool, Any
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi import mlab
from tvtk.api import tvtk

# Initialize Qt Application first
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

class MayaviQWidget(QWidget):
    def __init__(self, visualization):
        super(MayaviQWidget, self).__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.visualization = visualization
        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'visualization') and self.visualization is not None:
            self.visualization.close()
            self.visualization = None
        super().close()
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.close()
        event.accept()

class Visualization(HasTraits):
    """Visualization class for 3D plots."""
    scene = Instance(MlabSceneModel, ())
    equation1 = String("x**2")
    equation2 = String("2*x + 1")
    num_disks = Int(5)
    disk_volume1 = Float(0.0)
    disk_volume2 = Float(0.0)
    washer_volume = Float(0.0)
    exact_volume1 = Float(0.0)
    exact_volume2 = Float(0.0)
    exact_volume = Float(0.0)
    zoom = Bool(True)  # Changed to True for default zoom
    riemann_type = String('left')
    parent = Any()
    display_plot1 = Bool(True)
    display_plot2 = Bool(True)
    display_plot3 = Bool(True)
    display_f1 = Bool(True)
    display_f2 = Bool(True)
    display_washer = Bool(True)
    view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene), show_label=False),
        resizable=True,
        width=1200,
        height=800,
    )

    def __init__(self, parent=None):
        HasTraits.__init__(self)  # Initialize parent class without arguments
        self.parent = parent
        self._closing = False
        self.norm_scale = 7
        self._camera_state = None

        # Initial plot when scene is activated
        self.scene.on_trait_change(self._on_scene_activated, 'activated')
        self.norm_scale = 7

    def _on_scene_activated(self):
        """Called when the scene is activated."""
        try:
            if hasattr(self.scene, 'camera'):
                self.scene.on_trait_change(self._camera_changed, 'camera.position')
                self.scene.on_trait_change(self._camera_changed, 'camera.focal_point')
                self.scene.on_trait_change(self._camera_changed, 'camera.view_up')
                self.scene.on_trait_change(self._camera_changed, 'camera.view_angle')
                self.scene.on_trait_change(self._camera_changed, 'camera.clipping_range')
            self.draw_plot()
        except Exception as e:
            print(f"Error in scene activation: {e}")

    def _camera_changed(self, name, old, new):
        """Handle camera changes and update status."""
        # Camera updates are now handled by MainWindow
        pass

    def f1(self, x):
        """First function: parabola"""
        if isinstance(x, (int, float)):
            x = float(x)
        return eval(self.equation1, {'x': x, 'np': np})
    
    def f2(self, x):
        """Second function: line"""
        if isinstance(x, (int, float)):
            x = float(x)
        return eval(self.equation2, {'x': x, 'np': np})

    def create_axes(self, x_range, y_range, z_range, offset, title, x_center, x_scale):
        """Create axes for the plot."""
        if self.zoom:
            # In zoom mode, calculate transformed coordinates using same transformation as x_norm
            x_origin = (0 - x_center) * x_scale + offset  # Transform origin (0) to new space
            x_left = (x_range[0] - x_center) * x_scale + offset
            x_right = (x_range[1] - x_center) * x_scale + offset
            
            # Draw axes with transformed origin
            self.scene.mlab.plot3d([x_left, x_right], [0,0], [0,0], 
                                  color=(0,0,0), tube_radius=0.08)  # x-axis
            self.scene.mlab.plot3d([x_origin, x_origin], y_range, [0,0], 
                                  color=(0,0,0), tube_radius=0.08)  # y-axis
            self.scene.mlab.plot3d([x_origin, x_origin], [0,0], z_range, 
                                  color=(0,0,0), tube_radius=0.08)  # z-axis
            # Add title relative to transformed origin
            self.scene.mlab.text(x_origin - 1.2, y_range[1] + 0.8, title, z=0, width=0.12, color=(0,0,0))
        else:
            # In non-zoom mode, use original offset for all axes
            self.scene.mlab.plot3d([x_range[0] + offset, x_range[1] + offset], [0,0], [0,0], 
                                  color=(0,0,0), tube_radius=0.08)  # x-axis
            self.scene.mlab.plot3d([offset, offset], y_range, [0,0], 
                                  color=(0,0,0), tube_radius=0.08)  # y-axis
            self.scene.mlab.plot3d([offset, offset], [0,0], z_range, 
                                  color=(0,0,0), tube_radius=0.08)  # z-axis
            # Add title relative to original offset
            self.scene.mlab.text(offset - 1.2, y_range[1] + 0.8, title, z=0, width=0.12, color=(0,0,0))

    def draw_dashed_line(self, x1, y1, x2, y2, curr_offset, z=0):
        """Draw a dashed line between two points."""
        # Parameters for dashed line
        dash_length = 0.3  # Length of each dash
        gap_length = 0.2   # Length of gap between dashes
        
        # Calculate total length and direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction vector
        dx = dx / length
        dy = dy / length
        
        # Calculate number of segments
        segment_length = dash_length + gap_length
        num_segments = int(length / segment_length)
        
        # Draw dashed line segments
        for i in range(num_segments):
            start_fraction = i * segment_length / length
            end_fraction = min((i * segment_length + dash_length) / length, 1.0)
            
            x_start = x1 + dx * length * start_fraction
            y_start = y1 + dy * length * start_fraction
            x_end = x1 + dx * length * end_fraction
            y_end = y1 + dy * length * end_fraction
            
            self.scene.mlab.plot3d(
                [x_start + curr_offset, x_end + curr_offset],
                [y_start, y_end],
                [z, z],
                color=(0,0,0),  # Black color
                tube_radius=0.05
            )

    def draw_intersection_lines(self, x_pos, y_pos, x_range, y_range, curr_offset):
        """Draw intersection lines."""
        # Draw vertical dashed line at intersection point
        self.draw_dashed_line(x_pos, y_range[0], x_pos, y_pos, curr_offset)
        
        # Draw horizontal dashed line to y-axis
        self.draw_dashed_line(x_range[0], y_pos, x_pos, y_pos, curr_offset)

    def create_single_disk(self, x_left, x_right, radius, theta, color, opacity):
        """Helper function to create a single disk"""
        # Position disk based on Riemann sum type
        if self.riemann_type == 'left':
            # For left Riemann sum, x_left is the evaluation point
            disk_x_left = x_left
            disk_x_right = x_left + (x_right - x_left)
        else:  # right
            # For right Riemann sum, x_right is the evaluation point
            disk_x_left = x_right - (x_right - x_left)
            disk_x_right = x_right

        # Create circular edges
        for x in [disk_x_left, disk_x_right]:
            x_circle = x * np.ones_like(theta)
            y_circle = radius * np.cos(theta)
            z_circle = radius * np.sin(theta)
            self.scene.mlab.plot3d(x_circle, y_circle, z_circle, color=color, tube_radius=0.01, opacity=1.0)
        
        # Fill disk with triangular mesh (both faces and sides)
        # Front and back faces
        theta_mesh, r_mesh = np.meshgrid(theta, np.linspace(0, radius, 20))
        for x in [disk_x_left, disk_x_right]:
            x_mesh = x * np.ones_like(theta_mesh)
            y_mesh = r_mesh * np.cos(theta_mesh)
            z_mesh = r_mesh * np.sin(theta_mesh)
            self.scene.mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=opacity)
        
        # Side surface
        theta_mesh, x_mesh = np.meshgrid(theta, [disk_x_left, disk_x_right])
        y_mesh = radius * np.cos(theta_mesh)
        z_mesh = radius * np.sin(theta_mesh)
        self.scene.mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=opacity)

    def create_disk_or_washer(self, x_pos, f1_val, f2_val=None, method='washer', num_points=40, thickness=0.2, volume_color=(0, 1, 0)):
        """Create a disk or washer at specified x position."""
        # Reduce angular resolution for better performance
        theta = np.linspace(np.pi, 2*np.pi+0.05, num_points)  # Half circle
        
        # Position disk based on Riemann sum type
        if self.riemann_type == 'left':
            # x_pos is the left edge
            x_left = x_pos
            x_right = x_pos + thickness
        else:  # right
            # x_pos is the right edge
            x_left = x_pos - thickness
            x_right = x_pos
        
        if method == 'disk':
            r = np.abs(f1_val)
            self.create_single_disk(x_left, x_right, r, theta, color=volume_color, opacity=0.3)
        
        else:  # washer
            # Create washer (two concentric circles)
            r_outer = max(np.abs(f1_val), np.abs(f2_val))
            r_inner = min(np.abs(f1_val), np.abs(f2_val))
            
            # Create circular edges with reduced points
            for x in [x_left, x_right]:
                # Outer circle
                x_circle = x * np.ones_like(theta)
                y_outer = r_outer * np.cos(theta)
                z_outer = r_outer * np.sin(theta)
                self.scene.mlab.plot3d(x_circle, y_outer, z_outer, color=volume_color, tube_radius=0.01, opacity=1.0)
                
                # Inner circle
                y_inner = r_inner * np.cos(theta)
                z_inner = r_inner * np.sin(theta)
                self.scene.mlab.plot3d(x_circle, y_inner, z_inner, color=volume_color, tube_radius=0.01, opacity=1.0)
            
            # Fill washer with triangular mesh (both faces and sides)
            # Front and back faces
            theta_mesh, r_mesh = np.meshgrid(theta, np.linspace(r_inner, r_outer, 20))
            
            # Front and back faces
            for x in [x_left, x_right]:
                x_mesh = x * np.ones_like(theta_mesh)
                y_mesh = r_mesh * np.cos(theta_mesh)
                z_mesh = r_mesh * np.sin(theta_mesh)
                self.scene.mlab.mesh(x_mesh, y_mesh, z_mesh, color=volume_color, opacity=1.0)
            
            # Side surfaces
            theta_mesh, x_mesh = np.meshgrid(theta, [x_left, x_right])
            
            # Outer side surface
            y_mesh = r_outer * np.cos(theta_mesh)
            z_mesh = r_outer * np.sin(theta_mesh)
            self.scene.mlab.mesh(x_mesh, y_mesh, z_mesh, color=volume_color, opacity=1.0)
            
            # Inner side surface
            y_mesh = r_inner * np.cos(theta_mesh)
            z_mesh = r_inner * np.sin(theta_mesh)
            self.scene.mlab.mesh(x_mesh, y_mesh, z_mesh, color=volume_color, opacity=1.0)

    def find_intersection(self):
        """Find intersection points of f1 and f2"""
        def equation(x):
            return self.f1(x) - self.f2(x)
        
        # Find two intersection points
        x1 = fsolve(equation, -1)[0]
        x2 = fsolve(equation, 2)[0]
        return x1, x2

    def calculate_volumes(self, x1, x2):
        """Calculate volumes using both integration and disk/washer methods"""
        # Calculate exact volume using integration
        def integrand(x, f1, f2=None):
            if f2 is None:
                return np.pi * f1(x)**2
            return np.pi * (f1(x)**2 - f2(x)**2)
        
        # Calculate exact volumes
        self.exact_volume1, _ = quad(lambda x: integrand(x, self.f1), x1, x2)
        self.exact_volume2, _ = quad(lambda x: integrand(x, self.f2), x1, x2)
        self.exact_volume, _ = quad(lambda x: integrand(x, self.f1, self.f2), x1, x2)
        
        # Calculate approximate volumes using disks/washers
        dx = (x2 - x1) / self.num_disks
        
        # Calculate disk volumes for individual functions
        self.disk_volume1 = 0
        self.disk_volume2 = 0
        self.washer_volume = 0
        
        for x in np.linspace(x1, x2, self.num_disks):
            # For first function (disk method)
            r1 = abs(self.f1(x))
            self.disk_volume1 += np.pi * r1**2 * dx
            
            # For second function (disk method)
            r2 = abs(self.f2(x))
            self.disk_volume2 += np.pi * r2**2 * dx
            
            # For combined (washer method)
            outer_r = max(r1, r2)
            inner_r = min(r1, r2)
            self.washer_volume += np.pi * (outer_r**2 - inner_r**2) * dx
            
    def apply_zoom_transformation(self, x_int_range, y_int_range):
        """Applies zoom transformation to the x and y ranges."""
        zoom_factor = 3
        
        if self.zoom:
            scale_factor = (zoom_factor * self.norm_scale) / max(x_int_range, y_int_range)
            x_scale = scale_factor
            y_scale = scale_factor
        else:
            y_scale = (zoom_factor * self.norm_scale) / y_int_range
            x_scale = 1.0  # No x scaling in normal mode

        return x_scale, y_scale

    def draw_plot(self):
        # Don't try to draw if we're closing
        if self._closing:
            return
            
        try:
            # Clear the scene
            self.scene.mlab.clf()
            
            # Find intersection points and calculate ranges
            x_intersect1, x_intersect2 = self.find_intersection()
            y_intersect1, y_intersect2 = self.f1(x_intersect1), self.f1(x_intersect2)
            
            # Calculate volumes once
            self.calculate_volumes(x_intersect1, x_intersect2)
            
            # Calculate ranges within intersection region
            x_int_range = abs(x_intersect2 - x_intersect1)
            # Calculate center point between intersections for balanced zoom transformation
            x_center = (x_intersect1 + x_intersect2) / 2
            
            # Sample points within intersection region for y range
            x_int_sample = np.linspace(x_intersect1, x_intersect2, 50)
            y1_int = self.f1(x_int_sample)
            y2_int = self.f2(x_int_sample)
            y_int_values = np.concatenate([y1_int, y2_int])
            y_int_range = np.max(y_int_values) - np.min(y_int_values)

            
            # Calculate scale factors based on zoom state
            x_scale, y_scale = self.apply_zoom_transformation(x_int_range, y_int_range)
            
            # Add margin to x range
            x_margin = x_int_range * 0.1  # 10% margin
            x_min = x_intersect1 - x_margin
            x_max = x_intersect2 + x_margin
            
            # Create points for plotting curves
            x = np.linspace(x_min, x_max, 50)
            y1 = self.f1(x)
            y2 = self.f2(x)
            
            # Calculate step size for Riemann sum and disk parameters
            thickness = 0.2  # disk thickness
            dx = (x_intersect2 - x_intersect1) / self.num_disks
            
            # Create points for disks/washers based on Riemann sum type
            if self.riemann_type == 'left':
                # For left Riemann sum:
                # First disk's left edge should be exactly at x_intersect1
                # Each disk's center is thickness/2 to the right of its left edge
                x_points = np.linspace(x_intersect1, x_intersect2 - dx, self.num_disks)  # left edges
                x_points = x_points + thickness/2  # shift to center points
            else:  # right
                # For right Riemann sum:
                # Last disk's right edge should be exactly at x_intersect2
                # Each disk's center is thickness/2 to the left of its right edge
                x_points = np.linspace(x_intersect1 + dx, x_intersect2, self.num_disks)  # right edges
                x_points = x_points - thickness/2  # shift to center points

            # Calculate normalized coordinates
            if self.zoom:
                # Apply zoom transformation to x coordinates
                x_norm = (x - x_center) * x_scale
                x_min_norm = (x_min - x_center) * x_scale
                x_max_norm = (x_max - x_center) * x_scale
                x_points_norm = (x_points - x_center) * x_scale
                dx_norm = dx * x_scale
            else:
                x_norm = x
                x_min_norm = x_min
                x_max_norm = x_max
                x_points_norm = x_points
                dx_norm = dx

                # Only apply normalization to y-values
            y1_norm = y1 * y_scale
            y2_norm = y2 * y_scale
            
            # Calculate normalized y boundaries
            y_values_norm = np.concatenate([y1_norm, y2_norm])
            y_min_norm = np.min(y_values_norm)
            y_max_norm = np.max(y_values_norm)
            y_range_norm = y_max_norm - y_min_norm
            
            # Add margins proportional to the normalized range
            y_margin = y_range_norm * 0.1  # 10% margin
            y_min = y_min_norm - y_margin
            y_max = y_max_norm + y_margin
            
            # Fixed extensions for consistent spacing
            y_extension = y_range_norm * 0.25  # 25% of normalized range
            y_min_ext = y_min - y_extension
            y_max_ext = y_max + y_extension
            
            # Z range matches Y range for proper circular rotation
            z_max = (y_max_ext - y_min_ext) / 2
            z_min = -z_max
            
            # Create arrays for axes
            x_range_norm = np.array([x_min_norm, x_max_norm])
            y_range_norm = np.array([y_min_ext, y_max_ext])
            z_range = np.array([z_min, z_max])
            
            # Calculate plot width in normalized coordinates
            plot_width = abs(x_max_norm - x_min_norm)
            
            # Define minimum spacing between plots
            MIN_SPACING = 15.0  # Minimum spacing we want to maintain
            
            # Calculate spacing based on plot width and zoom state
            if self.zoom:
                # In zoom mode, use spacing proportional to plot width but at least MIN_SPACING
                PLOT_SPACING = max(plot_width * 2, MIN_SPACING)
            else:
                # In normal mode, use larger spacing
                PLOT_SPACING = max(plot_width * 1.5, MIN_SPACING)
                
            # Count visible plots
            visible_plots = sum([self.display_plot1, self.display_plot2, self.display_plot3])
            if visible_plots == 0:
                return
                
            # Calculate offsets based on visible plots and plot width
            if visible_plots == 1:
                offsets = [0]  # Center single plot
            elif visible_plots == 2:
                # Two plots centered with consistent spacing
                half_spacing = PLOT_SPACING / 2
                offsets = [-half_spacing, half_spacing]
            else:
                # Three plots with consistent spacing
                offsets = [-PLOT_SPACING, 0, PLOT_SPACING]
            
            # Create list of plots to draw
            plots_to_draw = []
            if self.display_plot1:
                plots_to_draw.append((0, 'y = ' + self.equation1))
            if self.display_plot2:
                plots_to_draw.append((1, 'y = ' + self.equation2))
            if self.display_plot3:
                plots_to_draw.append((2, 'Combined'))               

            # Draw each enabled plot
            for offset_idx, (plot_idx, title) in enumerate(plots_to_draw):
                curr_offset = offsets[offset_idx]  # Use offset based on current index
                                
                # Create axes
                self.create_axes(x_range_norm, y_range_norm, z_range, curr_offset, title, x_center, x_scale)

                # Draw intersection lines for appropriate plots
                if self.display_f1 and self.display_f2:
                    # Convert intersection points to normalized coordinates
                    x_int1_norm = (x_intersect1 - x_center) * x_scale if self.zoom else x_intersect1
                    x_int2_norm = (x_intersect2 - x_center) * x_scale if self.zoom else x_intersect2
                    y_int1_norm = y_intersect1 * y_scale
                    y_int2_norm = y_intersect2 * y_scale
                    
                    # Only draw intersection markers for plots 1 and 2
                    if plot_idx in [0, 1]:
                        self.draw_intersection_lines(x_int1_norm, y_int1_norm, x_range_norm, y_range_norm, curr_offset)
                        self.draw_intersection_lines(x_int2_norm, y_int2_norm, x_range_norm, y_range_norm, curr_offset)

                # Plot function(s)
                if plot_idx == 0:  # First plot - disk method for f1
                    if self.display_f1:
                        self.scene.mlab.plot3d(x_norm + curr_offset, y1_norm, np.zeros_like(x), color=(1,0,0), tube_radius=0.1)
                    for x_pos, x_pos_norm in zip(x_points, x_points_norm):
                        radius = abs(self.f1(x_pos) * y_scale)
                        self.create_single_disk(x_pos_norm + curr_offset - dx_norm/2, x_pos_norm + curr_offset + dx_norm/2, 
                                                radius, np.linspace(0, 2*np.pi, 40), (1,0,0), 0.3)
                        
                elif plot_idx == 1:  # Second plot - disk method for f2
                    if self.display_f2:
                        self.scene.mlab.plot3d(x_norm + curr_offset, y2_norm, np.zeros_like(x), color=(0,0,1), tube_radius=0.1)
                    for x_pos, x_pos_norm in zip(x_points, x_points_norm):
                        radius = abs(self.f2(x_pos) * y_scale)
                        self.create_single_disk(x_pos_norm + curr_offset - dx_norm/2, x_pos_norm + curr_offset + dx_norm/2, 
                                                radius, np.linspace(0, 2*np.pi, 40), (0,0,1), 0.3)
                        
                else:  # Third plot - washer method
                    if self.display_f1:
                        self.scene.mlab.plot3d(x_norm + curr_offset, y1_norm, np.zeros_like(x), color=(1,0,0), tube_radius=0.1)
                    if self.display_f2:
                        self.scene.mlab.plot3d(x_norm + curr_offset, y2_norm, np.zeros_like(x), color=(0,0,1), tube_radius=0.1)
                    
                    # Draw intersection lines
                    self.draw_intersection_lines(x_intersect1, y_intersect1, x_range_norm, y_range_norm, curr_offset)
                    self.draw_intersection_lines(x_intersect2, y_intersect2, x_range_norm, y_range_norm, curr_offset)
                    
                    # Create washers
                    for x_pos, x_pos_norm in zip(x_points, x_points_norm):
                        r1 = abs(self.f1(x_pos) * y_scale)
                        r2 = abs(self.f2(x_pos) * y_scale)
                        self.create_disk_or_washer(x_pos_norm + curr_offset, r1, r2, 
                                                    method='washer', num_points=40, thickness=dx_norm, volume_color=(0,1,0))

                # Add volume text
                y_text_pos = y_min_ext #- y_range_norm * 0.4
                if plot_idx == 0:
                    text = f'Disk Vol: {self.disk_volume1:.2f}\nExact Vol: {self.exact_volume1:.2f}'
                elif plot_idx == 1:
                    text = f'Disk Vol: {self.disk_volume2:.2f}\nExact Vol: {self.exact_volume2:.2f}'
                else:
                    text = f'Washer Vol: {self.washer_volume:.2f}\nExact Vol: {self.exact_volume:.2f}'
                self.scene.mlab.text(curr_offset - 1.5, y_text_pos, text, z=0, width=0.12, color=(0,0,0))

            # Set default camera position if no previous position exists
            if self._camera_state is None:
                # Set to a perspective similar to the reference image
                self.scene.camera.position = [4,4,54]  # Adjusted for 3/4 view from front-left
                self.scene.camera.focal_point = [3.8,3.6,0.4]
                self.scene.camera.view_up = [0, 1, 0.1]
                self.scene.camera.clipping_range = [38,103]
                self.scene.camera.view_angle = 35.0  # Slightly wider angle to match image
            else:
                try:
                    # Restore previous camera position
                    self.scene.camera.position = self._camera_state['position']
                    self.scene.camera.focal_point = self._camera_state['focal_point']
                    self.scene.camera.view_angle = self._camera_state['view_angle']
                    self.scene.camera.view_up = self._camera_state['view_up']
                    self.scene.camera.clipping_range = self._camera_state['clipping_range']
                except Exception:
                    # If camera state restoration fails, use default position
                    self.scene.camera.position = [4,4,54]
                    self.scene.camera.focal_point = [3.8,3.6,0.4]
                    self.scene.camera.view_up = [0, 1, 0.1]
                    self.scene.camera.clipping_range = [38,103]
                    self.scene.camera.view_angle = 35.0
                
            self._camera_state = {
                'position': list(self.scene.camera.position),
                'focal_point': list(self.scene.camera.focal_point),
                'view_angle': float(self.scene.camera.view_angle),
                'view_up': list(self.scene.camera.view_up),
                'clipping_range': list(self.scene.camera.clipping_range)
            }

                                    
        except Exception as e:
            print(f"Error in draw_plot: {e}")
            # Don't let camera status errors affect the rest of the application
            pass

    @on_trait_change('equation1, equation2, num_disks, zoom, riemann_type, display_plot1, display_plot2, display_plot3, display_f1, display_f2, display_washer')
    def update_plot(self):
        """Update the plot when any of the plot parameters change."""
        if hasattr(self.scene, 'activated') and self.scene.activated:
            self.draw_plot()

    def close(self):
        """Clean up resources when closing."""
        self._closing = True
        if hasattr(self, 'scene'):
            # Clear the scene first
            self.scene.mlab.clf()
            # Remove the scene
            self.scene = None

class MainWindow(QMainWindow):
    """Main window for the application."""
    def __init__(self):
        super().__init__()

        # Initialize closing flag
        self._closing = False

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create left panel for inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel)

        # Add input fields
        input_font = QFont()
        input_font.setPointSize(12)

        # Function inputs
        f1_label = QLabel('f₁(x) = ')
        f1_label.setFont(input_font)
        self.f1_input = QLineEdit('x**2')
        self.f1_input.setFont(input_font)

        f2_label = QLabel('f₂(x) = ')
        f2_label.setFont(input_font)
        self.f2_input = QLineEdit('2*x + 1')
        self.f2_input.setFont(input_font)

        # Number of disks input
        disks_label = QLabel('Number of Disks:')
        disks_label.setFont(input_font)
        self.disks_input = QSpinBox()
        self.disks_input.setFont(input_font)
        self.disks_input.setMinimum(1)
        self.disks_input.setValue(5)

        # Riemann sum selection
        riemann_label = QLabel('Riemann Sum:')
        riemann_label.setFont(input_font)
        self.riemann_type_input = QComboBox()
        self.riemann_type_input.setFont(input_font)

        # Zoom checkbox
        self.zoom_checkbox = QCheckBox('Zoom')
        self.zoom_checkbox.setChecked(True)  # Set zoom checked by default
        self.zoom_checkbox.setFont(input_font)

        # Display plot checkboxes
        display_label = QLabel('Display Plot:')
        display_label.setFont(input_font)
        self.display_plot1_checkbox = QCheckBox('Function 1')
        self.display_plot1_checkbox.setChecked(True)
        self.display_plot1_checkbox.setFont(input_font)
        self.display_plot2_checkbox = QCheckBox('Function 2')
        self.display_plot2_checkbox.setChecked(True)
        self.display_plot2_checkbox.setFont(input_font)
        self.display_plot3_checkbox = QCheckBox('Combined')
        self.display_plot3_checkbox.setChecked(True)
        self.display_plot3_checkbox.setFont(input_font)

        # Apply button
        self.apply_button = QPushButton('Apply')
        self.apply_button.setFont(input_font)
        self.apply_button.clicked.connect(self.update_plot)

        # Add widgets to left layout
        left_layout.addWidget(f1_label)
        left_layout.addWidget(self.f1_input)
        left_layout.addWidget(f2_label)
        left_layout.addWidget(self.f2_input)
        left_layout.addWidget(disks_label)
        left_layout.addWidget(self.disks_input)
        left_layout.addWidget(riemann_label)
        left_layout.addWidget(self.riemann_type_input)
        left_layout.addWidget(self.zoom_checkbox)
        left_layout.addWidget(display_label)
        left_layout.addWidget(self.display_plot1_checkbox)
        left_layout.addWidget(self.display_plot2_checkbox)
        left_layout.addWidget(self.display_plot3_checkbox)
        left_layout.addWidget(self.apply_button)
        left_layout.addStretch()

        # Create right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel)

        # Create visualization
        self.visualization = Visualization()
        self.visualization.scene.on_trait_change(self._setup_camera_listeners, 'activated')
        
        # Create Mayavi widget
        self.mayavi_widget = MayaviQWidget(self.visualization)
        right_layout.addWidget(self.mayavi_widget)

        # Create status bar
        self.status_bar = QStatusBar()
        self.status_bar.setFont(input_font)
        self.setStatusBar(self.status_bar)

        # Set window title
        self.setWindowTitle('Area Volume Calculator')
        self.setGeometry(100, 100, 1200, 800)

        # Initialize plot parameters from UI - this will be called by _on_scene_activated
        self.visualization.display_plot1 = self.display_plot1_checkbox.isChecked()
        self.visualization.display_plot2 = self.display_plot2_checkbox.isChecked()
        self.visualization.display_plot3 = self.display_plot3_checkbox.isChecked()
        self.visualization.display_f1 = self.display_plot1_checkbox.isChecked()
        self.visualization.display_f2 = self.display_plot2_checkbox.isChecked()
        self.visualization.display_washer = self.display_plot3_checkbox.isChecked()

    def _setup_camera_listeners(self):
        """Set up camera trait listeners after scene is activated."""
        if hasattr(self.visualization.scene, 'camera') and self.visualization.scene.camera is not None:
            self.visualization.scene.on_trait_change(self.update_status, 'camera.position')
            self.visualization.scene.on_trait_change(self.update_status, 'camera.focal_point')
            self.visualization.scene.on_trait_change(self.update_status, 'camera.view_up')
            self.visualization.scene.on_trait_change(self.update_status, 'camera.view_angle')
            self.visualization.scene.on_trait_change(self.update_status, 'camera.clipping_range')

    def update_status(self, name, old, new):
        """Handle camera changes and update status."""
        if self._closing:  # Skip updates if closing
            return
            
        try:
            if not hasattr(self.visualization.scene, 'camera') or self.visualization.scene.camera is None:
                return
                
            camera = self.visualization.scene.camera
            pos = [f"{x:.1f}" for x in camera.position]
            focal = [f"{x:.1f}" for x in camera.focal_point]
            up = [f"{x:.1f}" for x in camera.view_up]
            clip = [f"{x:.1f}" for x in camera.clipping_range]
            
            status = (
                f"Camera: [{', '.join(pos)}] | "
                f"Focal: [{', '.join(focal)}] | "
                f"Up: [{', '.join(up)}] | "
                f"View Angle: {camera.view_angle:.1f}° | "
                f"Clip: [{', '.join(clip)}]"
            )
            
            self.status_bar.showMessage(status)
        except Exception as e:
            print(f"Error updating camera status: {e}")
            # Don't let camera status errors affect the rest of the application
            pass

    def update_plot(self):
        try:
            # Update visualization parameters
            self.visualization.equation1 = self.f1_input.text()
            self.visualization.equation2 = self.f2_input.text()
            self.visualization.num_disks = self.disks_input.value()
            self.visualization.riemann_type = self.riemann_type_input.currentText()
            self.visualization.zoom = self.zoom_checkbox.isChecked()
            
            # Update plot visibility flags
            self.visualization.display_plot1 = self.display_plot1_checkbox.isChecked()
            self.visualization.display_plot2 = self.display_plot2_checkbox.isChecked()
            self.visualization.display_plot3 = self.display_plot3_checkbox.isChecked()
            self.visualization.display_f1 = self.display_plot1_checkbox.isChecked()
            self.visualization.display_f2 = self.display_plot2_checkbox.isChecked()
            self.visualization.display_washer = self.display_plot3_checkbox.isChecked()
            
            # Update plot
            self.visualization.draw_plot()
        except Exception as e:
            print(f"Error in update_plot: {e}")

    def closeEvent(self, event):
        """Handle window close event."""
        self._closing = True
        if hasattr(self, 'mayavi_widget'):
            self.mayavi_widget.close()
            self.mayavi_widget = None
        super().closeEvent(event)

if __name__ == '__main__':
    window = MainWindow()
    window.show()
    try:
        app.exec_()
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Ensure cleanup happens even if there's an error
        if hasattr(window, 'mayavi_widget') and window.mayavi_widget is not None:
            window.mayavi_widget.close()
            window.mayavi_widget = None
