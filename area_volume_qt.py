import os
os.environ['ETS_TOOLKIT'] = 'qt'
os.environ['QT_API'] = 'pyqt5'

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QSplitter, QTabWidget, QSizePolicy,
                           QLabel, QLineEdit, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi import mlab
from tvtk.api import tvtk
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    height=800, width=1200, show_label=False),
               resizable=True)

    def __init__(self):
        HasTraits.__init__(self)
        self.equation1 = 'x**2'
        self.equation2 = '2*x + 1'
        self.num_disks = 20
        self.disk_volume = 0
        self.washer_volume = 0
        self.exact_volume = 0
        self.exact_volume1 = 0
        self.exact_volume2 = 0
        self.washer_volume1 = 0
        self.washer_volume2 = 0

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

    def create_axes(self, x_range, y_range, z_range, x_offset=0, name=''):
        """Create custom axes with labels and ticks."""
        # X-axis in black
        self.scene.mlab.plot3d(x_range + x_offset, np.zeros_like(x_range), np.zeros_like(x_range), 
                             color=(0, 0, 0), tube_radius=0.15)
        self.scene.mlab.text3d(x_range[-1] + x_offset + 1, 0, 0, 'x', color=(0, 0, 0), scale=0.5)
        
        # Y-axis in black
        self.scene.mlab.plot3d([x_offset]*len(y_range), y_range, np.zeros_like(y_range), 
                             color=(0, 0, 0), tube_radius=0.15)
        self.scene.mlab.text3d(x_offset, y_range[-1] + 1, 0, name, color=(0, 0, 0), scale=0.5)
        
        # Z-axis in black
        self.scene.mlab.plot3d([x_offset]*len(z_range), np.zeros_like(z_range), z_range, 
                             color=(0, 0, 0), tube_radius=0.15)
        self.scene.mlab.text3d(x_offset, 0, z_range[-1] + 1, 'z', color=(0, 0, 0), scale=0.5)

    def create_single_disk(self, x_left, x_right, radius, theta, color, opacity):
        """Helper function to create a single disk"""
        # Create circular edges
        for x in [x_left, x_right]:
            x_circle = x * np.ones_like(theta)
            y_circle = radius * np.cos(theta)
            z_circle = radius * np.sin(theta)
            self.scene.mlab.plot3d(x_circle, y_circle, z_circle, color=(0,0,0), tube_radius=0.01, opacity=opacity)
            xvert = np.zeros_like(theta) + x
            self.scene.mlab.plot3d(xvert, np.linspace(-radius,radius,len(xvert)), np.zeros_like(theta), color=(0,0,0), tube_radius=0.05, opacity=1.0)   
            
        # Fill disk with triangular mesh (both faces and side)
        # Front and back faces
        theta_mesh, r_mesh = np.meshgrid(theta, np.linspace(0, radius, 20))
        for x in [x_left, x_right]:
            x_mesh = x * np.ones_like(theta_mesh)
            y_mesh = r_mesh * np.cos(theta_mesh)
            z_mesh = r_mesh * np.sin(theta_mesh)
            self.scene.mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=opacity)
        
        # Side surface
        theta_mesh, x_mesh = np.meshgrid(theta, [x_left, x_right])
        y_mesh = radius * np.cos(theta_mesh)
        z_mesh = radius * np.sin(theta_mesh)
        self.scene.mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=opacity)

    def create_disk_or_washer(self, x_pos, f1_val, f2_val=None, method='washer', num_points=40, thickness=0.2, volume_color=(0, 1, 0)):
        """Create a disk or washer at specified x position."""
        # Reduce angular resolution for better performance
        theta = np.linspace(np.pi, 2*np.pi+0.05, num_points)
        x_left = x_pos - thickness/2
        x_right = x_pos + thickness/2
        
        if method == 'separate_disks':
            # Create two separate disks with different colors
            r1 = np.abs(f1_val)
            self.create_single_disk(x_left, x_right, r1, theta, color=volume_color, opacity=1.0)
            
            # Second function disk
            r2 = np.abs(f2_val)
            self.create_single_disk(x_left, x_right, r2, theta, color=volume_color, opacity=1.0)
        
        elif method == 'disk':
            r = np.abs(f1_val)
            self.create_single_disk(x_left, x_right, r, theta, color=volume_color, opacity=1.0)
        
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
            # Reduce radial resolution for better performance
            radial_points = 20  # Reduced from 100
            theta_mesh, r_mesh = np.meshgrid(theta, np.linspace(r_inner, r_outer, radial_points))
            
            # Front and back faces
            for x in [x_left, x_right]:
                x_mesh = x * np.ones_like(theta_mesh)
                y_mesh = r_mesh * np.cos(theta_mesh)
                z_mesh = r_mesh * np.sin(theta_mesh)
                self.scene.mlab.mesh(x_mesh, y_mesh, z_mesh, color=volume_color, opacity=1.0)
            
            # Outer and inner side surfaces - combined into one mesh call
            theta_mesh, x_mesh = np.meshgrid(theta, [x_left, x_right])
            for r in [r_outer, r_inner]:
                y_mesh = r * np.cos(theta_mesh)
                z_mesh = r * np.sin(theta_mesh)
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
        x_points = np.linspace(x1, x2, self.num_disks)
        
        # Calculate disk volumes for individual functions
        self.disk_volume1 = 0
        self.disk_volume2 = 0
        self.washer_volume = 0
        
        for x in x_points:
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
            
    def draw_plot(self):
        try:
            # Store current camera position if it exists
            current_camera = None
            if hasattr(self.scene, 'camera'):
                current_camera = {
                    'position': self.scene.camera.position,
                    'focal_point': self.scene.camera.focal_point,
                    'view_angle': self.scene.camera.view_angle,
                    'view_up': self.scene.camera.view_up,
                    'clipping_range': self.scene.camera.clipping_range
                }
            
            # Clear any existing visualization and set background
            self.scene.mlab.clf()
            self.scene.background = (1, 1, 1)
            
            # Find intersection points and calculate ranges
            x_intersect1, x_intersect2 = self.find_intersection()
            y_intersect1, y_intersect2 = self.f1(x_intersect1), self.f1(x_intersect2)
            
            # Calculate volumes once
            self.calculate_volumes(x_intersect1, x_intersect2)
            
            # Calculate plot ranges once
            x_range = x_intersect2 - x_intersect1
            x_extension = x_range * 0.35
            x_min = x_intersect1 - x_extension
            x_max = x_intersect2 + x_extension
            
            # Create points for plotting - reduce number of points
            x = np.linspace(x_min, x_max, 50)  # Reduced from 100
            y1 = self.f1(x)
            y2 = self.f2(x)
            
            # Calculate y range with extension
            y_min = min(self.f1(x_intersect1), self.f2(x_intersect2))
            y_max = max(self.f1(x_intersect1), self.f2(x_intersect2))
            y_range = y_max - y_min
            y_extension = y_range * 0.25
            y_min_ext = y_min - y_extension
            y_max_ext = y_max + y_extension   
            y_min_ext = -y_max_ext
            
            # Z range should match Y range for proper circular rotation
            z_max = max(abs(y_max_ext), abs(y_min_ext))
            z_min = -z_max
            
            # Create arrays for axes once
            x_range = np.array([x_min, x_max])
            y_range = np.array([y_min_ext, y_max_ext])
            z_range = np.array([z_min, z_max])
            
            # Plot offsets
            offset = 12
            offsets = [-offset, 0, offset]
            titles = ['y = ' + self.equation1, 'y = ' + self.equation2, 'Combined']
            volumes = [
                [f'Disk Vol: {self.disk_volume1:.2f}', f'Exact Vol: {self.exact_volume1:.2f}'],
                [f'Disk Vol: {self.disk_volume2:.2f}', f'Exact Vol: {self.exact_volume2:.2f}'],
                [f'Washer Vol: {self.washer_volume:.2f}', f'Exact Vol: {self.exact_volume:.2f}']
            ]
            
            # Draw disks/washers
            x_points = np.linspace(x_intersect1, x_intersect2, self.num_disks + 1)  # Add 1 to get correct number of intervals
            dx = (x_intersect2 - x_intersect1) / self.num_disks
            
            # Create all three plots with shared calculations
            for i, (curr_offset, title, volume_texts) in enumerate(zip(offsets, titles, volumes)):
                # Create axes
                self.create_axes(x_range, y_range, z_range, curr_offset, title)
                
                # Plot function(s)
                if i == 0:  # First plot - disk method for f1
                    self.scene.mlab.plot3d(x + curr_offset, y1, np.zeros_like(x), color=(1,0,0), tube_radius=0.1)
                    for x_pos in x_points[:-1]:  # Exclude last point to get correct number of disks
                        radius = abs(self.f1(x_pos))
                        self.create_single_disk(x_pos + curr_offset - dx/2, x_pos + curr_offset + dx/2, 
                                             radius, np.linspace(0, 2*np.pi, 40), (1,0,0), 0.3)  # Red color
                        
                elif i == 1:  # Second plot - disk method for f2
                    self.scene.mlab.plot3d(x + curr_offset, y2, np.zeros_like(x), color=(0,0,1), tube_radius=0.1)
                    for x_pos in x_points[:-1]:  # Exclude last point to get correct number of disks
                        radius = abs(self.f2(x_pos))
                        self.create_single_disk(x_pos + curr_offset - dx/2, x_pos + curr_offset + dx/2, 
                                             radius, np.linspace(0, 2*np.pi, 40), (0,0,1), 0.3)  # Blue color
                        
                else:  # Third plot - washer method
                    self.scene.mlab.plot3d(x + curr_offset, y1, np.zeros_like(x), color=(1,0,0), tube_radius=0.1)
                    self.scene.mlab.plot3d(x + curr_offset, y2, np.zeros_like(x), color=(0,0,1), tube_radius=0.1)
                    for x_pos in x_points[:-1]:  # Exclude last point to get correct number of washers
                        self.create_disk_or_washer(x_pos + curr_offset, self.f1(x_pos), self.f2(x_pos), 
                                                 method='washer', num_points=40, thickness=dx, volume_color=(0,1,0))
                
                # Add volume texts with vertical spacing
                for j, volume_text in enumerate(volume_texts):
                    y_offset = y_min_ext - y_extension/2 - (j * y_extension/2)
                    self.scene.mlab.text3d(curr_offset - x_extension/2, y_offset, z_max,
                                         volume_text, color=(0,0,0), scale=0.5)
                
                # Draw intersection markers efficiently
                x_marker = np.array([x_min, x_max])
                # Combined vertical and horizontal lines for intersection points
                intersection_points = [(x_intersect1, y_intersect1), (x_intersect2, y_intersect2)]
                for x_int, y_int in intersection_points:
                    # Vertical line
                    self.scene.mlab.plot3d([x_int + curr_offset]*2, [y_min_ext, y_max_ext], [0, 0],
                                         color=(0,0,0), tube_radius=0.05, opacity=0.5)
                    # Horizontal line
                    self.scene.mlab.plot3d([x_min + curr_offset, x_max + curr_offset], 
                                         [y_int, y_int], [0, 0],
                                         color=(0,0,0), tube_radius=0.05, opacity=0.5)
            
            # Set default camera position if no previous position exists
            if current_camera is None:
                # Set to a perspective similar to the reference image
                self.scene.camera.position = [-50, -100, 80]  # Adjusted for 3/4 view from front-left
                self.scene.camera.focal_point = [0, 0, 0]
                self.scene.camera.view_up = [0, 0, 1]
                self.scene.camera.clipping_range = [0.1, 1000.0]
                self.scene.camera.view_angle = 35.0  # Slightly wider angle to match image
            else:
                # Restore previous camera position
                self.scene.camera.position = current_camera['position']
                self.scene.camera.focal_point = current_camera['focal_point']
                self.scene.camera.view_angle = current_camera['view_angle']
                self.scene.camera.view_up = current_camera['view_up']
                self.scene.camera.clipping_range = current_camera['clipping_range']
                
        except Exception as e:
            print(f"Error in draw_plot: {str(e)}")

class MayaviQWidget(QWidget):
    def __init__(self, parent=None):
        super(MayaviQWidget, self).__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.visualization = Visualization()
        ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(ui)
        
        # Draw the plot after UI is set up
        self.visualization.draw_plot()

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Volume Visualization")
        self.setGeometry(100, 100, 1800, 900)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left pane
        left_tabs = QTabWidget()
        left_tabs.setMinimumWidth(300)
        
        # Functions tab
        functions_tab = QWidget()
        functions_layout = QVBoxLayout(functions_tab)
        
        # Add equation input fields
        eq1_label = QLabel("Equation #1:")
        self.eq1_input = QLineEdit()
        self.eq1_input.setText("x**2")  # Default equation
        
        eq2_label = QLabel("Equation #2:")
        self.eq2_input = QLineEdit()
        self.eq2_input.setText("2*x + 1")  # Default equation
        
        # Add number of disks/washers input
        disks_label = QLabel("Number of Disks/Washers:")
        self.disks_input = QLineEdit()
        self.disks_input.setText("20")  # Default value
        
        # Add Apply button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_equations)
        
        # Add widgets to functions layout
        functions_layout.addWidget(eq1_label)
        functions_layout.addWidget(self.eq1_input)
        functions_layout.addWidget(eq2_label)
        functions_layout.addWidget(self.eq2_input)
        functions_layout.addWidget(disks_label)
        functions_layout.addWidget(self.disks_input)
        functions_layout.addWidget(apply_button)
        functions_layout.addStretch()
        
        # Settings tab
        settings_tab = QWidget()
        
        # Add tabs to left pane
        left_tabs.addTab(functions_tab, "Functions")
        left_tabs.addTab(settings_tab, "Settings")
        
        # Right pane
        right_tabs = QTabWidget()
        
        # Add visualization tab
        mayavi_widget = MayaviQWidget(self)
        right_tabs.addTab(mayavi_widget, "3D View")
        
        # Add graph tab
        graph_tab = QWidget()
        right_tabs.addTab(graph_tab, "Graph")
        
        # Add both panes to splitter
        splitter.addWidget(left_tabs)
        splitter.addWidget(right_tabs)
        
        # Set initial sizes (30% left, 70% right)
        splitter.setSizes([540, 1260])
        
        # Store reference to mayavi widget
        self.mayavi_widget = mayavi_widget

    def apply_equations(self):
        """Apply the equations from UI and check for intersection"""
        eq1 = self.eq1_input.text()
        eq2 = self.eq2_input.text()
        
        # Get number of disks
        try:
            num_disks = int(self.disks_input.text())
            if num_disks <= 0:
                QMessageBox.warning(self, "Invalid Input", 
                                  "Number of disks must be positive.")
                return
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", 
                              "Please enter a valid number for disks/washers.")
            return
            
        # Check for intersection
        def intersection_func(x):
            return self.mayavi_widget.visualization.f1(x) - self.mayavi_widget.visualization.f2(x)
        
        try:
            # Try to find intersection points
            x_intersection = fsolve(intersection_func, 0)
            
            # Check if the solution is actually an intersection
            if abs(intersection_func(x_intersection)) > 1e-10:
                QMessageBox.warning(self, "No Intersection", 
                                  "The equations do not intersect in the visible range.")
                return
                
            # If we get here, the equations intersect, so update everything
            self.mayavi_widget.visualization.equation1 = eq1
            self.mayavi_widget.visualization.equation2 = eq2
            self.mayavi_widget.visualization.num_disks = num_disks
            
            # Redraw the plot
            self.mayavi_widget.visualization.draw_plot()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", 
                              "Error checking intersection: " + str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
