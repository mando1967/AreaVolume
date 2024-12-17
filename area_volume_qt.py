import os
os.environ['ETS_TOOLKIT'] = 'qt'
os.environ['QT_API'] = 'pyqt5'

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QSplitter, QTabWidget, QSizePolicy)
from PyQt5.QtCore import Qt
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi import mlab
from tvtk.api import tvtk
import numpy as np
from scipy.optimize import fsolve

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    height=800, width=1200, show_label=False),
               resizable=True)

    def __init__(self):
        HasTraits.__init__(self)
        self.equation1 = 'x**2'
        self.equation2 = '2*x + 1'

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

    def create_disk_or_washer(self, x_pos, f1_val, f2_val=None, method='washer', num_points=100, thickness=0.2, volume_color=(0, 1, 0)):
        """Create a disk or washer at specified x position."""
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
            
            # Create circular edges
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
            theta_mesh, r_mesh = np.meshgrid(theta, np.linspace(r_inner, r_outer, num_points))
            for x in [x_left, x_right]:
                x_mesh = x * np.ones_like(theta_mesh)
                y_mesh = r_mesh * np.cos(theta_mesh)
                z_mesh = r_mesh * np.sin(theta_mesh)
                self.scene.mlab.mesh(x_mesh, y_mesh, z_mesh, color=volume_color, opacity=1.0)
            
            # Outer side surface
            theta_mesh, x_mesh = np.meshgrid(theta, [x_left, x_right])
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

    def draw_plot(self):
        try:
            # Clear any existing visualization
            self.scene.mlab.clf()
            
            # Find intersection points and calculate ranges
            x_intersect1, x_intersect2 = self.find_intersection()
            y_intersect1, y_intersect2 = self.f1(x_intersect1), self.f1(x_intersect2)  # Use f1 or f2, they're equal at intersection
            x_range = x_intersect2 - x_intersect1
            x_extension = x_range * 0.35  # 35% extension
            x_min = x_intersect1 - x_extension
            x_max = x_intersect2 + x_extension
            
            # Create points for plotting
            x = np.linspace(x_min, x_max, 100)
            y1 = self.f1(x)
            y2 = self.f2(x)
            
            # Calculate y range with extension
            y_min = min(min(y1), min(y2))
            y_max = max(max(y1), max(y2))
            y_range = y_max - y_min
            y_extension = y_range * 0.25
            y_min_ext = y_min - y_extension
            y_max_ext = y_max + y_extension
            
            # Z range should match Y range for proper circular rotation
            z_max = max(abs(y_max_ext), abs(y_min_ext))
            z_min = -z_max
            
            # Create arrays for axes
            x_range = np.array([x_min, x_max])
            y_range = np.array([y_min_ext, y_max_ext])
            z_range = np.array([z_min, z_max])
            
            # Plot the functions with offset
            offset = 12  # Reduced space between plots (was 20)
            
            # First plot (leftmost) - Parabola with volume
            self.create_axes(x_range, y_range, z_range, -offset, 'y = ' + self.equation1)
            self.scene.mlab.plot3d(x - offset, y1, np.zeros_like(x), color=(1,0,0), tube_radius=0.1)
            
            # Draw intersection markers
            # Vertical lines at intersection points
            for x_int in [x_intersect1, x_intersect2]:
                self.scene.mlab.plot3d([x_int-offset]*len(y_range), y_range, np.zeros_like(y_range), 
                                     color=(0,0,0), tube_radius=0.05, opacity=0.5)
            # Horizontal lines at intersection y-values
            x_marker = np.array([x_min, x_max])
            for y_int in [y_intersect1, y_intersect2]:
                self.scene.mlab.plot3d(x_marker - offset, np.ones_like(x_marker)*y_int, np.zeros_like(x_marker), 
                                     color=(0,0,0), tube_radius=0.05, opacity=0.5)
            
            # Add volume visualization
            x_points = np.linspace(x_intersect1, x_intersect2, 10)  # Only between intersection points
            for x_pos in x_points:
                self.create_disk_or_washer(x_pos - offset, self.f1(x_pos), method='disk', volume_color=(1,0,0))
            
            # Second plot (middle) - Line with volume
            self.create_axes(x_range, y_range, z_range, 0, 'y = ' + self.equation2)
            self.scene.mlab.plot3d(x, y2, np.zeros_like(x), color=(0,0,1), tube_radius=0.1)
            
            # Draw intersection markers
            # Vertical lines at intersection points
            for x_int in [x_intersect1, x_intersect2]:
                self.scene.mlab.plot3d([x_int]*len(y_range), y_range, np.zeros_like(y_range), 
                                     color=(0,0,0), tube_radius=0.05, opacity=0.5)
            # Horizontal lines at intersection y-values
            for y_int in [y_intersect1, y_intersect2]:
                self.scene.mlab.plot3d(x_marker, np.ones_like(x_marker)*y_int, np.zeros_like(x_marker), 
                                     color=(0,0,0), tube_radius=0.05, opacity=0.5)
            
            # Add volume visualization
            for x_pos in x_points:  # Using same x_points as above
                self.create_disk_or_washer(x_pos, self.f2(x_pos), method='disk', volume_color=(0,0,1))
            
            # Third plot (rightmost) - Combined with washers
            self.create_axes(x_range, y_range, z_range, offset, 'Combined')
            self.scene.mlab.plot3d(x + offset, y1, np.zeros_like(x), color=(1,0,0), tube_radius=0.1)
            self.scene.mlab.plot3d(x + offset, y2, np.zeros_like(x), color=(0,0,1), tube_radius=0.1)
            
            # Draw intersection markers
            # Vertical lines at intersection points
            for x_int in [x_intersect1, x_intersect2]:
                self.scene.mlab.plot3d([x_int+offset]*len(y_range), y_range, np.zeros_like(y_range), 
                                     color=(0,0,0), tube_radius=0.05, opacity=0.5)
            # Horizontal lines at intersection y-values
            for y_int in [y_intersect1, y_intersect2]:
                self.scene.mlab.plot3d(x_marker + offset, np.ones_like(x_marker)*y_int, np.zeros_like(x_marker), 
                                     color=(0,0,0), tube_radius=0.05, opacity=0.5)
            
            # Add washer visualization
            for x_pos in x_points:  # Using same x_points as above
                self.create_disk_or_washer(x_pos + offset, self.f1(x_pos), self.f2(x_pos), volume_color=(0,1,0))
            
            # Set up the camera for a better view
            self.scene.mlab.view(azimuth=45, elevation=45, distance=100, focalpoint=(0, 0, 0))
            self.scene.background = (1,1,1)
            
        except Exception as e:
            print(f"Error in draw_plot: {e}")

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
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStretchFactor(1, 1)  # Make the right pane stretch
        layout.addWidget(splitter)
        
        # Left pane with tabs
        left_tabs = QTabWidget()
        left_tabs.setMinimumWidth(300)
        left_tabs.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        # Add tabs to left pane
        tab1 = QWidget()
        tab2 = QWidget()
        left_tabs.addTab(tab1, "Functions")
        left_tabs.addTab(tab2, "Settings")
        
        # Right pane with tabs
        right_tabs = QTabWidget()
        right_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
