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
        # X-axis in red
        self.scene.mlab.plot3d(x_range + x_offset, np.zeros_like(x_range), np.zeros_like(x_range), 
                             color=(1, 0, 0), tube_radius=0.1)
        self.scene.mlab.text3d(x_range[-1] + x_offset + 1, 0, 0, 'x', color=(1, 0, 0), scale=0.5)
        
        # Y-axis in black
        self.scene.mlab.plot3d([x_offset]*len(y_range), y_range, np.zeros_like(y_range), 
                             color=(0, 0, 0), tube_radius=0.1)
        self.scene.mlab.text3d(x_offset, y_range[-1] + 1, 0, name, color=(0, 0, 0), scale=0.5)
        
        # Z-axis in black
        self.scene.mlab.plot3d([x_offset]*len(z_range), np.zeros_like(z_range), z_range, 
                             color=(0, 0, 0), tube_radius=0.1)
        self.scene.mlab.text3d(x_offset, 0, z_range[-1] + 1, 'z', color=(0, 0, 0), scale=0.5)

    def create_disk_or_washer(self, x_pos, f1_val, f2_val=None, num_points=100, thickness=0.2, volume_color=(0, 1, 0)):
        """Create a disk or washer at specified x position."""
        # Split into front and back halves for better visualization
        theta_back = np.linspace(np.pi, 2*np.pi, num_points)  # Back half (π to 2π)
        theta_front = np.linspace(0, np.pi, num_points)  # Front half (0 to π)
        
        x_left = x_pos - thickness/2
        x_right = x_pos + thickness/2
        
        # Create washer (two concentric circles)
        r_outer = max(np.abs(f1_val), np.abs(f2_val)) if f2_val is not None else np.abs(f1_val)
        r_inner = min(np.abs(f1_val), np.abs(f2_val)) if f2_val is not None else 0
        
        # Create circular edges with thicker lines
        for x in [x_left, x_right]:
            # Outer circle
            x_circle = x * np.ones_like(theta_back)
            y_outer = r_outer * np.cos(theta_back)
            z_outer = r_outer * np.sin(theta_back)
            self.scene.mlab.plot3d(x_circle, y_outer, z_outer, color=volume_color, tube_radius=0.05)
            
            if f2_val is not None:
                # Inner circle
                y_inner = r_inner * np.cos(theta_back)
                z_inner = r_inner * np.sin(theta_back)
                self.scene.mlab.plot3d(x_circle, y_inner, z_inner, color=volume_color, tube_radius=0.05)
        
        # Create the surfaces - back half (more opaque)
        theta_surf = theta_back
        x_surf = np.array([x_left, x_right])
        theta_grid, x_grid = np.meshgrid(theta_surf, x_surf)
        
        # Outer surface - back
        y_surf_outer = r_outer * np.cos(theta_grid)
        z_surf_outer = r_outer * np.sin(theta_grid)
        self.scene.mlab.mesh(x_grid, y_surf_outer, z_surf_outer, color=volume_color, opacity=0.5)
        
        if f2_val is not None:
            # Inner surface - back
            y_surf_inner = r_inner * np.cos(theta_grid)
            z_surf_inner = r_inner * np.sin(theta_grid)
            self.scene.mlab.mesh(x_grid, y_surf_inner, z_surf_inner, color=volume_color, opacity=0.5)
            
        # Create the surfaces - front half (more transparent)
        theta_surf = theta_front
        theta_grid, x_grid = np.meshgrid(theta_surf, x_surf)
        
        # Outer surface - front
        y_surf_outer = r_outer * np.cos(theta_grid)
        z_surf_outer = r_outer * np.sin(theta_grid)
        self.scene.mlab.mesh(x_grid, y_surf_outer, z_surf_outer, color=volume_color, opacity=0.2)
        
        if f2_val is not None:
            # Inner surface - front
            y_surf_inner = r_inner * np.cos(theta_grid)
            z_surf_inner = r_inner * np.sin(theta_grid)
            self.scene.mlab.mesh(x_grid, y_surf_inner, z_surf_inner, color=volume_color, opacity=0.2)

    def draw_plot(self):
        try:
            # Clear any existing visualization
            self.scene.mlab.clf()
            
            # Create a simple test plot first
            x = np.linspace(-5, 5, 100)
            y1 = self.f1(x)
            y2 = self.f2(x)
            
            # Calculate ranges for axes
            x_range = np.array([-5, 5])
            y_range = np.array([-5, 25])  # Extended for parabola
            z_range = np.array([-5, 5])
            
            # Plot the functions with offset
            offset = 20  # Space between plots
            
            # First plot (leftmost) - Parabola with volume
            self.create_axes(x_range, y_range, z_range, -offset, 'y = ' + self.equation1)
            self.scene.mlab.plot3d(x - offset, y1, np.zeros_like(x), color=(1,0,0), tube_radius=0.1)
            # Add volume visualization
            x_points = np.linspace(-5, 5, 10)
            for x_pos in x_points:
                self.create_disk_or_washer(x_pos - offset, self.f1(x_pos), volume_color=(1,0,0))
            
            # Second plot (middle) - Line with volume
            self.create_axes(x_range, y_range, z_range, 0, 'y = ' + self.equation2)
            self.scene.mlab.plot3d(x, y2, np.zeros_like(x), color=(0,0,1), tube_radius=0.1)
            # Add volume visualization
            for x_pos in x_points:
                self.create_disk_or_washer(x_pos, self.f2(x_pos), volume_color=(0,0,1))
            
            # Third plot (rightmost) - Combined with washers
            self.create_axes(x_range, y_range, z_range, offset, 'Combined')
            self.scene.mlab.plot3d(x + offset, y1, np.zeros_like(x), color=(0,1,0), tube_radius=0.1)
            self.scene.mlab.plot3d(x + offset, y2, np.zeros_like(x), color=(0,0,1), tube_radius=0.1)
            # Add washer visualization
            for x_pos in x_points:
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
