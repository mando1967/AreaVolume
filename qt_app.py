import os
os.environ['ETS_TOOLKIT'] = 'qt'
os.environ['QT_API'] = 'pyqt5'

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi import mlab
import numpy as np
from scipy.optimize import fsolve

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    height=600, width=1200, show_label=False),
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

    def find_intersection(self):
        """Find intersection points of f1 and f2"""
        def equation(x):
            return self.f1(x) - self.f2(x)
        
        # Find two intersection points
        x1 = fsolve(equation, -1)[0]
        x2 = fsolve(equation, 2)[0]
        return x1, x2

    def create_disk_or_washer(self, x_pos, f1_val, f2_val=None, method='washer', 
                             num_points=100, thickness=0.2, volume_color=(0, 1, 0)):
        """Create a disk or washer at specified x position."""
        theta = np.linspace(0, 2*np.pi, num_points)
        x_left = x_pos - thickness/2
        x_right = x_pos + thickness/2
        
        if method == 'disk':
            r = np.abs(f1_val)
            self.create_single_disk(x_left, x_right, r, theta, volume_color)
        else:  # washer
            r_outer = max(np.abs(f1_val), np.abs(f2_val))
            r_inner = min(np.abs(f1_val), np.abs(f2_val))
            self.create_washer(x_left, x_right, r_outer, r_inner, theta, volume_color)

    def create_single_disk(self, x_left, x_right, radius, theta, color):
        """Create a single disk."""
        x_grid, theta_grid = np.meshgrid(np.array([x_left, x_right]), theta)
        y_grid = radius * np.cos(theta_grid)
        z_grid = radius * np.sin(theta_grid)
        self.scene.mlab.mesh(x_grid, y_grid, z_grid, color=color)

    def create_washer(self, x_left, x_right, r_outer, r_inner, theta, color):
        """Create a washer (disk with hole)."""
        self.create_single_disk(x_left, x_right, r_outer, theta, color)
        self.create_single_disk(x_left, x_right, r_inner, theta, (1,1,1))

    def create_axes(self, x_range, y_range, z_range, x_offset=0, name=''):
        """Create custom axes with labels and ticks."""
        x_range = np.array(x_range) + x_offset
        
        self.scene.mlab.plot3d(x_range, np.zeros_like(x_range), np.zeros_like(x_range), 
                             color=(1, 0, 0), tube_radius=0.01)
        self.scene.mlab.text3d(x_range[-1] + 0.2, 0, 0, 'x', color=(1, 0, 0), scale=0.3)
        
        self.scene.mlab.plot3d([x_offset]*len(y_range), y_range, np.zeros_like(y_range), 
                             color=(0, 0, 0), tube_radius=0.01)
        self.scene.mlab.text3d(x_offset, y_range[-1] + 0.2, 0, name, color=(0, 0, 0), scale=0.3)
        
        self.scene.mlab.plot3d([x_offset]*len(z_range), np.zeros_like(z_range), z_range, 
                             color=(0, 0, 1), tube_radius=0.01)
        self.scene.mlab.text3d(x_offset, 0, z_range[-1] + 0.2, 'z', color=(0, 0, 1), scale=0.3)

    def plot_volume_surface(self, x, y, color, x_offset=0, opacity=0.5):
        """Plot the volume surface with rotation."""
        theta = np.linspace(0, 2*np.pi, 50)
        X, Theta = np.meshgrid(x, theta)
        Y = np.outer(np.cos(theta), y)
        Z = np.outer(np.sin(theta), y)
        self.scene.mlab.mesh(X + x_offset, Y, Z, color=color, opacity=opacity)

    def plot_single_function(self, f, x_range, color, name, x_offset=0):
        """Plot a single function with its volumetric rotation."""
        x = np.linspace(x_range[0], x_range[1], 100)
        y = f(x)
        
        self.scene.mlab.plot3d(x + x_offset, y, np.zeros_like(x), color=color, tube_radius=0.05)
        self.plot_volume_surface(x, y, color, x_offset)
        
        x_pos = np.linspace(x_range[0], x_range[1], 5)
        for pos in x_pos:
            y_val = f(pos)
            self.create_disk_or_washer(pos + x_offset, y_val, method='disk', volume_color=color)

    def draw_plot(self):
        """Create the visualization"""
        try:
            # Clear any existing visualization
            self.scene.mlab.clf()
            
            # Find intersection points
            x1, x2 = self.find_intersection()
            x_range = x2 - x1
            x_extension = x_range * 0.35
            plot_x_range = (x1 - x_extension, x2 + x_extension)
            
            # Calculate plot ranges
            y_vals = [self.f1(np.linspace(plot_x_range[0], plot_x_range[1], 100)),
                     self.f2(np.linspace(plot_x_range[0], plot_x_range[1], 100))]
            y_min = min(np.min(y_vals[0]), np.min(y_vals[1])) - 1
            y_max = max(np.max(y_vals[0]), np.max(y_vals[1])) + 1
            y_range = np.array([y_min, y_max])
            z_range = np.array([-max(abs(y_range)), max(abs(y_range))])
            
            # Calculate offset
            plot_width = plot_x_range[1] - plot_x_range[0]
            offset = plot_width * 2.0
            
            # Create axes for all three plots
            self.create_axes(plot_x_range, y_range, z_range, -offset, 'y1 = ' + self.equation2)
            self.create_axes(plot_x_range, y_range, z_range, 0, 'y2 = ' + self.equation1)
            self.create_axes(plot_x_range, y_range, z_range, offset, 'y = Volume')
            
            # Plot the three visualizations
            self.plot_single_function(self.f2, plot_x_range, (1,0,0), 'y1', -offset)  # Line in red
            self.plot_single_function(self.f1, plot_x_range, (0,0,1), 'y2', 0)      # Parabola in blue
            
            # Plot combined visualization
            x = np.linspace(plot_x_range[0], plot_x_range[1], 100)
            y1, y2 = self.f1(x), self.f2(x)
            for x_pos in np.linspace(x1, x2, 5):
                f1_val = self.f1(x_pos)
                f2_val = self.f2(x_pos)
                self.create_disk_or_washer(x_pos + offset, f1_val, f2_val, method='washer',
                                         volume_color=(0, 1, 0))
            
            # Set up the camera
            self.scene.mlab.view(azimuth=45, elevation=45, distance='auto')
            self.scene.background = (1, 1, 1)
            
        except Exception as e:
            print(f"Error in draw_plot: {e}")

class MayaviQWidget(QWidget):
    def __init__(self, parent=None):
        super(MayaviQWidget, self).__init__(parent)
        layout = QVBoxLayout(self)
        
        self.visualization = Visualization()
        ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(ui)
        
        # Draw the plot after UI is set up
        self.visualization.draw_plot()

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Volume Visualization")
        self.setGeometry(100, 100, 1200, 600)
        
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        
        mayavi_widget = MayaviQWidget(self)
        layout.addWidget(mayavi_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
