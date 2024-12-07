import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve

def f1(x):
    """First function: parabola"""
    return x**2

def f2(x):
    """Second function: line"""
    return 2*x + 1

def rotate_around_x(x, y1, y2, num_points=50):
    """Generate points for rotation around x-axis for region between y1 and y2"""
    theta = np.linspace(0, np.pi, num_points)  # Changed to pi for 180-degree rotation
    X = np.repeat(x[:, np.newaxis], num_points, axis=1)
    
    # Create a surface for each y-value
    Y1 = np.outer(y1, np.cos(theta))
    Z1 = np.outer(y1, np.sin(theta))
    Y2 = np.outer(y2, np.cos(theta))
    Z2 = np.outer(y2, np.sin(theta))
    
    return X, Y1, Z1, Y2, Z2

def rotate_around_y(x, y1, y2, num_points=50):
    """Generate points for rotation around y-axis for region between y1 and y2"""
    theta = np.linspace(0, np.pi, num_points)  # Changed to pi for 180-degree rotation
    Y = np.repeat(y1[:, np.newaxis], num_points, axis=1)
    
    # Create a surface for each x-value
    X1 = np.outer(x, np.cos(theta))
    Z1 = np.outer(x, np.sin(theta))
    
    return X1, Y, Z1

def find_intersection():
    """Find intersection points of f1 and f2"""
    def equation(x):
        return f1(x) - f2(x)
    
    # Find two intersection points
    x1 = fsolve(equation, -1)[0]
    x2 = fsolve(equation, 2)[0]
    return x1, x2

def plot_volumetric_figure():
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Original functions
    ax1 = fig.add_subplot(131)
    x = np.linspace(-2, 3, 1000)
    ax1.plot(x, f1(x), 'b-', label='f1(x) = x²', linewidth=2)
    ax1.plot(x, f2(x), 'r-', label='f2(x) = 2x + 1', linewidth=2)
    
    # Find intersection points and highlight intersection region
    x1, x2 = find_intersection()
    x_fill = np.linspace(x1, x2, 100)
    y1_fill = f1(x_fill)  # Upper boundary
    y2_fill = f2(x_fill)  # Lower boundary
    ax1.fill_between(x_fill, y1_fill, y2_fill, alpha=0.3, color='g', label='Intersection')
    
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Original Functions')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot 2: Rotation around x-axis
    ax2 = fig.add_subplot(132, projection='3d')
    # Plot original functions in 3D
    ax2.plot(x_fill, f1(x_fill), np.zeros_like(x_fill), 'b-', linewidth=2, label='f1(x)')
    ax2.plot(x_fill, f2(x_fill), np.zeros_like(x_fill), 'r-', linewidth=2, label='f2(x)')
    
    # Plot rotated intersection region
    X, Y1, Z1, Y2, Z2 = rotate_around_x(x_fill, y1_fill, y2_fill)
    # Plot the upper and lower surfaces with improved shading
    ax2.plot_surface(X, Y1, Z1, alpha=0.3, color='green', shade=True, lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45))
    ax2.plot_surface(X, Y2, Z2, alpha=0.3, color='green', shade=True, lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45))
    # Plot vertical lines at the boundaries to close the volume
    for i in range(0, X.shape[1], 3):  # Plot every 3rd line to avoid clutter
        ax2.plot([X[0,i], X[0,i]], [Y1[0,i], Y2[0,i]], [Z1[0,i], Z2[0,i]], 'g-', alpha=0.2)
    # Add closing surfaces at the ends with shading
    for i in [0, -1]:  # First and last x-value
        y_points = np.linspace(Y2[i,0], Y1[i,0], 10)
        z_points = np.linspace(Z2[i,0], Z1[i,0], 10)
        Y_end, Z_end = np.meshgrid(y_points, z_points)
        X_end = np.full_like(Y_end, X[i,0])
        ax2.plot_surface(X_end, Y_end, Z_end, alpha=0.3, color='green', shade=True, lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45))
    
    ax2.set_title('180° Rotation around X-axis')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.view_init(elev=20, azim=-90)
    
    # Plot 3: Rotation around y-axis
    ax3 = fig.add_subplot(133, projection='3d')
    # Plot original functions in 3D
    ax3.plot(x_fill, f1(x_fill), np.zeros_like(x_fill), 'b-', linewidth=2, label='f1(x)')
    ax3.plot(x_fill, f2(x_fill), np.zeros_like(x_fill), 'r-', linewidth=2, label='f2(x)')
    
    # Plot rotated intersection region with improved shading
    X1, Y, Z1 = rotate_around_y(x_fill, y1_fill, y2_fill)
    # Plot the surface with shading
    ax3.plot_surface(X1, Y, Z1, alpha=0.3, color='green', shade=True, lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45))
    
    ax3.set_title('180° Rotation around Y-axis')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    ax3.view_init(elev=20, azim=0)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_volumetric_figure()
