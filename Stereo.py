import numpy as np        
from helpers import rpy2R
from math import inf  
import matplotlib.pyplot as plt

class Stereo:
    points = None
    n = 0
    
    # Common camera parameters
    f  = 1          # focal length
    bu = bv = 1     # scaling factors
    u0 = v0 = 0     # offsets
    
    # First camera position (0, 0, 0) and orientation (aligned to world, rows are i, j, k vectors)
    cam_pos = np.array([0.0, 0.0, 0.0])
    cam_or = np.eye(3)
    
    # Second camera position and orientation
    cam_pos2 = cam_pos
    cam_or2 = cam_or
    
    # 2d views of 3d scene
    view1 = None
    view2 = None
    
    def __init__(self, filename = None, n = 20):
        if filename is None: 
            self.n = n
            self.random_set_points(n)
        else: 
            self.set_points(filename)
        
        self.set_view1()
        
    # Internal functions
    def read_data(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        data = []

        for line in lines:
            data.append(list(map(float, line.split())))
            
        data = np.asarray(data).astype(np.float64)
        
        return data

    def write_data(self, filename, data):
        lines = []
        for i in range(len(data)):
            lines.append(" ".join(["%11.8f" % v for v in data[i]]) + "\n")

        with open(filename, "w") as f:  
            f.writelines(lines)
            
    def projectPoints(self, camera):
        ''' Returns 2D projections (u, v) of 3d scene points specified 
            with respect to world coordinates, on a camera at position 
            cam_pos, and orientation cam_or. 
        '''
        cam_pos, cam_or = (self.cam_pos, self.cam_or) if camera == 1 else (self.cam_pos2, self.cam_or2) 

        # Compute extrinsic matrix to tranform points from world coordinates to camera coordinates
        R = cam_or.T
        t = -np.dot(R, cam_pos.reshape(-1,1))
        extrinsic = np.concatenate((R, t), axis = 1) 
        
        world_coords = np.concatenate((self.points, np.ones((self.n, 1))), axis = 1)
        
        # scene points in camera coordinates
        s = np.dot(extrinsic, world_coords.T).T
        
        i_f = np.array([[1.0,0.0,0.0]]).T
        j_f = np.array([[0.0,1.0,0.0]]).T
        k_f = np.array([[0.0,0.0,1.0]]).T
        
        u = ( self.f * self.bu * np.dot(s, i_f ))/( np.dot(s, k_f) ) + self.u0
        v = ( self.f * self.bv * np.dot(s, j_f ))/( np.dot(s, k_f) ) + self.v0
        
        # set points behind camera to [inf, inf, inf]
        u[s[:, 2] < 0] = inf
        v[s[:, 2] < 0] = inf
        
        return u, v
        
    def transformCamera(self, display=True, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        ''' Transforms camera from an initial location and pose aligned to world coordinates, 
            then displays the camera image with matplotlib.pyplot
        '''
        self.set_second_camera(roll, pitch, yaw, x, y, z)
        
        # Project points with new position and orientation
        self.set_view2()
        
        # Display camera image 
        if display:
            plt.plot(self.view2[:,0], self.view2[:,1], 'b*'), plt.axis([-self.bu*self.f, self.bu*self.f, -self.bv*self.f, self.bv*self.f]), plt.gca().invert_yaxis()
            plt.show()
        return self.view2
        
    # Setters 
    def set_points(self, filename):
        ''' Sets 3d points and count of points (n) from file input
        '''
        data = self.read_data(filename)
        self.points = data
        self.n = int(data.shape[0])
        
    def random_set_points(self, n):
        ''' Sets 3d points randomly from desired number of points (n)
        '''
        # Define n random points in 1x1x1 cube centred at (0,0,2)
        self.points = np.random.rand(n,3)*2 - 1.0 + np.array([[0,0,2.0]])
    
    def set_second_camera(self, roll, pitch, yaw, x, y, z):
        ''' Sets rpy-xyz parameters of second camera relative to the first at (0,0,0,0,0,0)
        '''
        self.cam_or2 = rpy2R(roll, pitch, yaw)
        self.cam_pos2 = self.cam_pos + np.array([x, y, z])
        
    def write_points(self, filename):
        ''' Writes 3d points to file output
        '''
        self.write_data(filename, self.points)
    
    def set_view1(self):
        ''' Sets and returns coordinates in first camera view
        '''
        u, v = self.projectPoints(1)
        self.view1 = np.concatenate((u, v), axis = 1)
        return self.view1
    
    def set_view2(self):
        ''' Sets and returns coordinates in second camera view (if camera 2 is not set, returns None)
        '''
        if self.cam_pos2 is not None and self.cam_or2 is not None:
            u, v = self.projectPoints(2)
            self.view2 = np.concatenate((u, v), axis = 1)
        return self.view2
        


           