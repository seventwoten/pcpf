import cv2
import numpy as np
from math import sin, cos, pi, inf
import matplotlib.pyplot as plt
from scipy.linalg import null_space

def projectPoints(p, cam_pos, cam_or, f=1, bu=1, bv=1, u0=0, v0=0):
    ''' 
        Returns 2D projections (u, v) of 3d scene points specified 
        with respect to world coordinates, on a camera at position 
        cam_pos, and orientation cam_or. 
    '''
    
    # Compute extrinsic matrix to tranform points from world coordinates to camera coordinates
    R = cam_or.T
    t = -np.dot(R, cam_pos.reshape(-1,1))
    extrinsic = np.concatenate((R, t), axis = 1) 
    
    world_coords = np.concatenate((p, np.ones((p.shape[0], 1))), axis = 1)
    
    # scene points in camera coordinates
    s = np.dot(extrinsic, world_coords.T).T
    
    i_f = np.array([[1.0,0.0,0.0]]).T
    j_f = np.array([[0.0,1.0,0.0]]).T
    k_f = np.array([[0.0,0.0,1.0]]).T
    
    u = ( f * bu * np.dot(s, i_f ))/( np.dot(s, k_f) ) + u0
    v = ( f * bv * np.dot(s, j_f ))/( np.dot(s, k_f) ) + v0
    
    # set points behind camera to [inf, inf, inf]
    u[s[:, 2] < 0] = inf
    v[s[:, 2] < 0] = inf
    
    return u, v
    
def translateCamera(x=0, y=0, z=0):
    ''' 
        Translates camera from world coordinates, 
        then displays the camera image with matplotlib.pyplot
    '''

    cam_pos = np.array([x, y, z])
    u, v = projectPoints(p, cam_pos, i_f, j_f, k_f, f, bu, bv, u0, v0)
    
    plt.plot(u, v, 'b*'), plt.axis([-bu, bu, -bv, bv])
    plt.show()
    return np.concatenate((u, v), axis = 1)

def rpy2R(roll, pitch, yaw):
    ''' Converts roll, pitch, yaw (right-handed and in radians) to a rotation matrix.
    '''
    
    cos_r = cos(roll)
    cos_p = cos(pitch)
    cos_y = cos(yaw)
    
    sin_r = sin(roll)
    sin_p = sin(pitch)
    sin_y = sin(yaw)
    
    R = np.array([[cos_r*cos_y,  cos_r*sin_y*sin_p - sin_r*cos_p,  cos_r*sin_y*cos_p + sin_r*sin_p],
                  [sin_r*cos_y,  sin_r*sin_y*sin_p + cos_r*cos_p,  sin_r*sin_y*cos_p - cos_r*sin_p],
                  [-sin_y     ,  cos_y*sin_p                    ,  cos_y*cos_p                    ]])
    
    return R

def xyz2T(x, y, z):
    T = np.array([[  0, -z,  y ],
                  [  z,  0, -x ],
                  [ -y,  x,  0 ]])
    return T

def transformCamera(p, x=0, y=0, z=0, 
                     roll=0, pitch=0, yaw=0, 
                     f=1, bu=1, bv=1, u0=0, v0=0):
    ''' 
        Transforms camera from an initial location and pose aligned to world coordinates, 
        then displays the camera image with matplotlib.pyplot
    '''
    
    # Starting point
    cam_pos = np.array([0.0, 0.0, 0.0])
    cam_or = np.eye(3)
    
    # Rotate camera
    cam_or2 = rpy2R(roll, pitch, yaw)
    print(cam_or2)
    
    # Translate camera
    cam_pos2 = cam_pos + np.array([x, y, z])
    print(cam_pos2)
    
    # Call projectPoints with new position and orientation
    u, v = projectPoints(p, cam_pos2, cam_or2, f, bu, bv, u0, v0)
    
    # Display camera image 
    plt.plot(u, v, 'b*'), plt.axis([-bu, bu, -bv, bv]), plt.gca().invert_yaxis()
    plt.show()
    return np.concatenate((u, v), axis = 1)
    
def getEpilineDeviations(line, pts1):
    ''' 
        Returns squared vertical distances of img 1 points from epipolar line. Shape: (n_points, 1)

        Parameters
        ----------
        line  - coefficients of epiline on img 1, (a, b, c). Shape: (1, 3)
        pts1  - points in img 1. Shape: (n_points, 2)
        
    '''
    # Add one-padding for matrix multiplication
    pts = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis = 1)
    
    # Compute result of au + bv + c = d. Shape: (n_points, 1)
    d = np.dot(pts, line.T)
    
    # Vertical distance from line is d/b. Compute square of this distance:
    return (d/line[0,1]) ** 2
    
def generateSamples(n_samples, ranges):
    ''' Returns n_samples uniformly-distributed sample guesses in 
        n_dims dimensions, with shape (n_dims, n_samples).
        ranges has shape (n_dims, 2), with two values in each row, 
        specifying the start and end in each dimension.
    ''' 
    n_dims = ranges.shape[0]
    samples = np.random.rand(n_dims, n_samples) * (ranges[:, 1, None] - ranges[:, 0, None]) + ranges[:, 0, None]
    
    return samples

    
def computeScore(t, pts1, pts2, n_corr, epsilon, epipole_t):
    ''' Computes score of sample state t
    '''
    score = 0
    mismatches = 0
    
    T = xyz2T(t[0], t[1], t[2])
    R = rpy2R(t[3], t[4], t[5])
    E = np.dot(T, R)
    
    # Compute epipole to ignore nearest points
    epipole = null_space(E.T)
    epipole = (epipole / epipole[2])[:2,0]  # Normalise to image point (u, v, 1)

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 1, E) # Shape: (n_pts2, 3)
    for j in range(lines1.shape[0]):
        sqdists = getEpilineDeviations(lines1[j, :], pts1)
        
        # Ignore points in pts1 too close to epipole
        for k in range(pts1.shape[0]):
            if np.linalg.norm(pts1[k]-epipole) < epipole_t:
                print(k)
                sqdists[k] = inf
                
        if np.any(sqdists < epsilon): 
            if sqdists[j,0] > epsilon or j >= n_corr:
                # print("not a match:" + str(j) + " " + str(np.argmin(sqdists)))
                mismatches += 1
            score += 1
            
    return score, mismatches
    
def ParticleFilter(S, sigma, pts1, pts2, n_corr, epsilon = 1, epipole_t = 0.01, norm_mode = None):
    ''' S         - Represents state-weight pairs. Shape: (dim+1, m) 
                    The first dim rows store m sample states, and the last row stores their importance weights. 
        sigma     - Standard deviation of Gaussian used for resampling in dim dimensions
        pts1      - Points from first image
        pts2      - Points from second image (used to draw epilines on first image)
        n_corr    - The first n_corr points in each image are true correspondences, and the rest are noise points
        epsilon   - Threshold of squared vertical deviation, for counting a point as "near" to an epiline
        epipole_t - Threshold for ignoring a point too close to epipole
        norm_mode - Mode of normalisation for importance weights. 
                    Default  : divide by total of scores over all samples.
                    "softmax": take exp() of each score and normalise over all samples. 
    '''
    dim = S.shape[0] - 1
    m   = S.shape[1]
    weights = np.zeros((1, m))
    normaliser = 0
    S_new = np.empty((dim+1, 0))
    
    # for debugging 
    score_list = []
    mismatches = 0
    matches = 0
    
    for i in range(m):
        # Sample with replacement
        ind = np.random.choice(m, 1, p=S[dim,:].tolist())[0]
        
        # Perturb sample state
        pt = S[:dim,ind]
        t = np.random.normal(loc=pt, scale=sigma, size=None)
        
        score, sample_mismatches = computeScore(t, pts1, pts2, n_corr, epsilon, epipole_t)
        score_list.append(score)
        matches += score
        mismatches += sample_mismatches
        
        # Add new point to output S_new
        new_pt = np.array([[*t, score]]).T
        S_new = np.concatenate((S_new, new_pt), axis = 1)
        
        normaliser += score
    
    # Normalise weights in S_new
    if norm_mode == "softmax": 
        # Use softmax function
        S_new[dim,:] = np.exp(S_new[dim,:]) / np.sum(np.exp(S_new[dim,:]))
        
    else:
        S_new[dim,:] = S_new[dim,:]/normaliser
    
    
    
    return S_new, score_list, mismatches, matches
    
