import cv2
import numpy as np
from math import sin, cos, pi, inf, sqrt, atan2, asin, exp
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
    
def R2rpy(R):

    sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular :
        r = atan2( R[1,0], R[0,0])
        p = atan2( R[2,1], R[2,2])
        y = atan2(-R[2,0], sy)

    else :
        p = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        r = 0
        
    return r, p, y
    
def xyz2T(x, y, z):
    T = np.array([[  0, -z,  y ],
                  [  z,  0, -x ],
                  [ -y,  x,  0 ]])
    return T

def transformCamera(p, display=True, x=0, y=0, z=0, 
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
    # print(cam_or2)
    
    # Translate camera
    cam_pos2 = cam_pos + np.array([x, y, z])
    # print(cam_pos2)
    
    # Call projectPoints with new position and orientation
    u, v = projectPoints(p, cam_pos2, cam_or2, f, bu, bv, u0, v0)
    
    # Display camera image 
    if display:
        plt.plot(u, v, 'b*'), plt.axis([-bu*f, bu*f, -bv*f, bv*f]), plt.gca().invert_yaxis()
        plt.show()
    return np.concatenate((u, v), axis = 1)
    
def getEpilineDeviations(line, pts1):
    ''' 
        Returns vertical distances of img 1 points from epipolar line. Shape: (n_points, 1)

        Parameters
        ----------
        line  - coefficients of epiline on img 1, (a, b, c). Shape: (1, 3)
        pts1  - points in img 1. Shape: (n_points, 2)
        
    '''
    # Add one-padding for matrix multiplication
    pts = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis = 1)
    
    # Compute result of au + bv + c = d. Shape: (n_points, 1)
    d = np.dot(pts, line.T)
    
    # Vertical distance from line is d/b
    return np.abs(d/line[0,1])
    
def generateSamples(n_samples, ranges):
    ''' Returns n_samples uniformly-distributed sample guesses in 
        n_dims dimensions, with shape (n_dims, n_samples).
        ranges has shape (n_dims, 2), with two values in each row, 
        specifying the start and end in each dimension.
    ''' 
    n_dims = ranges.shape[0]
    samples = np.random.rand(n_dims, n_samples) * (ranges[:, 1, None] - ranges[:, 0, None]) + ranges[:, 0, None]
    
    return samples

def sphericalToCartesian(r, theta, phi):
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    return x, y, z

def cartesianToSpherical(x, y, z):
    r     = sqrt(x**2 + y**2 + z**2)
    theta = atan2(y,x)
    phi   = atan2(sqrt(x**2 + y**2),z)
    return r, theta, phi    
    
def computeScore(t, orig_pts1, pts2, n_corr, epsilon, epipole_t):
    ''' Computes score of sample state t
    '''
    score = 0
    mismatches = 0
    pts1 = orig_pts1.copy()
    
    T = xyz2T(*sphericalToCartesian(1.0, t[0], t[1]))
    R = rpy2R(t[2], t[3], t[4])
    E = np.dot((1/sqrt(2))*T, R)
    
    if t.shape[0] == 7:
        # Find F using intrinsic parameters (A matrices are inverses of K matrices)
        # Currently, t[5] and t[6] are log(fk) for camera 1 and 2
        fk_inv_1 = exp(-t[5])
        fk_inv_2 = exp(-t[6])
        A1 = np.array([[ fk_inv_1, 0, 0], [0, fk_inv_1, 0], [0,0,1]])
        A2 = np.array([[ fk_inv_2, 0, 0], [0, fk_inv_2, 0], [0,0,1]])
        E  = A1 @ E @ A2

    # Compute epipole to ignore nearest points
    epipole = null_space(E.T, )
    epipole = (epipole / epipole[2])[:2,0]  # Normalise to image point (u, v, 1)

    if t.shape[0] == 7:
        epipole = epipole * exp(t[5]) # Then scale by fk

    # Set near-epipole points in pts1 to np.nan
    pts1[np.linalg.norm(pts1-epipole, axis=1) < epipole_t] = np.nan
    
    # Vectorised method for finding Sampson's distance
    pts1_uvf = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis = 1)
    pts2_uvf = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis = 1)
    
    epilines1 = E @ pts2_uvf.T
    ab1 = (epilines1[:2,:]).reshape(2,-1)      # 2  x n
    
    epilines2 = pts1_uvf @ E
    ab2 = (epilines2[:,:2].T).reshape(2,-1)    # 2  x n
    
    d = pts1_uvf @ epilines1                   # n' x n
    
    denom = np.sum(np.vstack((ab1, ab2))**2, axis = 0)**0.5  # 1  x n
    dists = np.abs(d / denom)                  # n' x n
    
    # Count matches: enforce one-to-one matching and prioritise closest matches first
    all_matches = dists < epsilon
    indices = np.array(np.where(all_matches)).T
    indices_sorted = indices[np.argsort(dists[all_matches])]
    
    curr_rows = list(np.unique(indices_sorted[:,0]))
    curr_cols = list(np.unique(indices_sorted[:,1]))
    
    score = 0
    corr_indices = []
    for i in range(indices_sorted.shape[0]):
        if curr_rows and curr_cols:
            if indices_sorted[i, 0] in curr_rows and indices_sorted[i, 1] in curr_cols:
                curr_rows.remove(indices_sorted[i, 0])
                curr_cols.remove(indices_sorted[i, 1])
                score += 1
                
                # Store correspondence
                corr_indices.append(indices_sorted[i])
                
                # Check mismatches if ground truth is known
                if n_corr:
                    if indices_sorted[i, 0] != indices_sorted[i, 1] or indices_sorted[i, 1] >= n_corr:
                        mismatches += 1

    return score, mismatches, np.array(corr_indices)
    
def ParticleFilter(S, sigma, pts1, pts2, n_corr, epsilon = 0.01, epipole_t = 0.1, norm_mode = None, resampling = None):
    ''' S         - Represents state-weight pairs. Shape: (dim+1, m) 
                    The first dim rows store m sample states, and the last row stores their importance weights. 
        sigma     - Standard deviation of Gaussian used for resampling in dim dimensions
        pts1      - Points from first image
        pts2      - Points from second image (used to draw epilines on first image)
        n_corr    - The first n_corr points in each image are true correspondences, and the rest are noise points.
                    If zero, indicates ground truth is unknown.
        epsilon   - Threshold of squared vertical deviation, for counting a point as "near" to an epiline
        epipole_t - Threshold for ignoring a point too close to epipole
        norm_mode - Mode of normalisation for importance weights. 
                    Default  : divide by total of scores over all samples.
                    "softmax": take exp() of each score and normalise over all samples. 
       resampling - Method for generating new offspring points each iteration. 
                    None     - Gaussian resampling, with given sigma, from neighbourhood in dim dimensions 
                    ComputeF - use estimated correspondences to compute F/E matrix as next guess
                   
    '''
    dim = S.shape[0] - 1
    m   = S.shape[1]
    weights = np.zeros((1, m))
    S_new = np.empty((dim+1, 0))
    
    # for debugging 
    score_list = []
    mismatches = 0
    matches = 0
    
    # Normalise weights in S
    if norm_mode == "softmax": 
        # Use softmax function
        max_corr = min(pts1.shape[0], pts2.shape[0])
        S[dim,:] = (np.exp(S[dim,:] / max_corr) - 1) / np.sum(np.exp(S[dim,:] / max_corr) - 1)
        
    else:
        S[dim,:] = S[dim,:] / np.sum(S[dim,:])


    for i in range(m):
        # Sample with replacement
        ind = np.random.choice(m, 1, p=S[dim,:].tolist())[0]
        
        if resampling == "ComputeF":
            t = S[:dim,ind]
            score, sample_mismatches, corr_indices = computeScore(t, pts1, pts2, n_corr, epsilon, epipole_t)
            
            score = 0
            
            # Compute F from corr_indices (Essential matrix for now)
            if len(corr_indices) >= 6:
                F, inliers = cv2.findEssentialMat(pts2[corr_indices[:,1]], pts1[corr_indices[:,0]], threshold = 0.002)
                
                if inliers is not None: 
                    score = sum(inliers)[0]
                    
                    # Add one new point to output S_new
                    points, R, t, mask = cv2.recoverPose(F, pts2, pts1)
                    r, theta, phi = cartesianToSpherical(t[0],t[1],t[2])
                    roll, p, y = R2rpy(R)
                    
                    t = [theta, phi, roll, p, y]
            
        else:
            # Perturb sample state
            pt = S[:dim,ind]
            t = np.random.normal(loc=pt, scale=sigma, size=None)
            t[:5] = (t[:5] + pi) % (2 * pi) - pi
            
            score, sample_mismatches, corr_indices = computeScore(t, pts1, pts2, n_corr, epsilon, epipole_t)
            
        # Keep track of score and matches
        matches += score
        mismatches += sample_mismatches
            
        # Add new point to output S_new
        if resampling != "ComputeF" or score > 0:
            new_pt = np.array([[*t, score]]).T
            S_new = np.concatenate((S_new, new_pt), axis = 1)
            score_list.append(score)
    
    return S_new, score_list, mismatches, matches
    
