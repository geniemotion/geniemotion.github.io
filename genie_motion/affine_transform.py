import numpy as np
from scipy.spatial.transform import Rotation as Rot

class AffineWarper:
    def __init__(self):
        # Initialize as Identity transformation (no change)
        self.R = np.eye(3)
        self.t = np.zeros(3)

    def fit(self, source_points, target_points):
        """
        Calculates optimal Rigid Transformation (Rotation R and Translation t)
        to align Source -> Target using SVD (Procrustes Analysis).
        
        Args:
            source_points: (N, 3) numpy array
            target_points: (N, 3) numpy array
        """
        # 1. Compute Centroids
        mu_s = np.mean(source_points, axis=0)
        mu_t = np.mean(target_points, axis=0)

        # 2. Center the data
        S_centered = source_points - mu_s
        T_centered = target_points - mu_t

        # 3. Compute Covariance Matrix H
        H = np.dot(S_centered.T, T_centered)

        # 4. SVD Decompositon
        U, S, Vt = np.linalg.svd(H)
        
        # 5. Compute Rotation R = V * U^T
        self.R = np.dot(Vt.T, U.T)

        # 6. Special Case: Reflection
        # If determinant is -1, it means we have a reflection (mirror), 
        # which is physically impossible for a rigid robot object. We fix this.
        if np.linalg.det(self.R) < 0:
            Vt[-1, :] *= -1
            self.R = np.dot(Vt.T, U.T)

        # 7. Compute Translation t = mu_t - R * mu_s
        self.t = mu_t - np.dot(self.R, mu_s)

    def predict(self, points):
        """
        Applies the transformation x' = R*x + t
        Args:
            points: (N, 3) or (3,) numpy array
        Returns:
            Transformed points
        """
        # Handle single point vs array of points
        points = np.atleast_2d(points)
        
        # (R @ x.T).T + t  -> equivalent to dot(x, R.T) + t
        transformed = np.dot(points, self.R.T) + self.t
        
        return transformed.squeeze()

    def get_jacobian(self, point=None):
        """
        Returns the Jacobian of the transformation.
        For an Affine transform, J is always the Rotation Matrix R.
        
        Args:
            point: Ignored (kept for compatibility with GP class API), 
                   since Affine Jacobian is constant in space.
        Returns:
            (3, 3) Jacobian Matrix
        """
        return self.R