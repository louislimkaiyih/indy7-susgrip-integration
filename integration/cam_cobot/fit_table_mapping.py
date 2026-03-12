import numpy as np

# Camera XY in meters
cam_xy = np.array([
    [-0.0096, -0.0016],   # Center
    [ 0.2078, -0.1173],   # Bottom Left
    [-0.2065, -0.1092],   # Bottom Right
    [-0.2133,  0.1058],   # Top Right
    [ 0.2119,  0.1118],   # Top Left
    [-0.0048, -0.1125],   # Bottom Mid
], dtype=float)

# Robot XY in mm
robot_xy = np.array([
    [288.85, 188.01],     # Center
    [495.65,  53.58],     # Bottom Left
    [287.73, 414.82],     # Bottom Right
    [ 93.96, 313.31],     # Top Right
    [299.13, -59.40],     # Top Left
    [389.56, 237.66],     # Bottom Mid
], dtype=float)

# Build affine model:
# robot_x = a1*cam_x + a2*cam_y + a3
# robot_y = b1*cam_x + b2*cam_y + b3
A = np.column_stack([
    cam_xy[:, 0],
    cam_xy[:, 1],
    np.ones(len(cam_xy))
])

coef_x, _, _, _ = np.linalg.lstsq(A, robot_xy[:, 0], rcond=None)
coef_y, _, _, _ = np.linalg.lstsq(A, robot_xy[:, 1], rcond=None)

def cam_to_robot_xy(cam_x_m, cam_y_m):
    robot_x_mm = coef_x[0] * cam_x_m + coef_x[1] * cam_y_m + coef_x[2]
    robot_y_mm = coef_y[0] * cam_x_m + coef_y[1] * cam_y_m + coef_y[2]
    return robot_x_mm, robot_y_mm

# Check fit on the same 6 points
predicted = np.array([cam_to_robot_xy(x, y) for x, y in cam_xy])
errors = np.linalg.norm(predicted - robot_xy, axis=1)

print("coef_x =", coef_x)
print("coef_y =", coef_y)
print()

for i, err in enumerate(errors, start=1):
    print(f"Point {i}: predicted={predicted[i-1]}, actual={robot_xy[i-1]}, error={err:.2f} mm")

print()
print(f"Mean error = {errors.mean():.2f} mm")
print(f"Max error  = {errors.max():.2f} mm")

# Example usage
test_cam_x = 0.1031
test_cam_y = -0.0198
test_robot_x, test_robot_y = cam_to_robot_xy(test_cam_x, test_cam_y)
print()
print(f"Example input cam XY = ({test_cam_x}, {test_cam_y}) m")
print(f"Predicted robot XY   = ({test_robot_x:.2f}, {test_robot_y:.2f}) mm")
