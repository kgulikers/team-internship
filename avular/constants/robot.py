"""
Robot & environment geometry constants.
"""

# distance between left and right wheels (m)
TRACK_WIDTH = 0.6

# wheel radius (m)
WHEEL_RADIUS = 0.1175

# thickness of the ground plane (m)
GROUND_THICKNESS = 0.10

# tiny margin to avoid interpenetration (m)
MARGIN = 0.005

# computed Z-height so the wheels just sit above ground
ZERO_INIT_Z = WHEEL_RADIUS + GROUND_THICKNESS / 2 + MARGIN
