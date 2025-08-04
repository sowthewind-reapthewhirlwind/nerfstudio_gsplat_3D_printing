import bpy
import math
from mathutils import Vector
import random

# Parameters
num_cameras = 256       # Number of cameras to create
radius = 20            # Radius of the circle around the object
height = 3                  # Height of the cameras (same plane as the object)
look_at = Vector((0, 0, 0))  # Coordinates of the object to look at

# Optional: Remove existing cameras
for obj in bpy.context.scene.objects:
    if obj.type == 'CAMERA':
        bpy.data.objects.remove(obj, do_unlink=True)

for i in range(num_cameras):
    # Calculate angle for camera placement
    angle = i * (2 * math.pi / num_cameras)
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    z = height + random.randint(0,20)

    # Add a new camera at the calculated position
    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object

    # Point the camera at the object
    direction = look_at - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()