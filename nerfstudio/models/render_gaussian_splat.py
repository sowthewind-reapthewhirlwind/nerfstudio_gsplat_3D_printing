
import numpy as np
import torch
from gsplat.rendering import rasterization

# Step 1: Load the data from the .npz file
input_file = r"./gaussian_splatting_data.npz"
data = np.load(input_file)

# Step 2: Convert the loaded data into PyTorch tensors
means_crop = torch.tensor(data['points'], dtype=torch.float32)
quats_crop = torch.tensor(data['quats'], dtype=torch.float32)
scales_crop = torch.tensor(data['scales'], dtype=torch.float32)
colors_crop = torch.tensor(data['colors'], dtype=torch.float32)
opacities_crop = torch.tensor(data['opacities'], dtype=torch.float32)
viewmat = torch.tensor(data['viewmats'], dtype=torch.float32)  # View matrices
K = torch.tensor(data['Ks'], dtype=torch.float32)  # Intrinsic camera parameters

# Step 3: Extract other rendering parameters
W = data['width'].item()  # Image width
H = data['height'].item()  # Image height
render_mode = "RGB"  # Example render mode, modify as needed
sh_degree_to_use = None  # Modify if needed
near_plane = data['near_plane'].item()
far_plane = data['far_plane'].item()
backgrounds = torch.tensor(data['backgrounds'], dtype=torch.float32)  # Background color

# Step 4: Call the rasterization function with the extracted and converted data
render, alpha, self_info = rasterization(
    means=means_crop,
    quats=quats_crop,  # rasterization does normalization internally
    scales=torch.exp(scales_crop),  # Apply exponential to the scales as per your function
    opacities=torch.sigmoid(opacities_crop).squeeze(-1),
    colors=colors_crop,
    viewmats=viewmat,  # [1, 4, 4]
    Ks=K,  # [1, 3, 3]
    width=W,
    height=H,
    packed=False,
    near_plane=near_plane,
    far_plane=far_plane,
    render_mode=render_mode,
    sh_degree=sh_degree_to_use,
    sparse_grad=False,
    absgrad=False,  # Example value, modify as needed
    rasterize_mode="classic",  # Example value, modify as needed
    # radius_clip=3.0,  # Uncomment and modify if needed
)

# Output rendered result (you can modify this as per your workflow)
print("Rendered colors: ", render)
print("Rendered alpha: ", alpha)
