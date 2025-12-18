import torch
from torchvision.models.optical_flow import raft_large, raft_small, raft

# raft_large(
#             weights="Raft_Large_Weights.DEFAULT", progress=False
#         ).to(device)

class RAFT:
    def __init__(self, device="cuda", weights="Raft_Large_Weights.DEFAULT", num_flow_updates=12):
        self.model = raft_large(
                    weights=weights, 
                    progress=False
                ).to(device)
        self.device = device
        self.num_flow_updates = num_flow_updates
        # Set num_flow_updates if the model supports it
        if hasattr(self.model, 'num_flow_updates'):
            self.model.num_flow_updates = num_flow_updates

    def disparity_estimation(self, framel, framer):
        flow = self.__call__(framel, framer)
        flow = flow[:, 0, :, :]
        flow[flow > 0] = 0.0
        flow = -flow
        return flow

    def depth_estimation(self, framel, framer, baseline):
        flow = self.__call__(framel, framer)
        flow = flow[:, 0, :, :]
        depth = baseline / -flow

        valid = torch.logical_and((depth > 0), (depth <= 1.0))
        depth[~valid] = 1.0

        return depth

    def __call__(self, framel, framer):
        """
        Forward pass through RAFT.
        Returns final dense flow (B, 2, H, W).
        """
        framel = framel.to(self.device)
        framer = framer.to(self.device)

        flow_predictions = self.model(framel, framer)
        
        # Return the final flow prediction (last iteration)
        # flow_predictions is a list of flows from each iteration
        if isinstance(flow_predictions, (list, tuple)):
            flow = flow_predictions[-1]  # Final flow
        else:
            flow = flow_predictions
        
        # Ensure shape is (B, 2, H, W)
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)
        
        return flow

if __name__ == "__main__":
    import PIL.Image as pil
    import torchvision
    raft = RAFT(device="cuda", weights="Raft_Large_Weights.DEFAULT", num_flow_updates=12)
    img1_path = "/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/dataset1/keyframe3/image_02/data/0000000001.png"
    img2_path = "/mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/dataset1/keyframe3/image_02/data/0000000002.png"
    img1 = torchvision.transforms.ToTensor()(pil.open(img1_path))
    img2 = torchvision.transforms.ToTensor()(pil.open(img2_path))
    img1 = img1.unsqueeze(0).to("cuda")
    img2 = img2.unsqueeze(0).to("cuda")
    flow = raft(img1, img2)




