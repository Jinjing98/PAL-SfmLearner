import torch
from torchvision.models.optical_flow import raft_large, raft_small, raft

# raft_large(
#             weights="Raft_Large_Weights.DEFAULT", progress=False
#         ).to(device)

class RAFT:
    def __init__(self, device="cuda", weights="Raft_Large_Weights.DEFAULT"):
        # self.model = raft(
        #     weights="Raft_Weights.DEFAULT", progress=False
        # ).to(device)
        # self.model = raft_small(
        #             weights="Raft_Small_Weights.DEFAULT", progress=False
        #         ).to(device)
        self.model = raft_large(
                    weights=weights, 
                    progress=False
                ).to(device)
        # self.model = self.model.eval()
        self.device = device

    def disparity_estimation(self, framel, framer):
        flow = self.__call__(framel, framer)
        flow = flow[:, 0, :, :]
        flow[flow > 0] = 0.0
        flow = -flow
        return flow#.detach().cpu()

    def depth_estimation(self, framel, framer, baseline):
        flow = self.__call__(framel, framer)
        flow = flow[:, 0, :, :]
        depth = baseline / -flow

        valid = torch.logical_and((depth > 0), (depth <= 1.0))
        depth[~valid] = 1.0

        return depth#.detach().cpu()

    def __call__(self, framel, framer):
        framel = framel.to(self.device)
        framer = framer.to(self.device)

        flow_predictions = self.model(framel, framer)#[-1]
        outputs = {}
        assert self.model.num_flow_udpates == 12, "num_flow_udpates must be 4"
        for scale_raw in range(self.model.num_flow_udpates):
            if scale_raw % 4 == 0:
                scale = scale_raw // 4
                outputs[("position", scale)] = flow_predictions[scale_raw]
            else:
                print(f"scale_raw {scale_raw} is not divisible by 4")
        return outputs

        # flow in the shape of (B, 2, H, W) or (2, H, W)
        # return flow#.detach().cpu()
