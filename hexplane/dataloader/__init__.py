from .dnerf_dataset import DNerfDataset
from .neural_3D_dataset_NDC import Neural3D_NDC_Dataset
from .phototourism_dataset import PhotoTourismDataset
from .llff import LLFFDataset
def get_train_dataset(cfg, is_stack=False):
    if cfg.data.dataset_name == "dnerf":
        train_dataset = DNerfDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
        )
    elif cfg.data.dataset_name == "neural3D_NDC":
        train_dataset = Neural3D_NDC_Dataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )
    elif cfg.data.dataset_name=="llff":
         train_dataset=LLFFDataset(
         cfg.data.datadir,
         split="train",
         hold_every=8,
         is_stack=is_stack,
         downsample=4,
         blender2opencv=np.eye(4),
         white_bg=False,
         near_far=[0.0,1.0],
         scene_bbox=torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]]),
         centre=torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3),
         invradius=1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3),

        )
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset


def get_test_dataset(cfg, is_stack=True):
    if cfg.data.dataset_name == "dnerf":
        test_dataset = DNerfDataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
        )
    elif cfg.data.dataset_name == "neural3D_NDC":
        test_dataset = Neural3D_NDC_Dataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )
    elif cfg.data.dataset_name=="llff":
        test_dataset=LLFFDataset(

         cfg.data.datadir,
         split="test",
         hold_every=8,
         is_stack=is_stack,
         downsample=4,
         blender2opencv=np.eye(4),
         white_bg=False,
         near_far=[0.0,1.0],
         scene_bbox=torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]]),
         centre=torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3),
         invradius=1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3),


            
        )
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset
