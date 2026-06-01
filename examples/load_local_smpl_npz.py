from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from local_smpl_viewer import MODEL_SPECS, model_layer_kwargs, resolve_device, resolve_model_path


POSE_KEYS = ("poses", "pose", "full_pose")
TRANS_KEYS = ("trans", "transl", "smpl_transl", "translation", "translations")
ROOT_KEYS = ("poses_root", "root_orient", "global_orient", "smpl_global_orient")
BODY_KEYS = ("poses_body", "body_pose", "smpl_body_pose")
BETA_KEYS = ("betas", "smpl_betas", "shape", "shapes")


def first_key(data, keys):
    for key in keys:
        if key in data:
            return key
    return None


def as_frames(array):
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[None]
    return array.astype(np.float32)


def slice_frames(array, start_frame=None, end_frame=None, stride=1):
    return array[start_frame:end_frame:stride]


def load_smpl_data(path):
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "keys"):
        return data

    array = np.asarray(data)
    if array.dtype == object and array.size == 1:
        item = array.item()
        if hasattr(item, "keys"):
            return item

    if array.ndim >= 1 and np.issubdtype(array.dtype, np.number):
        return {"poses": array}

    raise ValueError(
        f"Unsupported SMPL data in {path}. Expected a .npz, a .npy saved from a dict, "
        "or a numeric pose array."
    )


def hand_dof(smpl_layer):
    if not hasattr(smpl_layer.bm, "NUM_HAND_JOINTS"):
        return 0
    if smpl_layer.bm.use_pca:
        return smpl_layer.bm.num_pca_comps
    return smpl_layer.bm.NUM_HAND_JOINTS * 3


def parse_smpl_npz(npz_path, smpl_layer, start_frame=None, end_frame=None, stride=1):
    data = load_smpl_data(npz_path)
    keys = set(data.keys())

    root_key = first_key(data, ROOT_KEYS)
    body_key = first_key(data, BODY_KEYS)

    if root_key is not None and body_key is not None:
        poses_body = as_frames(data[body_key])
        poses_root = as_frames(data[root_key])
        poses_left_hand = as_frames(data["poses_left_hand"]) if "poses_left_hand" in keys else None
        poses_right_hand = as_frames(data["poses_right_hand"]) if "poses_right_hand" in keys else None
    else:
        pose_key = first_key(data, POSE_KEYS)
        if pose_key is None:
            available = ", ".join(sorted(keys))
            raise KeyError(f"Could not find a pose key. Expected one of {POSE_KEYS}, got keys: {available}")

        poses = as_frames(data[pose_key])
        body_end = 3 + smpl_layer.bm.NUM_BODY_JOINTS * 3
        if poses.shape[1] < body_end:
            raise ValueError(
                f"Pose array has shape {poses.shape}, but {smpl_layer.model_type} needs at least {body_end} values "
                "per frame: root(3) + body joints."
            )

        poses_root = poses[:, :3]
        poses_body = poses[:, 3:body_end]

        h_dof = hand_dof(smpl_layer)
        left_end = body_end + h_dof
        right_end = left_end + h_dof
        poses_left_hand = poses[:, body_end:left_end] if h_dof and poses.shape[1] >= left_end else None
        poses_right_hand = poses[:, left_end:right_end] if h_dof and poses.shape[1] >= right_end else None

    original_n_frames = poses_body.shape[0]
    poses_body = slice_frames(poses_body, start_frame, end_frame, stride)
    poses_root = slice_frames(poses_root, start_frame, end_frame, stride)

    n_frames = poses_body.shape[0]
    h_dof = hand_dof(smpl_layer)
    if poses_left_hand is None:
        poses_left_hand = np.zeros((n_frames, h_dof), dtype=np.float32)
    else:
        poses_left_hand = slice_frames(poses_left_hand, start_frame, end_frame, stride)
    if poses_right_hand is None:
        poses_right_hand = np.zeros((n_frames, h_dof), dtype=np.float32)
    else:
        poses_right_hand = slice_frames(poses_right_hand, start_frame, end_frame, stride)

    beta_key = first_key(data, BETA_KEYS)
    if beta_key is not None:
        betas = as_frames(data[beta_key])
    else:
        betas = np.zeros((1, smpl_layer.num_betas), dtype=np.float32)
    if betas.shape[0] == original_n_frames:
        betas = slice_frames(betas, start_frame, end_frame, stride)

    trans_key = first_key(data, TRANS_KEYS)
    if trans_key is None:
        trans = np.zeros((n_frames, 3), dtype=np.float32)
    else:
        trans = slice_frames(as_frames(data[trans_key]), start_frame, end_frame, stride)

    return {
        "poses_body": poses_body,
        "poses_root": poses_root,
        "poses_left_hand": poses_left_hand,
        "poses_right_hand": poses_right_hand,
        "betas": betas,
        "trans": trans,
    }


def to_torch(array):
    return torch.as_tensor(array, dtype=C.f_precision, device=C.device)


def make_mesh_from_params(params, smpl_layer, name):
    vertices, _ = smpl_layer.fk(
        poses_body=to_torch(params["poses_body"]),
        poses_root=to_torch(params["poses_root"]),
        poses_left_hand=to_torch(params["poses_left_hand"]),
        poses_right_hand=to_torch(params["poses_right_hand"]),
        betas=to_torch(params["betas"]),
        trans=to_torch(params["trans"]),
    )

    return Meshes(
        vertices.detach().cpu().numpy(),
        smpl_layer.faces.detach().cpu().numpy(),
        color=(160 / 255, 160 / 255, 160 / 255, 1.0),
        flat_shading=False,
        name=name,
    )


def parse_args():
    parser = ArgumentParser(description="Load a local SMPL parameter .npz or .npy file in aitviewer.")
    parser.add_argument("npz", type=Path, help="Path to a .npz/.npy with SMPL parameters.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="smpl", help="Body model type.")
    parser.add_argument("--model-path", type=Path, help="Override the local model path.")
    parser.add_argument("--gender", choices=("neutral", "female", "male"), help="Override model gender.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="Parse and load without opening a window.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.npz.exists():
        raise FileNotFoundError(args.npz)
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    spec = MODEL_SPECS[args.model]
    model_path = resolve_model_path(args.model, args.model_path)
    gender = args.gender or spec.gender
    device = resolve_device(args.device)
    C.update_conf(OmegaConf.create({"smplx_models": str(model_path), "device": device}))

    smpl_layer = SMPLLayer(
        model_type=spec.model_type,
        gender=gender,
        device=C.device,
        **model_layer_kwargs(spec.model_type, model_path),
    )
    params = parse_smpl_npz(args.npz, smpl_layer, args.start_frame, args.end_frame, args.stride)

    print(f"Loaded SMPL data: {args.npz}")
    print(f"  model:  {spec.model_type}")
    print(f"  model path: {model_path}")
    print(f"  frames: {params['poses_body'].shape[0]}")
    print(f"  body pose shape: {params['poses_body'].shape}")
    print(f"  trans shape: {params['trans'].shape}")

    mesh = make_mesh_from_params(params, smpl_layer, name=args.npz.stem)

    if args.dry_run:
        print("Dry run completed; not opening viewer.")
        return

    viewer = Viewer()
    viewer.scene.add(mesh)
    viewer.scene.camera.position = np.array([0.0, 1.0, 4.0])
    viewer.scene.fps = args.fps
    viewer.playback_fps = args.fps
    viewer.run_animations = True
    viewer.run()


if __name__ == "__main__":
    main()
