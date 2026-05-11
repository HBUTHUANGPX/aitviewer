from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from smplx.utils import Struct

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer


DEFAULT_SMPLH_MODEL_CANDIDATES = (
    Path("/home/hpx/HPX_Loco/loco-mujoco/datasets/smplh/SMPLH_NEUTRAL.pkl"),
    Path("/home/hpx/2025_5_24/loco-mujoco/datasets/smpl/SMPLH_NEUTRAL.pkl"),
)
DEFAULT_SMPL_MODEL_CANDIDATES = (
    Path("/home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz"),
    Path("/home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.pkl"),
)
DEFAULT_SMPLX_MODEL_CANDIDATES = (
    Path("/home/hpx/HPX_LOCO_2/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.pkl"),
    Path("/home/hpx/HPX_LOCO_2/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz"),
)


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    gender: str
    candidates: Tuple[Path, ...]


MODEL_SPECS = {
    "smplh": ModelSpec("smplh", "neutral", DEFAULT_SMPLH_MODEL_CANDIDATES),
    "smpl": ModelSpec("smpl", "neutral", DEFAULT_SMPL_MODEL_CANDIDATES),
    "smplx": ModelSpec("smplx", "neutral", DEFAULT_SMPLX_MODEL_CANDIDATES),
}


def first_existing_path(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_args():
    parser = ArgumentParser(description="Open a local SMPL/SMPL-H/SMPL-X model in aitviewer.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="smplh", help="Body model type to load.")
    parser.add_argument("--model-path", type=Path, help="Override the built-in local model path candidates.")
    parser.add_argument("--gender", choices=("neutral", "female", "male"), help="Override the default model gender.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to create.")
    parser.add_argument("--animate", action="store_true", help="Rotate the root a little over time.")
    parser.add_argument("--dry-run", action="store_true", help="Load the model and print info without opening a window.")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        return "cuda:0"
    return device_arg


def resolve_model_path(model_key, override):
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"Model path does not exist: {override}")
        return override

    candidates = MODEL_SPECS[model_key].candidates
    model_path = first_existing_path(candidates)
    if model_path is None:
        lines = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(f"No model file found for {model_key}. Checked:\n{lines}")
    return model_path


def model_layer_kwargs(model_type, model_path):
    suffix = model_path.suffix.lower().lstrip(".")
    if model_type == "smpl" and suffix == "npz":
        model_data = np.load(model_path, allow_pickle=True)
        return {"data_struct": Struct(**model_data)}
    if suffix in {"pkl", "npz"}:
        return {"ext": suffix}
    return {}


def make_mesh_sequence(smpl_layer, frames, animate):
    poses_body = torch.zeros((frames, smpl_layer.bm.NUM_BODY_JOINTS * 3), dtype=C.f_precision, device=C.device)
    poses_root = torch.zeros((frames, 3), dtype=C.f_precision, device=C.device)
    betas = torch.zeros((1, smpl_layer.num_betas), dtype=C.f_precision, device=C.device)
    hand_dof = 0

    if hasattr(smpl_layer.bm, "NUM_HAND_JOINTS"):
        if smpl_layer.bm.use_pca:
            hand_dof = smpl_layer.bm.num_pca_comps
        else:
            hand_dof = smpl_layer.bm.NUM_HAND_JOINTS * 3

    poses_left_hand = torch.zeros((frames, hand_dof), dtype=C.f_precision, device=C.device)
    poses_right_hand = torch.zeros((frames, hand_dof), dtype=C.f_precision, device=C.device)

    if animate and frames > 1:
        poses_root[:, 1] = torch.linspace(0.0, 2.0 * np.pi, frames, dtype=C.f_precision, device=C.device)

    vertices, _ = smpl_layer.fk(
        poses_body=poses_body,
        poses_root=poses_root,
        betas=betas,
        poses_left_hand=poses_left_hand,
        poses_right_hand=poses_right_hand,
    )

    return Meshes(
        vertices.detach().cpu().numpy(),
        smpl_layer.faces.detach().cpu().numpy(),
        color=(160 / 255, 160 / 255, 160 / 255, 1.0),
        flat_shading=False,
        name=f"{smpl_layer.model_type.upper()} local model",
    )


def main():
    args = parse_args()
    spec = MODEL_SPECS[args.model]
    model_path = resolve_model_path(args.model, args.model_path)
    gender = args.gender or spec.gender
    device = resolve_device(args.device)

    C.update_conf(OmegaConf.create({"smplx_models": str(model_path), "device": device}))

    print(f"Loading {spec.model_type.upper()} model")
    print(f"  path:   {model_path}")
    print(f"  gender: {gender}")
    print(f"  device: {C.device}")

    smpl_layer = SMPLLayer(
        model_type=spec.model_type,
        gender=gender,
        device=C.device,
        **model_layer_kwargs(spec.model_type, model_path),
    )
    smpl_mesh = make_mesh_sequence(
        smpl_layer,
        frames=args.frames,
        animate=args.animate,
    )

    print(f"Loaded vertices: {smpl_layer.bm.get_num_verts()}")
    print(f"Loaded body joints: {smpl_layer.bm.NUM_BODY_JOINTS}")

    if args.dry_run:
        print("Dry run completed; not opening viewer.")
        return

    viewer = Viewer()
    viewer.scene.add(smpl_mesh)
    viewer.scene.camera.position = np.array([0.0, 1.0, 4.0])
    viewer.run_animations = args.animate
    viewer.run()


if __name__ == "__main__":
    main()
