# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np
import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.spheres import Spheres
from aitviewer.scene.node import Node
from aitviewer.utils import local_to_global
from aitviewer.utils import to_numpy as c2c
from aitviewer.utils import to_torch


class FLAMESequence(Node):
    """
    Represents a temporal sequence of FLAME face model parameters.
    Does not inherit from SMPLSequence because FLAME is sufficiently different that
    would make an inheritence messy.
    """

    def __init__(
        self,
        flame_layer,
        poses_root=None,
        poses_neck=None,
        poses_jaw=None,
        poses_leye=None,
        poses_reye=None,
        expression=None,
        betas=None,
        trans=None,
        device=None,
        dtype=None,
        show_landmarks=False,
        is_rigged=True,
        show_joint_angles=False,
        z_up=False,
        post_fk_func=None,
        icon="\u0091",
        skeleton_radius=0.004,
        rb_radius=0.005,
        rb_length=0.025,
        **kwargs,
    ):
        """
        Initializer.
        :param flame_layer: The FLAMELayer that maps parameters to vertices and landmarks.
        :param poses_root: An optional array of shape (N, 3) for the global head orientation.
        :param poses_neck: An optional array of shape (N, 3) with neck pose parameters.
        :param poses_jaw: An optional array of shape (N, 3) with jaw pose parameters.
        :param poses_leye: An optional array of shape (N, 3) with left eye pose parameters.
        :param poses_reye: An optional array of shape (N, 3) with right eye pose parameters.
        :param expression: An optional array of shape (N, E) with expression coefficients.
        :param betas: An optional array of shape (S,) or (1, S) with shape parameters.
        :param trans: An optional array of shape (N, 3) with global translation.
        :param device: The pytorch device for computations.
        :param dtype: The pytorch data type.
        :param show_landmarks: Whether to display the landmarks as spheres.
        :param is_rigged: Whether to display the skeleton.
        :param show_joint_angles: Whether to display joint angle frames.
        :param z_up: Whether the input data uses Z-up convention.
        :param post_fk_func: Optional postprocessing function called after FK.
        :param skeleton_radius: Radius of skeleton bone spheres.
        :param rb_radius: Radius of joint angle sphere indicators.
        :param rb_length: Length of joint angle arrow indicators.
        :param kwargs: Remaining arguments for rendering.
        """
        n_frames = next(
            (
                a.shape[0]
                for a in (expression, poses_root, poses_neck, poses_jaw, poses_leye, poses_reye, trans)
                if a is not None
            ),
            1,
        )

        if device is None:
            device = C.device
        if dtype is None:
            dtype = C.f_precision

        super().__init__(n_frames=n_frames, icon=icon, gui_material=False, **kwargs)

        self.flame_layer = flame_layer
        self.post_fk_func = post_fk_func
        self.dtype = dtype
        self.device = device

        # Make sure that we have non-None values for global orient, translation and shape.
        poses_root = poses_root if poses_root is not None else torch.zeros([n_frames, 3])
        betas = betas if betas is not None else torch.zeros([n_frames, flame_layer.num_betas])
        trans = trans if trans is not None else torch.zeros([n_frames, 3])

        self.poses_root = to_torch(poses_root, dtype=dtype, device=device)
        self.poses_neck = to_torch(poses_neck, dtype=dtype, device=device)
        self.poses_jaw = to_torch(poses_jaw, dtype=dtype, device=device)
        self.poses_leye = to_torch(poses_leye, dtype=dtype, device=device)
        self.poses_reye = to_torch(poses_reye, dtype=dtype, device=device)
        self.expression = to_torch(expression, dtype=dtype, device=device)
        self.betas = to_torch(betas, dtype=dtype, device=device)
        self.trans = to_torch(trans, dtype=dtype, device=device)

        if len(self.betas.shape) == 1:
            self.betas = self.betas.unsqueeze(0)

        self._show_joint_angles = show_joint_angles
        self._is_rigged = is_rigged or show_joint_angles

        if z_up and not C.z_up:
            self.rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), self.rotation)

        self.vertices, self.joints, self.landmarks, self.faces, self.skeleton = self.fk()

        if self._is_rigged:
            self.skeleton_seq = Skeletons(
                self.joints,
                self.skeleton,
                radius=skeleton_radius,
                gui_affine=False,
                color=(1.0, 177 / 255, 1 / 255, 1.0),
                name="Skeleton",
            )
            self._add_node(self.skeleton_seq)

        global_oris = self._compute_global_oris()
        self.rbs = RigidBodies(
            self.joints,
            global_oris,
            radius=rb_radius,
            length=rb_length,
            gui_affine=False,
            name="Joint Angles",
        )
        self._add_node(self.rbs, enabled=self._show_joint_angles)

        self.mesh_seq = Meshes(
            self.vertices,
            self.faces,
            is_selectable=False,
            gui_affine=False,
            color=kwargs.get("color", (160 / 255, 160 / 255, 160 / 255, 1.0)),
            name="Mesh",
        )
        self._add_node(self.mesh_seq)

        self._show_landmarks = False
        if show_landmarks:
            self.landmark_seq = Spheres(
                c2c(self.landmarks),
                radius=0.002,
                color=(1.0, 177 / 255, 1 / 255, 1.0),
                gui_affine=False,
                name="Landmarks",
            )
            self.landmark_seq.enabled = False
            self._add_node(self.landmark_seq)
            self._show_landmarks = True

    def _get_full_local_poses(self, frame_id=None):
        """Local poses in kinematic order [root(3), neck(3), jaw(3), leye(3), reye(3)]."""
        s = slice(frame_id, frame_id + 1) if frame_id is not None else slice(None)
        n = 1 if frame_id is not None else self.n_frames

        def _get(attr):
            t = getattr(self, attr)
            if t is None:
                return torch.zeros([n, 3], dtype=self.dtype, device=self.device)
            return t[s]

        return torch.cat(
            [_get("poses_root"), _get("poses_neck"), _get("poses_jaw"), _get("poses_leye"), _get("poses_reye")],
            dim=-1,
        )

    def _compute_global_oris(self, frame_id=None):
        """Global joint orientations as a (N, J, 3, 3) numpy array."""
        local_poses = self._get_full_local_poses(frame_id)
        global_oris = local_to_global(local_poses, self.skeleton[:, 0], output_format="rotmat")
        n = local_poses.shape[0]
        return c2c(global_oris.reshape((n, -1, 3, 3)))

    def fk(self, current_frame_only=False):
        """Evaluate the FLAME model for the current frame or all frames."""
        s = slice(self.current_frame_id, self.current_frame_id + 1) if current_frame_only else slice(None)

        verts, joints, landmarks = self.flame_layer.fk(
            poses_root=self.poses_root[s],  # Always non-None
            poses_neck=self.poses_neck[s] if self.poses_neck is not None else None,
            poses_jaw=self.poses_jaw[s] if self.poses_jaw is not None else None,
            poses_leye=self.poses_leye[s] if self.poses_leye is not None else None,
            poses_reye=self.poses_reye[s] if self.poses_reye is not None else None,
            expression=self.expression[s] if self.expression is not None else None,
            betas=self.betas[s] if self.betas.shape[0] == self.n_frames else self.betas,  # Always non-None
            trans=self.trans[s],  # Always non-None
        )

        if self.post_fk_func:
            verts, joints = self.post_fk_func(self, verts, joints, current_frame_only)

        skeleton = self.flame_layer.skeletons().T
        faces = self.flame_layer.bm.faces.astype(np.int64)
        joints = joints[:, : skeleton.shape[0]]

        if current_frame_only:
            return c2c(verts)[0], c2c(joints)[0], c2c(landmarks)[0], c2c(faces), c2c(skeleton)
        else:
            return c2c(verts), c2c(joints), c2c(landmarks), c2c(faces), c2c(skeleton)

    def redraw(self, **kwargs):
        current_frame_only = kwargs.get("current_frame_only", False)
        vertices, joints, self.landmarks, self.faces, self.skeleton = self.fk(current_frame_only)

        if current_frame_only:
            f = self.current_frame_id
            self.vertices[f] = vertices
            self.joints[f] = joints

            if self._is_rigged:
                self.skeleton_seq.current_joint_positions = joints

            global_oris = self._compute_global_oris(f)
            self.rbs.current_rb_ori = global_oris[0]
            self.rbs.current_rb_pos = joints

            self.mesh_seq.current_vertices = vertices
        else:
            self.vertices = vertices
            self.joints = joints

            if self._is_rigged:
                self.skeleton_seq.joint_positions = joints

            global_oris = self._compute_global_oris()
            self.rbs.rb_ori = global_oris
            self.rbs.rb_pos = joints

            self.mesh_seq.vertices = vertices

        if self._show_landmarks:
            landmarks = c2c(self.landmarks)
            if current_frame_only:
                self.landmark_seq.current_sphere_positions = landmarks[0]
            else:
                self.landmark_seq.sphere_positions = landmarks

        super().redraw(**kwargs)

    @property
    def color(self):
        return self.mesh_seq.color

    @color.setter
    def color(self, color):
        self.mesh_seq.color = color

    @property
    def bounds(self):
        return self.mesh_seq.bounds

    @property
    def current_bounds(self):
        return self.mesh_seq.current_bounds

    @property
    def vertex_normals(self):
        return self.mesh_seq.vertex_normals

    def render_outline(self, *args, **kwargs):
        self.mesh_seq.render_outline(*args, **kwargs)
