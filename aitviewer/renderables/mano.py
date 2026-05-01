# Copyright (C) 2022-2026  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
from typing import IO, Union

import numpy as np
import torch

from aitviewer.models.mano import MANOLayer
from aitviewer.renderables.smpl import SMPLSequence


class MANOSequence(SMPLSequence):
    """
    Represents a temporal sequence of MANO poses.
    """

    def __init__(
        self,
        poses_hand,
        mano_layer,
        poses_root=None,
        betas=None,
        trans=None,
        device=None,
        dtype=None,
        include_root=True,
        normalize_root=False,
        is_rigged=True,
        show_joint_angles=False,
        z_up=False,
        post_fk_func=None,
        icon="\u0092",
        skeleton_radius=0.003,
        rb_radius=0.005,
        rb_length=0.025,
        **kwargs,
    ):
        """
        Initializer.
        :param poses_hand: An array (numpy ar pytorch) of shape (F, N_JOINTS*3) containing the pose parameters of the
          hand.
        :param mano_layer: The MANO layer that maps parameters to joint positions and/or dense surfaces.
        :param poses_root: An array (numpy or pytorch) of shape (F, 3) containing the global root orientation.
        :param betas: An array (numpy or pytorch) of shape (N_BETAS, ) containing the shape parameters.
        :param trans: An array (numpy or pytorch) of shape (F, 3) containing a global translation that is applied to
          all joints and vertices.
        :param device: The pytorch device for computations.
        :param dtype: The pytorch data type.
        :param include_root: Whether or not to include root information. If False, no root translation and no root
          rotation is applied.
        :param normalize_root: Whether or not to normalize the root. If True, the global root translation in the first
          frame is zero and the global root orientation is the identity.
        :param is_rigged: Whether or not to display the joints as a skeleton.
        :param show_joint_angles: Whether or not the coordinate frames at the joints should be visualized.
        :param z_up: Whether or not the input data assumes Z is up. If so, the data will be rotated such that Y is up.
        :param post_fk_func: User specified postprocessing function that is called after evaluating the SMPL model,
          the function signature must be: def post_fk_func(self, vertices, joints, current_frame_only),
          and it must return new values for vertices and joints with the same shapes.
          Shapes are:
            if current_frame_only is False: vertices (F, V, 3) and joints (F, N_JOINTS, 3)
            if current_frame_only is True:  vertices (1, V, 3) and joints (1, N_JOINTS, 3)
        :skeleton_radius: Size indication for spheres used to represent the skeleton.
        :rb_radius: Size indication for sphere of rigid bodies used to represent the joints.
        :rb_length: Size indication for arrow length of rigid bodies used to represent the joints.
        :param kwargs: Remaining arguments for rendering.
        """
        # We treat MANO hand poses as "body" poses so that we can reuse the SMPLSequence renderable.
        assert len(poses_hand.shape) == 2
        self.mano_layer = mano_layer

        super(MANOSequence, self).__init__(
            poses_body=poses_hand,
            smpl_layer=mano_layer,
            poses_root=poses_root,
            betas=betas,
            trans=trans,
            device=device,
            dtype=dtype,
            include_root=include_root,
            normalize_root=normalize_root,
            is_rigged=is_rigged,
            show_joint_angles=show_joint_angles,
            z_up=z_up,
            post_fk_func=post_fk_func,
            icon=icon,
            skeleton_radius=skeleton_radius,
            rb_radius=rb_radius,
            rb_length=rb_length,
            **kwargs,
        )

    @classmethod
    def t_pose(cls, mano_layer=None, betas=None, frames=1, **kwargs):
        """Creates a SMPL sequence whose single frame is a SMPL mesh in T-Pose."""

        if mano_layer is None:
            mano_layer = MANOLayer()

        hand_pose_dim = mano_layer.bm.num_pca_comps if mano_layer.bm.use_pca else 3 * mano_layer.bm.NUM_HAND_JOINTS
        poses = np.zeros([frames, hand_pose_dim])
        return cls(poses, mano_layer, betas=betas, **kwargs)

    @classmethod
    def from_npz(cls, file: Union[IO, str], mano_layer: MANOLayer = None, **kwargs):
        raise NotImplementedError()

    def export_to_npz(self, file: Union[IO, str]):
        raise NotImplementedError()

    def to_joint_angles(self, poses):
        """
        Makes sure poses are in joint angles for visualization purposes. This is not required by
        the base class, but might be required by subclasses when they represent poses not as
        joint angles but for instance as PCA components.
        :param poses: A torch tensor of shape (N, D) or (D, ) where D is the joint angle dimension, e.g. N_JOINTS * 3 for SMPL body poses or N_PCA_COMP for MANO hand poses.
        """
        if self.mano_layer.bm.use_pca:
            if len(poses.shape) == 1:
                p = poses.unsqueeze(0)
            else:
                p = poses
            result = torch.einsum("bi,ij->bj", [p, self.mano_layer.bm.hand_components])
            if len(poses.shape) == 1:
                result = result.squeeze(0)
            return result
        else:
            return poses

    def to_joint_angles_inv(self, poses):
        """
        Does the inverse of `to_joint_angles`.
        :param poses: A torch tensor of shape (N, N_JOINTS*3) or (N_JOINTS*3, ).
        """
        if self.mano_layer.bm.use_pca:
            if len(poses.shape) == 1:
                p = poses.unsqueeze(0)
            else:
                p = poses
            result = torch.einsum("bi,ji->bj", [p, self.mano_layer.bm.hand_components])
            if len(poses.shape) == 1:
                result = result.squeeze(0)
            return result
        else:
            return poses

    @property
    def poses(self):
        # This is used for editing, which only works with joint angles, so we make
        # sure it returns joint angles.
        pb = self.to_joint_angles(self.poses_body)
        return torch.cat((self.poses_root, pb), dim=-1)
