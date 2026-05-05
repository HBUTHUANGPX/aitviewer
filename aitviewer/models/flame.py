# Copyright (C) 2022-2026  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import torch

from aitviewer.models.smpl import SMPLLayer


class FLAMELayer(SMPLLayer):
    """A wrapper for the FLAME face model using the smplx package as backend."""

    def __init__(self, num_expression_coeffs=10, device=None, dtype=None, **flame_model_args):
        """
        Initializer.
        :param num_expression_coeffs: Number of expression coefficients.
        :param device: The pytorch device for computations. Defaults to the configured device.
        :param dtype: The pytorch data type. Defaults to the configured floating point precision.
        :param flame_model_args: Additional keyword arguments passed to smplx.create.
        """
        super(FLAMELayer, self).__init__(
            model_type="flame",
            gender="neutral",
            device=device,
            dtype=dtype,
            num_expression_coeffs=num_expression_coeffs,
            **flame_model_args,
        )

    def skeletons(self):
        """Return how the joints are connected in the kinematic chain where skeleton[0, i] is the parent of
        joint skeleton[1, i]."""
        parents = torch.stack(
            [
                self.bm.parents,
                torch.arange(0, len(self.bm.parents), device=self.bm.parents.device),
            ]
        )
        return parents

    def fk(
        self,
        poses_root=None,
        poses_neck=None,
        poses_jaw=None,
        poses_leye=None,
        poses_reye=None,
        expression=None,
        betas=None,
        trans=None,
        **kwargs,
    ):
        """
        Evaluate the FLAME model given shape, expression and pose parameters.
        :param poses_root: A tensor of shape (N, 3) for the global head orientation, or None.
        :param poses_neck: A tensor of shape (N, 3) for the neck pose, or None.
        :param poses_jaw: A tensor of shape (N, 3) for the jaw pose, or None.
        :param poses_leye: A tensor of shape (N, 3) for the left eye pose, or None.
        :param poses_reye: A tensor of shape (N, 3) for the right eye pose, or None.
        :param expression: A tensor of shape (N, E) for the expression coefficients, or None.
        :param betas: A tensor of shape (N, S) containing the shape parameters.
        :param trans: A tensor of shape (N, 3) for the global translation, or None.
        :return: A tuple (vertices, joints, landmarks).
        """
        batch_size = expression.shape[0] if expression is not None else betas.shape[0]
        device = betas.device
        dtype = betas.dtype

        if poses_root is None:
            poses_root = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if poses_neck is None:
            poses_neck = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if poses_jaw is None:
            poses_jaw = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if poses_leye is None:
            poses_leye = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if poses_reye is None:
            poses_reye = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if expression is None:
            expression = torch.zeros([batch_size, self.bm.num_expression_coeffs], dtype=dtype, device=device)
        if trans is None:
            trans = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        if len(betas.shape) == 1 or betas.shape[0] == 1:
            betas = betas.expand(batch_size, -1).contiguous()
        betas = betas[:, : self.num_betas]

        output = self.bm(
            global_orient=poses_root,
            betas=betas,
            expression=expression,
            jaw_pose=poses_jaw,
            leye_pose=poses_leye,
            reye_pose=poses_reye,
            neck_pose=poses_neck,
            transl=trans,
        )

        joints = output.joints[:, : self.bm.NUM_JOINTS]
        landmarks = output.joints[:, self.bm.NUM_JOINTS :]

        return output.vertices, joints, landmarks
