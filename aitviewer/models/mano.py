# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import torch

from aitviewer.models.smpl import SMPLLayer
from aitviewer.utils.so3 import aa2rot_torch as aa2rot
from aitviewer.utils.so3 import rot2aa_torch as rot2aa


class MANOLayer(SMPLLayer):
    """A wrapper for the MANO hand model."""

    def __init__(
        self,
        is_rhand=True,
        use_pca=True,
        num_pca_comps=6,
        flat_hand_mean=False,
        device=None,
        dtype=None,
        **mano_model_params,
    ):
        """
        Initializer. Making some useful MANO params explicit.
        :param is_rhand: Whether to load the right hand model. If False, loads the left hand model.
        :param use_pca: Whether to use PCA components to represent the hand pose. If False, joint
          angles in axis-angle format are expected directly.
        :param num_pca_comps: Number of PCA components to use when use_pca is True.
        :param flat_hand_mean: If True, the mean hand pose is a flat hand. If False, the mean pose
          is the average hand pose from the MANO dataset.
        :param device: The pytorch device for computations. Defaults to the configured device.
        :param dtype: The pytorch data type. Defaults to the configured floating point precision.
        :param mano_model_params: Additional keyword arguments passed to smplx.create.
        """
        all_params = {
            "is_rhand": is_rhand,
            "use_pca": use_pca,
            "num_pca_comps": num_pca_comps,
            "flat_hand_mean": flat_hand_mean,
            **mano_model_params,
        }
        super(MANOLayer, self).__init__(model_type="mano", device=device, dtype=dtype, **all_params)

    def skeletons(self):
        """Return how the joints are connected in the kinematic chain where skeleton[0, i] is the parent of
        joint skeleton[1, i]."""
        parents = torch.stack(
            [
                self.bm.parents,
                torch.arange(0, len(self.bm.parents), device=self.bm.parents.device),
            ]
        )
        return {
            "all": parents,
            "body": parents[:, : self.bm.NUM_HAND_JOINTS + 1],
        }

    def fk(self, poses_hand, betas, poses_root=None, trans=None, normalize_root=False):
        """
        Convert mano pose data (joint angles and shape parameters) to positional data (joint and mesh vertex positions).
        :param poses_hand: A tensor of shape (N, N_JOINTS*3), i.e. joint angles in angle-axis format or PCA format (N, N_PCA_COMPONENTS).
        :param betas: A tensor of shape (N, N_BETAS) containing the betas/shape parameters.
        :param poses_root: Orientation of the root or None. If specified expected shape is (N, 3).
        :param trans: translation that is applied to vertices and joints or None, this is the 'transl' parameter
          of the MANO Model. If specified expected shape is (N, 3).
        :param normalize_root: If set, it will normalize the root such that its orientation is the identity in the
          first frame and its position starts at the origin.
        :return: The resulting vertices and joints.
        """

        batch_size = poses_hand.shape[0]
        device = poses_hand.device

        if poses_root is None:
            poses_root = torch.zeros([batch_size, 3]).to(dtype=poses_hand.dtype, device=device)
        if trans is None:
            trans = torch.zeros([batch_size, 3]).to(dtype=poses_hand.dtype, device=device)

        # Batch shapes if they don't match batch dimension.
        if len(betas.shape) == 1 or betas.shape[0] == 1:
            betas = betas.repeat(poses_hand.shape[0], 1)
        betas = betas[:, : self.num_betas]

        if normalize_root:
            # Make everything relative to the first root orientation.
            root_ori = aa2rot(poses_root)
            first_root_ori = torch.inverse(root_ori[0:1])
            root_ori = torch.matmul(first_root_ori, root_ori)
            poses_root = rot2aa(root_ori)
            trans = torch.matmul(first_root_ori.unsqueeze(0), trans.unsqueeze(-1)).squeeze()
            trans = trans - trans[0:1]

        output = self.bm(
            hand_pose=poses_hand,
            betas=betas,
            global_orient=poses_root,
            transl=trans,
        )

        return output.vertices, output.joints

    def forward(self, *args, **kwargs):
        # This is a convenience function so that external code can use MANO layers
        # with the `poses_hand` parameter instead of the otherwise somewhat confusing
        # `poses_body`` parameter. However, internally `MANOSequence` reuses everything from
        # `SMPLSequence` and just pretends `poses_hand` are `poses_body`. So we catch this
        # case here.
        if "poses_body" in kwargs:
            kwargs["poses_hand"] = kwargs.pop("poses_body")
        # Drop body-model kwargs that MANO doesn't accept.
        for key in ["poses_left_hand", "poses_right_hand", "poses_jaw", "poses_leye", "poses_reye", "expression"]:
            kwargs.pop(key, None)
        return self.fk(*args, **kwargs)
