# Copyright (C) 2022-2026  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np

from aitviewer.models.mano import MANOLayer
from aitviewer.renderables.mano import MANOSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    v = Viewer()

    # Quickly create a a default hand, which is the right hand, using PCA.
    # Note that MANO hands are editable like SMPL bodies, but when using PCA
    # the editing is funky because the PCA conversion is lossy.
    mano_seq_rh = MANOSequence.reference_pose()
    v.scene.add(mano_seq_rh)

    # Create a left hand using joint angles.
    mano_layer = MANOLayer(is_rhand=False, use_pca=False)
    mano_pose = np.zeros((1, 15 * 3))
    mano_seq_lh = MANOSequence(poses_hand=mano_pose, mano_layer=mano_layer, position=np.array([0.5, 0.0, 0.0]))
    v.scene.add(mano_seq_lh)

    v.run()
