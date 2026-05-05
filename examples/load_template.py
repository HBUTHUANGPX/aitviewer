# Copyright (C) 2022-2026  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.models.flame import FLAMELayer
from aitviewer.models.mano import MANOLayer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.flame import FLAMESequence
from aitviewer.renderables.mano import MANOSequence
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    smplh_template = SMPLSequence.reference_pose(
        SMPLLayer(model_type="smplh", gender="neutral", device=C.device), name="SMPL-H"
    )
    mano_template = MANOSequence.reference_pose(
        MANOLayer(is_rhand=False, device=C.device),
        position=np.array((-1.0, 0.0, 0.0)),
        name="MANO",
    )
    flame_template = FLAMESequence(
        FLAMELayer(),
        position=np.array((1.0, 0.0, 0.0)),
        name="FLAME",
    )

    # Display in viewer.
    v = Viewer()
    v.scene.add(smplh_template, mano_template, flame_template)
    v.run()
