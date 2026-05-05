# Copyright (C) 2022-2026  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    v = Viewer()
    smpl_layer = SMPLLayer(model_type="smplx", gender="neutral")
    v.scene.add(SMPLSequence.reference_pose(smpl_layer))
    v.run()
