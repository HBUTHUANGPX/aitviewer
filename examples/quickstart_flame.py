# Copyright (C) 2022-2026  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np
import torch

from aitviewer.models.flame import FLAMELayer
from aitviewer.renderables.flame import FLAMESequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # For this to work you must download the FLAME model and landmark files.
    # Download the FLAME model from https://flame.is.tue.mpg.de (this was tested
    # with the open FLAME model) and place it into C.smplx_models/flame/FLAME_NEUTRAL.pkl.
    # Make sure to rename it to FLAME_NEUTRAL.pkl
    #
    # Then we also need landmark files, which we downloaded from the RingNet repo here:
    # https://github.com/soubhiksanyal/RingNet/tree/master/flame_model
    # Download the files "flame_static_embedding.pkl" and "flame_dynamic_embedding.npy"
    # and place them into C.smplx_models/flame keeping that name.

    # Quickest possible way to just get a face.
    # FLAMELayer is a thin wrapper around the FLAME layer in the smplx package.
    flame_layer = FLAMELayer()

    # The FLAMESequence evaluates the FLAME model to display the results.
    # It is similar in spirit to SMPLSequence but because the two models are
    # sufficiently different, they are independent classes.
    flame_seq1 = FLAMESequence(flame_layer, position=np.array([-0.25, 0.0, 0.0]))

    # A bit fancier, showing face contour landmarks and animating expressions.
    flame_layer2 = FLAMELayer(use_face_contour=True)

    expression = torch.zeros(300, flame_layer.bm.num_expression_coeffs)
    expression[:, 0] = torch.linspace(-2.0, 2.0, 300)

    flame_seq2 = FLAMESequence(
        flame_layer2, expression=expression, show_landmarks=True, position=np.array([0.25, 0.0, 0.0])
    )

    v = Viewer()
    v.scene.add(flame_seq1, flame_seq2)
    v.scene.origin.enabled = False
    v.run()
