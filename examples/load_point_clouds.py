# Copyright (C) 2022-2026  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import trimesh

from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    v = Viewer()
    sphere = trimesh.load("resources/planet/planet.obj")
    point_cloud = PointClouds(sphere.vertices[None])
    v.scene.add(point_cloud)
    v.run()
