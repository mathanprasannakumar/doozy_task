import trimesh 

mesh = trimesh.load("../blender/bolt_nut.stl")

mesh.export("bolt_nut.ply")

mesh = trimesh.load("../blender/hex_bolt_30.stl")

mesh.export("hex_bolt_30.ply")