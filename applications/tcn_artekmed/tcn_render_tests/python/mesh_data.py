import slangpy as spy
import numpy as np
import trimesh
from PIL.Image import Image

class Mesh:

    @staticmethod
    def from_obj(device: spy.Device, mesh_path: str):
        mesh = trimesh.load_mesh(mesh_path)
        positions = mesh.vertices.astype("float32")
        normals = mesh.vertex_normals.astype("float32")
        texcoords = mesh.visual.uv.astype("float32")  # type: ignore
        indices = mesh.faces.astype("uint16")

        image : Image = mesh.visual.material.image  # type: ignore
        image_shape = list(image.size) + [4]

        image_data = (
            np.frombuffer(image.tobytes(), dtype=np.uint8)
            .reshape(image_shape)
            .astype(np.float32)
            / 255
        )

        return Mesh(device, positions, indices, normals=normals, texcoords=texcoords, image=image_data)

    def __init__(self,
                 device: spy.Device,
                 positions: np.ndarray,
                 indices: np.ndarray,
                 normals: np.ndarray=None,
                 texcoords: np.ndarray=None,
                 image: np.ndarray=None):

        # derive from indices.dtype?
        self.index_format = spy.IndexFormat.uint16
        self.vertex_count = indices.size if indices is not None else 0

        self.position_buffer = device.create_buffer(
            size=positions.nbytes,
            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
            data=positions,
        )

        self.index_buffer = device.create_buffer(
            size=indices.nbytes,
            usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
            data=indices,
        )

        if normals is not None:
            self.normal_buffer = device.create_buffer(
                size=normals.nbytes,
                usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                data=normals,
            )
        else:
            self.normal_buffer = None

        if texcoords is not None:
            self.uv_buffer = device.create_buffer(
                size=texcoords.nbytes,
                usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                data=texcoords,
            )
        else:
            self.uv_buffer = None



        if image is not None:
            loader = spy.TextureLoader(device)
            self.texture = loader.load_texture(spy.Bitmap(image))
        else:
            self.texture = None

    @property
    def has_normals(self):
        return self.normal_buffer is not None

    @property
    def has_texcoords(self):
        return self.uv_buffer is not None

    @property
    def has_texture(self):
        return self.texture is not None