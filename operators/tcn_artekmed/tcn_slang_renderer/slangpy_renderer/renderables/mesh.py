"""
Mesh renderable data class.
"""
import slangpy as spy
import numpy as np
import trimesh
from PIL.Image import Image
import threading

from .base import Renderable


class Mesh(Renderable):
    """
    Textured mesh renderable with vertices, normals, and texture coordinates.
    """

    @staticmethod
    def from_obj(device: spy.Device, mesh_path: str):
        """
        Load mesh from OBJ file.

        Args:
            device: Slangpy device
            mesh_path: Path to OBJ file

        Returns:
            Mesh instance
        """
        mesh = trimesh.load_mesh(mesh_path)
        positions = mesh.vertices.astype("float32")
        normals = mesh.vertex_normals.astype("float32")
        texcoords = mesh.visual.uv.astype("float32")
        indices = mesh.faces.astype("uint16")

        image: Image = mesh.visual.material.image
        image_shape = list(image.size) + [1]

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
                 normals: np.ndarray = None,
                 texcoords: np.ndarray = None,
                 image: np.ndarray = None,
                 sync_gpu: bool = True):
        """
        Initialize mesh.

        Args:
            device: Slangpy device
            positions: Vertex positions (Nx3)
            indices: Triangle indices (Mx3)
            normals: Vertex normals (Nx3)
            texcoords: Texture coordinates (Nx2)
            image: Texture image (HxWxC)
            sync_gpu: If True, immediately create GPU buffers
        """
        super().__init__(device)
        self.buffer_lock = threading.Lock()
        self.renderer = None  # Will be set by MeshRenderer

        # Pending updates storage
        self._pending_data = {
            'positions': positions if not sync_gpu else None,
            'indices': indices if not sync_gpu else None,
            'normals': normals,
            'texcoords': texcoords,
            'image': image,
        }

        # derive from indices.dtype?
        self.index_format = spy.IndexFormat.uint16
        self.vertex_count = positions.size if positions is not None else 0

        self.position_buffer = None
        self.index_buffer = None
        self.normal_buffer = None
        self.uv_buffer = None
        self.texture = None

        self._is_dirty = False

        if sync_gpu:
            # Initialize buffers immediately
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

            if texcoords is not None:
                self.uv_buffer = device.create_buffer(
                    size=texcoords.nbytes,
                    usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                    data=texcoords,
                )

            if image is not None:
                loader = spy.TextureLoader(device)
                self.texture = loader.load_texture(spy.Bitmap(image))

    @property
    def has_normals(self):
        return self.normal_buffer is not None

    @property
    def has_texcoords(self):
        return self.uv_buffer is not None

    @property
    def has_texture(self):
        return self.texture is not None

    @property
    def is_dirty(self):
        return self._is_dirty

    def update(self, positions: np.ndarray = None,
               indices: np.ndarray = None,
               normals: np.ndarray = None,
               texcoords: np.ndarray = None,
               image: np.ndarray = None):
        """
        Thread-safe: Call this from any thread to stage data for the next frame.

        Args:
            positions: Updated vertex positions
            indices: Updated triangle indices
            normals: Updated vertex normals
            texcoords: Updated texture coordinates
            image: Updated texture image
        """
        with self.buffer_lock:
            if positions is not None:
                self._pending_data['positions'] = positions
            if indices is not None:
                self._pending_data['indices'] = indices
            if normals is not None:
                self._pending_data['normals'] = normals
            if texcoords is not None:
                self._pending_data['texcoords'] = texcoords
            if image is not None:
                self._pending_data['image'] = image
            self._is_dirty = True

    def sync_gpu(self):
        """
        Call this once per frame from the main rendering thread
        before dispatching shaders.
        """
        with self.buffer_lock:
            if self.is_dirty:
                if self._pending_data['positions'] is not None:
                    data = self._pending_data['positions']
                    if self.position_buffer is not None and self.position_buffer.size == data.nbytes:
                        self.position_buffer.copy_from_numpy(data)
                    else:
                        self.position_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                            data=data
                        )
                    self._pending_data['positions'] = None

                if self._pending_data['indices'] is not None:
                    data = self._pending_data['indices']
                    self.vertex_count = data.size
                    if self.index_buffer is not None and self.index_buffer.size == data.nbytes:
                        self.index_buffer.copy_from_numpy(data)
                    else:
                        self.index_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
                            data=data
                        )
                    self._pending_data['indices'] = None

                if self._pending_data['normals'] is not None:
                    data = self._pending_data['normals']
                    if self.normal_buffer is not None and self.normal_buffer.size == data.nbytes:
                        self.normal_buffer.copy_from_numpy(data)
                    else:
                        self.normal_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                            data=data
                        )
                    self._pending_data['normals'] = None

                if self._pending_data['texcoords'] is not None:
                    data = self._pending_data['texcoords']
                    if self.uv_buffer is not None and self.uv_buffer.size == data.nbytes:
                        self.uv_buffer.copy_from_numpy(data)
                    else:
                        self.uv_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                            data=data
                        )
                    self._pending_data['texcoords'] = None

                if self._pending_data['image'] is not None:
                    loader = spy.TextureLoader(self.device)
                    self.texture = loader.load_texture(spy.Bitmap(self._pending_data['image']))
                    self._pending_data['image'] = None

                self._is_dirty = False

    def render(self,
               pass_encoder: spy.RenderPassEncoder,
               window_size: tuple[int, int],
               view_matrix: np.ndarray,
               proj_matrix: np.ndarray,
               extra_args: dict = None,
               ):
        """
        Render this mesh using its associated renderer.

        Args:
            pass_encoder: Active render pass encoder
            window_size: Window dimensions (width, height)
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            extra_args: Additional rendering parameters
        """
        if self.renderer is not None:
            self.renderer.render(
                pass_encoder,
                self,
                window_size,
                view_matrix,
                proj_matrix,
                self.pose,
                extra_args,
            )
