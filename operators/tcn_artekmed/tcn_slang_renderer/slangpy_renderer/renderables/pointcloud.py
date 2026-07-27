"""
Pointcloud renderable data class.
"""
import slangpy as spy
import numpy as np
import trimesh
from PIL.Image import Image
import threading

from .base import Renderable


class Pointcloud(Renderable):
    """
    Pointcloud data representation for rendering.
    Supports dynamic updates from streaming data pipelines.
    """

    @staticmethod
    def from_ply(device: spy.Device, ply_path: str, image_path: str = None):
        """
        Load pointcloud from PLY file.

        Args:
            device: Slangpy device
            ply_path: Path to PLY file
            image_path: Optional path to texture image

        Returns:
            Pointcloud instance
        """
        pointcloud = trimesh.load_mesh(ply_path)
        positions = pointcloud.vertices.astype("float32")
        normals = None
        texcoords = None

        image_data = None
        if image_path is not None:
            image: Image = Image.open(image_path)
            image_shape = list(image.size) + [4]

            image_data = (
                np.frombuffer(image.tobytes(), dtype=np.uint8)
                .reshape(image_shape)
                .astype(np.float32)
                / 255
            )

        return Pointcloud(device,
                          positions=positions,
                          normals=normals,
                          texcoords=texcoords,
                          image=image_data,
                          sync_gpu=True)

    def __init__(self,
                 device: spy.Device,
                 positions: np.ndarray = None,
                 normals: np.ndarray = None,
                 texcoords: np.ndarray = None,
                 image: np.ndarray = None,
                 sync_gpu: bool = False):
        """
        Initialize pointcloud.

        Args:
            device: Slangpy device
            positions: Vertex positions (NxMx3 or Nx3)
            normals: Vertex normals (NxMx3 or Nx3)
            texcoords: Texture coordinates (NxMx2 or Nx2)
            image: Texture image (HxWxC)
            sync_gpu: If True, immediately sync to GPU
        """
        super().__init__(device)
        self.buffer_lock = threading.Lock()
        self.renderer = None  # Will be set by renderer
        self.vertices = positions  # Store for vertex count

        # Pending updates storage
        self._pending_data = {
            'positions': positions,
            'normals': normals,
            'texcoords': texcoords,
            'image': image,
        }

        self.position_buffer = None
        self.normal_buffer = None
        self.uv_buffer = None
        self.texture = None
        self.texture_upload_buffer = None

        self._is_dirty = True

        program = self.device.load_program("update_rgba32_texture.slang", ["upload_rgba32f"])
        self.update_rgba32_texture_kernel = self.device.create_compute_kernel(program)

        if sync_gpu:
            self.sync_gpu()

    @property
    def has_vertices(self):
        return self.position_buffer is not None

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
               normals: np.ndarray = None,
               texcoords: np.ndarray = None,
               image: np.ndarray = None):
        """
        Thread-safe: Call this from any thread to stage data for the next frame.

        Args:
            positions: Updated vertex positions
            normals: Updated vertex normals
            texcoords: Updated texture coordinates
            image: Updated texture image
        """
        with self.buffer_lock:
            if positions is not None:
                self._pending_data['positions'] = positions
            if normals is not None:
                self._pending_data['normals'] = normals
            if texcoords is not None:
                self._pending_data['texcoords'] = texcoords
            if image is not None:
                self._pending_data['image'] = image
            self._is_dirty = True

    def sync_gpu(self, command_encoder=None):
        """
        Call this once per frame from the main rendering thread
        before dispatching shaders.
        """
        from ..utils.cuda_helpers import copy_cupy_array_into_slangpy_buffer

        with self.buffer_lock:
            if self.is_dirty:
                # Re-use your existing logic but applied to the staged data
                if self._pending_data['positions'] is not None:
                    data = self._pending_data['positions']
                    self.vertices = data  # Update vertex count reference
                    if self.position_buffer is None or self.position_buffer.size != data.nbytes:
                        self.position_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource | spy.BufferUsage.shared,
                        )
                    copy_cupy_array_into_slangpy_buffer(data, self.position_buffer, data.shape)
                    self._pending_data['positions'] = None

                if self._pending_data['normals'] is not None:
                    data = self._pending_data['normals']
                    if self.normal_buffer is None or self.normal_buffer.size != data.nbytes:
                        self.normal_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource | spy.BufferUsage.shared,
                        )
                    copy_cupy_array_into_slangpy_buffer(data, self.normal_buffer, data.shape)
                    self._pending_data['normals'] = None

                if self._pending_data['texcoords'] is not None:
                    data = self._pending_data['texcoords']
                    if self.uv_buffer is None or self.uv_buffer.size != data.nbytes:
                        self.uv_buffer = self.device.create_buffer(
                            size=data.nbytes,
                            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource | spy.BufferUsage.shared,
                        )
                    copy_cupy_array_into_slangpy_buffer(data, self.uv_buffer, data.shape)
                    self._pending_data['texcoords'] = None

                if self._pending_data['image'] is not None:
                    image = self._pending_data['image']
                    if self.texture is None or self.texture.array_length != image.nbytes:
                        desc = spy.TextureDesc(dict(
                            width=image.shape[1],
                            height=image.shape[0],
                            format=spy.Format.rgba8_unorm,
                            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.shared,
                            type=spy.TextureType.texture_2d,
                            mip_count=1,
                            sample_count=1,
                        ))
                        self.texture = self.device.create_texture(desc)

                    # Copy via CPU — handle both cupy and numpy arrays
                    if hasattr(image, "get"):
                        # CuPy array — convert to numpy
                        host_image = image.get()
                    else:
                        host_image = np.asarray(image)
                    self.texture.copy_from_numpy(host_image)

                    self._pending_data['image'] = None

                self._is_dirty = False

    def render(self,
               pass_encoder: spy.RenderPassEncoder,
               window_size: tuple[int, int],
               view_matrix: np.ndarray,
               proj_matrix: np.ndarray,
               extra_args: dict = None):
        """
        Render this pointcloud using its associated renderer.

        Args:
            pass_encoder: Active render pass encoder
            window_size: Window dimensions (width, height)
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            extra_args: Additional rendering parameters
        """
        extra_args = extra_args or {}
        extra_args["depthWidth"] = self.vertices.shape[1]
        extra_args["depthHeight"] = self.vertices.shape[0]

        if self.renderer is not None:
            self.renderer.render(
                pass_encoder,
                self,
                window_size,
                view_matrix,
                proj_matrix,
                self.pose,
                extra_args
            )
