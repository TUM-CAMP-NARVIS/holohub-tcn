import slangpy as spy
import numpy as np
import trimesh
from PIL.Image import Image
import threading

from renderable import Renderable

class Pointcloud(Renderable):
    """
    Pointcloud data representation for rendering.
    """

    @staticmethod
    def from_ply(device: spy.Device, ply_path: str, image_path: str=None):
        pointcloud = trimesh.load_mesh(ply_path)
        positions = pointcloud.vertices.astype("float32")
        normals = None
        texcoords = None
        # normals = pointcloud.vertex_normals.astype("float32")
        # texcoords = pointcloud.visual.uv.astype("float32")  # type: ignore

        image_data = None
        if image_path is not None:
            image : Image = Image.open(image_path)  # type: ignore
            image_shape = list(image.size) + [4]

            image_data = (
                np.frombuffer(image.tobytes(), dtype=np.uint8)
                .reshape(image_shape)
                .astype(np.float32)
                / 255
            )

        return Pointcloud(device, positions,
                          normals=normals,
                          texcoords=texcoords,
                          image=image_data,
                          sync_gpu=True)

    def __init__(self,
                 device: spy.Device,
                 positions: np.ndarray,
                 normals: np.ndarray=None,
                 texcoords: np.ndarray=None,
                 image: np.ndarray=None,
                 sync_gpu: bool=False):

        super().__init__(device)
        self.buffer_lock = threading.Lock()
        self.renderer = None  # Will be set by PointcloudRenderer
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

        self._is_dirty = False

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


    def update(self, positions: np.ndarray=None,
               normals: np.ndarray=None,
               texcoords: np.ndarray=None,
               image: np.ndarray=None):
        """
        Thread-safe: Call this from any thread to stage data for the next frame.
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


    def sync_gpu(self):
        """
        Call this once per frame from the main rendering thread
        before dispatching shaders.
        """
        with self.buffer_lock:
            # Re-use your existing logic but applied to the staged data
            if self._pending_data['positions'] is not None:
                data = self._pending_data['positions']
                self.vertices = data  # Update vertex count reference
                if self.position_buffer is not None and self.position_buffer.size == data.nbytes:
                    self.position_buffer.copy_from_numpy(data)
                else:
                    self.position_buffer = self.device.create_buffer(
                        size=data.nbytes,
                        usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                        data=data
                    )
                self._pending_data['positions'] = None

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

    def render(self, command_encoder: spy.CommandEncoder,
               window_size: tuple[int, int],
               output_texture: spy.Texture,
               depth_texture: spy.Texture,
               view_matrix: np.ndarray,
               proj_matrix: np.ndarray,
               camera_pos: list = None,
               clear_color: list = None):
        """
        Render this pointcloud using its associated renderer.
        """
        if self.is_dirty:
            self.sync_gpu()

        if self.renderer is not None:
            self.renderer.render(
                command_encoder,
                self,
                window_size,
                output_texture,
                depth_texture,
                view_matrix,
                proj_matrix,
                self.pose,
                camera_pos,
                clear_color
            )
