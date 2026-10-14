# -*- coding: utf-8 -*-

import torch.nn as nn
from typing import Tuple, List, Optional


class Point2MeshOutput(object):
    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None
        self.center = None
        self.pc = None


class Latent2MeshOutput(object):

    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None


class AlignedMeshOutput(object):

    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None
        self.surface = None
        self.image = None
        self.text: Optional[str] = None
        self.shape_text_similarity: Optional[float] = None
        self.shape_image_similarity: Optional[float] = None


# Trivexa: ShapeAsLatentPLModule (a Lightning training wrapper) removed.


class ShapeAsLatentModule(nn.Module):
    latent_shape: Tuple[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def query_geometry(self, *args, **kwargs):
        raise NotImplementedError


# Trivexa: AlignedShapeAsLatentPLModule (a Lightning training wrapper) removed.


class AlignedShapeAsLatentModule(nn.Module):
    shape_model: ShapeAsLatentModule
    latent_shape: Tuple[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_shape_model_only(self):
        raise NotImplementedError

    def encode_image_embed(self, *args, **kwargs):
        raise NotImplementedError

    def encode_text_embed(self, *args, **kwargs):
        raise NotImplementedError

    def encode_shape_embed(self, *args, **kwargs):
        raise NotImplementedError


class TexturedShapeAsLatentModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def query_geometry(self, *args, **kwargs):
        raise NotImplementedError

    def query_color(self, *args, **kwargs):
        raise NotImplementedError
