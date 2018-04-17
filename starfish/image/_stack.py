import os

import numpy
from slicedimage import Reader, Writer

from ._base import ImageBase


class ImageStack(ImageBase):
    def __init__(self, slicedimage):
        self._slicedimage = slicedimage
        self._num_hybs = slicedimage.get_dimension_shape('hyb')
        self._num_chs = slicedimage.get_dimension_shape('ch')
        self._tile_shape = tuple(slicedimage.default_tile_shape)

        self._data = numpy.zeros((self._num_hybs, self._num_chs) + self._tile_shape)

        for tile in slicedimage.get_matching_tiles():
            h = tile.indices['hyb']
            c = tile.indices['ch']
            self._data[h, c, :] = tile.numpy_array

        if len(self._tile_shape) == 2:
            self._is_volume = False
        else:
            self._is_volume = True

    @classmethod
    def from_image_stack(cls, image_stack_name_or_url, baseurl):
        slicedimage = Reader.parse_doc(image_stack_name_or_url, baseurl)

        return ImageStack(slicedimage)

    @property
    def numpy_array(self):
        return self._data

    @numpy_array.setter
    def numpy_array(self, data):
        for tile in self._slicedimage.get_matching_tiles():
            h = tile.indices['hyb']
            c = tile.indices['ch']
            tile.numpy_array = data[h, c, :]

    @property
    def shape(self):
        if self._data is None:
            return None
        else:
            return self._data.shape

    @property
    def num_hybs(self):
        return self._num_hybs

    @property
    def num_chs(self):
        return self._num_chs

    @property
    def tile_shape(self):
        return self._tile_shape

    @property
    def is_volume(self):
        return self._is_volume

    def clone_shape(self):
        return ImageStack(self._slicedimage.clone_shape())

    def write(self, filepath, tile_opener=None):
        seen_x_coords, seen_y_coords = set(), set()
        for tile in self._slicedimage.get_matching_tiles():
            seen_x_coords.add(tile.coordinates['x'])
            seen_y_coords.add(tile.coordinates['y'])

        sorted_x_coords = sorted(seen_x_coords)
        sorted_y_coords = sorted(seen_y_coords)
        x_coords_to_idx = {coords: idx for idx, coords in enumerate(sorted_x_coords)}
        y_coords_to_idx = {coords: idx for idx, coords in enumerate(sorted_y_coords)}
        # TODO: should handle Z.

        if tile_opener is None:
            def tile_opener(toc_path, tile, ext):
                tile_basename = os.path.splitext(toc_path)[0]
                xcoord = tile.coordinates['x']
                ycoord = tile.coordinates['y']
                xcoord = tuple(xcoord) if isinstance(xcoord, list) else xcoord
                ycoord = tuple(ycoord) if isinstance(ycoord, list) else ycoord
                xval = x_coords_to_idx[xcoord]
                yval = y_coords_to_idx[ycoord]
                return open(
                    "{}-X{}-Y{}-H{}-C{}.{}".format(
                        tile_basename,
                        xval,
                        yval,
                        tile.indices['hyb'],
                        tile.indices['ch'],
                        ext,
                    ),
                    "wb")

        Writer.write_to_path(
            self._slicedimage,
            filepath,
            pretty=True,
            tile_opener=tile_opener)

    def max_proj(self, dim):
        valid_dims = ['hyb', 'ch', 'z']
        if dim not in valid_dims:
            msg = "Dimension: {} not supported. Expecting one of: {}".format(dim, valid_dims)
            raise ValueError(msg)

        if dim == 'hyb':
            res = numpy.max(self._data, axis=0)
        elif dim == 'ch':
            res = numpy.max(self._data, axis=1)
        elif dim == 'z' and len(self._tile_shape) > 2:
            res = numpy.max(self._data, axis=4)
        else:
            res = self.data

        return res
