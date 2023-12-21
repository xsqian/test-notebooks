import os

os.environ['MLRUN_DBPATH'] = 'dbpath'
os.environ['V3IO_USERNAME'] = 'avia'
os.environ['V3IO_ACCESS_KEY'] = 'access_key'

import mlrun
from typing import Tuple
import mlrun.datastore.datastore
from mlrun import ArtifactType
from mlrun.datastore import DataItem
from mlrun.package import DefaultPackager
from PIL import Image
import io
from mlrun.artifacts import Artifact
import imagehash


class ImagePackager(DefaultPackager):
    """
    ``builtins.str`` packager.
    """

    PACKABLE_OBJECT_TYPE = str
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.FILE
    DEFAULT_UNPACKING_ARTIFACT_TYPE = ArtifactType.PATH

    @classmethod
    def pack_image(
            cls, key, obj: Image
    ) -> Tuple[Artifact, dict]:
        """
        Pack an Image as byte array.

        :param key:
        :param obj:            The image object to pack.
        :return: The packed image artifact and instructions.
        """

        # Proceed by path type (file or directory):
        instructions = {"image_format": "PNG"}
        obj.save("./a.png", "PNG")
        byte_array = cls.convert_image_to_bytes(obj)

        artifact = Artifact(key=key, src_path="./a.png")
        cls.add_future_clearing_path("./a.png")
        return artifact, instructions

    @classmethod
    def unpack_image(
            cls, data_item: DataItem
    ) -> str:
        """
        Unpack a data item representing a path string. If the path is of a file, the file is downloaded to a local
        temporary directory and its path is returned. If the path is of a directory, the archive is extracted and the
        directory path extracted is returned.

        :param data_item:      The data item to unpack.
        :return: The unpacked string.
        """

        # Mark the downloaded file for future clear:
        image = cls.convert_bytes_to_image(data_item.get())

        return image

    @staticmethod
    def convert_image_to_bytes(unpacked_image):
        # Convert the image to bytes

        img_byte_arr = io.BytesIO()
        unpacked_image.save(img_byte_arr, format=unpacked_image.format)
        img_byte_arr = img_byte_arr.getvalue()

        # Display the type of img_byte_arr to confirm conversion
        type(img_byte_arr)
        image_hash = imagehash.average_hash(unpacked_image)
        # return byte value
        # return self._TEMPLATE.format(
        #     self.metadata.description or self.metadata.key, self.metadata.key, data_uri
        # )
        return img_byte_arr

    @staticmethod
    def convert_bytes_to_image(byte_array):
        # Assuming byte_array is your byte array containing image data
        img_from_byte_arr = Image.open(io.BytesIO(byte_array))
        return img_from_byte_arr


flower_image_path = '/Users/Avi_Asulin/PycharmProjects/mlrun/test-notebooks/test_packager/flower.jpeg'
im = Image.open(flower_image_path)
pack = ImagePackager()
image_artifact, instructions = pack.pack_image('flower_key', im)
proj = mlrun.get_or_create_project('image-pack')
# ctx = mlrun.get_or_create_ctx(name='ctx')
art = proj.log_artifact(image_artifact)
di = art.to_dataitem()
im2 =pack.unpack_image(di)
im2.show()
print("he")
