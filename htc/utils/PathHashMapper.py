# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import hashlib

from htc.tivita.DataPath import DataPath


class PathHashMapper:
    def __init__(self, paths: list[DataPath] = None, global_salt: str = "kd3Fqim1QajPPlnCXyXS"):
        """
        This class can be used to hash image names and convert them back to the original data path. Use case is for inter and intra rater results so that the raters don't know which image they are currently annotating. SHA256 will be used as hash function.

        >>> paths = [DataPath.from_image_name("P058#2020_05_13_18_09_26")]
        >>> m = PathHashMapper(paths)
        >>> h = m.path_to_hash(paths[0])
        >>> h
        'bd5ffba9672d9ed1904c0ba8975808cd09d0df57e3e8f28326d077cebb991cb7'
        >>> m.hash_to_path(h).image_name()
        'P058#2020_05_13_18_09_26'

        Args:
            paths: List of data paths where you have a hash value and want to convert it back. This is necessary to convert from a hash back to a data path as this can only be done based on a list of known hashes/paths.
            global_salt: Salt which is appended to the image name. Each salt will create a new set of unique hashes for your images making it harder to find out the original image. You must use the same salt for hashing and unhashing.
        """
        self.global_salt = global_salt
        self.path_hashes = {self.path_to_hash(p): p for p in paths} if paths is not None else {}

    def path_to_hash(self, path: DataPath) -> str:
        """
        Converts the data path to a unique hash.

        Args:
            path: Data path object.

        Returns: Hash value as hexdigest.
        """
        name = path.image_name() + self.global_salt
        return hashlib.sha256(name.encode()).hexdigest()

    def hash_to_path(self, sha256_hash: str) -> DataPath:
        """
        Converts the hash back to the data path. The data path must be part of the list provided to this class.

        Args:
            sha256_hash: The hash value as hexdigest.

        Returns: Data path object.
        """
        assert len(sha256_hash) == 64, (
            f"The hash {sha256_hash} is not a valid sha256 hash (it should be 64 characters long)"
        )
        assert self.path_hashes is not None, "No paths available for the reverse mapping"
        assert sha256_hash in self.path_hashes, (
            f"No path mapping for the hash {sha256_hash} found. Does the hash belong to a path provided in the paths"
            " list?"
        )

        return self.path_hashes[sha256_hash]
