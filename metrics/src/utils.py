# Authors: Ruslan Mammadov  <ruslanmammadov48@gmail.com>
# Copyright (C) 2021 Ruslan Mammadov and DynaGroup i.T. GmbH

"""
Utils required for metrics.
"""

import wget
import shutil
import os

from pathlib import Path


def download_and_extract_zip(link, out):
    """
    Download the file from link in out path.

    :param link: Link from where to download the file
    :param out: Path where to extract zip, including the desired file name

    I decided to implement it myself instead of using libraries to enable specifying name and
    getting better status information.
    """
    # Where to save initial zip file.
    temp_file = "cache/temp_file.zip"

    # If it exists, delete it.
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # Download the zip file
    wget.download(link, temp_file)

    # Specifying temporary directory where to extract the file
    parent_dir = Path(out).parent
    temp_dir = os.path.join(parent_dir, "dir_temp")
    # Delete if it exist
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    # Create the directory
    os.mkdir(temp_dir)

    # Unzip the file
    shutil.unpack_archive(temp_file, temp_dir)

    # Find the name of unzipped file
    file_name_temp = list(os.walk(temp_dir))[0][1][0]
    if os.path.exists(out):
        shutil.rmtree(out)

    # Rename and move to the destination according to out parameter
    os.rename(os.path.join(temp_dir, file_name_temp), out)

    # Delete cache files
    shutil.rmtree(temp_dir)
    os.remove(temp_file)

def remove_keys(dictionary, keys):
    """
    Removes keys from dictionary and returns updated version.

    :param dictionary: Dictionary which keys should be removed
    :param keys: Keys to remove
    :return: Updated dictionary
    """
    for key in keys:
        dictionary.pop(key)
    return dictionary

