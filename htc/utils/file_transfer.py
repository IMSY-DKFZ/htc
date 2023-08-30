# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import subprocess
from pathlib import Path

from htc.settings import settings


def upload_file_s3(local_path: Path, remote_path: str) -> str:
    """
    Uploads a local file to our S3 storage and makes it publicly available.

    In order to use this function, make sure you have access to our vault (e130-hyperspectal-tissue-classification) and that you installed and configured the Minio client correctly (based on: https://agcompute.sci.dkfz-heidelberg.de/s3-cos-documentation/quickstart/quickstart_cli/):
    ```bash
    # Download the Minio client to /usr/local/bin so that it will be available in your PATH
    sudo wget -O /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc

    # Make it executable
    sudo chmod +x /usr/local/bin/mc

    # Add an alias for our S3 storage (this will ask you for your S3 access and secret key)
    mc alias set dkfz-s3 https://s3.dkfz.de

    # Make sure that the generated configuration is properly set
    cat ~/.mc/config.json
    ```

    Args:
        local_path: Path to the local file which should be uploaded.
        remote_path: Relative path in our vault.

    Returns: The public URL which can be used to download the file.
    """
    # Delete the file first to avoid quota issues
    res = subprocess.run(
        f"mc rm dkfz-s3/e130-hyperspectal-tissue-classification/{remote_path}",
        shell=True,
        capture_output=True,
        text=True,
    )

    # It is ok if the object does not exist (we only report other errors)
    if res.returncode != 0 and "Object does not exist" not in res.stderr:
        settings.log.warning(
            "Something went wrong while deleting any existing file in the vault first, upload may not"
            f" work:\n{res.stderr = }\n{res.stdout = }\n{res.returncode=})"
        )

    remote_location = f"dkfz-s3/e130-hyperspectal-tissue-classification/{remote_path}"
    res = subprocess.run(f"mc cp --attr 'x-amz-acl=public-read' {local_path} {remote_location}", shell=True)
    assert res.returncode == 0, f"Could not upload the file {local_path} to the DKFZ S3 storage"

    return f"https://e130-hyperspectal-tissue-classification.s3.dkfz.de/{remote_path}"


def upload_content_s3(content: str, remote_path: str) -> str:
    """
    Uploads the given text to a remote file on our S3 storage.

    The setup is similar to upload_file_s3().

    Args:
        content: The content of the remote file.
        remote_path: Relative path in our vault.

    Returns: The public URL which can be used to download the file.
    """
    remote_location = f"dkfz-s3/e130-hyperspectal-tissue-classification/{remote_path}"
    res = subprocess.run(
        f"mc pipe --attr 'x-amz-acl=public-read' {remote_location}", shell=True, input=content, text=True
    )
    assert res.returncode == 0, f"Could not upload the content to the remote file {remote_path} on the DKFZ S3 storage"

    return f"https://e130-hyperspectal-tissue-classification.s3.dkfz.de/{remote_path}"
