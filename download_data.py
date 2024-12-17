"""Fetcher for RAMP data stored in OSF

To adapt it for another challenge, change the CHALLENGE_NAME and upload
public/private data as `tar.gz` archives in dedicated OSF folders named after
the challenge.
"""
import tarfile
import argparse
from zlib import adler32
from pathlib import Path
from osfclient.api import OSF
from osfclient.exceptions import UnauthorizedException

PUBLIC_PROJECT = "t4uf8"

CHALLENGE_NAME = 'map_estimation'
RAMP_FOLDER_CONFIGURATION = {
    'public': dict(
        code='t4uf8',
        files=['public.tar.gz.001', 'public.tar.gz.002', 'public.tar.gz.003'],
        data_checksum=1199509737,
    ),
    'private': dict(
        code='g5t7q',
        files=["test.h5", "private_test.h5"],
        data_checksum=None,
    )
}


def get_connection_info(code, username=None, password=None):
    "Get connection to OSF and info relative to public/private data."
    osf = OSF(username=username, password=password)

    try:
        project = osf.project(code)
        store = project.storage('osfstorage')
    except UnauthorizedException:
        raise ValueError("Invalid credentials for RAMP private storage.")
    return store


def get_one_element(container, name):
    "Get one element from OSF container with a comprehensible failure error."
    elements = [f for f in container if f.name == name]
    container_name = (
        container.name if hasattr(container, 'name') else CHALLENGE_NAME
    )
    assert len(elements) == 1, (
        f'There is no element named {name} in {container_name} from the RAMP '
        'OSF account.'
    )
    return elements[0]


def hash_folder(folder_path):
    """Return the Adler32 hash of an entire directory."""
    folder = Path(folder_path)

    # Recursively scan the folder and compute a checksum
    checksum = 1
    for f in sorted(folder.rglob('*')):
        if f.is_file():
            checksum = adler32(f.read_bytes(), checksum)
        else:
            checksum = adler32(f.name.encode(), checksum)

    return checksum


def checksum_data(data_dir, cksum, raise_error=False):
    print("Checking the data...", end='', flush=True)
    local_checksum = hash_folder(data_dir)
    if raise_error and cksum != local_checksum:
        raise ValueError(
            f"The checksum does not match. Expecting {cksum} but "
            f"got {local_checksum}. The archive seems corrupted. Try to "
            f"remove {data_dir} and re-run this command."
        )
    print("Done.")

    return cksum == local_checksum


def download_split_archive_from_osf(folder, split_files, data_dir):
    """
    Merge split files, extract the tar.gz file, and delete the merged file.

    Args:
        split_files (list of Path): Ordered list of split file paths.
        extract_path (str): Directory where files will be extracted.
    """
    output_tar = (data_dir / split_files[0]).with_suffix("")
    assert "001" not in output_tar.name, output_tar.name

    N = len(split_files)
    with open(output_tar, 'wb') as f:
        for i, fname in enumerate(split_files):
            print(f"Downloading {output_tar.name} ({i+1}/{N})...\r")
            osf_file = get_one_element(folder.files, fname)
            osf_file.write_to(f)
    print("Downloading done.".ljust(40))
    return output_tar


def download_from_osf(folder, filename, data_dir):
    # Download the archive in the data
    target_path = data_dir / filename
    osf_file = get_one_element(folder.files, filename)
    print(f"Downloading {filename}...\r", end='', flush=True)
    with open(target_path, 'wb') as f:
        osf_file.write_to(f)
    print("Downloading done.".ljust(40))
    return target_path


def setup_data(private_data=None, username=None, password=None):
    "Download and uncompress the data from OSF."
    public_data_path = Path("./data/")
    if private_data is not None:
        chunk = 'private'
        data_path = Path(private_data)
    else:
        chunk = 'public'
        data_path = public_data_path

    config = RAMP_FOLDER_CONFIGURATION[chunk]
    cksum = config['data_checksum']

    if not data_path.exists() or cksum is None:
        data_path.mkdir(exist_ok=True)
    elif checksum_data(data_path, cksum, raise_error=False):
        print("Data already downloaded and verified.")
        return

    # Get the connection to OSF and find the folder in the OSF project
    print("Checking the data URL...", end='', flush=True)
    store = get_connection_info(
        config['code'], username=username, password=password
    )
    challenge_folder = get_one_element(store.folders, CHALLENGE_NAME)
    print('Ok.')

    # Download the public data
    if chunk == 'public':
        archive = download_split_archive_from_osf(
            challenge_folder, config['files'], data_path
        )
        with tarfile.open(archive) as tar:
            tar.extractall(data_path)

        # Remove intermediate folder public
        [f.rename(data_path / f.name)
         for f in (data_path / "public").glob("*")]
        (data_path / "public").rmdir()
        archive.unlink()

        checksum_data(data_path, cksum, raise_error=True)
    else:
        private_test = download_from_osf(
            challenge_folder, "private_test.h5", data_path
        )
        download_from_osf(challenge_folder, "test.h5", public_data_path)
        private_test.rename(data_path / "test.h5")
        (public_data_path / "train.h5").symlink_to(private_data / "train.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f'Data loader for the {CHALLENGE_NAME} challenge on RAMP.'
    )
    parser.add_argument('--private-data', type=Path, default=None,
                        help='If this flag is used, download the private data '
                        'from OSF. This requires the username and password '
                        'options to be provided.')
    parser.add_argument('--username', type=str, default=None,
                        help='Username for downloading private OSF data.')
    parser.add_argument('--password', type=str, default=None,
                        help='Password for downloading private OSF data.')
    args = parser.parse_args()

    if args.private_data is not None:
        setup_data(private_data=args.private_data, username=args.username,
                   password=args.password)
    else:
        setup_data()
