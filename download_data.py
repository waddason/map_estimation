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

CHALLENGE_NAME = 'map_estimation'
PUBLIC_PROJECT = "t4uf8"
PRIVATE_PROJECT = "g5t7q"
PRIVATE_CKSUM = None
PUBLIC_CKSUM = None


def get_folder(code, username=None, password=None):
    "Get connection to OSF and info relative to public/private data."
    # Get the connection to OSF and find the folder in the OSF project
    osf = OSF(username=username, password=password)

    try:
        project = osf.project(code)
        store = project.storage('osfstorage')
    except UnauthorizedException:
        raise ValueError("Invalid credentials for RAMP private storage.")
    return get_one_element(store.folders, CHALLENGE_NAME)


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

    return local_checksum == cksum


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


def setup_data(data_path, private=False, username=None, password=None):
    "Download and uncompress the data from OSF."
    data_path = Path(data_path)
    cksum = PRIVATE_CKSUM if private else PUBLIC_CKSUM

    if not data_path.exists() or cksum is None:
        data_path.mkdir(exist_ok=True)
    elif checksum_data(data_path, cksum, raise_error=False):
        print("Data already downloaded and verified.")
        return

    # Download the public data
    split_files = [
        'public.tar.gz.001', 'public.tar.gz.002', 'public.tar.gz.003'
    ]
    public_folder = get_folder(PUBLIC_PROJECT)
    archive = download_split_archive_from_osf(
        public_folder, split_files, data_path
    )
    with tarfile.open(archive) as tar:
        tar.extractall(data_path)

    # Remove intermediate folder public and make a copy of test.h5
    # for public validation.h5
    [f.rename(data_path / f.name)
        for f in (data_path / "public").glob("*")]
    (data_path / "public").rmdir()
    archive.unlink()
    (data_path / "test.h5").symlink_to(data_path / "validation.h5")

    if private:
        private_folder = get_folder(PRIVATE_PROJECT, username, password)
        test = download_from_osf(
            private_folder, "private_test.h5", data_path
        )
        validation = download_from_osf(private_folder, "test.h5", data_path)
        validation.rename(data_path / "validation.h5")
        test.rename(data_path / "test.h5")

    checksum_data(data_path, cksum, raise_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f'Data loader for the {CHALLENGE_NAME} challenge on RAMP.'
    )
    parser.add_argument('--data-path', type=Path, default=Path("data"),
                        help='If this flag is used, download the private data '
                        'from OSF. This requires the username and password '
                        'options to be provided.')
    parser.add_argument('--private', action="store_true",
                        help='If this flag is used, download the private data '
                        'from OSF. This requires the username and password '
                        'options to be provided.')
    parser.add_argument('--username', type=str, default=None,
                        help='Username for downloading private OSF data.')
    parser.add_argument('--password', type=str, default=None,
                        help='Password for downloading private OSF data.')
    args = parser.parse_args()

    setup_data(args.data_path, private=args.private,
               username=args.username, password=args.password)
