import os
import tarfile
import subprocess
from pathlib import Path
import argparse

import owncloud


def make_tarfile(output_filename: Path, source_dir: Path):
    print(f'Tarring {source_dir} to {output_filename}')
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
    subprocess.run(["tar", "-czvf", output_filename, source_dir])


def upload_and_share(filepath: Path, oc_output_path: str):

    # change to environment variables
    url = 'https://uni-bielefeld.sciebo.de'
    username = 'djohnson1@uni-bielefeld.de'
    password = 'vGtrV9qefVFdJ7'

    oc = owncloud.Client(url)

    oc.login(username, password)
    _ = oc.put_file(f'{oc_output_path}/{filepath.name}', filepath)
    link_info: owncloud.ShareInfo = oc.share_file_with_link(f'{oc_output_path}/{filepath.name}')
    oc.logout()

    return link_info


def main(args):
    
    output_filepath = args.output / f'{args.source_dir.stem}.tar.gz'
    if args.tar:
        make_tarfile(output_filepath, args.source_dir)

    print('Uploading and Sharing to Sciebo')
    shareinfo = upload_and_share(output_filepath, args.remote_dir)
    print(f'File URL: {shareinfo.get_link()}/download')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_dir', 
                        help='Path to folder to Tar and Upload',
                        default='./blockies_datasets/',
                        type=Path)
    parser.add_argument('-o', '--output', 
                        help='Path to output the tar file', 
                        default='./', 
                        type=Path)
    parser.add_argument('-r', '--remote_dir', 
                        help='Directory on Sciebot to upload the file', 
                        default='1. Research/1. HCXAI/1. Projects/blockies_datasets/')
    parser.add_argument('-t', '--tar',
                        help='Tar and gzip source folder',
                        action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    main(args)