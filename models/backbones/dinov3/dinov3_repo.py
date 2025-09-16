import os

from .git_utils import GIT_ROOT



external_dir_path = os.path.join(GIT_ROOT, 'submodules')
if not os.path.exists(external_dir_path):
    os.makedirs(external_dir_path)

DINOV3_REPO_PATH = os.path.join(external_dir_path, 'dinov3')

#DNOV3_REPO_PATH = <or replace with your path>
if not os.path.exists(DINOV3_REPO_PATH):
    raise RuntimeError("Need to set DINOV3_REPO_PATH")
