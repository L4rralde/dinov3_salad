import os


if not 'DINO3_SALAD_ROOT' in os.environ:
    raise RuntimeError("Please, first set $DINO3_SALAD_ROOT environment variable")

external_dir_path = os.path.join(os.environ['DINO3_SALAD_ROOT'], 'submodules')
if not os.path.exists(external_dir_path):
    os.makedirs(external_dir_path)

DINOV3_REPO_PATH = os.path.join(external_dir_path, 'dinov3')

#DNOV3_REPO_PATH = <or replace with your path>
if not os.path.exists(DINOV3_REPO_PATH):
    raise RuntimeError("Need to set DINOV3_REPO_PATH")
