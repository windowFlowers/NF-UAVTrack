import os, sys
from pathlib import Path


def setup_repo(repo):
    """Run from inside the official tracker repository (several of them resolve paths relative to the cwd)."""
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f'tracker repository not found: {repo}')
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    return repo


def resolve(repo, path):
    """A config / checkpoint path given on the command line; relative paths are relative to the tracker repo."""
    p = Path(path)
    return p if p.is_absolute() else Path(repo) / p
