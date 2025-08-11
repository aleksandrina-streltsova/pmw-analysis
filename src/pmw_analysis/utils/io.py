import pathlib


def rmtree(f: pathlib.Path):
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()


def combine_paths(path_base: pathlib.Path | str, path_rel: pathlib.Path | str, path_rel_base: pathlib.Path | str | None = None):
    """
    Combines a base path with a relative path, optionally accounting for a base path for the relative path.
    """
    if isinstance(path_base, str):
        path_base = pathlib.Path(path_base)
    if isinstance(path_rel, str):
        path_rel = pathlib.Path(path_rel)

    if path_rel_base is not None:
        if isinstance(path_rel_base, str):
            path_rel_base = pathlib.Path(path_rel_base)
        path_rel = path_rel.absolute().relative_to(path_rel_base)

    return path_base / path_rel


def file_to_dir(path: pathlib.Path | str):
    """
    Return a directory Path composed of the parent directory and stem name of the input file path.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    return path.parent / path.stem