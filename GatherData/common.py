# !/usr/bin/python
import os


def replace_path(path, new_root):
    new_path = path.split(os.sep)
    for i_part, root_part in enumerate(new_root.split(os.sep)):
        new_path[i_part] = root_part
    if new_path[0] == '':
        new_path = os.path.join(os.sep, *new_path)
    else:
        new_path = os.path.join(*new_path)
    return new_path


def create_dirs(root, new_root, dirs):
    for name in dirs:
        new_path = replace_path(root, new_root)
        new_path = os.path.join(new_path, name)

        if not os.path.exists(new_path):
            try:
                print('CREATE {}'.format(new_path))
                os.makedirs(new_path)
            except OSError as exc:
                pass
