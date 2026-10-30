# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""Bounded immutable NPZ storage for PR-0012 higher-dimensional integer counts."""
import io,json,math,zipfile
from pathlib import Path
import numpy as np
import pr0012_io as disk

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def open_npz(path):
    """Validate sizes/types before allocation. No pickle or executable archive data."""
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)>64 or sum(i.file_size for i in infos)>32*1024**2:
            raise ValueError('Invalid/oversized NPZ')
        for item in infos:
            if '/' in item.filename or not item.filename.endswith('.npy'):raise ValueError('Invalid NPZ member')
            with z.open(item) as f:
                version=np.lib.format.read_magic(f)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:raise ValueError('Unsupported NPY')
                shape,_,dtype=reader(f)
                if dtype.hasobject or dtype.kind not in 'fiuU' or len(shape)>8 or any(n<0 for n in shape):
                    raise ValueError('Invalid array type/shape')
                if math.prod(shape)*dtype.itemsize!=item.file_size-f.tell():raise ValueError('NPY size mismatch')
    return np.load(path,allow_pickle=False)


def save_npz(path,**arrays):
    """Immutable logical file: retries may confirm identical contents, never replace."""
    path=Path(path);arrays={k:np.asarray(v) for k,v in arrays.items()}
    if path.exists():
        with open_npz(path) as old:
            if set(old.files)!=set(arrays):raise ValueError('Existing NPZ fields differ')
            for k,a in arrays.items():
                b=old[k]
                equal=np.array_equal(a,b,equal_nan=True) if a.dtype.kind=='f' else np.array_equal(a,b)
                if a.dtype!=b.dtype or not equal:raise ValueError('Existing evidence differs: '+str(path))
        return
    b=io.BytesIO();np.savez(b,**arrays);disk.atomic(path,b.getvalue())


def save_json(path,data):
    path=Path(path)
    if path.exists():
        if canonical(disk.read(path))!=canonical(data):raise ValueError('Existing JSON differs: '+str(path))
    else:disk.write(path,data)


