import os
import io
import pickle

import requests

import torch
from torch.serialization import (
    _get_restore_location, _maybe_decode_ascii, _open_file_like, _open_zipfile_reader
)

from tqdm import tqdm


def safe_load(f, map_location=None, pickle_module=pickle, pickle_file='data.pkl', **pickle_load_args):
    with _open_file_like(f, 'rb') as opened_file:
        with _open_zipfile_reader(opened_file) as zip_file:
            restore_location = _get_restore_location(map_location)

            loaded_storages = {}

            def load_tensor(data_type, size, key, location):
                name = f'data/{key}'
                dtype = data_type(0).dtype

                storage = zip_file.get_storage_from_record(name, size, dtype).storage()
                loaded_storages[key] = restore_location(storage, location)

            def persistent_load(saved_id):
                assert isinstance(saved_id, tuple)
                typename = _maybe_decode_ascii(saved_id[0])
                data = saved_id[1:]

                assert typename == 'storage', \
                    f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
                data_type, key, location, size = data
                if key not in loaded_storages:
                    load_tensor(data_type, size, key, _maybe_decode_ascii(location))
                storage = loaded_storages[key]
                return storage

            load_module_mapping = {
                'torch.tensor': 'torch._tensor'
            }

            class UnpicklerWrapper(pickle_module.Unpickler):
                def find_class(self, mod_name, name):
                    try:
                        mod_name = load_module_mapping.get(mod_name, mod_name)
                        return super().find_class(mod_name, name)
                    except Exception:
                        pass

            # Load the data (which may in turn use `persistent_load` to load tensors)
            data_file = io.BytesIO(zip_file.get_record(pickle_file))

            unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()

            torch._utils._validate_loaded_sparse_tensors()

            return result


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)
