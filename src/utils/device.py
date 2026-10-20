"""GPU-only runtime checks; importing this module does not import PyTorch."""
import argparse
import csv
import io
import json
import os
import re
import shlex
import sys
import subprocess


def cuda_request(requested):
    """Normalize legacy auto selections and reject CPU/non-CUDA execution."""
    if requested == 'auto':
        return 'cuda'
    if not isinstance(requested, str) or not re.fullmatch(r'cuda(?::[0-9]+)?', requested):
        raise ValueError('GPU-only execution: device must be cuda or cuda:N; CPU execution is disabled.')
    return requested


def gpu_requests(device, devices):
    """Resolve profile GPU selection without importing or probing CUDA."""
    device = cuda_request(device)
    if not isinstance(devices, list):
        raise ValueError('GPU devices must be a list of cuda:N strings, or [] for automatic selection.')
    if not devices:
        return [] if device == 'cuda' else [device]
    if any(not isinstance(value, str) or not re.fullmatch(r'cuda:[0-9]+', value) for value in devices):
        raise ValueError('GPU devices must use explicit indices, for example ["cuda:0", "cuda:1"].')
    normalized = [f'cuda:{int(value.split(":")[1])}' for value in devices]
    if len(set(normalized)) != len(normalized):
        raise ValueError('GPU devices must be distinct.')
    return normalized


def compute_processes():
    """Return UUID -> compute PIDs, or None when NVIDIA monitoring is unavailable."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=3, stdin=subprocess.DEVNULL)
        if result.returncode:
            return None
        processes = {}
        for row in csv.reader(io.StringIO(result.stdout)):
            if not row:
                continue
            uuid, pid = (item.strip() for item in row)
            processes.setdefault(uuid, set()).add(int(pid))
        return processes
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def available_gpus(device='cuda', devices=None, min_free_memory_mb=1024, require_dgl=False):
    """Discover usable idle GPU slots in the worker's CUDA visibility namespace."""
    requests = gpu_requests(device, devices if devices is not None else [])
    torch = require_cuda_available()
    count = torch.cuda.device_count()
    indices = [int(value.split(':')[1]) for value in requests] if requests else list(range(count))
    visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
    tokens = [token.strip() for token in visibility.split(',')] if visibility is not None else None
    processes = compute_processes()
    gpus, skipped = [], []
    for index in indices:
        selected = f'cuda:{index}'
        try:
            if index >= count:
                raise RuntimeError(f'Only {count} CUDA device(s) are visible.')
            props = torch.cuda.get_device_properties(index)
            uuid = str(getattr(props, 'uuid', ''))
            if re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid):
                uuid = 'GPU-' + uuid
            if processes is not None and processes.get(uuid, set()) - {os.getpid()}:
                raise RuntimeError('GPU already has another compute process.')
            free, total = torch.cuda.mem_get_info(index)
            if free < min_free_memory_mb * 1024**2:
                raise RuntimeError(f'Only {free // 1024**2} MiB free; {min_free_memory_mb} MiB required.')
            require_cuda(selected, require_dgl)
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(index)
            if free < min_free_memory_mb * 1024**2:
                raise RuntimeError(f'Only {free // 1024**2} MiB free after CUDA setup; '
                                   f'{min_free_memory_mb} MiB required.')
            # UUIDs preserve CUDA's mapping even when physical device order differs.
            token = tokens[index] if tokens else (uuid if uuid.startswith(('GPU-', 'MIG-')) else str(index))
            gpus.append(dict(device=selected, gpu=props.name, uuid=uuid, visibility_token=token,
                             free_memory_mb=free // 1024**2, total_memory_mb=total // 1024**2,
                             torch=str(torch.__version__), cuda=torch.version.cuda))
            if require_dgl:
                import dgl
                gpus[-1]['dgl'] = dgl.__version__
        except (RuntimeError, ValueError, AssertionError, IndexError) as error:
            skipped.append(dict(device=selected, reason=str(error)))
    if not gpus:
        reasons = '; '.join(f'{row["device"]}: {row["reason"]}' for row in skipped)
        raise RuntimeError('No available CUDA GPUs; no jobs can start. ' + reasons)
    return dict(gpus=gpus, skipped=skipped, process_monitoring=processes is not None)


def require_cuda_available():
    """Reject unavailable CUDA before loading checkpoints or model data."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('GPU execution requires CUDA-enabled PyTorch in the experiment interpreter. '
                           'Install requirements.txt there.') from error
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required: CUDA is unavailable in this interpreter. '
                           'Check the NVIDIA driver, CUDA_VISIBLE_DEVICES and CUDA-enabled PyTorch. '
                           'CPU fallback is disabled.')
    return torch


def require_cuda(requested='cuda', require_dgl=False):
    """Require a usable selected GPU and, for LGKDE, CUDA graph transfers."""
    requested = cuda_request(requested)
    torch = require_cuda_available()
    device = torch.device(requested)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise RuntimeError(f'GPU required: {requested} is not visible; '
                           f'{torch.cuda.device_count()} CUDA device(s) are available.')
    device = torch.device('cuda', index)
    try:
        torch.cuda.set_device(device)
        # Allocation plus a kernel catches unusable contexts and unsupported GPU builds.
        with torch.cuda.device(device):
            probe = torch.zeros(1, device=device) + 1
            torch.cuda.synchronize(device)
            del probe
    except (RuntimeError, AssertionError) as error:
        raise RuntimeError(f'GPU required: {device} cannot execute CUDA operations: {error}') from error
    if require_dgl:
        try:
            import dgl
            edge = torch.tensor([0], dtype=torch.int64)
            graph = dgl.graph((edge, edge), num_nodes=1).to(device)
            torch.cuda.synchronize(device)
            del graph
        except Exception as error:
            command = (f'{shlex.quote(sys.executable)} -m pip install --force-reinstall --no-deps '
                       '--no-index --find-links https://data.dgl.ai/wheels/cu121/repo.html '
                       "'dgl==1.1.3+cu121'")
            raise RuntimeError(f'LGKDE requires CUDA-enabled DGL on {device}. '
                               f'Repair the CUDA 12.1 installation with:\n{command}\n'
                               f'DGL error: {error}') from error
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dgl', action='store_true')
    parser.add_argument('--list-gpus', action='store_true')
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--min-free-memory-mb', type=int, default=1024)
    args = parser.parse_args()
    try:
        if args.list_gpus:
            if args.min_free_memory_mb < 1:
                raise ValueError('Minimum free GPU memory must be positive.')
            print(json.dumps(available_gpus(args.device, args.devices,
                                            args.min_free_memory_mb, args.dgl)), flush=True)
            return
        device = require_cuda(args.device, args.dgl)
        import torch
        details = dict(device=str(device), gpu=torch.cuda.get_device_name(device),
                       torch=torch.__version__, cuda=torch.version.cuda)
        if args.dgl:
            import dgl
            details['dgl'] = dgl.__version__
        print(json.dumps(details), flush=True)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
