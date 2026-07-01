# Gaudi (HPU) disaggregated weight-sync checkpoint engine, shipped as a verl
# custom_backend_module plugin (zero verl-core edits).
#
# Why this exists: verl's HCCL checkpoint engine is gated to Ascend NPU and its
# stateless communicator (vllm_ascend PyHccl) is Ascend-only; vllm_gaudi ships no
# stateless cross-job communicator. A real Habana PyHccl needs unverified libhccl
# C symbols, and a torch.distributed "hccl" subgroup is impossible here (both
# endpoints already own an initialized default process group in separate Ray
# worlds). So we route weight bytes HPU->CPU->TCPStore->CPU->HPU using vLLM's
# StatelessProcessGroup object channel — the one cross-job primitive already
# proven to work on this Gaudi env. For a 0.5B model / few-step proof the CPU
# detour bandwidth is irrelevant.
#
# It (a) defines a CPU-mediated stateless communicator matching the interface the
# engine calls (broadcast/all_reduce/destroyComm/.comm), and (b) execs the
# upstream HCCL engine source with surgical device swaps and registers it under
# the name "hccl".
import os
import socket
from datetime import timedelta

import torch
from torch.distributed import TCPStore
from vllm.distributed.utils import StatelessProcessGroup

import verl.checkpoint_engine as _ce_pkg
from verl.checkpoint_engine.base import CheckpointEngineRegistry  # noqa: F401  (used by exec'd source)
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.net_utils import is_ipv6


class HpuStatelessCommunicator:
    """CPU-mediated stand-in for a PyNccl/PyHccl communicator.

    Moves device tensors through the StatelessProcessGroup TCPStore object channel
    instead of a device collective library. Implements exactly the methods the
    HCCL checkpoint engine calls: broadcast(tensor, src), all_reduce(tensor),
    destroyComm(comm), plus a `.comm` handle attribute.
    """

    def __init__(self, pg: "StatelessProcessGroup", device):
        self.pg = pg
        self.comm = pg  # opaque handle the engine reads as `.comm`
        self.rank = pg.rank
        self.world_size = pg.world_size

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        if self.rank == src:
            self.pg.broadcast_obj(tensor.detach().to("cpu"), src)
        else:
            cpu = self.pg.broadcast_obj(None, src)
            # in-place: the engine reuses send_buf/recv_buf across buckets
            tensor.copy_(cpu.to(tensor.device))

    def all_reduce(self, tensor: torch.Tensor):
        parts = self.pg.all_gather_obj(tensor.detach().to("cpu"))
        acc = parts[0]
        for p in parts[1:]:
            acc = acc + p
        tensor.copy_(acc.to(tensor.device))

    def destroyComm(self, comm):  # noqa: N802 (match upstream API name)
        self.pg = None
        self.comm = None


def _hpu_stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """Build a StatelessProcessGroup (TCPStore rendezvous) + HpuStatelessCommunicator.

    Mirrors verl.utils.distributed.stateless_init_process_group / create_process_group
    but returns the CPU-mediated Habana communicator instead of PyNccl.
    """
    launch_server = rank == 0
    if launch_server:
        if is_ipv6(master_address):
            listen_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind((master_address, master_port))
        listen_socket.listen()
        listen_fd = listen_socket.fileno()
    else:
        listen_socket = None
        listen_fd = None

    store = TCPStore(
        host_name=master_address,
        port=master_port,
        world_size=world_size,
        is_master=launch_server,
        timeout=timedelta(seconds=300),
        use_libuv=False,
        master_listen_fd=listen_fd,
    )
    pg = StatelessProcessGroup(
        rank=rank,
        world_size=world_size,
        store=store,
        socket=listen_socket,
        data_expiration_seconds=3600,
    )
    return HpuStatelessCommunicator(pg, device)


# --- load the upstream HCCL engine source, device-port it, register as "hccl" ---
_src_path = os.path.join(os.path.dirname(_ce_pkg.__file__), "hccl_checkpoint_engine.py")
_src = open(_src_path).read()

# drop the NPU-only imports + guard (we provide device-agnostic equivalents)
_src = _src.replace("from verl.utils.device import is_torch_npu_available\n", "")
_src = _src.replace("from verl.utils.distributed import stateless_init_process_group\n", "")
_src = _src.replace(
    'if not is_torch_npu_available(check_device=False):\n'
    '    raise ImportError("HCCLCheckpointEngine is unavailable because the torch.npu module is not available.")\n',
    "",
)
# device swaps: torch.npu.* -> verl device facade; device="npu" -> active device
_src = _src.replace("torch.npu.current_device()", "get_torch_device().current_device()")
_src = _src.replace('device="npu"', "device=get_device_name()")
_src = _src.replace("torch.npu.empty_cache()", "get_torch_device().empty_cache()")
_src = _src.replace("torch.npu.synchronize()", "get_torch_device().synchronize()")
# use the CPU-mediated communicator, and register under the "hccl" backend name
_src = _src.replace("stateless_init_process_group(", "_hpu_stateless_init_process_group(")
_src = _src.replace('@CheckpointEngineRegistry.register("nccl")', '@CheckpointEngineRegistry.register("hccl")')

exec(compile(_src, _src_path, "exec"), globals())
