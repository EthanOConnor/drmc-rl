"""Batched CUDA twin of the CPU ``drm_reach_bfs_full`` planner.

``CudaReachFull.solve`` returns, per instance, the same costs, offsets,
lengths and packed controller scripts as ``NativeReachabilityRunner.bfs_full``
for an unconstrained profile, byte for byte (see ``drm_reach_full.cu``). An
instance with a non-zero status must be answered by the CPU planner; the
caller never receives a partial CUDA answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import time

import cuda.bindings.driver as drv
import numpy as np

from drmc_rl.planning.cuda.host import INSTANCE_DTYPE, N_POSES, _check, _compile_cubin, _HERE
from drmc_rl.planning.native_reach import NativeReachability

_SOURCE = _HERE / "drm_reach_full.cu"
KEYS_PER_SC = 18432
STATUS_NAMES = {1: "script_chain", 2: "script_capacity", 4: "frontier_capacity",
                8: "threshold_capacity", 16: "bad_arguments"}


@dataclass
class FullBatch:
    costs: np.ndarray      # (n, 512) u16
    offsets: np.ndarray    # (n, 512) u16
    lengths: np.ndarray    # (n, 512) u16
    scripts: np.ndarray    # (n, script_capacity) u8
    used: np.ndarray       # (n,) i32
    status: np.ndarray     # (n,) i32
    nodes: np.ndarray      # (n,) u32 frontier entries

    def reach(self, i: int) -> NativeReachability:
        used = int(self.used[i])
        return NativeReachability(self.costs[i].copy(), self.offsets[i].copy(), self.lengths[i].copy(),
                                  self.scripts[i, :used].copy())


class CudaReachFull:
    """Owns the CUDA context, module and one BFS workspace per resident block.

    Not thread-safe; use one owner thread.
    """

    def __init__(self, device: int = 0, *, blocks_per_sm: int = 1, block_threads: int = 1024,
                 script_capacity: int = 32768, frontier_factor: float = 1.0, max_batch: int = 1024):
        _check(drv.cuInit(0))
        self.dev = _check(drv.cuDeviceGet(device))
        self.ctx = _check(drv.cuDevicePrimaryCtxRetain(self.dev))
        _check(drv.cuCtxSetCurrent(self.ctx))
        major = _check(drv.cuDeviceGetAttribute(
            drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.dev))
        minor = _check(drv.cuDeviceGetAttribute(
            drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.dev))
        sms = _check(drv.cuDeviceGetAttribute(
            drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self.dev))
        self.arch = f"sm_{major}{minor}"
        cubin = _compile_cubin(_SOURCE.read_text(), self.arch, log_prefix="drm_reach_full")
        self.module = _check(drv.cuModuleLoadData(cubin))
        self.kernel = _check(drv.cuModuleGetFunction(self.module, b"drm_reach_full_kernel"))
        self.stream = _check(drv.cuStreamCreate(0))
        self.slots = int(sms * blocks_per_sm)
        self.block_threads = int(block_threads)
        self.script_capacity = int(script_capacity)
        self.frontier_factor = float(frontier_factor)
        self.max_batch = int(max_batch)
        self.scr = 0
        self.arena = None
        self.cursor = _check(drv.cuMemAlloc(8))
        self._io_capacity = 0
        self._io = {}

    # -- memory -------------------------------------------------------------

    def _slot_layout(self, scr: int):
        nkeys = KEYS_PER_SC * scr
        fr_cap = int(nkeys * 8 * self.frontier_factor)
        fr_cap = max(1, min(fr_cap, (2**32 - 1) // 108 - 1))
        slot = nkeys * (32 + 4 + 4 + 4 + 16 + 4 + 1) + fr_cap * 5
        slot = (slot + 255) // 256 * 256
        return nkeys, fr_cap, slot

    def _ensure_arena(self, scr: int) -> None:
        if scr <= self.scr:
            return
        if self.arena is not None:
            _check(drv.cuStreamSynchronize(self.stream))
            drv.cuMemFree(self.arena)
            self.arena = None
        nkeys, fr_cap, slot = self._slot_layout(scr)
        self.arena = _check(drv.cuMemAlloc(slot * self.slots))
        _check(drv.cuMemsetD8(self.arena, 0, slot * self.slots))
        for index in range(self.slots):
            base = int(self.arena) + index * slot
            _check(drv.cuMemsetD32(drv.CUdeviceptr(base), 0xFFFFFFFF, nkeys * 8))               # sord
            _check(drv.cuMemsetD32(drv.CUdeviceptr(base + nkeys * 36), 0xFFFFFFFF, nkeys))      # kord (after sord, acc)
        self.scr, self.nkeys, self.fr_cap, self.slot_bytes = scr, nkeys, fr_cap, slot

    def _ensure_io(self, n: int) -> None:
        if n <= self._io_capacity:
            return
        for pointer in self._io.values():
            drv.cuMemFree(pointer)
        capacity = max(n, 64)
        sizes = dict(insts=capacity * INSTANCE_DTYPE.itemsize, costs=capacity * N_POSES * 2,
                     offsets=capacity * N_POSES * 2, lengths=capacity * N_POSES * 2,
                     scripts=capacity * self.script_capacity, used=capacity * 4, status=capacity * 4,
                     nodes=capacity * 4)
        self._io = {name: _check(drv.cuMemAlloc(size)) for name, size in sizes.items()}
        self._io_capacity = capacity

    # -- public -------------------------------------------------------------

    @staticmethod
    def pack(columns, spawns, thresholds, max_frames: int = 2048) -> np.ndarray:
        """(n,8) columns, FrameState spawns and speed thresholds -> instances."""
        n = len(spawns)
        out = np.zeros(n, dtype=INSTANCE_DTYPE)
        out["cols"] = np.asarray(columns, dtype=np.uint16).reshape(n, 8)
        for i, spawn in enumerate(spawns):
            out["sx"][i] = int(spawn.x)
            out["sy"][i] = int(spawn.y)
            out["srot"][i] = int(spawn.rot) & 3
            out["sc"][i] = min(255, max(0, int(spawn.speed_counter)))
            out["hv"][i] = int(spawn.hor_velocity) & 0xFF
            out["hd"][i] = int(getattr(spawn.hold_dir, "value", spawn.hold_dir))
            out["p"][i] = int(spawn.frame_parity) & 1
            out["rh"][i] = int(getattr(spawn.rot_hold, "value", spawn.rot_hold))
        thresholds = np.asarray(thresholds, dtype=np.int64)
        if (thresholds < 0).any() or (thresholds > 127).any():
            raise ValueError("speed thresholds must be in [0,127]")
        out["thr"] = thresholds
        out["max_frames"] = max_frames
        return out

    def solve(self, instances: np.ndarray) -> FullBatch:
        instances = np.ascontiguousarray(instances, dtype=INSTANCE_DTYPE)
        n = len(instances)
        empty = np.zeros((0, N_POSES), np.uint16)
        if n == 0:
            return FullBatch(empty, empty, empty, np.zeros((0, self.script_capacity), np.uint8),
                             np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.uint32))
        if n > self.max_batch:
            parts = [self.solve(instances[i:i + self.max_batch]) for i in range(0, n, self.max_batch)]
            width = max(p.scripts.shape[1] for p in parts)
            scripts = np.concatenate([np.pad(p.scripts, ((0, 0), (0, width - p.scripts.shape[1])))
                                      for p in parts])
            return FullBatch(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                               ("costs", "offsets", "lengths")), scripts,
                             *(np.concatenate([getattr(p, f) for p in parts]) for f in
                               ("used", "status", "nodes")))
        self._ensure_arena(max(14, int(instances["thr"].max()) + 1))
        self._ensure_io(n)
        io = self._io
        _check(drv.cuMemcpyHtoDAsync(io["insts"], instances.ctypes.data, instances.nbytes, self.stream))
        _check(drv.cuMemsetD8Async(self.cursor, 0, 8, self.stream))
        self._launch(self.kernel, [
            (io["insts"], "p"), (n, "i"), (self.cursor, "p"), (self.arena, "p"),
            (self.slot_bytes // 256, "I"), (self.scr, "i"), (self.fr_cap, "I"),
            (self.script_capacity, "i"), (io["costs"], "p"), (io["offsets"], "p"),
            (io["lengths"], "p"), (io["scripts"], "p"), (io["used"], "p"), (io["status"], "p"),
            (io["nodes"], "p")], min(self.slots, n), self.block_threads)
        return self._collect(n, ok=CudaReachFull._full_ok)

    # -- paced stage 1 -------------------------------------------------------

    @staticmethod
    def pack_paced(columns, spawns, thresholds, planner_args, max_frames: int = 2048) -> np.ndarray:
        """Instances for ``solve_paced``; ``planner_args`` as ``Pace.planner_args``."""
        out = CudaReachFull.pack(columns, spawns, thresholds, max_frames)
        for i, args in enumerate(planner_args):
            reaction = int(args.get("reaction_frames", 0))
            edge, motion = int(args.get("edge_interval", 0)), int(args.get("motion_interval", 0))
            buttons = int(args.get("max_buttons", 3))
            if not (0 <= reaction <= 255 and 0 <= edge <= 255 and 0 <= motion <= 255 and 0 <= buttons <= 255):
                raise ValueError("paced profile out of range")
            out["flags"][i] = reaction
            out["_pad"][i] = edge | motion << 8 | buttons << 16
        return out

    def solve_paced(self, instances: np.ndarray):
        """Stage 1 of ``drm_reach_bfs_paced``: returns (FullBatch, unresolved wanted (n,512) u8,
        post-reaction state (n,8) u8). Status 32 means the CPU would continue past the
        simple routes; the caller resolves it with the unrestricted v4 costs or the CPU."""
        instances = np.ascontiguousarray(instances, dtype=INSTANCE_DTYPE)
        n = len(instances)
        if n > self.max_batch:
            raise ValueError("paced batch larger than max_batch; chunk it")
        if getattr(self, "paced_kernel", None) is None:
            self.paced_kernel = _check(drv.cuModuleGetFunction(self.module, b"drm_reach_paced_stage1_kernel"))
            self.paced_scratch = _check(drv.cuMemAlloc(self.slots * N_POSES * 2 * 1024))
            self.paced_io = None
        if self.paced_io is None or self.paced_io[0] < n:
            if self.paced_io is not None:
                drv.cuMemFree(self.paced_io[1]); drv.cuMemFree(self.paced_io[2])
            capacity = max(n, self.max_batch)
            self.paced_io = (capacity, _check(drv.cuMemAlloc(capacity * N_POSES)),
                             _check(drv.cuMemAlloc(capacity * 8)))
        self._ensure_io(n)
        io = self._io
        _check(drv.cuMemcpyHtoDAsync(io["insts"], instances.ctypes.data, instances.nbytes, self.stream))
        _check(drv.cuMemsetD8Async(self.cursor, 0, 8, self.stream))
        self._launch(self.paced_kernel, [
            (io["insts"], "p"), (n, "i"), (self.cursor, "p"), (self.paced_scratch, "p"),
            (self.script_capacity, "i"), (io["costs"], "p"), (io["offsets"], "p"), (io["lengths"], "p"),
            (io["scripts"], "p"), (io["used"], "p"), (io["status"], "p"), (self.paced_io[1], "p"),
            (self.paced_io[2], "p")], min(self.slots, n), 512)
        wanted = np.empty((n, N_POSES), np.uint8)
        initial = np.empty((n, 8), np.uint8)
        batch = self._collect(n, extra=((wanted, self.paced_io[1]), (initial, self.paced_io[2])),
                              ok=lambda status: (status == 0) | (status == 32))
        return batch, wanted, initial

    def _launch(self, kernel, args, grid, block):
        holders = []
        for value, kind in args:
            dtype = np.uint64 if kind == "p" else np.int32 if kind == "i" else np.uint32
            holders.append(np.array([int(value)], dtype=dtype))
        pointers = np.array([h.ctypes.data for h in holders], dtype=np.uint64)
        _check(drv.cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1, 0, self.stream, pointers.ctypes.data, 0))

    def _collect(self, n, extra=(), ok=lambda status: status == 0):
        io = self._io
        costs = np.empty((n, N_POSES), np.uint16)
        offsets = np.empty((n, N_POSES), np.uint16)
        lengths = np.empty((n, N_POSES), np.uint16)
        used = np.empty(n, np.int32)
        status = np.empty(n, np.int32)
        nodes = np.zeros(n, np.uint32)
        pairs = [(costs, io["costs"]), (offsets, io["offsets"]), (lengths, io["lengths"]),
                 (used, io["used"]), (status, io["status"]), *extra]
        if ok is CudaReachFull._full_ok:
            pairs.append((nodes, io["nodes"]))
        for array, pointer in pairs:
            _check(drv.cuMemcpyDtoHAsync(array.ctypes.data, pointer, array.nbytes, self.stream))
        self._wait()
        width = int(min(self.script_capacity, max(1, used[ok(status)].max(initial=1))))
        scripts = np.zeros((n, width), np.uint8)
        copy = drv.CUDA_MEMCPY2D()
        copy.srcMemoryType = drv.CUmemorytype.CU_MEMORYTYPE_DEVICE
        copy.srcDevice = io["scripts"]
        copy.srcPitch = self.script_capacity
        copy.dstMemoryType = drv.CUmemorytype.CU_MEMORYTYPE_HOST
        copy.dstHost = scripts.ctypes.data
        copy.dstPitch = width
        copy.WidthInBytes = width
        copy.Height = n
        _check(drv.cuMemcpy2DAsync(copy, self.stream))
        self._wait()
        return FullBatch(costs, offsets, lengths, scripts, used, status, nodes)

    @staticmethod
    def _full_ok(status):
        return status == 0

    def _wait(self) -> None:
        # Poll so the interpreter lock is free for the rollout thread while
        # the kernel runs (a blocking synchronize may hold it).
        while True:
            (err,) = drv.cuStreamQuery(self.stream)
            if err == drv.CUresult.CUDA_SUCCESS:
                return
            if err != drv.CUresult.CUDA_ERROR_NOT_READY:
                _check((err,))
            time.sleep(0.0002)

    def close(self) -> None:
        for pointer in self._io.values():
            drv.cuMemFree(pointer)
        self._io = {}
        if getattr(self, "paced_io", None) is not None:
            drv.cuMemFree(self.paced_io[1]); drv.cuMemFree(self.paced_io[2])
            self.paced_io = None
        for name in ("arena", "cursor", "paced_scratch"):
            if getattr(self, name, None) is not None:
                drv.cuMemFree(getattr(self, name))
                setattr(self, name, None)
        if getattr(self, "stream", None) is not None:
            drv.cuStreamDestroy(self.stream)
            self.stream = None
        if getattr(self, "module", None) is not None:
            drv.cuModuleUnload(self.module)
            self.module = None
        if getattr(self, "ctx", None) is not None:
            drv.cuDevicePrimaryCtxRelease(self.dev)
            self.ctx = None
