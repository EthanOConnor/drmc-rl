"""Host layer for the CUDA reachability planner.

Torch-free by design ("lots of planner/BFS and no torch"): only numpy +
cuda-bindings (driver API + NVRTC, both shipped by the [rl] extra's wheels).
Torch interop is duck-typed via __cuda_array_interface__ where needed.

Usage:
    from drmc_rl.planning.cuda.host import CudaReach
    ctx = CudaReach()
    costs = ctx.solve_costs(cols, parity, thr)   # (n,8)u16 -> (n,512)u16
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cuda.bindings.driver as drv
import cuda.bindings.nvrtc as nvrtc
import numpy as np

_HERE = Path(__file__).resolve().parent
_KERNEL_SRC = _HERE / "drm_reach.cu"
_CACHE_DIR = _HERE / ".cache"

INSTANCE_DTYPE = np.dtype(
    {
        "names": [
            "cols", "sx", "sy", "srot", "sc", "hv", "hd", "p", "rh",
            "thr", "flags", "max_frames", "_pad",
        ],
        "formats": [
            "(8,)u2", "u1", "u1", "u1", "u1", "u1", "u1", "u1", "u1",
            "u1", "u1", "u2", "u4",
        ],
        "offsets": [0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28],
        "itemsize": 32,
    }
)

N_POSES = 512
GRID_W_COLS = 8


class CudaError(RuntimeError):
    pass


def _check(ret):
    """Unwrap cuda-bindings (err, *vals) tuples; raise on failure."""
    err = ret[0]
    if isinstance(err, drv.CUresult):
        if err != drv.CUresult.CUDA_SUCCESS:
            _, name = drv.cuGetErrorName(err)
            _, desc = drv.cuGetErrorString(err)
            raise CudaError(f"{name.decode()}: {desc.decode()}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise CudaError(nvrtc.nvrtcGetErrorString(err)[1].decode())
    else:
        raise CudaError(f"unexpected return: {ret!r}")
    vals = ret[1:]
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


def _compile_cubin(src: str, arch: str, log_prefix: str = "drm_reach") -> bytes:
    """NVRTC-compile `src` for `arch` (e.g. 'sm_86'), with an on-disk cache."""
    key = hashlib.sha256((arch + "\x00" + src).encode()).hexdigest()[:24]
    cache = _CACHE_DIR / f"{log_prefix}-{arch}-{key}.cubin"
    if cache.exists():
        return cache.read_bytes()

    prog = _check(nvrtc.nvrtcCreateProgram(src.encode(), b"drm_reach.cu", 0, [], []))
    opts = [f"--gpu-architecture={arch}".encode(), b"-lineinfo", b"--std=c++17"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    log_size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
    if log_size > 1:
        buf = b" " * log_size
        _check(nvrtc.nvrtcGetProgramLog(prog, buf))
        log = buf.decode(errors="replace").strip("\x00 \n")
    else:
        log = ""
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise CudaError(f"NVRTC compile failed:\n{log}")

    cubin_size = _check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * cubin_size
    _check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    _check(nvrtc.nvrtcDestroyProgram(prog))

    _CACHE_DIR.mkdir(exist_ok=True)
    cache.write_bytes(cubin)
    return cubin


class CudaReach:
    """Owns the CUDA context, JIT-compiled module, and transfer buffers.

    One instance per process/GPU. Not thread-safe (annotation shards are
    separate processes; the env service runs single-owner).
    """

    def __init__(self, device: int = 0, max_batch: int = 65536,
                 blocks_per_sm: int = 1, block_threads: int = 768):
        # Default config from the tf3090 sweep (tools/bench_reach_cuda.py):
        # wide blocks win — per-depth BFS parallelism matters more than
        # instance-level concurrency, and fewer resident instances thrash L2
        # less. 768x1 = 14.7k solves/s vs 8k at 128x4.
        _check(drv.cuInit(0))
        self.dev = _check(drv.cuDeviceGet(device))
        self.ctx = _check(drv.cuDevicePrimaryCtxRetain(self.dev))
        _check(drv.cuCtxSetCurrent(self.ctx))

        major = _check(drv.cuDeviceGetAttribute(
            drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.dev))
        minor = _check(drv.cuDeviceGetAttribute(
            drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.dev))
        self.n_sms = _check(drv.cuDeviceGetAttribute(
            drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self.dev))
        self.arch = f"sm_{major}{minor}"

        cubin = _compile_cubin(_KERNEL_SRC.read_text(), self.arch)
        self.module = _check(drv.cuModuleLoadData(cubin))
        self.k_identity = _check(drv.cuModuleGetFunction(
            self.module, b"drm_reach_identity_kernel"))
        self.k_debug12 = _check(drv.cuModuleGetFunction(
            self.module, b"drm_reach_debug_phase12_kernel"))
        self.k_costs = _check(drv.cuModuleGetFunction(
            self.module, b"drm_reach_costs_kernel"))
        self.k_scripts = _check(drv.cuModuleGetFunction(
            self.module, b"drm_reach_scripts_kernel"))
        self._k_ws_probe = _check(drv.cuModuleGetFunction(
            self.module, b"drm_reach_ws_size_probe"))

        self.max_batch = int(max_batch)
        self.grid_blocks = self.n_sms * blocks_per_sm
        self.block_threads = block_threads

        self.stream = _check(drv.cuStreamCreate(0))

        # Transfer buffers: pinned host staging + device mirrors.
        inst_bytes = self.max_batch * INSTANCE_DTYPE.itemsize
        cost_bytes = self.max_batch * N_POSES * 2
        self._h_insts_ptr = _check(drv.cuMemHostAlloc(inst_bytes, 0))
        self._h_costs_ptr = _check(drv.cuMemHostAlloc(cost_bytes, 0))
        self.h_insts = np.ctypeslib.as_array(
            (np.ctypeslib.ctypes.c_char * inst_bytes).from_address(int(self._h_insts_ptr)),
        ).view(INSTANCE_DTYPE)
        self.h_costs = np.ctypeslib.as_array(
            (np.ctypeslib.ctypes.c_char * cost_bytes).from_address(int(self._h_costs_ptr)),
        ).view(np.uint16).reshape(self.max_batch, N_POSES)
        self.d_insts = _check(drv.cuMemAlloc(inst_bytes))
        self.d_costs = _check(drv.cuMemAlloc(cost_bytes))
        self.d_cursor = _check(drv.cuMemAlloc(8))

        # Workspace arena: one slot per resident block. Size queried from the
        # device struct itself so Python can never drift from the kernel ABI.
        probe = _check(drv.cuMemAlloc(8))
        self._launch(self._k_ws_probe, [probe], 1, 1)
        ws_size_buf = np.zeros(1, dtype=np.uint64)
        _check(drv.cuStreamSynchronize(self.stream))
        _check(drv.cuMemcpyDtoH(ws_size_buf.ctypes.data, probe, 8))
        drv.cuMemFree(probe)
        self.ws_size = int(ws_size_buf[0])
        self.d_ws = _check(drv.cuMemAlloc(self.grid_blocks * self.ws_size))
        # The solve kernels rely on visited/accum/flags starting zeroed; each
        # solve cleans up after itself (touched-list clearing).
        _check(drv.cuMemsetD8(self.d_ws, 0, self.grid_blocks * self.ws_size))

    # -- helpers ----------------------------------------------------------

    def _launch(self, kernel, args: list, grid: int, block: int, shmem: int = 0):
        """cuLaunchKernel with numpy-packed scalar/pointer args."""
        holders = []
        for a in args:
            if isinstance(a, (drv.CUdeviceptr,)):
                holders.append(np.array([int(a)], dtype=np.uint64))
            elif isinstance(a, int):
                holders.append(np.array([a], dtype=np.int32))
            else:
                raise TypeError(f"unsupported arg {a!r}")
        ptrs = np.array([h.ctypes.data for h in holders], dtype=np.uint64)
        _check(drv.cuLaunchKernel(
            kernel, grid, 1, 1, block, 1, 1, shmem, self.stream,
            ptrs.ctypes.data, 0))

    def _pack_instances(self, cols, parity, thr, *, sx=3, sy=0, srot=0, sc=0,
                        hv=0, hd=0, rh=0, max_frames=2048) -> int:
        cols = np.ascontiguousarray(cols, dtype=np.uint16)
        if cols.ndim != 2 or cols.shape[1] != GRID_W_COLS:
            raise ValueError(f"cols must be (n, 8) u16, got {cols.shape}")
        n = cols.shape[0]
        if n > self.max_batch:
            raise ValueError(f"batch {n} > max_batch {self.max_batch}; chunk it")
        h = self.h_insts[:n]
        h["cols"] = cols
        for name, val in (("sx", sx), ("sy", sy), ("srot", srot), ("sc", sc),
                          ("hv", hv), ("hd", hd), ("p", parity), ("rh", rh),
                          ("thr", thr)):
            h[name] = np.broadcast_to(np.asarray(val, dtype=np.uint8), (n,))
        h["flags"] = 0
        h["max_frames"] = np.broadcast_to(
            np.asarray(max_frames, dtype=np.uint16), (n,))
        h["_pad"] = 0
        return n

    # -- public API --------------------------------------------------------

    def solve_costs_identity(self, cols, parity, thr, **spawn) -> np.ndarray:
        """Stage-0 data-flow test path: runs the identity kernel."""
        n = self._pack_instances(cols, parity, thr, **spawn)
        _check(drv.cuMemsetD8Async(self.d_cursor, 0, 8, self.stream))
        _check(drv.cuMemcpyHtoDAsync(
            self.d_insts, self._h_insts_ptr, n * INSTANCE_DTYPE.itemsize, self.stream))
        grid = min(self.grid_blocks, n)
        self._launch(self.k_identity,
                     [self.d_insts, n, self.d_cursor, self.d_costs],
                     grid, self.block_threads)
        _check(drv.cuMemcpyDtoHAsync(
            self._h_costs_ptr, self.d_costs, n * N_POSES * 2, self.stream))
        _check(drv.cuStreamSynchronize(self.stream))
        return self.h_costs[:n].copy()

    def solve_costs(self, cols, parity, thr, **spawn) -> np.ndarray:
        """Exact planner costs, bit-identical to CPU drm_reach_bfs_v4.

        cols: (n, 8) u16 column bitboards; parity/thr: scalar or (n,) arrays;
        spawn kwargs (sx, sy, srot, sc, hv, hd, rh, max_frames) default to the
        annotation spawn. Returns (n, 512) u16 (0xFFFF = unreachable pose).
        """
        n = self._pack_instances(cols, parity, thr, **spawn)
        # The persistent kernel draws instances from one atomic queue. Real VS
        # batches mix nearly-empty reset boards with expensive stacked boards;
        # their solve times differ enough that input order otherwise leaves a
        # long, under-occupied tail. Cheap-first occupancy ordering keeps all
        # SMs fed. Stable order makes equal-complexity rows deterministic.
        order = None
        if n > self.grid_blocks:
            occupancy = np.bitwise_count(self.h_insts["cols"][:n]).sum(
                axis=1, dtype=np.uint16
            )
            order = np.argsort(occupancy, kind="stable")
            self.h_insts[:n] = self.h_insts[:n][order]
        _check(drv.cuMemsetD8Async(self.d_cursor, 0, 8, self.stream))
        _check(drv.cuMemcpyHtoDAsync(
            self.d_insts, self._h_insts_ptr, n * INSTANCE_DTYPE.itemsize, self.stream))
        grid = min(self.grid_blocks, n)
        self._launch(self.k_costs,
                     [self.d_insts, n, self.d_cursor, self.d_ws, self.d_costs],
                     grid, self.block_threads)
        _check(drv.cuMemcpyDtoHAsync(
            self._h_costs_ptr, self.d_costs, n * N_POSES * 2, self.stream))
        _check(drv.cuStreamSynchronize(self.stream))
        costs = self.h_costs[:n].copy()
        if order is not None:
            restored = np.empty_like(costs)
            restored[order] = costs
            return restored
        return costs

    SCRIPT_BUF_CAP = 24576
    PARENT_SLOT_U32 = 2 * 6336 * 8 * 64   # must match the kernel

    def _ensure_parent_arena(self):
        """Parent arena for the scripts re-BFS: ~26 MB per block slot,
        allocated on first solve_scripts call. No zeroing needed — parent
        walks are count-bounded and only follow entries written during the
        same solve."""
        if getattr(self, "d_parents", None) is None:
            self.d_parents = _check(drv.cuMemAlloc(
                self.grid_blocks * self.PARENT_SLOT_U32 * 4))
        return self.d_parents

    def solve_scripts(self, cols, parity, thr, **spawn):
        """Costs plus an optimal input script per macro-legal reachable pose.

        Returns (costs (n,512)u16, offsets (n,512)u16, lengths (n,512)u16,
        scripts (n, 24576)u8, status (n,)i32). status != 0 => fall back to the
        CPU planner for that instance (bit 1: some pose had no greedy-matched
        script; bit 2: script buffer overflow; bit 4: parity alarm — a greedy
        rollout beat the "exact" cost, which should be impossible).

        Scripts are optimal by construction (frame count == exact cost) but
        not byte-identical to CPU v1's tie-break choice; verify by replay.
        """
        n = self._pack_instances(cols, parity, thr, **spawn)
        d_parents = self._ensure_parent_arena()
        d_off = _check(drv.cuMemAlloc(n * N_POSES * 2))
        d_len = _check(drv.cuMemAlloc(n * N_POSES * 2))
        d_scr = _check(drv.cuMemAlloc(n * self.SCRIPT_BUF_CAP))
        d_st = _check(drv.cuMemAlloc(n * 4))
        try:
            _check(drv.cuMemsetD8Async(self.d_cursor, 0, 8, self.stream))
            _check(drv.cuMemcpyHtoDAsync(
                self.d_insts, self._h_insts_ptr, n * INSTANCE_DTYPE.itemsize, self.stream))
            grid = min(self.grid_blocks, n)
            self._launch(self.k_scripts,
                         [self.d_insts, n, self.d_cursor, self.d_ws, d_parents,
                          self.d_costs, d_off, d_len, d_scr, d_st],
                         grid, self.block_threads)
            costs = np.empty((n, N_POSES), dtype=np.uint16)
            off = np.empty((n, N_POSES), dtype=np.uint16)
            length = np.empty((n, N_POSES), dtype=np.uint16)
            scr = np.empty((n, self.SCRIPT_BUF_CAP), dtype=np.uint8)
            st = np.empty(n, dtype=np.int32)
            _check(drv.cuStreamSynchronize(self.stream))
            _check(drv.cuMemcpyDtoH(costs.ctypes.data, self.d_costs, costs.nbytes))
            _check(drv.cuMemcpyDtoH(off.ctypes.data, d_off, off.nbytes))
            _check(drv.cuMemcpyDtoH(length.ctypes.data, d_len, length.nbytes))
            _check(drv.cuMemcpyDtoH(scr.ctypes.data, d_scr, scr.nbytes))
            _check(drv.cuMemcpyDtoH(st.ctypes.data, d_st, st.nbytes))
            return costs, off, length, scr, st
        finally:
            for d in (d_off, d_len, d_scr, d_st):
                drv.cuMemFree(d)

    def debug_phase12(self, cols, parity, thr, gd_cap: int = 128, **spawn):
        """Run phases 1+2 only; returns (wanted u8 (n,512), ub u16 (n,512),
        gd u8 (n,gd_cap,512), n_wanted i32 (n,)) for parity testing."""
        n = self._pack_instances(cols, parity, thr, **spawn)
        d_wanted = _check(drv.cuMemAlloc(n * N_POSES))
        d_ub = _check(drv.cuMemAlloc(n * N_POSES * 2))
        d_gd = _check(drv.cuMemAlloc(n * gd_cap * N_POSES))
        d_nw = _check(drv.cuMemAlloc(n * 4))
        try:
            _check(drv.cuMemsetD8Async(self.d_cursor, 0, 8, self.stream))
            _check(drv.cuMemcpyHtoDAsync(
                self.d_insts, self._h_insts_ptr, n * INSTANCE_DTYPE.itemsize, self.stream))
            grid = min(self.grid_blocks, n)
            self._launch(self.k_debug12,
                         [self.d_insts, n, self.d_cursor, self.d_ws,
                          d_wanted, d_ub, d_gd, gd_cap, d_nw],
                         grid, self.block_threads)
            wanted = np.empty((n, N_POSES), dtype=np.uint8)
            ub = np.empty((n, N_POSES), dtype=np.uint16)
            gd = np.empty((n, gd_cap, N_POSES), dtype=np.uint8)
            nw = np.empty(n, dtype=np.int32)
            _check(drv.cuStreamSynchronize(self.stream))
            _check(drv.cuMemcpyDtoH(wanted.ctypes.data, d_wanted, wanted.nbytes))
            _check(drv.cuMemcpyDtoH(ub.ctypes.data, d_ub, ub.nbytes))
            _check(drv.cuMemcpyDtoH(gd.ctypes.data, d_gd, gd.nbytes))
            _check(drv.cuMemcpyDtoH(nw.ctypes.data, d_nw, nw.nbytes))
            return wanted, ub, gd, nw
        finally:
            for d in (d_wanted, d_ub, d_gd, d_nw):
                drv.cuMemFree(d)

    def close(self):
        for p in ("d_insts", "d_costs", "d_cursor", "d_ws", "d_parents"):
            if getattr(self, p, None) is not None:
                drv.cuMemFree(getattr(self, p))
                setattr(self, p, None)
        for p in ("_h_insts_ptr", "_h_costs_ptr"):
            if getattr(self, p, None) is not None:
                drv.cuMemFreeHost(getattr(self, p))
                setattr(self, p, None)
        if getattr(self, "stream", None) is not None:
            drv.cuStreamDestroy(self.stream)
            self.stream = None
        if getattr(self, "module", None) is not None:
            drv.cuModuleUnload(self.module)
            self.module = None
        if getattr(self, "ctx", None) is not None:
            drv.cuDevicePrimaryCtxRelease(self.dev)
            self.ctx = None
