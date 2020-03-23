"""
Routine to record data from Ouster UDP stream, build frames, and signal average.
"""

# Imports
import argparse
import json
import time
from datetime import datetime as dt
import logging
import signal
from multiprocessing import Process, Queue
from functools import partial
from pathlib import Path
import threading

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import open3d as o3d

from os1 import OS1
from os1.utils import raw_points, RAW_NAMES, RAW_DTYPE


# -----------------------------------------------------------------------------
# Build packet queue and workers to process packets
# -----------------------------------------------------------------------------

unprocessed_packets = Queue()


def handler(packet):
    unprocessed_packets.put(packet)


def worker(FN_PREFIX, TIME, queueID, queue):

    # start timer
    DURATION = TIME + 2
    start_time = time.time()
    delta = 0

    # collect packets for specified time
    data = []
    while delta < DURATION:
        packet = queue.get()
        data.append(raw_points(packet))
        delta = time.time() - start_time

    # save packet data in a sorted, structured array
    dt0 = [(n, np.uint64) for n in RAW_NAMES]
    dt = {"names": RAW_NAMES, "formats": RAW_DTYPE, "aligned": True}
    D = np.array(data).reshape((-1, len(data[0]))).ravel().view(dt0).astype(dt)
    D = np.sort(D, order=RAW_NAMES[:3])
    np.savez(f"{FN_PREFIX}-DATA-Q{queueID}.npz", data=D)


def spawn_workers(n, FN_PREFIX, TIME, worker, *args, **kwargs):
    processes = []
    for i in range(n):
        w = partial(worker, FN_PREFIX, TIME, i)
        process = Process(target=w, args=args, kwargs=kwargs)
        process.start()
        processes.append(process)
    return processes


# -----------------------------------------------------------------------------
# Setup Ouster
# -----------------------------------------------------------------------------
MODES = ("512x10", "512x20", "1024x10", "1024x20", "2048x10")
TICKS_PER_REVOLUTION = 90112

def setup_Ouster(OS_IP="192.168.1.100", HOST_IP="192.168.1.2", H_RES=2048, V_RES=64):

    MODE = f"{H_RES}x10"
    os1 = OS1(OS_IP, HOST_IP, udp_port=7502, tcp_port=7501, mode=MODE)

    print("Configuring LIDAR")
    os1.start()
    
    ts = 0.1  # [s]
    for i in tqdm(range(int(25/ts)), desc="Configuring OS1"):
        time.sleep(ts)

    AZ = np.array(os1._beam_intrinsics["beam_azimuth_angles"])
    EL = np.array(os1._beam_intrinsics["beam_altitude_angles"])

    FN_PREFIX = dt.now().strftime("%Y%m%d-%H%M%S")

    return os1, FN_PREFIX, H_RES, V_RES, AZ, EL, MODE


# -----------------------------------------------------------------------------
# Data collector
# -----------------------------------------------------------------------------
def data_collector(os1, TIME, WORKERS):
    print("Collecting Data")

    workers = spawn_workers(A.n_workers, FN_PREFIX, A.time, worker, unprocessed_packets)
    # set timer used to kill thread 1 minute after data collection is done
    TIME_DELTA = 60

    try:
        os1.run_forever(handler)
    except KeyboardInterrupt:
        for w in workers:
            w.terminate()
        os1._server.shutdown()
        time.sleep(3)
        del os1


# -----------------------------------------------------------------------------
# Build frames
# -----------------------------------------------------------------------------


def frame_builder(data, nPts=2048 * 64):
    frames = np.unique(data["FrameID"])[1:-1]
    ix = (data["FrameID"] > np.min(frames)) & (data["FrameID"] < np.max(frames))
    data = data[ix]
    F = []
    for f in frames:
        ix = data["FrameID"] == f
        if np.sum(ix) == nPts:  # only append full frames
            F.append(np.sort(data[ix], order=RAW_NAMES[1:3]))
    return F


def point_cloud_maker(AZ, EL, FN_PREFIX, N_WORKERS, MAX_COV=0.05, fname=None):
    # Load data chunks from each worker and build frames
    D = []
    for i in range(N_WORKERS):
        fn = f"{FN_PREFIX}-DATA-Q{i}.npz"
        print(f"Loading {fn}")
        tmp = np.load(fn)
        D.append(tmp["data"])
    D = np.concatenate(D)
    print("Building frames")
    F = np.array(frame_builder(D))
    if fname is None:
        fname_all = f"{FN_PREFIX}-OS1-DATA-ALL.npz"
    else:
        fname_all = Path(fname).resolve().stem + "-ALL.npz"

    # Build point cloud colored by stretched reflectivity using signal-averaged ranges
    ix_ch = F[0]["ChannelID"]
    phi = 2 * np.pi * (F[0]["EncoderPosition"] / TICKS_PER_REVOLUTION + AZ[ix_ch] / 360)
    theta = 2 * np.pi * EL[ix_ch] / 360
    np.savez(fname_all, Frames=F, Azimuth=phi, Elevation=theta)
    
    Rm = np.mean(F[:]["Range"], axis=0) / 1000  # [m] from [mm]
    Rs = np.std(F[:]["Range"], axis=0) / 1000  # [m] from [mm]
    Cm = np.mean(F[:]["Reflectivity"], axis=0)
    Cs = np.std(F[:]["Reflectivity"], axis=0)
    x = +Rm * np.cos(theta) * np.cos(phi)
    y = -Rm * np.cos(theta) * np.sin(phi)
    z = +Rm * np.sin(theta)

    # Filter out noisy / bad points based on Coefficient of Variation (COV)
    COV_R = Rs / Rm
    COV_C = Cs / Cm
    ix = (Rs > 0) & (Rm > 0) & np.isfinite(COV_R + COV_C) & (COV_R < MAX_COV) & (COV_C < MAX_COV)
    x = x[ix]
    y = y[ix]
    z = z[ix]
    C = Cm[ix]

    # Convert to ply format and color by stretched reflectivity
    PC = np.vstack((x, y, z)).T
    C -= np.quantile(C, 0.025)
    C /= np.quantile(C, 0.975)
    C[C<0] = 0
    C[C>1] = 1
    P = o3d.geometry.PointCloud()
    P.points = o3d.utility.Vector3dVector(PC)
    P.colors = o3d.utility.Vector3dVector(np.vstack((C, C, C)).T)
    Q, _ = P.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
    if fname is None: fname = f"{FN_PREFIX}-OS1-DATA.ply"
    o3d.io.write_point_cloud(fname, Q, write_ascii=False)

    return Q, fname, fname_all


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def files_exist(FN_PREFIX, W):
    test = []
    for i in range(W):
        fn = f"{FN_PREFIX}-DATA-Q{i}.npz"
        test.append(Path(fn).is_file())
    return all(test)

def build_parser():
    DATE_TIME = dt.now().strftime("%Y%m%d-%H%M%S")
    fname_default = f"{DATE_TIME}-OS1-DATA.ply"
    parser = argparse.ArgumentParser(
        description="Ouster OS1 Data Collector and Processor"
    )
    parser.add_argument(
        "--os_IP", type=str, default="192.168.1.100", help="Ouster IP address",
    )
    parser.add_argument(
        "--host_IP", type=str, default="192.168.1.2", help="Host IP address",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4, help="Number of workers",
    )
    parser.add_argument(
        "--time", type=int, default=10, help="Duration of data collection in seconds",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=None,
        help='File name for storing results (default = "YYYYMMDD-hhmmss-OS1-DATA.ply"',
    )
    parser.add_argument(
        "--h_res", type=int, default=2048, help="Horizontal resolution (default=2048)",
    )
    parser.add_argument(
        "--v_res", type=int, default=64, help="Vertical resolution (default=64)",
    )
    parser.add_argument(
        "--max_CoV", type=float, default=0.1, help="Maximum coefficient of variation (default=0.1)",
    )
    parser.add_argument(
        "--viz",
        dest="visualize",
        action="store_false",
        help="Visualize the aligned point clouds (default = False)",
    )

    return parser


if __name__ == "__main__":
    # build parser and parse command-line arguments
    parser = build_parser()
    A = parser.parse_args()

    os1, FN_PREFIX, H_RES, V_RES, AZ, EL, MODE = \
        setup_Ouster(OS_IP=A.os_IP, HOST_IP=A.host_IP, H_RES=A.h_res, V_RES=A.v_res)

    th = threading.Thread(target=data_collector, args=(os1, A.time, A.n_workers), daemon=True)
    th.start()

    ts = 0.1  # [s]
    for i in tqdm(range(int((A.time+5)/ts)), desc="Collecting data"):
        time.sleep(ts)

    for i in tqdm(range(int(25/ts)), desc="Saving data"):
        time.sleep(ts)
        if files_exist(FN_PREFIX, A.n_workers):
            break
    
    time.sleep(3)
    th.join(timeout=2)

    Q, fname, fname_all = \
        point_cloud_maker(AZ, EL, FN_PREFIX, A.n_workers, MAX_COV=A.max_CoV, fname=A.fname)

    print(f"Time-averaged data saved to {fname}; Full dataset saved to {fname_all}")

    if A.visualize:
        ax = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([Q, ax])

