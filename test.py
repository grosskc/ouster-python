import json
import time
from datetime import datetime as dt
import signal
from multiprocessing import Process, Queue
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from os1 import OS1
from os1.utils import build_trig_table, xyz_points, build_raw_table, raw_points


OS1_IP = '192.168.1.100'
HOST_IP = '192.168.1.2'
TICKS_PER_REVOLUTION = 90112
WORKERS = 6
TIME = 20
unprocessed_packets = Queue()

def timeout_handler(signum, frame):
    print("Data collection is over")
    raise Exception("Data Collection Timer")


def format(L):
    out = ""
    for val in L: out += f"{val}, "
    return out[:-2]


def handler(packet):
    unprocessed_packets.put(packet)


def worker(FN_PREFIX, queueID, queue, beam_altitude_angles, beam_azimuth_angles, DURATION=TIME+2):
    build_raw_table(beam_altitude_angles, beam_azimuth_angles)
    start_time = time.time()
    delta = 0
    data = []
    while delta < DURATION:
        packet = queue.get()
        data.append(raw_points(packet))
        delta = time.time() - start_time
    # data = np.array(data)
    names = ["FrameID", "MeasurementID", "ChannelID", "Timestamp", "EncoderPosition", "Range", "Reflectivity", "Signal", "Noise"]
    dt = [np.uint16, np.uint16, np.uint16, np.uint64, np.uint32, np.uint32, np.uint16, np.uint16, np.uint16]
    dtypes0 = [(n,np.uint64) for n in names]
    dtypes = {"names": names, "formats": dt, "aligned": True}
    D = np.array(data).reshape((-1,len(data[0]))).ravel().view(dtypes0).astype(dtypes)
    D = np.sort(D, order=names[:3])
    np.savez(f"{FN_PREFIX}-DATA-Q{queueID}.npz", data=D)


def spawn_workers(n, FN_PREFIX, worker, *args, **kwargs):
    processes = []
    for i in range(n):
        w = partial(worker, FN_PREFIX, i)
        process = Process(target=w, args=args, kwargs=kwargs)
        process.start()
        processes.append(process)
    return processes


os1 = OS1(OS1_IP, HOST_IP)
beam_intrinsics = json.loads(os1.get_beam_intrinsics())
beam_alt_angles = beam_intrinsics['beam_altitude_angles']
beam_az_angles = beam_intrinsics['beam_azimuth_angles']

print("Configuring LIDAR")
os1.start()
time.sleep(25)

FN_PREFIX = dt.now().strftime('%Y%m%d-%H%M%S')
workers = spawn_workers(WORKERS, FN_PREFIX, worker, unprocessed_packets, beam_alt_angles, beam_az_angles)
print("Collecting Data")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TIME+45)
try:
    os1.run_forever(handler)
except Exception:
    for w in workers:
        w.terminate()
    os1._server.shutdown()
    del os1





# def frame_builder(data, nPts=2048*64):
#     dims = data.shape
#     data = np.reshape(data, (np.prod(dims[:-1]),-1))
#     frames = np.unique(data[:,0])[1:-1]
#     ix = (data[:,0] >= np.min(frames)) & (data[:,0] <= np.max(frames))
#     data = data[ix]
#     F = []
#     for f in frames:
#         ix = data[:, 0] == f
#         D = data[ix]
#         if len(D) == nPts:
#             ix = np.lexsort((D[:,1], D[:,4]))
#             F.append(D[ix])
#     return F

names = ["FrameID", "MeasurementID", "ChannelID", "Timestamp", "EncoderPosition", "Range", "Reflectivity", "Signal", "Noise"]

def frame_builder(data, nPts=2048*64):
    frames = np.unique(data["FrameID"])[1:-1]
    ix = (data["FrameID"] > np.min(frames)) & (data["FrameID"] < np.max(frames))
    data = data[ix]
    F = []
    for f in frames:
        ix = data["FrameID"] == f
        if np.sum(ix) == nPts:
            F.append(np.sort(data[ix],order=names[1:3]))
    return F


# Load data chunks from each worker and build frames
# FN_PREFIX = "20200320-171134"
FN_PREFIX = "20200321-160639"
D = []
for i in range(WORKERS):
    fn = f"{FN_PREFIX}-DATA-Q{i}.npz"
    tmp = np.load(fn)
    D.append(tmp["data"])
D = np.concatenate(D)
F = np.array(frame_builder(D))

OS1_AZ = np.array(beam_az_angles)
OS1_EL = np.array(beam_alt_angles)
ix_ch = np.uint16(F[0]["ChannelID"])
phi = 2*np.pi * (F[0]["EncoderPosition"] / TICKS_PER_REVOLUTION + OS1_AZ[ix_ch]/360)
theta = 2*np.pi * OS1_EL[ix_ch]/360

R = np.mean(F[:]["Range"],axis=0)
Rs = np.std(F[:]["Range"],axis=0)
C = np.mean(F[:]["Reflectivity"],axis=0)
Cs = np.std(F[:]["Reflectivity"],axis=0)
x = +R * np.cos(theta) * np.cos(phi)
y = -R * np.cos(theta) * np.sin(phi)
z = +R * np.sin(theta)


ix = (Rs > 0) & (Rs < np.quantile(Rs,0.95)) & (C > np.quantile(C,0.05)) & (C < np.quantile(C,0.95))
x=x[ix]; y=y[ix]; z=z[ix]; C=C[ix]

P = np.vstack((x,y,z)).T
C = (C-np.min(C)) / np.max(C-np.min(C))
c = np.vstack((C,C,C)).T
P = np2o3d(P, c)
Q, ind = P.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
np2ply(f"{FN_PREFIX}-DATA.ply", Q) 


def make_axes(origin=[0, 0, 0], sf=1.0, T=np.identity(4)):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=sf, origin=origin).transform(T)

ax = make_axes(sf=1000)
o3d.visualization.draw_geometries([Q, ax])




# Signal average
Fm = np.mean(F,axis=0)
Fs = np.std(F,axis=0)

# Convert to point cloud
OS1_AZ = np.array(beam_az_angles)
OS1_EL = np.array(beam_alt_angles)
ix_ch = np.uint16(Fm[:,4])
phi = 2*np.pi * (np.uint32(Fm[:,3]) / TICKS_PER_REVOLUTION + OS1_AZ[ix_ch]/360)
theta = 2*np.pi * OS1_EL[ix_ch]/360
R = Fm[:, 5]
C = Fm[:, -1]
x = +R * np.cos(theta) * np.cos(phi)
y = -R * np.cos(theta) * np.sin(phi)
z = +R * np.sin(theta)

err = Fs[:, 5]
ix = (err > 0) & (err < np.quantile(err,0.95)) & (C > np.quantile(C,0.05)) & (C < np.quantile(C,0.95))
x=x[ix]; y=y[ix]; z=z[ix]; C=C[ix]

P = np.vstack((x,y,z)).T
C = (C-np.min(C)) / np.max(C-np.min(C))
c = np.vstack((C,C,C)).T
P = np2o3d(P, c)
Q, ind = P.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
np2ply(f"{FN_PREFIX}-DATA.ply", Q) 

def make_axes(origin=[0, 0, 0], sf=1.0, T=np.identity(4)):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=sf, origin=origin).transform(T)

ax = make_axes(sf=1000)
o3d.visualization.draw_geometries([Q, ax])

import open3d as o3d

def np2o3d(pts, C=None):
    """
    Convert a numpy array of shape (N,3) into an Open3D point cloud.

    Parameters
    ----------
    pts : array like
        numpy array of shape (N,3) where N is number of points, and each point is ordered as (x,y,z)

    Returns
    -------
    pcd : Open3D
        Open3D representation of an unstructured point cloud
    """
    if type(pts) is not o3d.open3d.geometry.PointCloud:
        pts = np.asanyarray(pts)
        pcd_ = o3d.geometry.PointCloud()
        pcd_.points = o3d.utility.Vector3dVector(pts)
        if C is not None:
            pcd_.colors = o3d.utility.Vector3dVector(C)
        return pcd_
    else:
        return pts


def np2ply(fname, pts, write_ascii=True):
    """
    Save an unstructured point cloud in the PLY format.

    Parameters
    ----------
    fname : string
        filename to save the data to
    pts : array like
        numpy array of shape (N,3)

    Returns
    -------
    None
    """
    pcd = np2o3d(pts)
    o3d.io.write_point_cloud(fname, pcd, write_ascii=write_ascii)


np2ply(f"{FN_PREFIX}-DATA.ply", P)




fn = lambda i: f"{FN_PREFIX}-DATA-Q{i}.csv"
tmp=np.loadtxt(fn(0), dtype=np.uint64, delimiter=",")
types = [np.uint16, np.uint16, np.uint64, np.uint32, np.uint16, np.uint32, np.uint16]
names = ["FrameID", "MeasurementID", "Timestamp", "EncoderPosition", "ChannelID", "Range", "Reflectivity"]
data = np.concatenate(tuple(np.genfromtxt(fn(i), dtype=types, delimiter=',', names=names) for i in range(WORKERS)))
data = np.sort(data, order=['FrameID', 'EncoderPosition', 'ChannelID'])

frames = np.unique(data["FrameID"])[1:-1] # drop first and last frames as they'll be partial
F = []
for f in frames:
    ix = data["FrameID"] == f
    D = np.sort(data[ix],order=["MeasurementID", "ChannelID"])
    F.append(D)
D = np.concatenate(F)

MID = np.uint32(1+data["MeasurementID"])
CID = np.uint32(1+data["ChannelID"])
UID = MID * CID

UUID = np.unique(UID)
R = data["Range"]
Rm = np.zeros(UUID.shape)
for i,u in enumerate(UUID):
    ix = UID == u
    Rm[i] = np.mean(R[ix])
    if i % 100 == 0: print(i)


OS1_AZ = np.array(beam_az_angles)
phi = 2*np.pi * (data["EncoderPosition"] / TICKS_PER_REVOLUTION + OS1_AZ[data["ChannelID"]]/360)