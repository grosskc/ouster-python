import math
import struct
import sys

import numpy as np

from os1.packet import (
    AZIMUTH_BLOCK_COUNT,
    CHANNEL_BLOCK_COUNT,
    azimuth_block,
    azimuth_encoder_count,
    azimuth_frame_id,
    azimuth_measurement_id,
    azimuth_timestamp,
    azimuth_valid,
    channel_block,
    channel_range,
    channel_reflectivity,
    channel_signal_photons,
    channel_noise_photons,
    unpack,
)

# TODO This will require modification if we use a OS1-128
# The OS-16 will still contain 64 channels in the packet, but only
# every 4th channel starting at the 2nd will contain data .
OS_16_CHANNELS = (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62)
OS_64_CHANNELS = tuple(i for i in range(CHANNEL_BLOCK_COUNT))

RAW_NAMES = ["FrameID", "AzimuthID", "ChannelID", "Timestamp", "EncoderPosition", "Range", "Reflectivity", "Signal", "Noise"]
RAW_DTYPE = [np.uint16, np.uint16, np.uint16, np.uint64, np.uint32, np.uint32, np.uint16, np.uint16, np.uint16]

def raw_point(ChannelID, azimuth_block):
    """
    Returns a raw data point for a given `ChannelID` and `AzimuthID` within
    a single packet.

    Parameters
    ----------
    ChannelID : uint16
        number between 0 and `N_CHANNEL`-1 identifying the Ouster channel
    azimuth_block : list of packet data
        raw packet data from the azimuth block for processing

    Returns
    -------
    data : list
        [FrameID, AzimuthID, ChannelID, Timestamp, EncoderPosition, Range, Reflectivity, Signal, Noise]
    """
    channel = channel_block(ChannelID, azimuth_block)
    AzimuthID = azimuth_measurement_id(azimuth_block)
    FrameID = azimuth_frame_id(azimuth_block)
    Timestamp = azimuth_timestamp(azimuth_block)
    EncoderPosition = azimuth_encoder_count(azimuth_block)
    Range = channel_range(channel) # [mm]
    Reflectivity = channel_reflectivity(channel)
    Signal = channel_signal_photons(channel)
    Noise = channel_noise_photons(channel)

    return [FrameID, AzimuthID, ChannelID, Timestamp, EncoderPosition, Range, Reflectivity, Signal, Noise]


def raw_points(packet, os16=False):
    """
    Returns a list of raw data point lists for a given packet.

    Parameters
    ----------
    packet : bytearray
        UDP raw packet data
    azimuth_block : list of packet data
        raw packet data from the azimuth block for processing

    Returns
    -------
    data : list
        [FrameID, AzimuthID, ChannelID, Timestamp, EncoderPosition, Range, Reflectivity, Signal, Noise]
    """
    channels = OS_16_CHANNELS if os16 else OS_64_CHANNELS
    if not isinstance(packet, tuple):
        packet = unpack(packet)

    raw = []
    for b in range(AZIMUTH_BLOCK_COUNT):
        block = azimuth_block(b, packet)
        if azimuth_valid(block):
            for c in channels: 
                raw.append(raw_point(c, block))
    return raw


_unpack = struct.Struct("<I").unpack


def peek_encoder_count(packet):
    return _unpack(packet[12:16])[0]


def frame_handler(queue):
    """
    Handler that buffers packets until it has a full frame then puts
    them into a queue. Queue must have a put method.

    The data put into the queue will be a dict that contains a buffer
    of packets making up a frame and the rotation number.

        {
            'buffer': [packet1, packet2, ...],
            'rotation': 1
        }
    """
    buffer = []
    rotation_num = 0
    sentinel = None
    last = None

    def handler(packet):
        nonlocal rotation_num, sentinel, last, buffer

        encoder_count = peek_encoder_count(packet)
        if sentinel is None:
            sentinel = encoder_count

        if buffer and last and encoder_count >= sentinel and last <= sentinel:
            rotation_num += 1
            queue.put({"buffer": buffer, "rotation": rotation_num})
            buffer = []

        buffer.append(packet)
        last = encoder_count

    return handler
