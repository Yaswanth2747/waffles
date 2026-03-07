# network_udp.py

import struct
import socket
import time

# Keep payload below common Ethernet MTU after UDP/IP headers to avoid IP fragmentation drops.
CHUNK_SIZE = 1400
HEADER_FMT = "II"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
SEND_PACING_SEC = 0.0002


def send_data(sock, payload, addr):

    payload_size = len(payload)

    total_packets = (payload_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    for packet_id in range(total_packets):

        start = packet_id * CHUNK_SIZE
        end = start + CHUNK_SIZE

        chunk = payload[start:end]

        header = struct.pack(HEADER_FMT, packet_id, total_packets)

        sock.sendto(header + chunk, addr)
        # Small pacing avoids overrunning kernel/network buffers during large bursts.
        time.sleep(SEND_PACING_SEC)


def receive_data(sock):

    packets = {}

    first_packet = True
    total_packets = None

    while True:
        try:
            packet, addr = sock.recvfrom(CHUNK_SIZE + HEADER_SIZE + 100)
        except socket.timeout as exc:
            missing = "unknown"
            if total_packets is not None:
                missing = total_packets - len(packets)
            raise TimeoutError(
                f"Timed out while receiving UDP chunks, missing packets: {missing}"
            ) from exc

        packet_id, pkt_total = struct.unpack(
            HEADER_FMT, packet[:HEADER_SIZE]
        )

        if first_packet:
            t_first = time.perf_counter()
            total_packets = pkt_total
            first_packet = False

        payload = packet[HEADER_SIZE:]

        packets[packet_id] = payload
        if len(packets) == total_packets:
            break

    t_last = time.perf_counter()

    T_comm = t_last - t_first

    ordered = [packets[i] for i in range(total_packets)]

    data = b"".join(ordered)

    return data, addr, T_comm
