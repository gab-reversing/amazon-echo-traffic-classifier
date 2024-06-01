import os
import numpy as np
from scapy.all import rdpcap, TCP
from joblib import load
import sys

def extract_features(pcap_file):
    """
    Extracts features from a pcap file.

    Args:
        pcap_file (str): Path to the pcap file.

    Returns:
        dict: Dictionary containing extracted features.
    """
    packets = rdpcap(pcap_file)
    total_packets = 0
    ack_pkts_sent = 0
    pure_acks_sent = 0
    unique_bytes_sent = 0
    actual_data_pkts = 0
    actual_data_bytes = 0
    pushed_data_pkts = 0
    segment_sizes = []
    window_adv_sizes = []
    initial_window_bytes = 0
    data_xmit_start_time = None
    data_xmit_end_time = None
    idletime_max = 0
    last_packet_time = None
    for packet in packets:
        if TCP in packet:
            total_packets += 1
            tcp_layer = packet[TCP]
            packet_time = packet.time
            if data_xmit_start_time is None:
                data_xmit_start_time = packet_time
            data_xmit_end_time = packet_time
            if last_packet_time is not None:
                idle_time = packet_time - last_packet_time
                if idle_time > idletime_max:
                    idletime_max = idle_time
            last_packet_time = packet_time
            if tcp_layer.flags & 0x10:
                ack_pkts_sent += 1
                if len(tcp_layer.payload) == 0:
                    pure_acks_sent += 1
            if len(tcp_layer.payload) > 0:
                unique_bytes_sent += len(tcp_layer.payload)
                actual_data_pkts += 1
                actual_data_bytes += len(tcp_layer.payload)
            if tcp_layer.flags & 0x08:
                pushed_data_pkts += 1
            segment_sizes.append(len(tcp_layer.payload))
            window_adv_sizes.append(tcp_layer.window)
            if total_packets == 1:
                initial_window_bytes = tcp_layer.window
    max_segm_size = max(segment_sizes) if segment_sizes else 0
    min_segm_size = min(segment_sizes) if segment_sizes else 0
    avg_segm_size = np.mean(segment_sizes) if segment_sizes else 0
    max_win_adv = max(window_adv_sizes) if window_adv_sizes else 0
    min_win_adv = min(window_adv_sizes) if window_adv_sizes else 0
    avg_win_adv = np.mean(window_adv_sizes) if window_adv_sizes else 0
    data_xmit_time = data_xmit_end_time - data_xmit_start_time if data_xmit_start_time and data_xmit_end_time else 0
    throughput = unique_bytes_sent / data_xmit_time if data_xmit_time > 0 else 0
    features = {
        "total_packets": total_packets,
        "ack_pkts_sent": ack_pkts_sent,
        "pure_acks_sent": pure_acks_sent,
        "unique_bytes_sent": unique_bytes_sent,
        "actual_data_pkts": actual_data_pkts,
        "actual_data_bytes": actual_data_bytes,
        "pushed_data_pkts": pushed_data_pkts,
        "max_segm_size": max_segm_size,
        "min_segm_size": min_segm_size,
        "avg_segm_size": avg_segm_size,
        "max_win_adv": max_win_adv,
        "min_win_adv": min_win_adv,
        "avg_win_adv": avg_win_adv,
        "initial_window_bytes": initial_window_bytes,
        "data_xmit_time": data_xmit_time,
        "idletime_max": idletime_max,
        "throughput": throughput
    }
    return features

def load_model(model_filename):
    """
    Loads a trained model from a file.

    Args:
        model_filename (str): Name of the file containing the trained model.

    Returns:
        object: Trained model object.
    """
    return load(model_filename)

def prepare_features(features_dict):
    """
    Prepares features for classification.

    Args:
        features_dict (dict): Dictionary containing extracted features.

    Returns:
        numpy.ndarray: Array containing prepared features.
    """
    return np.array([
        features_dict["total_packets"],
        features_dict["ack_pkts_sent"],
        features_dict["pure_acks_sent"],
        features_dict["unique_bytes_sent"],
        features_dict["actual_data_pkts"],
        features_dict["actual_data_bytes"],
        features_dict["pushed_data_pkts"],
        features_dict["max_segm_size"],
        features_dict["min_segm_size"],
        features_dict["avg_segm_size"],
        features_dict["max_win_adv"],
        features_dict["min_win_adv"],
        features_dict["avg_win_adv"],
        features_dict["initial_window_bytes"],
        features_dict["data_xmit_time"],
        features_dict["idletime_max"],
        features_dict["throughput"]
    ]).reshape(1, -1)

def classify_pcap(pcap_file, clf):
    """
    Classifies a pcap file using a trained classifier.

    Args:
        pcap_file (str): Path to the pcap file to classify.
        clf (object): Trained classifier object.

    Returns:
        str: Predicted class.
    """
    features_dict = extract_features(pcap_file)
    features = prepare_features(features_dict)
    predicted_class = clf.predict(features)
    return predicted_class[0]

def main():
    """
    Main function to classify a pcap file using a trained model.

    The function expects a command-line argument specifying the path to the pcap file to classify.

    Usage: python classify_pcap.py <pcap_file>
    """
    if len(sys.argv) != 2:
        print("Usage: python classify_pcap.py <pcap_file>")
        sys.exit(1)
    pcap_file = sys.argv[1]
    if not os.path.isfile(pcap_file):
        print(f"File not found: {pcap_file}")
        sys.exit(1)
    model_filename = 'pcap_classifier.joblib'
    clf = load_model(model_filename)
    result = classify_pcap(pcap_file, clf)
    print(f"The predicted class for the pcap file is: {result}")

if __name__ == "__main__": main()