import os
import numpy as np
from scapy.all import rdpcap, TCP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

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
    if segment_sizes:
        max_segm_size = max(segment_sizes)
        min_segm_size = min(segment_sizes)
        avg_segm_size = np.mean(segment_sizes)
    else:
        max_segm_size = min_segm_size = avg_segm_size = 0
    if window_adv_sizes:
        max_win_adv = max(window_adv_sizes)
        min_win_adv = min(window_adv_sizes)
        avg_win_adv = np.mean(window_adv_sizes)
    else:
        max_win_adv = min_win_adv = avg_win_adv = 0
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

def create_dataset(base_folder):
    """
    Creates a dataset from pcap files in the specified folder.

    Args:
        base_folder (str): Path to the base folder containing pcap files.

    Returns:
        tuple: Tuple containing data and labels.
    """
    data = []
    labels = []
    current_progress = 0
    total_files = len(os.listdir(base_folder)) * 500
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            label = folder
            for pcap_file in os.listdir(folder_path):
                current_progress += 1
                print(f'{current_progress}/{total_files}')
                if pcap_file.endswith('.pcap'):
                    pcap_path = os.path.join(folder_path, pcap_file)
                    features_dict = extract_features(pcap_path)
                    features = [
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
                    ]
                    data.append(features)
                    labels.append(label)
    return np.array(data), np.array(labels)

def train_model(X_train, y_train):
    """
    Trains a random forest classifier.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.

    Returns:
        RandomForestClassifier: Trained classifier.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Evaluates a classifier using test data.

    Args:
        clf (RandomForestClassifier): Trained classifier.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
    """
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def save_model(clf, filename):
    """
    Saves a trained model to a file.

    Args:
        clf (RandomForestClassifier): Trained classifier.
        filename (str): Name of the file to save the model.
    """
    dump(clf, filename)
    print(f"Model saved as {filename}")

def main(base_folder='data', model_filename='pcap_classifier.joblib'):
    """
    Main function to create dataset, train model, evaluate model, and save model.

    Args:
        base_folder (str): Path to the base folder containing pcap files.
        model_filename (str): Name of the file to save the trained model.
    """
    print('Creating dataset..')
    data, labels = create_dataset(base_folder)
    print('Splitting data..')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42, stratify=labels)
    print('Training model..')
    clf = train_model(X_train, y_train)
    print('Evaluating model..')
    evaluate_model(clf, X_test, y_test)
    print('Saving model..')
    save_model(clf, model_filename)

if __name__ == "__main__": main()