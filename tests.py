import numpy as np
import pandas as pd
import pytest
from utils.filter import butterworth_lowpass_filter


def test_filter_preserves_non_numeric_columns():
    """Test that 'Time', 'frame#', and other non-numeric columns are not filtered."""
    
    # Create a sample DataFrame with time, frame number, and numeric data
    n_samples = 100
    time = np.linspace(0, 1, n_samples)
    frame_numbers = np.arange(n_samples)
    
    # Create some numeric data that would change if filtered
    signal = np.sin(2 * np.pi * 5 * time)  # 5 Hz signal
    
    data = pd.DataFrame({
        'Time': time,
        'frame#': frame_numbers,
        'signal_x': signal,
        'signal_y': signal * 2,
        'signal_z': signal * 3
    })
    
    # Apply the filter with a cutoff frequency
    fs = 100  # 100 Hz sampling frequency
    cutoff = 10  # 10 Hz cutoff
    filtered_data = butterworth_lowpass_filter(data, cutoff, fs, order=4)
    
    # Check that Time and frame# columns are unchanged
    assert 'Time' in filtered_data.columns, "Time column should be preserved"
    assert 'frame#' in filtered_data.columns, "frame# column should be preserved"
    
    # Check that Time and frame# values are exactly the same
    np.testing.assert_array_equal(
        filtered_data['Time'].values, 
        data['Time'].values,
        err_msg="Time column should not be filtered"
    )
    np.testing.assert_array_equal(
        filtered_data['frame#'].values, 
        data['frame#'].values,
        err_msg="frame# column should not be filtered"
    )
    
    # Check that numeric columns are filtered (should be different)
    assert not np.allclose(filtered_data['signal_x'].values, data['signal_x'].values), \
        "Numeric columns should be filtered"
    
    print("✓ Test passed: Non-numeric columns preserved correctly")


def test_filter_removes_high_frequency_noise():
    """Test that the filter can remove high-frequency noise from a signal."""
    
    # Create a synthetic signal: low frequency + high frequency noise
    n_samples = 1000
    fs = 200  # 200 Hz sampling frequency
    time = np.linspace(0, 5, n_samples)  # 5 seconds
    
    # Low frequency signal (2 Hz)
    low_freq = np.random.uniform(1.5, 2.5)  # Randomize low frequency a bit
    clean_signal = 5 * np.sin(2 * np.pi * low_freq * time)
    
    # High frequency noise (50 Hz)
    high_freq = np.random.uniform(45, 100)  # Randomize high frequency a bit
    noise = 2 * np.sin(2 * np.pi * high_freq * time)
    
    # Combined signal
    noisy_signal = clean_signal + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'Time': time,
        'noisy_signal': noisy_signal
    })
    
    # Apply low-pass filter with cutoff between low and high frequencies
    cutoff = 10  # 10 Hz cutoff (allows 2 Hz, removes 50 Hz)
    filtered_data = butterworth_lowpass_filter(data, cutoff, fs, order=4)
    
    # Extract filtered signal
    filtered_signal = filtered_data['noisy_signal'].values
    
    # Check that the filtered signal is closer to the clean signal than the noisy signal
    error_before = np.mean((noisy_signal - clean_signal) ** 2)
    error_after = np.mean((filtered_signal - clean_signal) ** 2)
    
    assert error_after < error_before * 0.1, \
        f"Filter should significantly reduce noise (error before: {error_before:.4f}, after: {error_after:.4f})"
    
    # Perform frequency analysis to verify high frequency is attenuated
    # FFT of noisy signal
    fft_noisy = np.fft.fft(noisy_signal)
    fft_filtered = np.fft.fft(filtered_signal)
    freqs = np.fft.fftfreq(n_samples, 1/fs)
    
    # Find power at high frequency (50 Hz)
    high_freq_idx = np.argmin(np.abs(freqs - high_freq))
    power_noisy = np.abs(fft_noisy[high_freq_idx])
    power_filtered = np.abs(fft_filtered[high_freq_idx])
    
    # High frequency should be attenuated by at least 90%
    attenuation_ratio = power_filtered / power_noisy
    assert attenuation_ratio < 0.1, \
        f"High frequency noise should be attenuated by at least 90% (attenuation ratio: {attenuation_ratio:.4f})"
    
    # Find power at low frequency (2 Hz)
    low_freq_idx = np.argmin(np.abs(freqs - low_freq))
    power_low_before = np.abs(fft_noisy[low_freq_idx])
    power_low_after = np.abs(fft_filtered[low_freq_idx])
    
    # Low frequency should be mostly preserved (at least 80%)
    preservation_ratio = power_low_after / power_low_before
    assert preservation_ratio > 0.8, \
        f"Low frequency signal should be preserved (preservation ratio: {preservation_ratio:.4f})"
    
    print("✓ Test passed: High-frequency noise removed successfully")
    print(f"  - Mean squared error reduced by {(1 - error_after/error_before)*100:.1f}%")
    print(f"  - High frequency attenuated to {attenuation_ratio*100:.1f}% of original")
    print(f"  - Low frequency preserved at {preservation_ratio*100:.1f}% of original")

def test_segmentation_of_gait_cycles():
    """Test the segmentation of gait cycles based on vertical GRF data."""
    
    from utils import segment
    grf_data = pd.read_csv('data/grf.csv')
    grf_y = grf_data['force_r_y'].values
    threshold = 60  # N

    gait_cycles = segment.segment_gait_cycles(grf_y, data=grf_data, threshold=threshold)
    assert isinstance(gait_cycles, list), "Output should be a list of DataFrames"
    assert len(gait_cycles) > 50, "There should be at least 50 gait cycles detected"
    assert all(isinstance(cycle, pd.DataFrame) for cycle in gait_cycles), "All elements should be DataFrames"
    assert len(gait_cycles) < 500, "There should be fewer than 500 gait cycles detected (300s of data--> ~300 gait cycles)"

    ensemble_average, std_cycle = segment.ensemble_average(gait_cycles)
    assert isinstance(ensemble_average, pd.DataFrame), "Ensemble average should be a DataFrame"
    assert isinstance(std_cycle, pd.DataFrame), "Standard deviation should be a DataFrame"

    assert 'time' in ensemble_average.columns, "Ensemble average should contain 'time' column"
    assert 0.9 < ensemble_average['time'].iloc[-1] < 1.03, "Ensemble average time should span approximately 0.97 seconds" 
    ensemble_average_grf_y = ensemble_average['force_r_y']
    assert ensemble_average_grf_y.max() > 800, "Ensemble max GRF should be greater than 800 N"
    assert ensemble_average_grf_y.min() < 30, "Ensemble min GRF should be less than 30 N"

    assert ensemble_average_grf_y[20] > 700, "Ensemble GRF at 20% should be greater than 700 N"
    assert ensemble_average_grf_y[50] > 300, "Ensemble GRF at 50% should be greater than 300 N"
    assert ensemble_average_grf_y[80] < 40, "Ensemble GRF at 80% should be less than 40 N"

    ensemble_average_std_y = std_cycle['force_r_y']
    assert ensemble_average_std_y.max() < 300, "Ensemble std GRF should be less than 300 N at all times"
    assert ensemble_average_std_y.min() > 0, "Ensemble std GRF should be greater than 0 N at all times"


def test_marker_error():
    """Test the marker error calculation."""
    from fk.forward_kinematics import forward_kinematics
    from utils.visualize import evaluate_marker_error
    from model.kintree import kintree

    q_all = pd.read_csv('data/angles_clean.csv')
    marker_data = pd.read_csv('data/markers.csv')
    errors = []
    for frame in range(len(q_all)):  # Test multiple frames

        curr_frame = frame
        q = q_all.iloc[curr_frame,1:] # Exclude time column, 
        markers = marker_data.iloc[curr_frame,2:] # Exclude time and frame column
        key = list(q_all.columns[1:]) # Exclude time column

        joints, markers_fk = forward_kinematics(q.values, key, kintree)
        marker_error = evaluate_marker_error(markers_fk, markers)
        errors.append(marker_error)

    marker_error = np.mean(errors)*1e3
    print(f"Average marker error over {len(errors)} frames: {marker_error:.2f} mm")
    assert marker_error < 10.0, f"Marker error should be less than 10 mm, got {marker_error:.2f} mm"

if __name__ == "__main__":
    # Run tests
    pass