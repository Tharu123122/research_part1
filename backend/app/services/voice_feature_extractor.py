from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly


BASE_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = BASE_DIR / "02_oral_cancer_voice_labeled" / "voice_labeled_manifest.csv"
OUT_PATH = BASE_DIR / "04_model_ready_outputs" / "features" / "voice_labeled_features.csv"


def pre_emphasis(signal: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    if signal.size < 2:
        return signal
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def frame_signal(signal: np.ndarray, sample_rate: int, frame_ms: int = 25, hop_ms: int = 10) -> np.ndarray:
    frame_len = int(sample_rate * frame_ms / 1000)
    hop_len = int(sample_rate * hop_ms / 1000)
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))
    frame_count = 1 + int((len(signal) - frame_len) / hop_len)
    indices = np.tile(np.arange(frame_len), (frame_count, 1)) + np.tile(
        np.arange(frame_count) * hop_len, (frame_len, 1)
    ).T
    return signal[indices] * np.hamming(frame_len)


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700 * (10 ** (mel / 2595) - 1)


def mfcc_features(signal: np.ndarray, sample_rate: int, nfilt: int = 26, num_ceps: int = 13) -> np.ndarray:
    emphasized = pre_emphasis(signal)
    frames = frame_signal(emphasized, sample_rate)
    nfft = 512
    magnitude = np.absolute(np.fft.rfft(frames, nfft))
    power = (1.0 / nfft) * (magnitude**2)

    low_mel = hz_to_mel(np.array([0]))[0]
    high_mel = hz_to_mel(np.array([sample_rate / 2]))[0]
    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    filter_bank = np.zeros((nfilt, int(nfft / 2 + 1)))
    for m in range(1, nfilt + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            filter_bank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            filter_bank[m - 1, k] = (right - k) / (right - center)

    energies = np.dot(power, filter_bank.T)
    energies = np.where(energies == 0, np.finfo(float).eps, energies)
    cepstra = dct(np.log(energies), type=2, axis=1, norm="ortho")[:, :num_ceps]
    return cepstra


def load_wav(path: Path, target_rate: int = 16000) -> tuple[int, np.ndarray]:
    sample_rate, signal = wavfile.read(path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    signal = signal.astype(np.float32)
    peak = np.max(np.abs(signal)) if signal.size else 0
    if peak > 0:
        signal = signal / peak
    if sample_rate != target_rate:
        gcd = np.gcd(sample_rate, target_rate)
        signal = resample_poly(signal, target_rate // gcd, sample_rate // gcd)
        sample_rate = target_rate
    return sample_rate, signal


def trim_silence(signal: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    if signal.size == 0:
        return signal
    mask = np.abs(signal) > threshold
    if not mask.any():
        return signal
    return signal[np.argmax(mask) : len(signal) - np.argmax(mask[::-1])]


def estimate_pitch(signal: np.ndarray, sample_rate: int) -> float:
    if len(signal) < sample_rate // 20:
        return 0.0
    segment = signal[: min(len(signal), sample_rate * 3)]
    segment = segment - np.mean(segment)
    corr = correlate(segment, segment, mode="full", method="fft")[len(segment) - 1 :]
    min_lag = int(sample_rate / 350)
    max_lag = int(sample_rate / 70)
    if max_lag >= len(corr):
        return 0.0
    corr[:min_lag] = 0
    lag = np.argmax(corr[min_lag:max_lag]) + min_lag
    return float(sample_rate / lag) if lag else 0.0


def voice_quality(signal: np.ndarray, sample_rate: int) -> tuple[float, float]:
    frames = frame_signal(signal, sample_rate, frame_ms=40, hop_ms=20)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    shimmer = float(np.std(rms) / (np.mean(rms) + 1e-8))

    pitches = []
    sampled_frames = frames[:: max(1, len(frames) // 40)]
    for frame in sampled_frames[:40]:
        pitch = estimate_pitch(frame, sample_rate)
        if 70 <= pitch <= 350:
            pitches.append(pitch)
    if len(pitches) < 2:
        jitter = 0.0
    else:
        periods = 1 / np.array(pitches)
        jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-8))
    return jitter, shimmer


def spectral_features(signal: np.ndarray, sample_rate: int) -> dict:
    frames = frame_signal(signal, sample_rate)
    magnitude = np.abs(np.fft.rfft(frames, 512))
    freqs = np.fft.rfftfreq(512, d=1 / sample_rate)
    mag_sum = magnitude.sum(axis=1) + 1e-8
    centroid = (magnitude * freqs).sum(axis=1) / mag_sum
    bandwidth = np.sqrt(((freqs - centroid[:, None]) ** 2 * magnitude).sum(axis=1) / mag_sum)
    cumulative = np.cumsum(magnitude, axis=1)
    rolloff_idx = [np.searchsorted(row, 0.85 * row[-1]) for row in cumulative]
    rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]
    zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
    }


def extract_features(row: pd.Series) -> dict:
    path = Path(row["file_path"])
    sample_rate, signal = load_wav(path)
    signal = trim_silence(signal)
    duration = len(signal) / sample_rate if sample_rate else 0
    cepstra = mfcc_features(signal, sample_rate)
    pitch = estimate_pitch(signal, sample_rate)
    jitter, shimmer = voice_quality(signal, sample_rate)

    features = {
        "relative_path": row["relative_path"],
        "split": row["split"],
        "speaker_id": row["speaker_id"],
        "gender": row["gender"],
        "voice_label": row["voice_label"],
        "duration_seconds": duration,
        "pitch_mean": pitch,
        "jitter": jitter,
        "shimmer": shimmer,
        "rms_energy": float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0,
    }
    features.update(spectral_features(signal, sample_rate))
    for i in range(cepstra.shape[1]):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(cepstra[:, i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(cepstra[:, i]))
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract voice features from labeled oral cancer speech files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick testing.")
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    if args.limit:
        manifest = manifest.head(args.limit)

    rows = []
    errors = []
    for index, row in manifest.iterrows():
        try:
            rows.append(extract_features(row))
        except Exception as exc:
            errors.append({"relative_path": row.get("relative_path"), "error": str(exc)})
        if (index + 1) % 250 == 0:
            print(f"Processed {index + 1} files...")

    features = pd.DataFrame(rows)
    features.to_csv(OUT_PATH, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(OUT_PATH.with_name("voice_feature_extraction_errors.csv"), index=False)
        warnings.warn(f"{len(errors)} files failed. See voice_feature_extraction_errors.csv")

    print(f"Saved {len(features)} voice feature rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
