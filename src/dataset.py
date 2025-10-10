"""Welding dataset module.

Provides WeldingDataset which can run in 'dummy' mode for unit tests (no external deps)
and a 'real' stub mode to be implemented later for reading videos, audio, and sensor CSVs.
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import random

try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - tests run with these available in this environment
    np = None
    torch = None
    Dataset = object


class WeldingDataset(Dataset):
    """Dataset for welding multimodal data.

    Modes:
    - 'dummy': generates random tensors for video/audio/sensor for fast local testing.
    - 'real': stub that expects a directory structure under `root_dir` (to be implemented).

    Returns a dict per sample with keys: 'video','audio','sensor','label','meta'.
    Shapes (dummy defaults):
      video: (num_frames, 3, H, W)
      audio: (1, mel_bins, audio_frames)
      sensor: (sensor_len, num_channels)
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "dummy",
        num_samples: int = 32,
        num_frames: int = 8,
        frame_size: int = 64,
        image_size: int = 224,
        num_angles: int = 5,
        audio_mel_bins: int = 64,
        audio_frames: int = 32,
        audio_sr: int = 16000,
        sensor_len: int = 128,
        sensor_channels: int = 6,
        seed: Optional[int] = 42,
    ) -> None:
        self.root_dir = root_dir
        self.mode = mode
        self.num_samples = int(num_samples)
        self.num_frames = int(num_frames)
        self.frame_size = int(frame_size)
        self.image_size = int(image_size)
        self.num_angles = int(num_angles)
        self.audio_mel_bins = int(audio_mel_bins)
        self.audio_frames = int(audio_frames)
        self.audio_sr = int(audio_sr)
        self.sensor_len = int(sensor_len)
        self.sensor_channels = int(sensor_channels)

        if seed is not None:
            random.seed(seed)
            if np is not None:
                np.random.seed(seed)

        if mode not in ("dummy", "real"):
            raise ValueError("mode must be 'dummy' or 'real'")

        # For dummy mode, create simple label mapping
        if mode == "dummy":
            # two classes: 0 normal, 1 anomaly
            self._labels = [0 if i < self.num_samples // 2 else 1 for i in range(self.num_samples)]
            self._ids = [f"dummy_{i:04d}" for i in range(self.num_samples)]
        else:
            # real mode will scan directories
            self._ids, self._labels = self._scan_real_files()

    def _scan_real_files(self) -> Tuple[List[str], List[int]]:
        """Scan root_dir and collect sample ids with labels.

        Data structure: Data/<category_folder>/<sample_id>/
        Category folders like "1_good_weld_*", "7_spatter", etc.
        """
        ids, labels = [], []
        if not os.path.isdir(self.root_dir):
            return ids, labels
        
        # Category mapping
        cat_map = {"good_weld": 0, "crater_cracks": 1, "burn_through": 2,
                   "excessive_penetration": 3, "porosity": 4, "spatter": 5}
        
        for cat_folder in sorted(os.listdir(self.root_dir)):
            cat_path = os.path.join(self.root_dir, cat_folder)
            if not os.path.isdir(cat_path):
                continue
            
            # Extract label from folder name
            label = 0  # default to good_weld
            for key, val in cat_map.items():
                if key in cat_folder.lower():
                    label = val
                    break
            
            # Scan sample folders
            for sample_folder in sorted(os.listdir(cat_path)):
                sample_path = os.path.join(cat_path, sample_folder)
                if os.path.isdir(sample_path):
                    ids.append(os.path.join(cat_folder, sample_folder))
                    labels.append(label)
        
        return ids, labels

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "dummy":
            return self._get_dummy(idx)
        return self._get_real(idx)

    def _get_dummy(self, idx: int) -> Dict[str, Any]:
        """Return a synthetic sample as numpy arrays or torch tensors if torch available."""
        label = int(self._labels[idx])
        sid = self._ids[idx]

        # video: (num_frames, 3, H, W)
        vshape = (self.num_frames, 3, self.frame_size, self.frame_size)
        # post_weld_images: (num_angles, 3, image_size, image_size)
        ishape = (self.num_angles, 3, self.image_size, self.image_size)
        ashape = (1, self.audio_mel_bins, self.audio_frames)
        sshape = (self.sensor_len, self.sensor_channels)

        if np is not None:
            video = np.random.randn(*vshape).astype(np.float32)
            images = np.random.randn(*ishape).astype(np.float32)
            audio = np.random.randn(*ashape).astype(np.float32)
            sensor = np.random.randn(*sshape).astype(np.float32)
        else:
            video = [[[[]]]]
            images = [[[[]]]]
            audio = [[[[]]]]
            sensor = [[[[]]]]

        if torch is not None:
            video = torch.from_numpy(video)
            images = torch.from_numpy(images)
            audio = torch.from_numpy(audio)
            sensor = torch.from_numpy(sensor)

        return {
            "video": video,
            "post_weld_images": images,
            "audio": audio,
            "sensor": sensor,
            "label": label,
            "meta": {"id": sid},
        }

    def _get_real(self, idx: int) -> Dict[str, Any]:
        """Load real sample from filesystem under `root_dir`.

        Expected sample layout (examples seen in dataset):
          <root_dir>/<class_folder>/<sample_id>/
            *.avi or images/  (video)
            images/           (post-weld multi-angle images)
            *.flac or *.wav    (audio)
            *.csv              (sensor)

        Returns same dict structure as dummy mode.
        """
        sid = self._ids[idx]
        sample_dir = os.path.join(self.root_dir, sid)
        if not os.path.isdir(sample_dir):
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

        # Load modalities with helper functions
        video = self._read_video(sample_dir)
        images = self._read_post_weld_images(sample_dir)
        audio = self._read_audio(sample_dir)
        sensor = self._read_sensor(sample_dir)

        label = int(self._labels[idx]) if idx < len(self._labels) else 0

        # convert to torch tensors if available
        if torch is not None:
            if isinstance(video, np.ndarray):
                video = torch.from_numpy(video)
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            if isinstance(sensor, np.ndarray):
                sensor = torch.from_numpy(sensor)

        return {
            "video": video,
            "post_weld_images": images,
            "audio": audio,
            "sensor": sensor,
            "label": label,
            "meta": {"id": sid},
        }

    def _read_video(self, sample_dir: str) -> Any:
        """Read frames from images/ folder or video file and return
        numpy array shaped (num_frames, 3, H, W) of dtype float32 in [0,1]."""
        # lazy import
        try:
            import cv2
        except Exception as e:
            raise RuntimeError("OpenCV (cv2) is required for real video loading") from e

        # prefer images/ folder
        images_dir = os.path.join(sample_dir, "images")
        frames: List[Any] = []
        if os.path.isdir(images_dir):
            names = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            for n in names:
                frames.append(os.path.join(images_dir, n))
        else:
            # find a video file (avi/mp4)
            for ext in (".avi", ".mp4", ".mov", ".mkv"):
                vf = os.path.join(sample_dir, os.path.basename(sample_dir) + ext)
                if os.path.isfile(vf):
                    # extract frames using cv2
                    cap = cv2.VideoCapture(vf)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    idxs = list(range(total))
                    # read and store all frame indices as temp files in memory
                    for i in idxs:
                        ret, frm = cap.read()
                        if not ret:
                            break
                        # write to frames as in-memory image (decode later)
                        frames.append(frm)
                    cap.release()
                    break

        # if frames are file paths, read them; if they are images already, use directly
        imgs: List[Any] = []
        # normalize and resize
        H = W = self.frame_size
        if frames and isinstance(frames[0], str):
            for p in frames:
                img = cv2.imread(p)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                imgs.append(img)
        else:
            for im in frames:
                # im already ndarray (BGR)
                try:
                    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                except Exception:
                    img = im
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                imgs.append(img)

        if not imgs:
            # fallback: return zeros
            arr = np.zeros((self.num_frames, 3, H, W), dtype=np.float32)
            return arr

        # sample num_frames evenly from imgs
        total = len(imgs)
        if total >= self.num_frames:
            indices = [int(round(i * (total - 1) / (self.num_frames - 1))) if self.num_frames > 1 else 0 for i in range(self.num_frames)]
        else:
            # repeat last frame to pad
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)

        out = np.stack([imgs[i] for i in indices], axis=0).astype(np.float32) / 255.0
        # convert to (T, C, H, W)
        out = out.transpose(0, 3, 1, 2)
        return out

    def _read_post_weld_images(self, sample_dir: str) -> Any:
        """Read multi-angle post-weld images from images/ folder.
        
        Returns numpy array shaped (num_angles, 3, H, W) of dtype float32 in [0,1].
        """
        try:
            import cv2
        except Exception as e:
            raise RuntimeError("OpenCV (cv2) is required for image loading") from e
        
        images_dir = os.path.join(sample_dir, "images")
        if not os.path.isdir(images_dir):
            # fallback zeros
            return np.zeros((self.num_angles, 3, self.image_size, self.image_size), dtype=np.float32)
        
        # read all jpg/png files
        img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        
        imgs = []
        H = W = self.image_size
        for fname in img_files:
            img = cv2.imread(os.path.join(images_dir, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs.append(img)
        
        if not imgs:
            return np.zeros((self.num_angles, 3, H, W), dtype=np.float32)
        
        # pad or truncate to num_angles
        total = len(imgs)
        if total >= self.num_angles:
            imgs = imgs[:self.num_angles]
        else:
            # pad with last image
            imgs = imgs + [imgs[-1]] * (self.num_angles - total)
        
        out = np.stack(imgs, axis=0).astype(np.float32) / 255.0
        # convert to (N, C, H, W)
        out = out.transpose(0, 3, 1, 2)
        return out

    def _read_audio(self, sample_dir: str) -> Any:
        """Load audio file and convert to mel-spectrogram numpy array shaped (1, n_mels, audio_frames)."""
        try:
            import librosa
            import soundfile as sf
        except Exception as e:
            raise RuntimeError("librosa and soundfile are required for real audio loading") from e

        # find audio file
        audio_path = None
        for ext in (".flac", ".wav", ".mp3"):
            p = os.path.join(sample_dir, os.path.basename(sample_dir) + ext)
            if os.path.isfile(p):
                audio_path = p
                break
        if audio_path is None:
            # fall back: no audio
            mel = np.zeros((1, self.audio_mel_bins, self.audio_frames), dtype=np.float32)
            return mel

        # load audio
        y, sr = librosa.load(audio_path, sr=self.audio_sr if hasattr(self, 'audio_sr') else 16000)
        # compute mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.audio_mel_bins)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # time dimension length
        T = mel_db.shape[1]
        if T >= self.audio_frames:
            # sample or crop to audio_frames evenly
            indices = [int(round(i * (T - 1) / (self.audio_frames - 1))) if self.audio_frames > 1 else 0 for i in range(self.audio_frames)]
            mel_db = mel_db[:, indices]
        else:
            # pad with minimum value
            pad_width = self.audio_frames - T
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=(mel_db.min() if mel_db.size else 0.0,))

        mel_db = mel_db.astype(np.float32)
        mel_db = np.expand_dims(mel_db, axis=0)
        return mel_db

    def _read_sensor(self, sample_dir: str) -> Any:
        """Read CSV sensor data, select numeric columns, resample/interpolate to self.sensor_len and z-score normalize.

        Returns numpy array shape (sensor_len, sensor_channels)
        """
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError("pandas is required for real sensor CSV loading") from e

        csv_path = None
        # look for any csv file in directory
        for f in os.listdir(sample_dir):
            if f.lower().endswith('.csv'):
                csv_path = os.path.join(sample_dir, f)
                break
        if csv_path is None:
            # fallback zeros
            return np.zeros((self.sensor_len, self.sensor_channels), dtype=np.float32)

        df = pd.read_csv(csv_path)
        # keep only numeric columns
        df_num = df.select_dtypes(include=["number"]).copy()
        if df_num.shape[1] == 0:
            return np.zeros((self.sensor_len, self.sensor_channels), dtype=np.float32)

        # if sensor_channels mismatch, adjust
        cols = df_num.columns.tolist()
        arr = df_num.to_numpy(dtype=np.float32)
        T, C = arr.shape

        # resample/interpolate along time axis to self.sensor_len
        if T == 0:
            out = np.zeros((self.sensor_len, min(self.sensor_channels, C)), dtype=np.float32)
        else:
            # create target positions and interpolate each channel
            xp = np.linspace(0, 1, T)
            x = np.linspace(0, 1, self.sensor_len)
            out = np.stack([np.interp(x, xp, arr[:, ch]) for ch in range(C)], axis=1)

        # if fewer channels than requested, pad with zeros; if more, truncate
        if out.shape[1] < self.sensor_channels:
            pad = np.zeros((self.sensor_len, self.sensor_channels - out.shape[1]), dtype=np.float32)
            out = np.concatenate([out, pad], axis=1)
        elif out.shape[1] > self.sensor_channels:
            out = out[:, : self.sensor_channels]

        # z-score normalize per-channel
        mean = out.mean(axis=0, keepdims=True)
        std = out.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        out = (out - mean) / std

        return out.astype(np.float32)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple collate that stacks tensors along batch dim.

    Accepts numpy arrays or torch tensors. If mixed, prefer torch if available.
    """
    if not batch:
        return {}

    first = batch[0]
    use_torch = torch is not None and isinstance(first["video"], getattr(torch, 'Tensor', object))

    videos = [b["video"] for b in batch]
    images = [b["post_weld_images"] for b in batch]
    audios = [b["audio"] for b in batch]
    sensors = [b["sensor"] for b in batch]
    labels = [b["label"] for b in batch]
    metas = [b.get("meta", {}) for b in batch]

    if use_torch:
        videos = torch.stack(videos, dim=0)
        images = torch.stack(images, dim=0)
        audios = torch.stack(audios, dim=0)
        sensors = torch.stack(sensors, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        import numpy as _np

        videos = _np.stack(videos, axis=0)
        images = _np.stack(images, axis=0)
        audios = _np.stack(audios, axis=0)
        sensors = _np.stack(sensors, axis=0)
        labels = _np.array(labels, dtype=_np.int64)

    return {
        "video": videos,
        "post_weld_images": images,
        "audio": audios,
        "sensor": sensors,
        "label": labels,
        "meta": metas,
    }


__all__ = ["WeldingDataset", "collate_fn"]
