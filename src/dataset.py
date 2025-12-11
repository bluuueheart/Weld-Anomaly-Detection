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
        root_dir: str = None,
        mode: str = "dummy",
        split: str = None,  # 'train', 'test', or None (all samples)
        manifest_path: str = "configs/manifest.csv",
        num_samples: int = 32,
        num_frames: int = 8,
        frame_size: int = 64,
        image_size: int = 224,
        num_angles: int = 5,
        audio_mel_bins: int = 64,
        audio_frames: int = 32,
        audio_sr: int = 16000,
        audio_type: str = "mel",  # 'mel' or 'stft'
        n_fft: int = 2048,  # FFT window size for STFT
        hop_length: int = 512,  # Hop length for STFT
        sensor_len: int = 128,
        sensor_channels: int = 6,
        seed: Optional[int] = 42,
        # Backward-compatibility aliases
        data_root: str = None,
        video_length: int = None,
        audio_sample_rate: int = None,
        audio_duration: int = None,
        sensor_length: int = None,
        dummy: bool = None,
        augment: bool = False,
    ) -> None:
        # Handle backward-compatibility aliases
        if data_root is not None:
            root_dir = data_root
        if root_dir is None:
            root_dir = "Data"  # default fallback
        if video_length is not None:
            num_frames = video_length
        if audio_sample_rate is not None:
            audio_sr = audio_sample_rate
        if audio_duration is not None:
            audio_frames = audio_duration
        if sensor_length is not None:
            sensor_len = sensor_length
        if dummy is not None:
            mode = "dummy" if dummy else "real"
        self.augment = bool(augment)
        
        self.root_dir = root_dir
        self.mode = mode
        self.split = split  # Store split parameter
        self.manifest_path = manifest_path
        self.num_samples = int(num_samples)
        self.num_frames = int(num_frames)
        self.frame_size = int(frame_size)
        self.image_size = int(image_size)
        self.num_angles = int(num_angles)
        self.audio_mel_bins = int(audio_mel_bins)
        self.audio_frames = int(audio_frames)
        self.audio_sr = int(audio_sr)
        self.audio_type = audio_type
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sensor_len = int(sensor_len)
        self.sensor_channels = int(sensor_channels)

        # Category name to label mapping
        self.CATEGORY_MAP = {
            "good": 0,
            "excessive_convexity": 1,
            "undercut": 2,
            "lack_of_fusion": 3,
            "porosity": 5,
            "spatter": 6,
            "burnthrough": 7,
            "porosity_w_excessive_penetration": 4,
            "excessive_penetration": 8,
            "excessive penetration": 8,  # Handle space variant
            "crater_cracks": 9,
            "warping": 10,
            "overlap": 11,
        }
        # Create reverse mapping for display
        self.label_to_name = {v: k for k, v in self.CATEGORY_MAP.items() if k != "excessive penetration"}

        if seed is not None:
            random.seed(seed)
            if np is not None:
                np.random.seed(seed)

        if mode not in ("dummy", "real"):
            raise ValueError("mode must be 'dummy' or 'real'")

        # For dummy mode, create simple label mapping
        if mode == "dummy":
            # Ensure balanced classes across num_samples
            # Generate labels that cycle through all classes
            num_classes = 6  # welding dataset has 6 classes
            self._labels = [i % num_classes for i in range(self.num_samples)]
            # Shuffle to mix classes (random already imported at module level)
            temp_list = list(zip(range(self.num_samples), self._labels))
            random.shuffle(temp_list)
            indices, self._labels = zip(*temp_list)
            self._labels = list(self._labels)
            self._ids = [f"dummy_{i:04d}" for i in range(self.num_samples)]
        else:
            # real mode will scan directories
            self._ids, self._labels = self._scan_real_files()

    def _load_manifest(self) -> Tuple[List[str], List[int], List[str]]:
        """Load manifest.csv and return (sample_paths, labels, splits).
        
        Returns:
            sample_paths: List of sample subdirectories (e.g., "2_good_weld_2_02-09-23_Fe410/04-01-23-0024-00")
            labels: List of integer labels
            splits: List of split names ("TRAIN" or "TEST")
        """
        import csv
        
        sample_paths, labels, splits = [], [], []
        
        if not os.path.isfile(self.manifest_path):
            return sample_paths, labels, splits
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = row.get('CATEGORY', '').strip()
                subdir = row.get('SUBDIRS', '').strip()
                split_val = row.get('SPLIT', '').strip().upper()
                
                if not subdir or not category:
                    continue
                
                # Map category to label
                category_key = category.lower().replace(' ', '_')
                label = self.CATEGORY_MAP.get(category_key, 0)
                
                sample_paths.append(subdir)
                labels.append(label)
                splits.append(split_val)
        
        return sample_paths, labels, splits

    def _scan_real_files(self) -> Tuple[List[str], List[int]]:
        """Scan root_dir and collect sample ids with labels.
        
        Uses manifest.csv if available, otherwise falls back to directory scanning.
        """
        # Try loading from manifest first
        if os.path.isfile(self.manifest_path):
            sample_paths, labels, splits = self._load_manifest()

            # Filter by split if specified
            if self.split is not None:
                split_filter = self.split.upper()
                filtered_paths, filtered_labels = [], []
                for path, label, split_val in zip(sample_paths, labels, splits):
                    if split_val == split_filter:
                        filtered_paths.append(path)
                        filtered_labels.append(label)
            else:
                filtered_paths, filtered_labels = sample_paths, labels

            # Remove entries that don't exist on disk to avoid DataLoader worker crashes
            final_paths, final_labels = [], []
            skipped_examples = []
            for p, l in zip(filtered_paths, filtered_labels):
                full = os.path.join(self.root_dir, p)
                if os.path.isdir(full):
                    final_paths.append(p)
                    final_labels.append(l)
                else:
                    skipped_examples.append(p)

            if skipped_examples:
                # Print a concise summary so the user knows some manifest entries were missing
                example = skipped_examples[:3]
                print(f"[WeldingDataset] Skipped {len(skipped_examples)} missing samples from manifest (examples: {example})")

            return final_paths, final_labels

        # If no manifest is present, fail fast to avoid silent train/val overlap.
        # Using the full filesystem scan as a fallback can silently introduce
        # data leakage (train and val getting the same samples). Require an
        # explicit manifest or run the resplit utility to generate one.
        msg = (
            f"[WeldingDataset] ERROR: manifest not found at '{self.manifest_path}'.\n"
            "Please create a manifest.csv (TRAIN/TEST split) or run scripts/resplit_dataset.py.\n"
            "Refusing to continue to avoid train/validation data leakage."
        )
        print(msg)
        raise RuntimeError(msg)

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
        # audio: shape depends on audio_type
        if self.audio_type == "stft":
            n_bins = self.n_fft // 2 + 1
            ashape = (1, n_bins, self.audio_frames)
        else:
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
        # For dummy mode, optionally apply simple augmentations when requested
        if self.augment and self.split == 'train':
            try:
                video = self._augment_video_numpy(video) if not isinstance(video, getattr(torch, 'Tensor', object)) else video
                images = self._augment_images_numpy(images) if not isinstance(images, getattr(torch, 'Tensor', object)) else images
            except Exception:
                # best-effort: if augmentation fails in dummy mode, ignore and continue
                pass
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
        # Apply augmentations only for training split and when enabled
        if self.augment and (self.split == 'train' or self.split is None):
            try:
                # operate on numpy if available; convert back to torch if necessary
                if isinstance(images, torch.Tensor):
                    imgs_np = images.cpu().numpy()
                    imgs_np = self._augment_images_numpy(imgs_np)
                    images = torch.from_numpy(imgs_np)
                elif isinstance(images, np.ndarray):
                    images = self._augment_images_numpy(images)

                if isinstance(video, torch.Tensor):
                    vid_np = video.cpu().numpy()
                    vid_np = self._augment_video_numpy(vid_np)
                    video = torch.from_numpy(vid_np)
                elif isinstance(video, np.ndarray):
                    video = self._augment_video_numpy(video)

                if isinstance(audio, torch.Tensor):
                    aud_np = audio.cpu().numpy()
                    aud_np = self._augment_audio_numpy(aud_np)
                    audio = torch.from_numpy(aud_np)
                elif isinstance(audio, np.ndarray):
                    audio = self._augment_audio_numpy(audio)

                if isinstance(sensor, torch.Tensor):
                    sen_np = sensor.cpu().numpy()
                    sen_np = self._augment_sensor_numpy(sen_np)
                    sensor = torch.from_numpy(sen_np)
                elif isinstance(sensor, np.ndarray):
                    sensor = self._augment_sensor_numpy(sensor)
            except Exception:
                # best-effort: do not fail load if augmentation errors
                pass
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
        """Load audio file and convert to mel-spectrogram or STFT numpy array.
        
        Returns:
            For mel: (1, n_mels, audio_frames)
            For stft: (1, n_bins, audio_frames) where n_bins = n_fft // 2 + 1
        """
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
            if self.audio_type == "stft":
                n_bins = self.n_fft // 2 + 1
                spec = np.zeros((1, n_bins, self.audio_frames), dtype=np.float32)
            else:
                spec = np.zeros((1, self.audio_mel_bins, self.audio_frames), dtype=np.float32)
            return spec

        # load audio
        y, sr = librosa.load(audio_path, sr=self.audio_sr if hasattr(self, 'audio_sr') else 16000)
        
        # compute spectrogram based on audio_type
        if self.audio_type == "stft":
            # Compute STFT
            stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            spec = np.abs(stft)  # magnitude spectrogram (n_bins, time)
        else:
            # compute mel spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.audio_mel_bins)
            spec = librosa.power_to_db(mel, ref=np.max)

        # time dimension length
        T = spec.shape[1]
        if T >= self.audio_frames:
            # sample or crop to audio_frames evenly
            indices = [int(round(i * (T - 1) / (self.audio_frames - 1))) if self.audio_frames > 1 else 0 for i in range(self.audio_frames)]
            spec = spec[:, indices]
        else:
            # pad with minimum value
            pad_width = self.audio_frames - T
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=(spec.min() if spec.size else 0.0,))

        spec = spec.astype(np.float32)
        spec = np.expand_dims(spec, axis=0)
        return spec

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

    # ----------------------
    # Augmentation helpers
    # ----------------------
    def _augment_images_numpy(self, images):
        """Apply in-place simple image augmentations to numpy images.

        images: (N, C, H, W), values in [0,1]
        """
        if images is None:
            return images

        # optional config from configs.dataset_config.AUGMENTATION
        try:
            from configs.dataset_config import AUGMENTATION as AUG
        except Exception:
            AUG = None

        # default parameters
        p_hflip = AUG.get("p_hflip", 0.5) if AUG else 0.5
        p_vflip = AUG.get("p_vflip", 0.0) if AUG else 0.0
        p_rotate = AUG.get("p_rotate", 0.3) if AUG else 0.3
        rotate_max = AUG.get("rotate_max_deg", 10.0) if AUG else 10.0
        p_rrc = AUG.get("p_random_resized_crop", 0.3) if AUG else 0.3
        rrc_scale_min = AUG.get("rrc_scale_min", 0.8) if AUG else 0.8
        brightness = AUG.get("brightness", 0.2) if AUG else 0.2
        contrast = AUG.get("contrast", 0.2) if AUG else 0.2
        saturation = AUG.get("saturation", 0.1) if AUG else 0.1
        hue = AUG.get("hue", 0.05) if AUG else 0.05
        p_blur = AUG.get("p_blur", 0.2) if AUG else 0.2
        blur_max = AUG.get("blur_max_ksize", 5) if AUG else 5
        p_noise = AUG.get("p_gauss_noise", 0.2) if AUG else 0.2
        noise_sigma = AUG.get("gauss_noise_sigma", 0.02) if AUG else 0.02
        p_cutout = AUG.get("p_cutout", 0.25) if AUG else 0.25
        cutout_min = AUG.get("cutout_area_min", 0.02) if AUG else 0.02
        cutout_max = AUG.get("cutout_area_max", 0.2) if AUG else 0.2

        imgs = images.astype(np.float32)
        N = imgs.shape[0]

        # helper: convert C,H,W -> H,W,C and back
        def chw_to_hwc(arr):
            return arr.transpose(1, 2, 0)

        def hwc_to_chw(arr):
            return arr.transpose(2, 0, 1)

        for i in range(N):
            img = imgs[i]
            H = img.shape[1]
            W = img.shape[2]

            # to HWC for cv2 ops
            img_hwc = chw_to_hwc(img)

            # random resized crop
            if random.random() < p_rrc:
                try:
                    import cv2
                    scale = random.uniform(rrc_scale_min, 1.0)
                    new_h = max(1, int(H * scale))
                    new_w = max(1, int(W * scale))
                    top = random.randint(0, H - new_h) if H - new_h > 0 else 0
                    left = random.randint(0, W - new_w) if W - new_w > 0 else 0
                    cropped = img_hwc[top:top+new_h, left:left+new_w]
                    img_hwc = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    pass

            # rotation
            if random.random() < p_rotate:
                try:
                    import cv2
                    ang = random.uniform(-rotate_max, rotate_max)
                    M = cv2.getRotationMatrix2D((W/2.0, H/2.0), ang, 1.0)
                    img_hwc = cv2.warpAffine((img_hwc * 255.0).astype(np.uint8), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    img_hwc = img_hwc.astype(np.float32) / 255.0
                except Exception:
                    pass

            # flips
            if random.random() < p_hflip:
                img_hwc = img_hwc[:, ::-1]
            if random.random() < p_vflip:
                img_hwc = img_hwc[::-1, :]

            # color jitter: brightness
            b = 1.0 + random.uniform(-brightness, brightness)
            img_hwc = img_hwc * b

            # contrast
            c = 1.0 + random.uniform(-contrast, contrast)
            mean = img_hwc.mean(axis=(0, 1), keepdims=True)
            img_hwc = (img_hwc - mean) * c + mean

            # saturation/hue via HSV when available
            if (saturation > 0 or hue > 0):
                try:
                    import cv2
                    img_cv = (np.clip(img_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
                    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV).astype(np.float32)
                    # saturation
                    s_factor = 1.0 + random.uniform(-saturation, saturation)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
                    # hue shift
                    h_factor = random.uniform(-hue, hue) * 180.0
                    hsv[:, :, 0] = (hsv[:, :, 0] + h_factor) % 180.0
                    img_cv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                    img_hwc = img_cv.astype(np.float32) / 255.0
                except Exception:
                    pass

            # blur
            if random.random() < p_blur:
                try:
                    import cv2
                    k = random.choice([k for k in range(1, blur_max+1) if k % 2 == 1] or [3])
                    img_hwc = cv2.GaussianBlur((img_hwc * 255.0).astype(np.uint8), (k, k), 0).astype(np.float32) / 255.0
                except Exception:
                    pass

            # gaussian noise
            if random.random() < p_noise:
                sigma = noise_sigma * (img_hwc.std() if img_hwc.size else 1.0)
                img_hwc = img_hwc + np.random.normal(scale=sigma, size=img_hwc.shape).astype(np.float32)

            # cutout / random erasing
            if random.random() < p_cutout:
                area = H * W
                erase_area = int(area * random.uniform(cutout_min, cutout_max))
                if erase_area > 0:
                    ew = max(1, int((erase_area ** 0.5)))
                    eh = ew
                    x = random.randint(0, max(0, W - ew))
                    y = random.randint(0, max(0, H - eh))
                    img_hwc[y:y+eh, x:x+ew, :] = 0.0

            img_hwc = np.clip(img_hwc, 0.0, 1.0)
            imgs[i] = hwc_to_chw(img_hwc)

        return imgs

    def _augment_video_numpy(self, video):
        """Apply video-level augmentations (consistent across frames).

        video: (T, C, H, W)
        """
        if video is None:
            return video

        try:
            from configs.dataset_config import AUGMENTATION as AUG
        except Exception:
            AUG = None

        p_hflip = AUG.get("p_hflip", 0.5) if AUG else 0.5
        p_vflip = AUG.get("p_vflip", 0.0) if AUG else 0.0
        p_rotate = AUG.get("p_rotate", 0.3) if AUG else 0.3
        rotate_max = AUG.get("rotate_max_deg", 10.0) if AUG else 10.0
        p_rrc = AUG.get("p_random_resized_crop", 0.3) if AUG else 0.3
        rrc_scale_min = AUG.get("rrc_scale_min", 0.8) if AUG else 0.8
        p_temporal = AUG.get("p_temporal_shift", 0.2) if AUG else 0.2
        max_shift = AUG.get("max_temporal_shift", 2) if AUG else 2

        vid = video.astype(np.float32)
        Tlen, C, H, W = vid.shape

        # sample geometric params once
        do_hflip = random.random() < p_hflip
        do_vflip = random.random() < p_vflip
        do_rotate = random.random() < p_rotate
        angle = random.uniform(-rotate_max, rotate_max) if do_rotate else 0.0
        do_rrc = random.random() < p_rrc
        rrc_scale = random.uniform(rrc_scale_min, 1.0) if do_rrc else 1.0

        # prepare affine if needed
        M = None
        if do_rotate:
            try:
                import cv2
                M = cv2.getRotationMatrix2D((W/2.0, H/2.0), angle, 1.0)
            except Exception:
                M = None

        # random resized crop coordinates
        if do_rrc:
            new_h = max(1, int(H * rrc_scale))
            new_w = max(1, int(W * rrc_scale))
            top = random.randint(0, H - new_h) if H - new_h > 0 else 0
            left = random.randint(0, W - new_w) if W - new_w > 0 else 0
        else:
            top = 0; left = 0; new_h = H; new_w = W

        # temporal shift (roll) sometimes
        if random.random() < p_temporal and Tlen > 1:
            shift = random.randint(-max_shift, max_shift)
            vid = np.roll(vid, shift, axis=0)

        # apply same transforms to each frame
        for t in range(Tlen):
            frame = vid[t]
            # C,H,W -> H,W,C
            frame_hwc = frame.transpose(1, 2, 0)

            # crop+resize
            try:
                import cv2
                if do_rrc:
                    cropped = frame_hwc[top:top+new_h, left:left+new_w]
                    frame_hwc = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
                # rotate
                if M is not None:
                    frame_hwc = cv2.warpAffine((frame_hwc * 255.0).astype(np.uint8), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    frame_hwc = frame_hwc.astype(np.float32) / 255.0
            except Exception:
                pass

            # flips
            if do_hflip:
                frame_hwc = frame_hwc[:, ::-1]
            if do_vflip:
                frame_hwc = frame_hwc[::-1, :]

            # color jitter small per-clip
            b = 1.0 + random.uniform(-0.15, 0.15)
            c = 1.0 + random.uniform(-0.15, 0.15)
            frame_hwc = frame_hwc * b
            mean = frame_hwc.mean(axis=(0, 1), keepdims=True)
            frame_hwc = (frame_hwc - mean) * c + mean

            # back to C,H,W
            vid[t] = np.clip(frame_hwc, 0.0, 1.0).transpose(2, 0, 1)

        return vid

    def _augment_audio_numpy(self, audio):
        """Simple mel-spectrogram augmentations: additive noise + time masking.

        audio: (1, n_mels, T)
        """
        if audio is None:
            return audio

        try:
            from configs.dataset_config import AUGMENTATION as AUG
        except Exception:
            AUG = None

        p_time = AUG.get("p_time_mask", 0.5) if AUG else 0.5
        time_max = AUG.get("time_mask_max_band", 0.15) if AUG else 0.15
        p_freq = AUG.get("p_freq_mask", 0.5) if AUG else 0.5
        freq_max = AUG.get("freq_mask_max_band", 0.2) if AUG else 0.2

        a = audio.astype(np.float32)
        # additive gaussian noise (small)
        sigma = 0.01 * (a.std() if a.size else 1.0)
        a = a + np.random.normal(scale=sigma, size=a.shape).astype(np.float32)

        # time mask (SpecAugment style)
        if a.shape[-1] > 4 and random.random() < p_time:
            T = a.shape[-1]
            mask_len = int(random.uniform(0.02, time_max) * T)
            start = random.randint(0, max(0, T - mask_len))
            a[:, :, start:start+mask_len] = a.min() if a.size else 0.0

        # freq mask
        if a.shape[1] > 4 and random.random() < p_freq:
            F = a.shape[1]
            mask_band = int(random.uniform(0.01, freq_max) * F)
            if mask_band > 0:
                start_f = random.randint(0, max(0, F - mask_band))
                a[:, start_f:start_f+mask_band, :] = a.min() if a.size else 0.0

        return a

    def _augment_sensor_numpy(self, sensor):
        """Sensor augmentation: small Gaussian noise."""
        if sensor is None:
            return sensor

        try:
            from configs.dataset_config import AUGMENTATION as AUG
        except Exception:
            AUG = None

        p_dropout = AUG.get("p_channel_dropout", 0.15) if AUG else 0.15
        p_scale = AUG.get("p_channel_scale", 0.3) if AUG else 0.3
        scale_max = AUG.get("channel_scale_max", 0.05) if AUG else 0.05

        s = sensor.astype(np.float32)
        # gaussian noise
        sigma = 0.01 * (s.std() if s.size else 1.0)
        s = s + np.random.normal(scale=sigma, size=s.shape).astype(np.float32)

        # per-channel small scaling
        if random.random() < p_scale:
            C = s.shape[1] if s.ndim > 1 else 1
            scales = 1.0 + np.random.uniform(-scale_max, scale_max, size=(C,))
            s = s * scales.reshape((1, -1)) if s.ndim > 1 else s * scales[0]

        # channel dropout
        if random.random() < p_dropout and s.ndim > 1:
            C = s.shape[1]
            drop_ch = random.randint(0, C-1)
            s[:, drop_ch] = 0.0

        return s

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for DataLoader (delegates to module-level function).
        
        This static method allows accessing collate_fn via dataset.collate_fn
        for backward compatibility with test code.
        """
        return _collate_fn_impl(batch)


def _collate_fn_impl(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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


# Module-level alias for backward compatibility
collate_fn = _collate_fn_impl

__all__ = ["WeldingDataset", "collate_fn"]
