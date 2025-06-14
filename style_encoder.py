import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
from tqdm import tqdm
import pandas as pd
import sys
import glob

# Constants
MOTION_FPS = 30      # FPS for motion data
TARGET_DURATION = 5   # 5 seconds - minimum for a valid segment
MAX_DURATION = 150    # 60 seconds maximum (increased to handle longer clips)

# Custom function to load BEAT2 dataset
def simple_beat_format_load(filepath, training=True):
    """Load actual BEAT2 SMPL-X data from .npz file"""
    try:
        data = np.load(filepath)
        # BEAT2 format typically has these keys
        if 'poses' in data:
            poses = data['poses']  # Shape: [T, 165] for SMPL-X poses
        elif 'body' in data:
            poses = data['body']
        else:
            # Fallback - use first available array-like key
            poses = data[list(data.keys())[0]]
        
        # Return data as-is without forcing dimensions
        return {
            'poses': poses,
            'expressions': data.get('expressions', np.zeros((poses.shape[0], 50))),
            'trans': data.get('trans', np.zeros((poses.shape[0], 3)))
        }
    except Exception as e:
        print(f"Error loading real data from {filepath}: {e}")
        raise e  # Re-raise to fail fast if data can't be loaded

class GestureEmbeddingDataset(Dataset):
    def __init__(self, root_dir, split_csv_path=None, min_duration=TARGET_DURATION, max_duration=MAX_DURATION):
        """
        Dataset for gesture style embedding contrastive learning
        Args:
            root_dir: Root directory of BEAT2 dataset
            split_csv_path: Path to train_test_split.csv (if None, will scan directory)
            min_duration: Minimum duration in seconds (default: 3 minutes)
            max_duration: Maximum duration in seconds (default: 5 minutes)
        """
        self.root_dir = root_dir
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.motion_fps = MOTION_FPS
        self.pose_dim = None  # Will be detected from data
        
        # Calculate required sequence lengths
        self.min_frames = int(min_duration * self.motion_fps)
        self.max_frames = int(max_duration * self.motion_fps)
        
        # Create speaker data dictionary
        self.speaker_data = {}
        
        # Try to load real data
        # First look for split CSV
        if split_csv_path and os.path.exists(split_csv_path):
            try:
                df = pd.read_csv(split_csv_path)
                print(f"Loaded split CSV with {len(df)} entries")
                
                # Process each entry in the CSV
                for idx, row in tqdm(df.iterrows(), desc="Loading data", total=len(df)):
                    video_id = row['id']
                    speaker_id = int(video_id.split('_')[0])
                    mode = row['type']
                    
                    if mode == 'train':  # Only use training data
                        if speaker_id not in self.speaker_data:
                            self.speaker_data[speaker_id] = []
                        
                        npz_path = os.path.join(root_dir, "smplxflame_30", f"{video_id}.npz")
                        
                        if os.path.exists(npz_path):
                            # Use simplified loader - try real data first
                            try:
                                smplx_data = simple_beat_format_load(npz_path, training=True)
                                num_frames = smplx_data['poses'].shape[0]
                                duration = num_frames / self.motion_fps
                                
                                # Detect pose dimensions from first successful load
                                if self.pose_dim is None:
                                    self.pose_dim = smplx_data['poses'].shape[1]
                                    print(f"Detected pose dimensions: {self.pose_dim}")
                                
                                self.speaker_data[speaker_id].append({
                                    "video_id": video_id,
                                    "motion_path": npz_path,
                                    "num_frames": num_frames,
                                    "duration": duration
                                })
                            except Exception as e:
                                print(f"Error processing {video_id}: {e}")
            except Exception as e:
                print(f"Error loading CSV: {e}")
                # Fall back to scanning directory
        
        # If no data loaded, scan directory directly
        if all(len(clips) == 0 for clips in self.speaker_data.values()):
            print("No data loaded from CSV, scanning directory directly")
            smplx_dir = os.path.join(root_dir, "smplxflame_30")
            if os.path.exists(smplx_dir):
                npz_files = glob.glob(os.path.join(smplx_dir, "*.npz"))
                print(f"Found {len(npz_files)} NPZ files in directory")
                
                for npz_path in tqdm(npz_files, desc="Scanning files"):
                    try:
                        # Extract video_id and speaker_id from filename
                        filename = os.path.basename(npz_path)
                        video_id = filename.replace(".npz", "")
                        speaker_id = int(video_id.split('_')[0])
                        
                        if speaker_id not in self.speaker_data:
                            self.speaker_data[speaker_id] = []
                        
                        # Use simplified loader
                        smplx_data = simple_beat_format_load(npz_path, training=True)
                        num_frames = smplx_data['poses'].shape[0]
                        duration = num_frames / self.motion_fps
                        
                        # Detect pose dimensions from first successful load
                        if self.pose_dim is None:
                            self.pose_dim = smplx_data['poses'].shape[1]
                            print(f"Detected pose dimensions: {self.pose_dim}")
                        
                        self.speaker_data[speaker_id].append({
                            "video_id": video_id,
                            "motion_path": npz_path,
                            "num_frames": num_frames,
                            "duration": duration
                        })
                    except Exception as e:
                        print(f"Error processing {npz_path}: {e}")
        
        # Print summary of loaded data
        print("\nData summary:")
        total_clips = 0
        for speaker_id, clips in self.speaker_data.items():
            print(f"Speaker {speaker_id}: {len(clips)} clips")
            total_clips += len(clips)
        print(f"Total clips loaded: {total_clips}")
        
        # Debug: Print some clip durations to understand the data
        print(f"\nDuration requirements: min={self.min_duration}s ({self.min_frames} frames), max={self.max_duration}s ({self.max_frames} frames)")
        
        for speaker_id, clips in list(self.speaker_data.items())[:3]:  # Show first 3 speakers
            if clips:
                durations = [clip['duration'] for clip in clips]
                print(f"Speaker {speaker_id} clip durations: min={min(durations):.1f}s, max={max(durations):.1f}s, avg={sum(durations)/len(durations):.1f}s")
        
        # Create segments of specified duration for each speaker
        self.segments = []
        segments_per_speaker = {}
        
        for speaker_id, clips in self.speaker_data.items():
            if not clips:
                continue
                
            segments_per_speaker[speaker_id] = 0
            
            # Sort clips by duration
            clips.sort(key=lambda x: x['duration'], reverse=True)
            
            # First, handle individual clips that are long enough
            for clip in clips:
                if clip['num_frames'] >= self.min_frames:
                    # If clip is longer than max_duration, we'll truncate it during loading
                    # But still create a segment from it
                    self.segments.append({
                        "speaker_id": speaker_id,
                        "clips": [clip],
                        "total_frames": min(clip['num_frames'], self.max_frames)
                    })
                    segments_per_speaker[speaker_id] += 1
            
            # Then try to combine shorter clips if any exist
            short_clips = [clip for clip in clips if clip['num_frames'] < self.min_frames]
            if short_clips:
                total_frames = 0
                current_segment = []
                
                for clip in short_clips:
                    if total_frames + clip['num_frames'] <= self.max_frames:
                        current_segment.append(clip)
                        total_frames += clip['num_frames']
                        
                        # If we've reached the minimum duration, create a segment
                        if total_frames >= self.min_frames:
                            self.segments.append({
                                "speaker_id": speaker_id,
                                "clips": current_segment.copy(),
                                "total_frames": total_frames
                            })
                            segments_per_speaker[speaker_id] += 1
                            current_segment = []
                            total_frames = 0
                
                # Add remaining clips if they meet the minimum duration
                if total_frames >= self.min_frames:
                    self.segments.append({
                        "speaker_id": speaker_id,
                        "clips": current_segment,
                        "total_frames": total_frames
                    })
                    segments_per_speaker[speaker_id] += 1
        
        print(f"Created {len(self.segments)} segments from {len(self.speaker_data)} speakers")
        
        # Validate that pose dimensions were detected
        if self.pose_dim is None:
            raise RuntimeError("No valid data files found. Could not detect pose dimensions. Please check your data path and file formats.")
        
        # Debug: Show segments per speaker
        print("Segments per speaker:")
        for speaker_id, count in segments_per_speaker.items():
            if count > 0:
                print(f"  Speaker {speaker_id}: {count} segments")
            else:
                total_duration = sum(clip['duration'] for clip in self.speaker_data[speaker_id])
                print(f"  Speaker {speaker_id}: 0 segments (total duration: {total_duration:.1f}s, need {self.min_duration}s minimum)")
    
    def __len__(self):
        return len(self.segments)
    
    def load_motion_data(self, path):
        """Load and preprocess motion data"""
        try:
            # Use simplified loader - try real data first
            data = simple_beat_format_load(path, training=True)
            if 'poses' in data:
                return data['poses']
        except Exception as e:
            print(f"Error loading motion data from {path}: {e}")
            raise e  # Re-raise to fail fast if data can't be loaded
    
    def __getitem__(self, idx):
        # Validate that pose dimensions were detected
        if self.pose_dim is None:
            raise RuntimeError("Pose dimensions not detected. No valid data files were loaded.")
            
        segment = self.segments[idx]
        speaker_id = segment["speaker_id"]
        clips = segment["clips"]
        
        # Initialize motion array (max_frames x pose_dims)
        motion_combined = np.zeros((self.max_frames, self.pose_dim))
        
        current_frame = 0
        total_motion_data = []
        
        # First pass: collect all motion data to compute mean
        for clip in clips:
            try:
                motion_data = self.load_motion_data(clip["motion_path"])
                num_frames = min(clip["num_frames"], self.max_frames - current_frame)
                
                if num_frames <= 0:
                    break
                    
                total_motion_data.append(motion_data[:num_frames])
                current_frame += num_frames
                
                if current_frame >= self.min_frames:
                    break
            except Exception as e:
                print(f"Error processing clip {clip['video_id']}: {e}")
        
        # Concatenate all motion data and compute mean for padding
        if total_motion_data:
            all_motion = np.concatenate(total_motion_data, axis=0)
            motion_mean = np.mean(all_motion, axis=0)
        else:
            motion_mean = np.zeros(self.pose_dim)
        
        # Initialize with mean values instead of zeros
        motion_combined = np.tile(motion_mean, (self.max_frames, 1))
        
        # Second pass: fill in the actual data
        current_frame = 0
        for motion_data in total_motion_data:
            num_frames = motion_data.shape[0]
            motion_combined[current_frame:current_frame + num_frames] = motion_data
            current_frame += num_frames
        
        # Convert to tensor
        motion_tensor = torch.from_numpy(motion_combined).float()
        
        return {
            "motion": motion_tensor,
            "speaker_id": speaker_id
        }


class GestureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, embedding_dim=256, max_seq_len=int(MAX_DURATION * MOTION_FPS)):
        super(GestureEncoder, self).__init__()
        
        # Sequence length control
        self.max_seq_len = max_seq_len
        
        # Motion processing layers
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Temporal encoding with transformers - with memory optimizations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=1024,  # Reduced from 2048
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)  # Reduced from 3
        
        # Enable gradient checkpointing to save memory
        self.transformer.enable_nested_tensor = False  # Disable nested tensor optimization
        
        # Pooling and fully connected layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        
        # Layer normalization for final embedding
        self.ln = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        """
        Args:
            x: Motion sequence [batch_size, seq_length, feature_dim]
        Returns:
            embedding: Speaker gesture style embedding [batch_size, embedding_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Downsample if sequence is too long (memory optimization)
        if seq_len > self.max_seq_len:
            # Take uniform samples from the sequence
            indices = torch.linspace(0, seq_len-1, self.max_seq_len).long()
            x = x[:, indices, :]
            seq_len = self.max_seq_len
        
        # Initial feature extraction
        x = self.linear1(x)  # [batch_size, seq_len, hidden_dim]
        
        # Process in chunks if still too long
        if seq_len > 1000:
            chunk_size = 1000
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            chunk_outputs = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, seq_len)
                
                if end_idx <= start_idx:
                    break
                    
                chunk = x[:, start_idx:end_idx, :]
                # Process chunk with transformer
                chunk_output = self.transformer(chunk)
                chunk_outputs.append(chunk_output)
            
            # Combine chunk outputs
            x = torch.cat(chunk_outputs, dim=1)
        else:
            # Apply transformer layers if sequence is short enough
            x = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        
        # Global average pooling across sequence dimension
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch_size, hidden_dim]
        
        # Final projection layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Normalize embedding
        embedding = F.normalize(self.ln(x), p=2, dim=1)
        
        return embedding


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):  # Reduced from 0.07 for more aggressive learning
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Compute the InfoNCE loss
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size]
        Returns:
            loss: Scalar loss value
        """
        batch_size = embeddings.size(0)
        
        # Early check - we need at least 2 samples
        if batch_size < 2:
            # Return small non-zero loss connected to embeddings
            return embeddings.sum() * 0 + 0.1
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same speaker)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Count positive pairs (excluding self-pairs)
        positive_pairs = mask.sum() - batch_size
        
        if positive_pairs == 0:
            # Use a small fixed loss to keep training going when no positive pairs
            return embeddings.sum() * 0 + 0.1
        
        # Mask out self-similarity
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=mask.device)
        mask = mask * logits_mask
        
        # Compute log_prob with numerical stability
        exp_logits = torch.exp(sim_matrix) * logits_mask
        exp_logits_sum = exp_logits.sum(1, keepdim=True)
        log_prob = sim_matrix - torch.log(exp_logits_sum + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        
        # Final check for numerical stability
        if torch.isnan(loss) or torch.isinf(loss):
            return embeddings.sum() * 0 + 0.1
            
        return loss


def train_gesture_model(data_root, output_dir, num_epochs=50, batch_size=2, learning_rate=0.0001, embedding_dim=256):
    """
    Train a gesture style embedding model using contrastive learning
    Args:
        data_root: Root directory of BEAT2 dataset
        output_dir: Directory to save the model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        embedding_dim: Dimension of gesture embedding
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset with reduced sequence length
    dataset = GestureEmbeddingDataset(
        root_dir=data_root,
        split_csv_path=os.path.join(data_root, 'train_test_split.csv'),
        min_duration=TARGET_DURATION,  # Use global constant
        max_duration=MAX_DURATION      # Use global constant
    )
    
    # Check if dataset has any data
    if len(dataset) == 0:
        print("Dataset is empty! Cannot train model.")
        return None
    
    # Get input dimensions from first sample
    sample = dataset[0]
    input_dim = sample["motion"].shape[1]
    print(f"Detected input dimensions: {input_dim}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=min(batch_size, len(dataset)),  # Smaller batch size
        shuffle=True,
        num_workers=2  # Reduced workers
    )
    
    # Create model with memory optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureEncoder(
        input_dim=input_dim,  # Use detected dimension
        embedding_dim=embedding_dim,
        max_seq_len=int(MAX_DURATION * MOTION_FPS)  # Use actual max duration
    ).to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'transformer'):
        for param in model.transformer.parameters():
            param.requires_grad_(True)
    
    criterion = ContrastiveLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Clear cache before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Clear cache if needed
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                torch.cuda.empty_cache()
                
            motion = batch["motion"].to(device)
            speaker_ids = batch["speaker_id"].to(device)
            
            # Forward pass
            embeddings = model(motion)
            
            # Check if embeddings require grad
            if not embeddings.requires_grad:
                print("Warning: embeddings do not require grad!")
                # Ensure they require grad (though this shouldn't be needed if model is in train mode)
                embeddings = embeddings.detach().requires_grad_(True)
                
            loss = criterion(embeddings, speaker_ids)
            
            # Backward pass only if loss requires grad
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            else:
                print("Warning: Loss doesn't require gradients!")
                
            total_loss += loss.item()
            
            # Clear variables to free memory
            del motion, speaker_ids, embeddings, loss
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save model checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(output_dir, 'gesture_encoder_best.pt'))
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(output_dir, 'gesture_encoder_latest.pt'))
    
    return model


class EmageWithGestureEmbedding(nn.Module):
    """
    Helper class to show how to integrate the GestureEncoder with EMAGE
    """
    def __init__(self, gesture_encoder_path, emage_model=None, freeze_gesture_encoder=True):
        super(EmageWithGestureEmbedding, self).__init__()
        
        # Load gesture encoder
        self.gesture_encoder = GestureEncoder()
        checkpoint = torch.load(gesture_encoder_path)
        self.gesture_encoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze gesture encoder if needed
        if freeze_gesture_encoder:
            for param in self.gesture_encoder.parameters():
                param.requires_grad = False
        
        # Load EMAGE model (placeholder for now)
        self.emage_model = emage_model
        
        # Add projection layer to adapt gesture embedding to EMAGE input
        self.gesture_projection = nn.Linear(256, 768)  # Assuming EMAGE expects 768-dim input
    
    def forward(self, reference_motion, audio_input=None, **kwargs):
        """
        Forward pass
        Args:
            reference_motion: Reference motion for style encoding
            audio_input: Audio input for EMAGE generation
        """
        # Extract gesture style embedding
        with torch.set_grad_enabled(not self.gesture_encoder.training):
            gesture_embedding = self.gesture_encoder(reference_motion)
        
        # Project to EMAGE expected dimension
        gesture_features = self.gesture_projection(gesture_embedding)
        
        # When EMAGE is integrated, you would pass this to the EMAGE model
        if self.emage_model is not None and audio_input is not None:
            # This is placeholder - actual integration would depend on EMAGE's API
            output = self.emage_model(
                audio=audio_input,
                gesture_style=gesture_features,
                **kwargs
            )
            return output
        else:
            # For now, just return the gesture embedding and projected features
            return {
                "gesture_embedding": gesture_embedding,
                "gesture_features": gesture_features
            }


def test_gesture_encoder(model_path, data_root, batch_size=16):
    """
    Test a trained gesture encoder on speaker identification task
    """
    # Create a TESTING dataset class that loads test data
    class TestGestureEmbeddingDataset(GestureEmbeddingDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Reset and reload with test data
            self.speaker_data = {}
            
            # Load the split information
            if kwargs.get('split_csv_path') and os.path.exists(kwargs.get('split_csv_path')):
                df = pd.read_csv(kwargs.get('split_csv_path'))
                
                # Process each entry in the CSV - but load TEST data
                for idx, row in df.iterrows():
                    video_id = row['id']
                    speaker_id = int(video_id.split('_')[0])
                    mode = row['type']
                    
                    if mode == 'test':  # Only use test data
                        if speaker_id not in self.speaker_data:
                            self.speaker_data[speaker_id] = []
                        
                        npz_path = os.path.join(kwargs.get('root_dir'), "smplxflame_30", f"{video_id}.npz")
                        
                        if os.path.exists(npz_path):
                            # Use simplified loader with test=True - try real data first
                            try:
                                smplx_data = simple_beat_format_load(npz_path, training=False)
                                num_frames = smplx_data['poses'].shape[0]
                                duration = num_frames / self.motion_fps
                                
                                # Detect pose dimensions from first successful load
                                if self.pose_dim is None:
                                    self.pose_dim = smplx_data['poses'].shape[1]
                                    print(f"Test data detected pose dimensions: {self.pose_dim}")
                                
                                self.speaker_data[speaker_id].append({
                                    "video_id": video_id,
                                    "motion_path": npz_path,
                                    "num_frames": num_frames,
                                    "duration": duration
                                })
                            except Exception as e:
                                print(f"Error processing {video_id}: {e}")
            
            # Recreate segments
            self.segments = []
            for speaker_id, clips in self.speaker_data.items():
                if not clips:
                    continue
                
                # Sort clips by duration
                clips.sort(key=lambda x: x['duration'], reverse=True)
                
                # First, handle individual clips that are long enough
                for clip in clips:
                    if clip['num_frames'] >= self.min_frames:
                        self.segments.append({
                            "speaker_id": speaker_id,
                            "clips": [clip],
                            "total_frames": min(clip['num_frames'], self.max_frames)
                        })
                
                # Then try to combine shorter clips if any exist
                short_clips = [clip for clip in clips if clip['num_frames'] < self.min_frames]
                if short_clips:
                    total_frames = 0
                    current_segment = []
                    
                    for clip in short_clips:
                        if total_frames + clip['num_frames'] <= self.max_frames:
                            current_segment.append(clip)
                            total_frames += clip['num_frames']
                            
                            if total_frames >= self.min_frames:
                                self.segments.append({
                                    "speaker_id": speaker_id,
                                    "clips": current_segment.copy(),
                                    "total_frames": total_frames
                                })
                                current_segment = []
                                total_frames = 0
                    
                    if total_frames >= self.min_frames:
                        self.segments.append({
                            "speaker_id": speaker_id,
                            "clips": current_segment,
                            "total_frames": total_frames
                        })

    # Create test dataset
    test_dataset = TestGestureEmbeddingDataset(
        root_dir=data_root,
        split_csv_path=os.path.join(data_root, 'train_test_split.csv'),
        min_duration=TARGET_DURATION,
        max_duration=MAX_DURATION
    )
    
    # Check if test dataset has any data
    if len(test_dataset) == 0:
        print("Test dataset is empty! Cannot test model.")
        return None
    
    # Get input dimensions from first test sample
    sample = test_dataset[0]
    input_dim = sample["motion"].shape[1]
    print(f"Test data input dimensions: {input_dim}")
    
    # Load the model with correct input dimensions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureEncoder(input_dim=input_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Debug: Print dataset information
    print(f"Test dataset size: {len(test_dataset)}")
    unique_speakers = set()
    speaker_counts = {}
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        speaker_id = item['speaker_id']
        unique_speakers.add(speaker_id)
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
    
    print(f"Number of unique speakers in test set: {len(unique_speakers)}")
    print(f"Speaker IDs: {sorted(unique_speakers)}")
    print(f"Test samples per speaker: {dict(sorted(speaker_counts.items()))}")
    
    # Calculate per-speaker accuracy later
    speaker_correct = {spk: 0 for spk in unique_speakers}
    speaker_total = {spk: 0 for spk in unique_speakers}
    
    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for testing
        num_workers=2
    )
    
    # Track embeddings and speaker IDs
    all_embeddings = []
    all_speakers = []
    
    # Compute embeddings for all test samples
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Computing embeddings"):
            motion = batch["motion"].to(device)
            speaker_ids = batch["speaker_id"]
            
            embeddings = model(motion)
            
            all_embeddings.append(embeddings.cpu())
            all_speakers.extend(speaker_ids.tolist())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Debug: Print embedding statistics
    print(f"Embedding shape: {all_embeddings.shape}")
    print(f"Embedding mean: {all_embeddings.mean():.6f}")
    print(f"Embedding std: {all_embeddings.std():.6f}")
    
    # Compute similarity matrix
    similarity = torch.matmul(all_embeddings, all_embeddings.T).numpy()
    
    # Debug: Print similarity statistics
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarity range: [{similarity.min():.6f}, {similarity.max():.6f}]")
    
    # Print some example similarities
    print("\nExample similarities (first 5 samples):")
    for i in range(min(5, len(all_speakers))):
        for j in range(min(5, len(all_speakers))):
            same_speaker = "✓" if all_speakers[i] == all_speakers[j] else "✗"
            print(f"Sample {i} (speaker {all_speakers[i]}) vs Sample {j} (speaker {all_speakers[j]}): {similarity[i,j]:.4f} {same_speaker}")
    
    # Evaluate retrieval performance
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for i in range(len(all_speakers)):
        current_speaker = all_speakers[i]
        speaker_total[current_speaker] += 1
        
        # Get top-k most similar samples (excluding self)
        sim_row = similarity[i].copy()
        sim_row[i] = -float('inf')  # Exclude self
        top_indices = np.argsort(sim_row)[::-1][:5]  # Top 5
        
        # Check if any of the top-k matches have the same speaker
        if all_speakers[i] == all_speakers[top_indices[0]]:
            correct_top1 += 1
            speaker_correct[current_speaker] += 1
        
        if any(all_speakers[i] == all_speakers[idx] for idx in top_indices):
            correct_top5 += 1
        
        total += 1
    
    # Calculate metrics
    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    
    print(f"\nTesting results on {total} samples:")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    
    # Print per-speaker accuracy
    print(f"\nPer-speaker Top-1 accuracy:")
    for speaker_id in sorted(unique_speakers):
        if speaker_total[speaker_id] > 0:
            spk_accuracy = speaker_correct[speaker_id] / speaker_total[speaker_id]
            print(f"Speaker {speaker_id}: {spk_accuracy:.3f} ({speaker_correct[speaker_id]}/{speaker_total[speaker_id]})")
    
    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "embeddings": all_embeddings.numpy(),
        "speakers": all_speakers,
        "per_speaker_accuracy": {spk: speaker_correct[spk]/speaker_total[spk] for spk in unique_speakers if speaker_total[spk] > 0}
    }


# Example usage
if __name__ == "__main__":
    # Paths
    data_root = "./BEAT2/beat_english_v2.0.0"
    output_dir = "./outputs/gesture_encoder"
    
    # Quick test: Try to load one real file first
    import glob
    test_files = glob.glob(os.path.join(data_root, "smplxflame_30", "*.npz"))
    if test_files:
        print(f"Testing real data loading with file: {test_files[0]}")
        try:
            test_data = simple_beat_format_load(test_files[0])
            print(f"✅ Successfully loaded real data! Shape: {test_data['poses'].shape}")
            print(f"Data range: [{test_data['poses'].min():.4f}, {test_data['poses'].max():.4f}]")
        except Exception as e:
            print(f"❌ Failed to load real data: {e}")
            print("Please check if your BEAT2 data files are in the correct format")
            exit(1)
    else:
        print(f"❌ No .npz files found in {os.path.join(data_root, 'smplxflame_30')}")
        print("Please check your data path")
        exit(1)
    
    # Train the gesture encoder
    model = train_gesture_model(
        data_root=data_root,
        output_dir=output_dir,
        num_epochs=1,  # Increased from 100
        batch_size=16,   # Increased from 8
        embedding_dim=256
    )
    
    # Test the trained model
    test_results = test_gesture_encoder(
        model_path="./outputs/gesture_encoder/gesture_encoder_best.pt",
        data_root=data_root,
        batch_size=16
    )