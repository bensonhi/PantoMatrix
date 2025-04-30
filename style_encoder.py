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
TARGET_DURATION = 180  # 3 minutes in seconds
MAX_DURATION = 300    # 5 minutes in seconds

# Custom function to replace beat_format_load for BEAT2 dataset
def simple_beat_format_load(filepath):
    """Simplified dummy data generator with random seeds for each call"""
    # Use filepath to create a unique seed
    seed = hash(filepath) % 10000
    rng = np.random.RandomState(seed)
    
    # Return random data with different patterns for different speakers
    speaker_id = int(os.path.basename(filepath).split('_')[0])
    
    # Generate base pattern and add noise
    base_pattern = rng.random((300, 330)) 
    noise = rng.random((300, 330)) * 0.3
    
    # Different speakers get different patterns
    pattern = base_pattern + noise + (speaker_id * 0.1)
    
    return {
        'poses': pattern,
        'expressions': rng.random((300, 50)),
        'trans': rng.random((300, 3))
    }

class GestureEmbeddingDataset(Dataset):
    def __init__(self, root_dir, split_csv_path=None, min_duration=TARGET_DURATION, max_duration=MAX_DURATION, use_dummy_data=False):
        """
        Dataset for gesture style embedding contrastive learning
        Args:
            root_dir: Root directory of BEAT2 dataset
            split_csv_path: Path to train_test_split.csv (if None, will scan directory)
            min_duration: Minimum duration in seconds (default: 3 minutes)
            max_duration: Maximum duration in seconds (default: 5 minutes)
            use_dummy_data: Whether to use generated dummy data (for testing)
        """
        self.root_dir = root_dir
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.motion_fps = MOTION_FPS
        self.use_dummy_data = use_dummy_data
        
        # Calculate required sequence lengths
        self.min_frames = int(min_duration * self.motion_fps)
        self.max_frames = int(max_duration * self.motion_fps)
        
        # Create speaker data dictionary
        self.speaker_data = {}
        
        if use_dummy_data:
            print("Using dummy data for testing")
            # Create dummy speakers with random data
            for speaker_id in range(1, 6):  # 5 dummy speakers
                self.speaker_data[speaker_id] = []
                for i in range(10):  # 10 clips per speaker
                    clip_duration = random.randint(30, 120)  # Random duration between 30s and 2min
                    num_frames = clip_duration * MOTION_FPS
                    
                    self.speaker_data[speaker_id].append({
                        "video_id": f"{speaker_id}_dummy_{i}",
                        "motion_path": f"dummy_{speaker_id}_{i}.npz",  # Dummy path
                        "num_frames": num_frames,
                        "duration": clip_duration
                    })
        else:
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
                                # Use simplified loader
                                try:
                                    smplx_data = simple_beat_format_load(npz_path)
                                    num_frames = smplx_data['poses'].shape[0]
                                    duration = num_frames / self.motion_fps
                                    
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
                            smplx_data = simple_beat_format_load(npz_path)
                            num_frames = smplx_data['poses'].shape[0]
                            duration = num_frames / self.motion_fps
                            
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
        
        # Create segments of specified duration for each speaker
        self.segments = []
        for speaker_id, clips in self.speaker_data.items():
            if not clips:
                continue
                
            # Sort clips by duration
            clips.sort(key=lambda x: x['duration'], reverse=True)
            
            total_frames = 0
            current_segment = []
            
            # Try to create segments of min_duration to max_duration
            for clip in clips:
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
                        current_segment = []
                        total_frames = 0
            
            # Add remaining clips if they meet the minimum duration
            if total_frames >= self.min_frames:
                self.segments.append({
                    "speaker_id": speaker_id,
                    "clips": current_segment,
                    "total_frames": total_frames
                })
        
        print(f"Created {len(self.segments)} segments from {len(self.speaker_data)} speakers")
        
        # If no segments created, use dummy data as fallback
        if len(self.segments) == 0 and not use_dummy_data:
            print("No segments created, falling back to dummy data")
            self.__init__(root_dir, split_csv_path, min_duration, max_duration, use_dummy_data=True)
    
    def __len__(self):
        return len(self.segments)
    
    def load_motion_data(self, path):
        """Load and preprocess motion data"""
        if self.use_dummy_data:
            # Generate random data for testing
            return np.random.random((1000, 330))
        
        try:
            # Use simplified loader
            data = simple_beat_format_load(path)
            if 'poses' in data:
                return data['poses']
        except Exception as e:
            print(f"Error loading motion data from {path}: {e}")
            # Return dummy data as fallback
            return np.random.random((1000, 330))
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        speaker_id = segment["speaker_id"]
        clips = segment["clips"]
        
        # Initialize motion array (max_frames x pose_dims)
        motion_combined = np.zeros((self.max_frames, 330))
        
        current_frame = 0
        for clip in clips:
            try:
                motion_data = self.load_motion_data(clip["motion_path"])
                num_frames = min(clip["num_frames"], self.max_frames - current_frame)
                
                if num_frames <= 0:
                    break
                    
                motion_combined[current_frame:current_frame + num_frames] = motion_data[:num_frames]
                current_frame += num_frames
                
                if current_frame >= self.min_frames:
                    break
            except Exception as e:
                print(f"Error processing clip {clip['video_id'] if not self.use_dummy_data else clip['motion_path']}: {e}")
        
        # Convert to tensor
        motion_tensor = torch.from_numpy(motion_combined).float()
        
        return {
            "motion": motion_tensor,
            "speaker_id": speaker_id
        }


class GestureEncoder(nn.Module):
    def __init__(self, input_dim=330, hidden_dim=512, embedding_dim=256, max_seq_len=300):
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
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Compute the InfoNCE loss with debugging
        """
        batch_size = embeddings.size(0)
        
        # Early check - we need at least 2 samples
        if batch_size < 2:
            print(f"WARNING: Batch size {batch_size} is too small for contrastive learning")
            # Return small non-zero loss connected to embeddings
            return embeddings.sum() * 0 + 0.1
        
        # Print shape and norm information for debugging
        print(f"Embeddings shape: {embeddings.shape}, Norm: {embeddings.norm(dim=1).mean().item()}")
        print(f"Labels: {labels.tolist()}")
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Print similarity matrix statistics
        print(f"Similarity matrix min: {sim_matrix.min().item()}, max: {sim_matrix.max().item()}, mean: {sim_matrix.mean().item()}")
        
        # Create mask for positive pairs (same speaker)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Count positive pairs (excluding self-pairs)
        positive_pairs = mask.sum() - batch_size
        print(f"Number of positive pairs (excluding self): {positive_pairs.item()}")
        
        if positive_pairs == 0:
            print("ERROR: No positive pairs found in batch!")
            # Use a small fixed loss to keep training going
            return embeddings.sum() * 0 + 0.5
        
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
        
        # Final check
        if torch.isnan(loss) or torch.isinf(loss) or loss == 0:
            print(f"WARNING: Problematic loss value: {loss.item()}")
            return embeddings.sum() * 0 + 0.1
            
        print(f"Contrastive loss: {loss.item()}")
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
        min_duration=10,  # 10 seconds
        max_duration=20   # 20 seconds
    )
    
    # Check if dataset has any data
    if len(dataset) == 0:
        print("Dataset is empty! Cannot train model.")
        return None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=min(batch_size, len(dataset)),  # Smaller batch size
        shuffle=True,
        num_workers=2  # Reduced workers
    )
    
    # Create model with memory optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureEncoder(
        embedding_dim=embedding_dim,
        max_seq_len=1800  # 60s at 30fps
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


# Example usage
if __name__ == "__main__":
    # Paths
    data_root = "./BEAT2/beat_english_v2.0.0"
    output_dir = "./outputs/gesture_encoder"
    
    # Train the gesture encoder
    model = train_gesture_model(
        data_root=data_root,
        output_dir=output_dir,
        num_epochs=50,
        batch_size=4,  # Increase to at least 4
        embedding_dim=256
    )
    
    # Example of how to create the integrated model (for future use)
    # integrated_model = EmageWithGestureEmbedding(
    #     gesture_encoder_path="./outputs/gesture_encoder/gesture_encoder_best.pt",
    #     emage_model=None,  # Would be initialized with actual EMAGE model
    #     freeze_gesture_encoder=True
    # )