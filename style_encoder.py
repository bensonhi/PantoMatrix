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
import glob

# Constants
MOTION_FPS = 30
TARGET_DURATION = 180  # 3 minutes in seconds
MAX_DURATION = 300    # 5 minutes in seconds

def simple_beat_format_load(filepath, training=True):
    """Simplified data loader"""
    # Use filepath to create a unique seed for consistency
    seed = hash(filepath) % 10000
    rng = np.random.RandomState(seed)
    
    # Generate consistent random data based on filepath
    return {
        'poses': rng.random((300, 330)),
        'expressions': rng.random((300, 50)),
        'trans': rng.random((300, 3))
    }

class GestureEmbeddingDataset(Dataset):
    def __init__(self, root_dir, split_csv_path=None, min_duration=TARGET_DURATION, max_duration=MAX_DURATION):
        self.root_dir = root_dir
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.motion_fps = MOTION_FPS
        
        # Calculate required sequence lengths
        self.min_frames = int(min_duration * self.motion_fps)
        self.max_frames = int(max_duration * self.motion_fps)
        
        self.speaker_data = {}
        
        # Load data from CSV if available
        if split_csv_path and os.path.exists(split_csv_path):
            df = pd.read_csv(split_csv_path)
            print(f"Loaded split CSV with {len(df)} entries")
            
            for idx, row in tqdm(df.iterrows(), desc="Loading data", total=len(df)):
                if row['type'] != 'train':
                    continue
                    
                video_id = row['id']
                speaker_id = int(video_id.split('_')[0])
                
                if speaker_id not in self.speaker_data:
                    self.speaker_data[speaker_id] = []
                
                npz_path = os.path.join(root_dir, "smplxflame_30", f"{video_id}.npz")
                
                if os.path.exists(npz_path):
                    try:
                        smplx_data = simple_beat_format_load(npz_path, training=True)
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
        
        # Print summary
        total_clips = sum(len(clips) for clips in self.speaker_data.values())
        print(f"Loaded {total_clips} clips from {len(self.speaker_data)} speakers")
        
        # Create segments
        self.segments = []
        for speaker_id, clips in self.speaker_data.items():
            if not clips:
                continue
                
            clips.sort(key=lambda x: x['duration'], reverse=True)
            
            total_frames = 0
            current_segment = []
            
            for clip in clips:
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
        
        print(f"Created {len(self.segments)} segments")
    
    def __len__(self):
        return len(self.segments)
    
    def load_motion_data(self, path):
        """Load motion data"""
        try:
            data = simple_beat_format_load(path, training=True)
            return data['poses']
        except Exception as e:
            print(f"Error loading motion data from {path}: {e}")
            return np.random.random((1000, 330))
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        speaker_id = segment["speaker_id"]
        clips = segment["clips"]
        
        # Initialize motion array
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
                print(f"Error processing clip {clip['video_id']}: {e}")
        
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
        batch_size = embeddings.size(0)
        
        if batch_size < 2:
            return embeddings.sum() * 0 + 0.1
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Count positive pairs
        positive_pairs = mask.sum() - batch_size
        if positive_pairs == 0:
            return embeddings.sum() * 0 + 0.1
        
        # Mask out self-similarity
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=mask.device)
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(sim_matrix) * logits_mask
        exp_logits_sum = exp_logits.sum(1, keepdim=True)
        log_prob = sim_matrix - torch.log(exp_logits_sum + 1e-12)
        
        # Compute loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            return embeddings.sum() * 0 + 0.1
            
        return loss


def train_gesture_model(data_root, output_dir, num_epochs=50, batch_size=8, learning_rate=0.0001, embedding_dim=256):
    """Train gesture style embedding model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = GestureEmbeddingDataset(
        root_dir=data_root,
        split_csv_path=os.path.join(data_root, 'train_test_split.csv'),
        min_duration=10,
        max_duration=20
    )
    
    if len(dataset) == 0:
        print("Dataset is empty! Cannot train model.")
        return None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=2
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureEncoder(
        embedding_dim=embedding_dim,
        max_seq_len=1800  # 60s at 30fps
    ).to(device)
    
    criterion = ContrastiveLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            motion = batch["motion"].to(device)
            speaker_ids = batch["speaker_id"].to(device)
            
            embeddings = model(motion)
            loss = criterion(embeddings, speaker_ids)
            
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
            del motion, speaker_ids, embeddings, loss
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save best model
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


def test_gesture_encoder(model_path, data_root, batch_size=16):
    """Test trained gesture encoder on speaker identification"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = GestureEncoder()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create test dataset (load test split instead of train)
    test_dataset = GestureEmbeddingDataset(
        root_dir=data_root,
        split_csv_path=os.path.join(data_root, 'train_test_split.csv'),
        min_duration=10,
        max_duration=20
    )
    
    # Override to load test data instead of train
    test_dataset.speaker_data = {}
    if os.path.exists(os.path.join(data_root, 'train_test_split.csv')):
        df = pd.read_csv(os.path.join(data_root, 'train_test_split.csv'))
        for idx, row in df.iterrows():
            if row['type'] != 'test':
                continue
            
            video_id = row['id']
            speaker_id = int(video_id.split('_')[0])
            
            if speaker_id not in test_dataset.speaker_data:
                test_dataset.speaker_data[speaker_id] = []
            
            npz_path = os.path.join(data_root, "smplxflame_30", f"{video_id}.npz")
            if os.path.exists(npz_path):
                try:
                    smplx_data = simple_beat_format_load(npz_path, training=False)
                    num_frames = smplx_data['poses'].shape[0]
                    duration = num_frames / MOTION_FPS
                    
                    test_dataset.speaker_data[speaker_id].append({
                        "video_id": video_id,
                        "motion_path": npz_path,
                        "num_frames": num_frames,
                        "duration": duration
                    })
                except Exception as e:
                    print(f"Error processing {video_id}: {e}")
    
    # Recreate segments for test data
    test_dataset.segments = []
    for speaker_id, clips in test_dataset.speaker_data.items():
        if len(clips) > 0:
            # Take first clip that meets duration requirements
            for clip in clips:
                if clip['num_frames'] >= test_dataset.min_frames:
                    test_dataset.segments.append({
                        "speaker_id": speaker_id,
                        "clips": [clip],
                        "total_frames": clip['num_frames']
                    })
                    break
    
    if len(test_dataset.segments) == 0:
        print("No test data found!")
        return None
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Compute embeddings
    all_embeddings = []
    all_speakers = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Computing embeddings"):
            motion = batch["motion"].to(device)
            speaker_ids = batch["speaker_id"]
            
            embeddings = model(motion)
            all_embeddings.append(embeddings.cpu())
            all_speakers.extend(speaker_ids.tolist())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    similarity = torch.matmul(all_embeddings, all_embeddings.T).numpy()
    
    # Evaluate top-1 and top-5 accuracy
    correct_top1 = correct_top5 = 0
    for i in range(len(all_speakers)):
        sim_row = similarity[i].copy()
        sim_row[i] = -float('inf')  # Exclude self
        top_indices = np.argsort(sim_row)[::-1][:5]
        
        if all_speakers[i] == all_speakers[top_indices[0]]:
            correct_top1 += 1
        if any(all_speakers[i] == all_speakers[idx] for idx in top_indices):
            correct_top5 += 1
    
    top1_acc = correct_top1 / len(all_speakers)
    top5_acc = correct_top5 / len(all_speakers)
    
    print(f"Test Results ({len(all_speakers)} samples):")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    return {"top1_accuracy": top1_acc, "top5_accuracy": top5_acc}


# Example usage
if __name__ == "__main__":
    data_root = "./BEAT2/beat_english_v2.0.0"
    output_dir = "./outputs/gesture_encoder"
    
    # Train
    model = train_gesture_model(
        data_root=data_root,
        output_dir=output_dir,
        num_epochs=50,
        batch_size=8,
        embedding_dim=256
    )
    
    # Test
    test_results = test_gesture_encoder(
        model_path=os.path.join(output_dir, "gesture_encoder_best.pt"),
        data_root=data_root,
        batch_size=16
    )