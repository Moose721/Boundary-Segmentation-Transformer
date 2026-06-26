import torch
from datasets import load_dataset
from torchvision.transforms import v2
import time

from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError

from torchcodec.decoders import VideoDecoder
from torchcodec.transforms import Resize, CenterCrop


class SilentLogger:
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

class StatefulClipBuffer():
    def __init__(self, 
                window_size=16,
                clip_size=256,
                target_fps=18.0,
                max_batch_clips=8):
        self.window_size = window_size
        self.clip_size = clip_size
        self.target_fps = target_fps
        self.max_batch_clips = max_batch_clips

        self.video_clip_ptr = 0
        self.sample_buffer = []
        self.train_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
        ])

        self.ydl_opts = {
            'cookiefile': 'premium_cookies.txt',
            'format': 'bestvideo[ext=mp4]/mp4',  
            'js_runtimes': { 'node': {'path': None} }, 
            'extractor_args': {
                'youtube': {
                    'player_skip': ['webpage', 'configs'],
                    #'player_client': ['web_safari', 'web_creator']
                }
            },
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'ignoreerrors': True,
            'logger': SilentLogger()
        }
    
    def downsample_frames(self, total_frames, source_fps, target_fps):
        """
        Selects the mathematically closest source frame indices for downsampling,
        ensuring a strictly unique and increasing subset (removing indices only).
        """
        if target_fps > source_fps:
            raise ValueError("Target FPS must be less than Source FPS for downsampling.")
            
        # Determine the precise number of frames the output should have
        duration_seconds = total_frames / source_fps
        total_target_frames = round(duration_seconds * target_fps)
        
        frames_to_keep = []
        for i in range(total_target_frames):
            # Calculate exactly where this target frame lands on the source timeline
            exact_source_idx = i * (source_fps / target_fps)
            
            # Round to the absolute nearest frame to minimize distortion
            closest_source_idx = round(exact_source_idx)
            
            # Guard against minor floating-point edge cases at the very end
            if closest_source_idx < total_frames:
                frames_to_keep.append(closest_source_idx)
        
        assert frames_to_keep[-1] < total_frames
        return frames_to_keep


    def initial_batch_process(self, batch):
        batch_uids = batch['video_uid']
        batch_nodes = batch['nodes']
        for uid, nodes in zip(batch_uids, batch_nodes):
            try:
                # 1. Get the direct stream URL
                url = f'https://www.youtube.com/watch?v={uid}'
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    direct_stream_url = info['url']
                
                w = int(info.get("width"))
                h = int(info.get("height"))
                target_size = 256

                new_w = target_size if w < h else int(w * (target_size / h))
                new_h = int(h * (target_size / w)) if w < h else target_size
        
                # 2. Decode the video stream into frames
                decoder = VideoDecoder(
                    direct_stream_url, 
                    device="cpu", 
                    seek_mode="exact",  
                    transforms=[
                        Resize(size=(new_h, new_w)),  # Resizes smaller edge to 256
                        CenterCrop(size=(224, 224))   # Crops the center to 224x224
                    ]
                )

                total_frames = decoder.metadata.num_frames
                native_fps = decoder.metadata.average_fps
              
                if native_fps < self.target_fps:
                    #print(f"Error processing {url}: frame-rate too low")
                    continue
  
                downsampled_frames = self.downsample_frames(total_frames, native_fps, self.target_fps)

                #remove frames that can't form full clip
                remainder = len(downsampled_frames) % self.clip_size
                downsampled_frames = downsampled_frames[:-remainder]
                
                total_windows = len(downsampled_frames) // self.window_size
                total_clips = len(downsampled_frames) // self.clip_size
            
                # Extract transition frames 
                transition_frames = []
                for node in nodes:
                    # Filter by your specific hierarchical level
                    start_time = node["start"]
                    end_time = node["end"]

                    if end_time - start_time >= 4.0: 
                        start_frame = int(round(start_time * self.target_fps))
                        end_frame = int(round(end_time * self.target_fps))
                        transition_frames.append(start_frame)
                        transition_frames.append(end_frame)

                transition_frames = list(set(transition_frames))

                # Generate non-overlapping window labels
                window_labels = []
                for i in range(total_windows):
                    start_frame = i * self.window_size
                    end_frame = start_frame + self.window_size - 1
                    has_transition = any(start_frame <= t <= end_frame for t in transition_frames)
                    window_label = 1 if has_transition else 0
                    window_labels.append(has_transition)

                window_labels = torch.tensor(window_labels).to(torch.int8)
        
                self.sample_buffer.append({
                    'decoder': decoder,
                    'downsampled_frames': downsampled_frames,
                    'window_labels': window_labels,
                    'total_clips': total_clips
                })

            except ExtractorError as e:
                #print(f"Error processing {url}: {e}")
                continue
            except DownloadError as e:
                #print(f"Error processing {url}: {e}")
                continue
            except Exception as e:
                #print(f"Error processing {url}: {e}")
                continue
    
    def __call__(self, batch):
        #print("Exact seek mode start")
        self.initial_batch_process(batch)
        print(f"Buffer length at start: {len(self.sample_buffer)}")
        
        features = []
        labels = []
        clips_processed = 0
        
        remaining_batch_clips = self.max_batch_clips
        while remaining_batch_clips > 0 and len(self.sample_buffer) > 0:
            sample_data = self.sample_buffer[0]
            decoder = sample_data['decoder']
            window_labels = sample_data['window_labels']
            total_clips = sample_data['total_clips']
            
            remaining_video_clips = total_clips - self.video_clip_ptr
            clip_start_idx = self.video_clip_ptr * self.clip_size
            clip_end_idx = (self.video_clip_ptr + remaining_batch_clips)*self.clip_size if remaining_video_clips > remaining_batch_clips else total_clips * self.clip_size
            label_start_idx = clip_start_idx // 16
            label_end_idx = clip_end_idx // 16
            try: 
                indices = sample_data['downsampled_frames'][clip_start_idx : clip_end_idx]
                feature = self.train_transform(decoder.get_frames_at(indices).data)
                label = window_labels[label_start_idx : label_end_idx]
            except RuntimeError as e:
                print("Runtime error while decoding frames occured - skipping to next video")
                self.video_clip_ptr = 0
                del self.sample_buffer[0]
                continue

            if remaining_video_clips > remaining_batch_clips:
                self.video_clip_ptr += remaining_batch_clips
                remaining_batch_clips = 0
            else:
                self.video_clip_ptr = 0
                remaining_batch_clips -= remaining_video_clips
                del self.sample_buffer[0]
            
            split_feature = torch.split(feature, self.clip_size, 0)
            split_label = torch.split(label, self.window_size, 0)
            assert len(split_feature) == len(split_label)
            features.extend(split_feature)
            labels.extend(split_label)

        assert len(features) == self.max_batch_clips
        assert len(labels) == self.max_batch_clips
        
        return {"features": features, "labels": labels}
            
