import torch
from datasets import load_dataset
from torchvision.transforms import v2

from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError

from torchcodec.decoders import VideoDecoder
from torchcodec.transforms import Resize, CenterCrop

train_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(),
        #v2.RandomApply([
        #    v2.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #], p=0.8),
        #v2.RandomGrayscale(p=0.2),
        #v2.Normalize(mean=[0.54, 0.5, 0.474], std=[0.234, 0.235, 0.231])
    ])


class StatefulClipBuffer():
    def __init__(self, 
                window_size=16,
                clip_size=256,
                target_fps=18.0,
                target_batch_size=64):
        self.window_size = window_size
        self.clip_size = clip_size
        self.target_fps = target_fps
        self.target_batch_size = target_batch_size
        self.window_size
        self.buffer = []
    
    def downsample_frames(total_frames, source_fps, target_fps):
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
                
        return frames_to_keep

    def process_youtube_videos(batch, transform=None):
        print("Spawned")
        window_size = 16
        clip_size = 256
        target_fps = 18.0
        max_clips_per_vid = 48

        ydl_opts = {
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
            'skip_download': True
        }


        features = []
        labels = []
        
        batch_uids = batch['video_uid']
        batch_nodes = batch['nodes']
        for uid, nodes in zip(batch_uids, batch_nodes):
            try:
                # 1. Get the direct stream URL
                url = f'https://www.youtube.com/watch?v={uid}'
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    direct_stream_url = info['url']
                
                #fps = info.get("fps")
                w = int(info.get("width"))
                h = int(info.get("height"))

                target_size = 256
                if w < h:
                    new_w = target_size
                    new_h = int(h * (target_size / w))
                else:
                    new_h = target_size
                    new_w = int(w * (target_size / h))

                # 2. Decode the video stream into frames
                decoder = VideoDecoder(
                    direct_stream_url, 
                    device="cpu", 
                    seek_mode="approximate",  
                    transforms=[
                        Resize(size=(new_h, new_w)),  # Resizes smaller edge to 256
                        CenterCrop(size=(224, 224))   # Crops the center to 224x224
                    ]
                )

                total_frames = decoder.metadata.num_frames
                native_fps = decoder.metadata.average_fps

                if native_fps < target_fps:
                    print(f"Error processing {url}: frame-rate too low")
                    continue
                #input("Please wait")
                frames_to_keep = downsample_frames(total_frames, native_fps, target_fps)
                
                #remove frames that can't form full clip
                remainder = len(frames_to_keep) % clip_size
                frames_to_keep = frames_to_keep[:-remainder]
                total_target_frames = len(frames_to_keep)
                
                num_windows = total_target_frames // window_size
                num_clips = min(total_target_frames // clip_size, max_clips_per_vid)
                print(num_clips)
                window_labels = []
                transition_frames = []
                # Extract transition frames 
                for node in nodes:
                    # Filter by your specific hierarchical level
                    start_time = node["start"]
                    end_time = node["end"]
                    
                    if end_time - start_time >= 4.0: 
                        start_frame = int(round(start_time * target_fps))
                        end_frame = int(round(end_time * target_fps))
                        
                        transition_frames.append(start_frame)
                        transition_frames.append(end_frame)
                
                transition_frames = list(set(transition_frames))

                # Generate non-overlapping window labels
                for i in range(num_windows):
                    start_frame = i*window_size
                    end_frame = start_frame + window_size - 1
                    has_transition = any(start_frame <= t <= end_frame for t in transition_frames)
                    window_label = 1 if has_transition else 0
                    window_labels.append(has_transition)

                window_labels = torch.tensor(window_labels).to(torch.int8)

                # 3. Extract Features
                for i in range(num_clips):
                    start_idx = i*clip_size
                    end_idx = start_idx + clip_size

                    indices = frames_to_keep[start_idx:end_idx]
                    feature = decoder.get_frames_at(indices).data

                    feature = train_transform(feature)
                    label = window_labels[start_idx // window_size : end_idx // window_size]

                    features.append(feature)
                    labels.append(label)

            except ExtractorError as e:
                print(f"Error processing {url}: {e}")
                continue
            except DownloadError as e:
                print(f"Error processing {url}: {e}")
                continue
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
        #print(len(features))
        return {"features": features, "labels": labels}

    def __call__(self, batch):
        batch_uids = batch['video_uid']
        batch_nodes = batch['nodes']
