from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import os
from fractions import Fraction
import torch
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

class EncodedGif():
    def __init__(
        self,
        file_path: str,
        video_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            file_path (str): the file path for the gif.
            video_name (Optional - str): the name of the video
        """
        self._file_path = file_path
        self._video_name = video_name

        try:
            self._container = Image.open(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open video {video_name}. {e}")

        self._video_start_pts = 0.0

        self._duration = self._calculate_duration()
        self._file_size = os.path.getsize(file_path) * 8

    def _calculate_duration(self) -> float:
        """
        Returns:
            duration: the video's duration in seconds.
        """
        duration = 0
        for i in range(self._container.n_frames):
            self._container.seek(i)
            duration += self._container.info.get('duration', 1)

        self._container.seek(0)
        return duration / 1000

    @property
    def rate(self) -> Union[str, Fraction]:
        """
        Returns:
            rate: the frame rate of the video
        """
        return self._container.n_frames / self._duration

    @property
    def bit_rate(self) -> int:
        """
        Returns:
            bit_rate: the bit rate of the underlying video
        """
        return self._file_size / self._duration

    @property
    def pix_fmt(self) -> int:
        """
        Returns:
            pix_fmt: the pixel format of the underlying video
        """
        return 'bgra'

    @property
    def name(self) -> Optional[str]:
        """
        Returns:
            name: the name of the stored video if set.
        """
        return self._video_name

    @property
    def duration(self) -> float:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self._duration

    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds). Returned frames will be in
        [start_sec, end_sec).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            clip_data:
                A dictionary mapping the entries at "video" to a tensors.

                "video": A tensor of the clip's RGB frames with shape:
                (channel, time, height, width). The frames are of type torch.float32 and
                in the range [0 - 255].

            Returns None if no video found within time range.
        """
        self._video = self._gif_decode_video(start_sec, end_sec)

        video_frames = None
        if self._video is not None or len(self._video) != 0:
            video_frames = [
                f
                for f, pts in self._video
                if pts >= start_sec and pts < end_sec
            ]

        if video_frames is None or len(video_frames) == 0:
            logger.debug(
                f"No video found within {start_sec} and {end_sec} seconds. "
                f"Video starts at time 0 and ends at {self.duration}."
            )

            video_frames = None

        if video_frames is not None:
            video_frames = torch.stack(video_frames).permute(3, 0, 1, 2).to(torch.float32)

        return {
            "video": video_frames
        }

    def close(self):
        """
        Closes the internal video container.
        """
        if self._container is not None:
            self._container.close()

    def _gif_decode_video(
        self, start_secs: float = 0.0, end_secs: float = math.inf
    ) -> float:
        """
        Selectively decodes a video between start_pts and end_pts in time units of the
        self._video's timebase.
        """
        try:
            video_and_pts = []
            now = 0
            for i in range(self._container.n_frames):
                self._container.seek(i)
                frame_duration = self._container.info.get('duration', 1) / 1000
                if ((now >= start_secs and now < end_secs) 
                or (now < start_secs and end_secs >= now + frame_duration)):
                    frame = self._container.convert("RGB")  # Convert to RGB
                    #frames.append(transforms.ToTensor()(frame))
                    video_and_pts.append((torch.from_numpy(np.array(frame)), now))
                elif now >= end_secs:
                    break
                now += frame_duration

            self._container.seek(0)

        except Exception as e:
            logger.debug(f"Failed to decode video: {self._video_name}. {e}")

        return video_and_pts