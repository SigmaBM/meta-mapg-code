import os
from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np
from gym.wrappers.monitoring import video_recorder


class VideoRecorder(object):
    """Wrap env to record rendered image as mp4 video

        (From openai/baselines)
    """
    def __init__(
        self,
        env: gym.Env,
        directory: str,
        record_video_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = 200,
    ) -> None:
        """
            :param directory: path to save videos
            :param record_video_trigger: fucntion that defines when to start recording, takes the
                current number of step, and returns whether we should start recoding or not.
            :param video_length: length of recorded video
        """
        self.env = env
        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.file_prefix = "env"
        self.file_infix = '{}'.format(os.getpid())
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0

    def __getattr__(self, key: str) -> Any:
        return getattr(self.env, key)

    def reset(self) -> np.ndarray:
        obs = self.env.reset()

        if self._video_enabled():
            self.start_video_recorder()

        return obs

    def start_video_recorder(self):
        self.close_video_recorder()

        base_path = os.path.join(self.directory, '{}.video.{}.video{:06}'.
                                 format(self.file_prefix, self.file_infix, self.step_id))
        self.video_recorder = video_recorder.VideoRecorder(env=self.env,
                                                           base_path=base_path,
                                                           metadata={'step_id': self.step_id})
        
        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def close_video_recorder(self):
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 0

    def _video_enabled(self):
        return self.record_video_trigger(self.step_id)

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)

        self.step_id += 1
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                print("Saving video to ", self.video_recorder.path)
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()

        return obs, rews, dones, infos
    
    def render(self) -> None:
        return self.env.render()

    def close(self) -> None:
        self.close_video_recorder()
        self.env.close()
    
    def __del__(self) -> None:
        self.close()
