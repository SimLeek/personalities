from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from displayarray.effects import crop, lens
from pnums import PInt
from torch.optim.optimizer import Optimizer

from personalities.base.actor_critic import ContinualProximalActorCritic
from personalities.base.nn_utils import Lambda
from personalities.base.odst import ODST
from personalities.util.pyramid_head_64 import PyramidHead64

class SingleEyeNet(object):
    def __init__(
            self,
            output_encoding_length,
            extra_input_encoding_length=0,
            frame_head: nn.Module = None,
            fcn: nn.Module = None,
            loss_criteria: nn.modules.loss._Loss = None,
            optimizer: Optimizer = None,
    ):
        if frame_head:
            self.frame_head = frame_head
        else:
            self.frame_head = PyramidHead64()

        if fcn:
            self.fcn = fcn
        else:
            self.fcn = nn.Sequential(
                ODST(
                    in_features=frame_head.output_vector_len + extra_input_encoding_length,
                    num_trees=64,
                    tree_dim=output_encoding_length,
                    flatten_output=False,
                    depth=6,
                ),
                Lambda(lambda x: x.mean(1)),
            )

        if loss_criteria:
            self.loss_criteria = loss_criteria
        else:
            self.loss_criteria = nn.SmoothL1Loss()

        if optimizer:
            self.optimizer = optimizer
        else:
            params = list(self.frame_head.parameters()) + list(self.fcn.parameters())
            self.optimizer = optim.Adam(params)

    def serialize(self):
        state = {
            "frame_head": self.frame_head,
            "optimizer": self.optimizer,
            "optimizer_state": self.optimizer.state_dict(),
            "loss": self.loss_criteria,
            "loss_state": self.loss_criteria.state_dict(),
        }
        return state

    @classmethod
    def deserialize(cls, cereal):
        sys = cls(
            frame_head=cereal["frame_head"],
            optimizer=cereal["optimizer"],
            loss_criteria=cereal["loss"],
        )
        sys.optimizer.load_state_dict(cereal["optimizer_state"])
        sys.loss_criteria.load_state_dict(cereal["loss_state"])

        return sys


class EyePositionEncoding(object):
    def __init__(self, center_bits=12, zoom_bits=6):
        self.center = PInt(0, 0, bits=center_bits)
        self.d_center = PInt(0, 0, bits=center_bits)
        self.zoom = PInt(0, bits=zoom_bits)
        self.d_zoom = PInt(0, bits=zoom_bits)

    @property
    def numel(self):
        accum = self.center.bits * 4
        accum += self.d_center.bits * 4
        accum += self.zoom.bits * 2
        accum += self.d_zoom.bits * 2
        return accum


class CropSettings(object):
    CAM_SIZE_REQUEST = (99999, 99999)
    POST_LENS = (64, 64, 3)


class VideoLearner(object):
    """An AI that moves around the input video space, cutting out and focusing on regions, to get the correct output."""

    ZOOM_MIN = 1.0 / 20
    ZOOM_MAX = 1.0
    CENTER_MIN = 0.001
    CENTER_MAX = 1

    def __init__(
            self,
            output_encoding: List[PInt],
            recognition_system=None,
            movement_encoding: EyePositionEncoding = EyePositionEncoding(),
            crop_settings: CropSettings = CropSettings(),
    ):
        """Create a virtual eye. Can run on video files, webcams, numpy arrays, etc."""

        self.movement_encoding = movement_encoding
        self.crop_settings = crop_settings
        self.output_encoding = output_encoding
        self.output_endocing_numel = sum([o.bits * o.ndim * 2 for o in self.output_encoding])
        self.recognition_system = recognition_system
        if self.recognition_system is None:
            self.recognition_system = SingleEyeNet(self.output_endocing_numel,
                                                   extra_input_encoding_length=movement_encoding.numel)

        self.pac_input = \
            self.movement_encoding.numel + self.recognition_system.frame_head.output_vector_len
        self.pac_output = self.movement_encoding.numel

        self.pac = ContinualProximalActorCritic(
            self.pac_input, self.pac_output, 64, memory_len=4
        )

        self._move_data_len = 0

        self.prev_loss = None

        self._lens_callback = lens.BarrelPyTorch()
        self._crop_callback = crop.Crop(
            output_size=self.crop_settings.POST_LENS
        )
        self.recognition_system.optimizer.zero_grad()
        self.is_training = False

        if torch.cuda.is_available():
            self.pac.cuda()
            self.recognition_system.frame_head.cuda()
            self.recognition_system.fcn.cuda()

    def forward(self, frame):
        if self.is_training:
            self.recognition_system.optimizer.zero_grad()
        frame = self._lens_callback(frame)
        frame = self._crop_callback(frame)
        frame_encoding = self.recognition_system.frame_head.forward(frame)
        final_encoding = self.recognition_system.fcn.forward(frame_encoding + self.last_action)
        self.last_encoding = final_encoding
        action = self.pac.get_action(final_encoding).squeeze()
        self.last_action = action
        self._set_focal_point(action)
        return final_encoding

    def train(self, target):
        self.is_training = True
        loss = self.recognition_system.loss_criteria(self.last_encoding, target)
        try:
            loss.backward()
        except RuntimeError as re:
            self.save()
            print("Runtime Error. Model was saved.")
            # traceback.print_exc()
            raise re
        if self.prev_loss is not None:
            diff_loss = loss - self.prev_loss
            if loss == 0:
                reward = 100_000
            else:
                reward = diff_loss / loss - self.bad_actions * 10
                reward = min(reward, 100_000)
            self.pac.memory.update(reward=reward, done=[0])
            self.pac.update_ppo()
        self.prev_loss = loss.item()
        self.recognition_system.optimizer.step()

    def _set_focal_point(self, focal_encoding: torch.Tensor):
        """Set where we'll focus in the next frame once we can move."""
        self._lens_callback.center = [
            self.state.center[0] * self._lens_callback.input_size[0],
            self.state.center[1] * self._lens_callback.input_size[0],
        ]
        self._crop_callback.center = [
            self.state.zoom * self._post_crop_callback.input_size[0] / 2,
            self.state.zoom * self._post_crop_callback.input_size[1] / 2,
        ]

        self._lens_callback.zoom = self.state.zoom

    def serialize_full(self):
        """
        Serialize this class so it can be saved as part of a larger system.

        This method save all information so that the model can be further trained.
        """
        ser = {
            "_move_data_len": self._move_data_len,
            "recognition_system": self.recognition_system.serialize(),
            "movement_encoding_widths": self.movement_encoding_widths,
            "crop_settings": self.crop_settings,
        }
        return ser

    @classmethod
    def deserialize_full(cls, ser):
        """
        Create an instance of this class from a serialized dictionary.

        This method loads all information so that the model can be further trained.
        """
        eye = cls(
            ser["cam"],
            SingleEyeNet.deserialize(ser["recognition_system"]),
            ser["movement_encoding_widths"],
            ser["crop_settings"],
        )
        eye._move_data_len = ser["_move_data_len"]
        return eye

    def save(self, filename="virtual_eye_local.torch"):
        """
        This saves this class for future training.

        For deployment on other devices, only the model and some class variables are needed.
        """
        state = self.serialize_full()
        torch.save(state, filename)

    @classmethod
    def load(cls, filename="virtual_eye_local.torch"):
        """Load this class from a file for further training."""
        state = torch.load(filename)
        return VideoLearner.deserialize_full(state)


if __name__ == "__main__":
    eye = VideoLearner()
    eye.run()
