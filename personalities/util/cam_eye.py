from displayarray.effects import crop, lens
from displayarray import display
from coordencode import ints_to_2d, int_to_1d
import math as m
import numpy as np
import torch.optim as optim
import torch
from torch.optim.optimizer import Optimizer
from personalities.util.cv_helpers import cv_image_to_pytorch, vector_to_2d_encoding
from personalities.util.simple_auto_encoder_256 import AutoEncoder
from collections.abc import Iterable
from typing import Optional, List
import torch.nn as nn


class CamYield(object):
    LOSS = True  # yield the loss from the recognition system
    ENCODING = True  # yield the learned encoding from the recognition system


class RecognitionSystem(object):
    def __init__(self,
                 model: nn.Module = None,
                 loss_criteria: nn.modules.loss._Loss = None,
                 optimizer: Optimizer = None):
        if model:
            self.model = model
        else:
            self.model = AutoEncoder(1024)

        if loss_criteria:
            self.loss_criteria = loss_criteria
        else:
            self.loss_criteria = nn.SmoothL1Loss()

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.model.parameters())

    def serialize(self):
        state = {
            "model": self.model,
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer,
            "optimizer_state": self.optimizer.state_dict(),
            "loss": self.loss_criteria,
            "loss_state": self.loss_criteria.state_dict(),
        }
        return state

    @classmethod
    def deserialize(cls, cereal):
        sys = cls(
            model=cereal['model'],
            optimizer=cereal['optimizer'],
            loss_criteria=cereal['loss']
        )
        sys.model.load_state_dict(cereal['model_state'])
        sys.optimizer.load_state_dict(cereal['optimizer_state'])
        sys.loss_criteria.load_state_dict(cereal['loss_state'])

        return sys


class MovementEncodingWidth(object):
    CENTER_X = 4
    CENTER_Y = 4
    CENTER_DX = 4
    CENTER_DY = 4
    ZOOM = 4
    DZOOM = 4
    BARREL = 4
    DBARREL = 4


class CropSettings(object):
    CAM_SIZE_REQUEST = (99999, 99999)
    PRE_LENS = (480, 640, 3)
    POST_LENS = (256, 256, 3)


class VirtualEyeWithLens(object):
    """
    Allows an AI to move around the input video space, cutting out and focusing on regions.

    Note that it does not take into account muscle fatigue or focus speed. Because this is virtual, muscle fatigue is
     nonexistent and focus speed is instantaneous. Since that's the case, looking everywhere around the image is
     actually the best strategy.
    """

    __slots__ = [
        "cam",
        "_pre_crop_callback",
        "_lens_callback",
        "_post_crop_callback",
        "yields",
        "_move_data_len",
        "recognition_system",
        "movement_encoding_widths",
        "state",
        "crop_settings",
        "bad_actions",
    ]

    ZOOM_MIN = 0.5
    ZOOM_MAX = 2.0
    BARREL_MIN = 0.5
    BARREL_MAX = 2.0
    CENTER_MIN = 0.001
    CENTER_MAX = 1

    class State(object):
        center: Optional[List[float]] = None
        zoom: Optional[float] = None
        barrel: Optional[float] = None
        movement: Optional[np.ndarray] = None
        frame: Optional[np.ndarray] = None
        unclipped_center: Optional[List[float]] = None
        unclipped_zoom: Optional[float] = None
        unclipped_barrel: Optional[float] = None

    def __init__(
            self,
            cam=(0,),
            recognition_system=RecognitionSystem(),
            yields: CamYield = CamYield(),
            movement_encoding_widths: MovementEncodingWidth = MovementEncodingWidth(),
            crop_settings: CropSettings = CropSettings(),
    ):
        """Create a virtual eye. Can run on video files, webcams, numpy arrays, etc."""

        self.yields = yields
        self.recognition_system = recognition_system
        self.movement_encoding_widths = movement_encoding_widths
        self.crop_settings = crop_settings

        self.bad_actions = 0

        # todo: I just realized this easily supports multiple cams. To take advantage of that, I should add which
        #  cam we're using to a neural encoding, as well as which cam we were just on. Otherwise, the AI has no way of
        #  knowing. Allowing things like rapid switching could be useful too, and could, for example, enable detection
        #  of shiny materials that are brighter in one eye than another. Or slight 3D shape. Right now its random
        #  switching.
        self._init_cam(cam)

        self.state = VirtualEyeWithLens.State()

        self._move_data_len = 0

        if torch.cuda.is_available():
            self.recognition_system.model.cuda()

    def run(self, csv_reward=False):
        from personalities.base.actor_critic import ContinualProximalActorCritic

        pac = None
        for encoding, loss in self:
            if pac is None:
                pac = ContinualProximalActorCritic(encoding.numel(), 4, 64, memory_len=4)
                pac.cuda()
            else:
                reward = (loss * 10 / (1 + self.bad_actions)) - self.bad_actions * 10
                print(loss.item(), reward.item(), end=', ')
                if csv_reward:
                    with open("reward_hist.csv", "a+") as reward_file:
                        reward_file.write(f"{reward},\n")
                pac.memory.update(reward=reward, done=[0])
                # self.loss = loss.detach().item()
                # self.reward = reward.detach().item()
            action = pac.get_action(encoding).squeeze()
            self.move_focal_point(
                action[:2].cpu().numpy() / 10,
                action[2].cpu().item() / 10,
                action[3].cpu().item() / 10,
            )
            pac.update_ppo()

    def set_focal_point(self, center_x_y: np.ndarray, zoom, barrel):
        """Set where we'll focus in the next frame once we can move."""
        self.state.unclipped_center = center_x_y.copy()
        self.state.unclipped_zoom = zoom
        self.state.unclipped_barrel = barrel
        self.state.center = center_x_y.clip(self.CENTER_MIN, self.CENTER_MAX)
        self.state.zoom = min(max(zoom, self.ZOOM_MIN), self.ZOOM_MAX)
        self.state.barrel = min(max(barrel, self.BARREL_MIN), self.BARREL_MAX)

    def move_focal_point(self, center_x_y: np.ndarray, zoom, barrel):
        center = self.state.center + center_x_y.copy().astype(self.state.unclipped_center.dtype)
        zoom = self.state.zoom + zoom
        barrel = self.state.barrel + barrel
        self.state.unclipped_center = center
        self.state.unclipped_zoom = zoom
        self.state.unclipped_barrel = barrel
        self.state.center = center.clip(self.CENTER_MIN, self.CENTER_MAX)
        self.state.zoom = min(max(zoom, self.ZOOM_MIN), self.ZOOM_MAX)
        self.state.barrel = min(max(barrel, self.BARREL_MIN), self.BARREL_MAX)

    def _update_prev(self, frame: np.ndarray, prev: State):
        """Update our memory."""
        prev.center = self._pre_crop_callback.center.copy()
        prev.zoom = self._lens_callback.zoom
        prev.barrel = self._lens_callback.barrel_power

        if prev.movement is None:
            prev_prev = VirtualEyeWithLens.State()
            prev_prev.center = (0, 0)
            prev_prev.zoom = 0
            prev_prev.barrel = 0
            self.set_focal_point(np.asarray([0, 0]), 0, 0)
            prev.movement = self._encode_focal_point_movement(prev_prev)

        if prev.frame is None or prev.frame.shape != frame.shape:
            prev.frame = np.zeros_like(frame)

        return prev

    def serialize_full(self):
        """
        Serialize this class so it can be saved as part of a larger system.

        This method save all information so that the model can be further trained.
        """
        ser = {
            "cam": self.cam.source_names,
            "yields": self.yields,
            "_move_data_len": self._move_data_len,
            "recognition_system": self.recognition_system.serialize(),
            "movement_encoding_widths": self.movement_encoding_widths,
            "state": self.state,
            "crop_settings": self.crop_settings,
            "bad_actions": self.bad_actions,
        }
        return ser

    @classmethod
    def deserialize_full(cls, ser):
        """
        Create an instance of this class from a serialized dictionary.

        This method loads all information so that the model can be further trained.
        """
        eye = cls(ser['cam'],
                  RecognitionSystem.deserialize(ser['recognition_system']),
                  ser['yields'],
                  ser['movement_encoding_widths'],
                  ser['crop_settings']
                  )
        eye._move_data_len = ser['_move_data_len']
        eye.state = ser['state']
        eye.bad_actions = ser['bad_actions']
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
        return VirtualEyeWithLens.deserialize_full(state)

    def train_recognizer(self, frame, prev_frame, movement):
        """Train our chosen recognition system."""
        self.recognition_system.optimizer.zero_grad()
        t_frame = cv_image_to_pytorch(frame)
        p_frame = cv_image_to_pytorch(prev_frame)
        m_frame = vector_to_2d_encoding(movement)
        guess_current_frame = self.recognition_system.model(p_frame, m_frame)
        loss_val = self.recognition_system.loss_criteria(
            guess_current_frame, t_frame
        )
        try:
            loss_val.backward()
        except RuntimeError as re:
            self.save()
            print("Runtime Error. Model was saved.")
            # traceback.print_exc()
            raise re

        self.recognition_system.optimizer.step()

        return loss_val

    def _handle_iter_yields(self, loss):
        """Setup whatever we decided to yield from the iterator."""
        yield_list = []

        if self.yields.ENCODING:
            yield_list.append(self.recognition_system.model.encoding)

        if self.yields.LOSS:
            yield_list.append(loss)

        return yield_list

    def __iter__(self):
        """
        Run the virtual eye in a for loop frame by frame.

        returns the encoding and loss in a tuple by default.
        """
        prev = VirtualEyeWithLens.State()
        while self.cam:
            if len(self.cam.frames) > 0:
                frame = self.cam.frames[0]

                prev = self._update_prev(frame, prev)

                loss = self.train_recognizer(frame, prev.frame, prev.movement)

                yield tuple(self._handle_iter_yields(loss))
                self.bad_actions = 0  # reset now that we've yielded

                prev.frame = frame.copy()

                self._update_focal_point()
                self._encode_focal_point_movement(prev)

                self.recognition_system.model.set_movement_data_len(
                    prev.movement.size
                )
        self.save()

    def _init_cam(self, cam):
        """Initialize the input camera."""
        self._pre_crop_callback = crop.Crop(
            output_size=self.crop_settings.PRE_LENS
        ).enable_mouse_control()
        self._lens_callback = lens.BarrelPyTorch()
        self._post_crop_callback = crop.Crop(
            output_size=self.crop_settings.POST_LENS
        ).enable_mouse_control()

        if not isinstance(cam, Iterable):
            cam = [cam]

        self.cam = (
            display(*cam, size=self.crop_settings.CAM_SIZE_REQUEST)
                .add_callback(self._pre_crop_callback)
                .add_callback(self._lens_callback)
                .add_callback(self._post_crop_callback)
                .wait_for_init()
        )

    def _update_focal_point(self):
        """Set the focal point to update in the next iteration."""
        self._pre_crop_callback.center = [
            self.state.center[0] * self._pre_crop_callback.input_size[0],
            self.state.center[1] * self._pre_crop_callback.input_size[1],
        ]
        self._lens_callback.center = [
            self.state.center[0] * self._lens_callback.input_size[0],
            self.state.center[1] * self._lens_callback.input_size[0],
        ]
        self._post_crop_callback.center = [
            self._post_crop_callback.input_size[0] / 2,
            self._post_crop_callback.input_size[1] / 2,
        ]

        self._lens_callback.zoom = self.state.zoom
        self._lens_callback.barrel_power = self.state.barrel

    def _punish_out_of_bounds_actions(self):
        """
        Punish the AI for out of bounds actions, linearly.

        The more it goes out of bounds, the more it gets punished.
        """
        if self.state.center[0] == self.CENTER_MIN:
            self.bad_actions += self.state.center[0] - self.state.unclipped_center[0]
        if self.state.center[1] == self.CENTER_MIN:
            self.bad_actions += self.state.center[1] - self.state.unclipped_center[1]
        if self.state.center[0] == self.CENTER_MAX:
            self.bad_actions += self.state.unclipped_center[0] - self.state.center[0]
        if self.state.center[1] == self.CENTER_MAX:
            self.bad_actions += self.state.unclipped_center[1] - self.state.center[1]

        if self.state.zoom == self.ZOOM_MIN:
            self.bad_actions += self.state.zoom - self.state.unclipped_zoom

        if self.state.zoom == self.ZOOM_MAX:
            self.bad_actions += self.state.unclipped_zoom - self.state.zoom

        if self.state.barrel == self.BARREL_MIN:
            self.bad_actions += self.state.barrel - self.state.unclipped_barrel
        if self.state.barrel == self.BARREL_MAX:
            self.bad_actions += self.state.unclipped_barrel - self.state.barrel

    def _punish_overly_rapid_actions(self, diff: State):
        if diff.center[0] == 0:
            self.bad_actions += (
                                        diff.center[0] - diff.unclipped_center[0]
                                ) / self._pre_crop_callback.input_size[0]
        if diff.center[1] == 0:
            self.bad_actions += (
                                        diff.center[1] - diff.unclipped_center[1]
                                ) / self._pre_crop_callback.input_size[1]
        if diff.center[0] >= self._pre_crop_callback.input_size[0] * 2:
            self.bad_actions += (
                                        diff.center[0] - self._pre_crop_callback.input_size[0] * 2
                                ) / self._pre_crop_callback.input_size[0]
        if diff.center[1] >= self._pre_crop_callback.input_size[1] * 2:
            self.bad_actions += (
                                        diff.center[1] - self._pre_crop_callback.input_size[0] * 2
                                ) / self._pre_crop_callback.input_size[1]

        if diff.zoom == 0:
            self.bad_actions += diff.zoom - diff.unclipped_zoom
        if diff.zoom > 1:
            self.bad_actions += diff.zoom - 1

        if diff.barrel == 0:
            self.bad_actions += diff.barrel - diff.unclipped_barrel
        if diff.barrel > 1:
            self.bad_actions += diff.barrel - 1

    def _setup_movement_values_for_encoding(self, prev: State):
        """The encoder currently only supports positive integers, so wen need to convert some of our values."""
        diff = VirtualEyeWithLens.State()

        center = [None, None]
        center[0] = (
                self._pre_crop_callback.center[0]
                + self._pre_crop_callback.input_size[0]
        )
        center[1] = (
                self._pre_crop_callback.center[1]
                + self._pre_crop_callback.input_size[1]
        )

        diff.unclipped_center = [None, None]
        diff.unclipped_center[0] = (
                self._pre_crop_callback.center[0]
                - prev.center[0]
                + self._pre_crop_callback.input_size[0]
        )
        diff.unclipped_center[1] = (
                self._pre_crop_callback.center[1]
                - prev.center[1]
                + self._pre_crop_callback.input_size[1]
        )
        diff.center = [None, None]
        diff.center[0] = max(0, diff.unclipped_center[0])
        diff.center[1] = max(0, diff.unclipped_center[1])

        diff.unclipped_zoom = self._lens_callback.zoom - prev.zoom + 1
        diff.zoom = max(diff.unclipped_zoom, 0)

        diff.unclipped_barrel = self._lens_callback.barrel_power - prev.barrel + 1
        diff.barrel = max(diff.unclipped_barrel, 0)

        return center, diff

    def _encode_focal_point_movement(self, prev: State):
        """Encode our motion in a way neural networks will be able to learn from it."""

        # todo: move the byte safety code to ints_to_2d
        try:
            center, diff = self._setup_movement_values_for_encoding(prev)

            encode_center = ints_to_2d(
                center[0],
                int(
                    m.ceil(m.log2(self._pre_crop_callback.input_size[0]) / 8) * 8
                ),  # todo: move this to ints_to_2d
                self.movement_encoding_widths.CENTER_X,
                center[1],
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[1]) / 8) * 8),
                self.movement_encoding_widths.CENTER_Y,
            )

            encode_change_in_center = ints_to_2d(
                diff.center[0],
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[0] * 2) / 8) * 8),
                self.movement_encoding_widths.CENTER_DX,
                diff.center[1],
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[1] * 2) / 8) * 8),
                self.movement_encoding_widths.CENTER_DY,
            )

            encode_zoom = int_to_1d(
                self._lens_callback.zoom * (2 ** 8),
                16,
                self.movement_encoding_widths.ZOOM,
            )

            encode_change_in_zoom = int_to_1d(
                diff.zoom * (2 ** 8), 16, self.movement_encoding_widths.DZOOM
            )

            encode_barrel = int_to_1d(
                self._lens_callback.barrel_power * (2 ** 8),
                16,
                self.movement_encoding_widths.BARREL,
            )

            encode_change_in_barrel = int_to_1d(
                diff.barrel * (2 ** 8), 16, self.movement_encoding_widths.DBARREL
            )

            prev_movement = np.concatenate(
                (
                    encode_center,
                    encode_change_in_center,
                    encode_zoom,
                    encode_change_in_zoom,
                    encode_barrel,
                    encode_change_in_barrel,
                ),
                axis=None,
            )

            self._punish_out_of_bounds_actions()
            self._punish_overly_rapid_actions(diff)
            return prev_movement

        except ValueError as ve:
            import traceback

            traceback.print_exc()
            print("The AI programmed an illegal value.")


if __name__ == "__main__":
    eye = VirtualEyeWithLens()
    eye.run()
