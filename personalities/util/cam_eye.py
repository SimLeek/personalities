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


class VirtualEye(object):
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
        "_prev_frame",
        "_prev_movement",
        "yields",
        "_move_data_len",
        "recognition_system",
        "movement_encoding_widths",
        "center_x_y",
        "zoom",
        "barrel",
        "unclipped_center_x_y",
        "unclipped_zoom",
        "unclipped_barrel",
        "crop_settings",
        "bad_actions",
    ]

    ZOOM_MIN = 0.5
    ZOOM_MAX = 2.0
    BARREL_MIN = 0.5
    BARREL_MAX = 2.0
    CENTER_MIN = 0.001
    CENTER_MAX = 1

    def __init__(
            self,
            cam=0,
            recognition_system=RecognitionSystem(),
            yields: CamYield = CamYield(),
            movement_encoding_widths: MovementEncodingWidth = MovementEncodingWidth(),
            crop_settings: CropSettings = CropSettings(),
    ):

        self.yields = yields
        self.recognition_system = recognition_system
        self.movement_encoding_widths = movement_encoding_widths
        self.crop_settings = crop_settings

        self.bad_actions = 0  # set to true whenever the eye was set past its limit

        self._init_cam(cam)

        self._prev_movement = None
        self._prev_frame = None
        self.center_x_y = None
        self.zoom = None
        self.barrel = None
        self.unclipped_center_x_y = None
        self.unclipped_zoom = None
        self.unclipped_barrel = None
        self._move_data_len = 0

        if torch.cuda.is_available():
            self.recognition_system.model.cuda()

    def set_focal_point(self, center_x_y: np.ndarray, zoom, barrel):
        self.unclipped_center_x_y = center_x_y.copy()
        self.unclipped_zoom = zoom
        self.unclipped_barrel = barrel
        self.center_x_y = center_x_y.clip(self.CENTER_MIN, self.CENTER_MAX)
        self.zoom = min(max(zoom, self.ZOOM_MIN), self.ZOOM_MAX)
        self.barrel = min(max(barrel, self.BARREL_MIN), self.BARREL_MAX)

    def __iter__(self):
        while self.cam:
            yield_list = []
            if len(self.cam.frames) > 0:
                prev_center = self._pre_crop_callback.center.copy()
                prev_zoom = self._lens_callback.zoom
                prev_barrel = self._lens_callback.barrel_power
                frame = self.cam.frames[0]

                if self._prev_movement is None:
                    self.set_focal_point(np.asarray([0, 0]), 0, 0)
                    self._encode_focal_point_movement((0, 0), 0, 0)

                if self._prev_frame is None or self._prev_frame.shape != frame.shape:
                    self._prev_frame = np.zeros_like(frame)

                self.recognition_system.optimizer.zero_grad()
                t_frame = cv_image_to_pytorch(frame)
                p_frame = cv_image_to_pytorch(self._prev_frame)
                m_frame = vector_to_2d_encoding(self._prev_movement)
                guess_current_frame = self.recognition_system.model(p_frame, m_frame)
                loss = self.recognition_system.loss_criteria(
                    guess_current_frame, t_frame
                )
                lossed = False
                while not lossed:
                    try:
                        loss.backward()
                        lossed = True
                    except RuntimeError as re:
                        breakpoint()
                        print("close some GPU using stuff.")
                self.recognition_system.optimizer.step()

                if self.yields.ENCODING:
                    yield_list.append(self.recognition_system.model.encoding)

                if self.yields.LOSS:
                    yield_list.append(loss)

                yield tuple(yield_list)
                self.bad_actions = 0  # reset now that we've yielded

                self._prev_frame = frame.copy()

                self._update_focal_point()
                self._encode_focal_point_movement(prev_center, prev_zoom, prev_barrel)

                self.recognition_system.model.set_movement_data_len(
                    self._prev_movement.size
                )

    def _init_cam(self, cam):
        self._pre_crop_callback = crop.Crop(
            output_size=self.crop_settings.PRE_LENS
        ).enable_mouse_control()
        self._lens_callback = lens.BarrelPyTorch()
        self._post_crop_callback = crop.Crop(
            output_size=self.crop_settings.POST_LENS
        ).enable_mouse_control()

        self.cam = (
            display(cam, size=self.crop_settings.CAM_SIZE_REQUEST)
                .add_callback(self._pre_crop_callback)
                .add_callback(self._lens_callback)
                .add_callback(self._post_crop_callback)
                .wait_for_init()
        )

    def _update_focal_point(self):
        self._pre_crop_callback.center = [
            self.center_x_y[0] * self._pre_crop_callback.input_size[0],
            self.center_x_y[1] * self._pre_crop_callback.input_size[1],
        ]
        self._lens_callback.center = [
            self.center_x_y[0] * self._lens_callback.input_size[0],
            self.center_x_y[1] * self._lens_callback.input_size[0],
        ]
        self._post_crop_callback.center = [
            self._post_crop_callback.input_size[0] / 2,
            self._post_crop_callback.input_size[1] / 2,
        ]

        self._lens_callback.zoom = self.zoom
        self._lens_callback.barrel_power = self.barrel

    def _encode_focal_point_movement(self, prev_center, prev_zoom, prev_barrel):
        # todo: move the byte safety code to ints_to_2d
        try:
            center0 = (
                    self._pre_crop_callback.center[0]
                    + self._pre_crop_callback.input_size[0]
            )
            center1 = (
                    self._pre_crop_callback.center[1]
                    + self._pre_crop_callback.input_size[1]
            )
            encode_center = ints_to_2d(
                center0,
                int(
                    m.ceil(m.log2(self._pre_crop_callback.input_size[0]) / 8) * 8
                ),  # todo: move this to ints_to_2d
                self.movement_encoding_widths.CENTER_X,
                center1,
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[1]) / 8) * 8),
                self.movement_encoding_widths.CENTER_Y,
            )
            if self.center_x_y[0] == 0.001:
                self.bad_actions += self.center_x_y[0] - self.unclipped_center_x_y[0]
            if self.center_x_y[1] == 0.001:
                self.bad_actions += self.center_x_y[1] - self.unclipped_center_x_y[1]
            if self.center_x_y[0] == 1:
                self.bad_actions += self.unclipped_center_x_y[0] - self.center_x_y[0]
            if self.center_x_y[1] == 1:
                self.bad_actions += self.unclipped_center_x_y[1] - self.center_x_y[1]

            d_center0_unclipped = (
                    self._pre_crop_callback.center[0]
                    - prev_center[0]
                    + self._pre_crop_callback.input_size[0]
            )
            d_center1_unclipped = (
                    self._pre_crop_callback.center[1]
                    - prev_center[1]
                    + self._pre_crop_callback.input_size[1]
            )
            d_center0 = max(0, d_center0_unclipped)
            d_center1 = max(0, d_center1_unclipped)

            encode_change_in_center = ints_to_2d(
                d_center0,
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[0] * 2) / 8) * 8),
                self.movement_encoding_widths.CENTER_DX,
                d_center1,
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[1] * 2) / 8) * 8),
                self.movement_encoding_widths.CENTER_DY,
            )
            if d_center0 == 0:
                self.bad_actions += (
                                            d_center0 - d_center0_unclipped
                                    ) / self._pre_crop_callback.input_size[0]
            if d_center1 == 0:
                self.bad_actions += (
                                            d_center1 - d_center1_unclipped
                                    ) / self._pre_crop_callback.input_size[1]
            if d_center0 >= self._pre_crop_callback.input_size[0] * 2:
                self.bad_actions += (
                                            d_center0 - self._pre_crop_callback.input_size[0] * 2
                                    ) / self._pre_crop_callback.input_size[0]
            if d_center1 >= self._pre_crop_callback.input_size[1] * 2:
                self.bad_actions += (
                                            d_center1 - self._pre_crop_callback.input_size[0] * 2
                                    ) / self._pre_crop_callback.input_size[1]

            encode_zoom = int_to_1d(
                self._lens_callback.zoom * (2 ** 8),
                16,
                self.movement_encoding_widths.ZOOM,
            )
            if self.zoom == self.ZOOM_MIN:
                self.bad_actions += self.zoom - self.unclipped_zoom

            if self.zoom == self.ZOOM_MAX:
                self.bad_actions += self.unclipped_zoom - self.zoom

            d_zoom_unclipped = self._lens_callback.zoom - prev_zoom + 1
            d_zoom = max((d_zoom_unclipped), 0)
            encode_change_in_zoom = int_to_1d(
                d_zoom * (2 ** 8), 16, self.movement_encoding_widths.DZOOM
            )
            # going faster than the encoder can encode is cheating, so set a bad_action
            if d_zoom == 0:
                self.bad_actions += d_zoom - d_zoom_unclipped
            if d_zoom > 1:
                self.bad_actions += d_zoom - 1

            encode_barrel = int_to_1d(
                self._lens_callback.barrel_power * (2 ** 8),
                16,
                self.movement_encoding_widths.BARREL,
            )
            if self.barrel == self.BARREL_MIN:
                self.bad_actions += self.barrel - self.unclipped_barrel
            if self.barrel == self.BARREL_MAX:
                self.bad_actions += self.unclipped_barrel - self.barrel

            d_barrel_unclipped = self._lens_callback.barrel_power - prev_barrel + 1
            d_barrel = max(d_barrel_unclipped, 0)
            encode_change_in_barrel = int_to_1d(
                d_barrel * (2 ** 8), 16, self.movement_encoding_widths.DBARREL
            )
            if d_barrel == 0:
                self.bad_actions += d_barrel - d_barrel_unclipped
            if d_barrel > 1:
                self.bad_actions += d_barrel - 1

            self._prev_movement = np.concatenate(
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
        except ValueError as ve:
            import traceback

            traceback.print_exc()
            print("The AI programmed an illegal value.")


if __name__ == "__main__":
    from personalities.base.actor_critic import ProximalActorCritic

    eye = VirtualEye()
    pac = None
    i = 1
    for encoding, loss in eye:
        if pac is None:
            pac = ProximalActorCritic(encoding.numel(), 4, 256)
            pac.cuda()
        else:
            reward = (loss / (1 + eye.bad_actions)) - eye.bad_actions * 10
            print(loss.item(), reward.item())
            with open("reward_hist.csv", "a+") as reward_file:
                reward_file.write(f"{reward},\n")
            pac.memory.update(reward=reward, done=[0])
        action = pac.get_action(encoding).squeeze()
        eye.set_focal_point(
            eye.center_x_y + action[:2].cpu().numpy(),
            eye.zoom + action[2].cpu().item(),
            eye.barrel + action[3].cpu().item(),
        )
        if i % 32 == 0:
            pac.update_ppo()
            pac.memory.reset()
        i += 1
