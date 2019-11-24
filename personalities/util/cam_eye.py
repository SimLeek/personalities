from displayarray.effects import crop, lens
from displayarray import display
from coordencode import ints_to_2d, int_to_1d
import math as m
import numpy as np
import torch.optim as optim
import torch

from personalities.util.simple_auto_encoder_256 import AutoEncoder

import torch.nn as nn


class CamYield(object):
    LOSS = True  # yield the loss from the recognition system
    ENCODING = True  # yield the learned encoding from the recognition system


class RecognitionSystem(object):
    def __init__(self, model=None, loss_criteria=None, optimizer=None):
        if model:
            self.model = model
        else:
            self.model = AutoEncoder(1024)

        if loss_criteria:
            self.loss_criteria = loss_criteria
        else:
            self.loss_criteria = nn.MSELoss()

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


# todo: move this to displayarray or relayarray
def cv_image_to_pytorch(image: np.ndarray, cuda: bool = True) -> torch.Tensor:
    torch_image = torch.from_numpy(image).float()
    if cuda:
        torch_image = torch_image.cuda()
    torch_image = torch_image.permute((2, 0, 1))
    torch_image = torch_image[None, ...]
    return torch_image


# todo: move this to displayarray or relayarray
def vector_to_pytorch(vector: np.ndarray, cuda: bool = True) -> torch.Tensor:
    torch_vec = torch.from_numpy(vector).float()
    if cuda:
        torch_vec = torch_vec.cuda()
    torch_vec = torch_vec[None, :, None, None]
    return torch_vec


class CamEye(object):
    """
    >>> from personalities.base.actor_critic import ProximalActorCritic
    >>> eye = CamEye()
    >>> pac = None
    >>> i = 1
    >>> for encoding, loss in eye:
    ...   if pac is None:
    ...     pac = ProximalActorCritic(encoding.numel(), 4, 256)
    ...     pac.cuda()
    ...   else:
    ...     reward = loss/(eye.bad_actions+1)  # ouch my eye
    ...     pac.memory.update(reward=reward, done=[0])
    ...   action = pac.get_action(encoding).squeeze()
    ...   eye.set_focal_point(action[:2].cpu().numpy(), action[2].cpu().item(), action[3].cpu().item())
    ...   if i%32==0:
    ...     pac.update_ppo()
    ...     pac.memory.reset()
    ...   i+=1
    """
    __slots__ = ['cam', '_pre_crop_callback', '_lens_callback', '_post_crop_callback', '_prev_frame', '_prev_movement',
                 'yields', '_move_data_len', 'recognition_system', 'movement_encoding_widths', 'center_x_y', 'zoom',
                 'barrel', 'crop_settings', 'bad_actions']

    def __init__(self,
                 cam=0,
                 recognition_system=RecognitionSystem(),
                 yields: CamYield = CamYield(),
                 movement_encoding_widths: MovementEncodingWidth = MovementEncodingWidth(),
                 crop_settings: CropSettings = CropSettings()):

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
        self._move_data_len = 0

        if torch.cuda.is_available():
            self.recognition_system.model.cuda()

    def set_focal_point(self, center_x_y: np.ndarray, zoom, barrel):
        self.center_x_y = center_x_y.clip(0.001, 1)
        self.zoom = min(max(zoom, 0.1), 10)
        self.barrel = min(max(barrel, 0.3), 10)

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
                m_frame = vector_to_pytorch(self._prev_movement)
                guess_current_frame = self.recognition_system.model(p_frame, m_frame)
                loss = self.recognition_system.loss_criteria(guess_current_frame, t_frame)
                loss.backward()
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

                self.recognition_system.model.set_movement_data_len(self._prev_movement.size)

    def _init_cam(self, cam):
        self._pre_crop_callback = crop.Crop(output_size=self.crop_settings.PRE_LENS).enable_mouse_control()
        self._lens_callback = lens.BarrelPyTorch()
        self._post_crop_callback = crop.Crop(output_size=self.crop_settings.POST_LENS).enable_mouse_control()

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
        self._post_crop_callback.center = [self._post_crop_callback.input_size[0] / 2,
                                           self._post_crop_callback.input_size[1] / 2]

        self._lens_callback.zoom = self.zoom
        self._lens_callback.barrel_power = self.barrel

    def _encode_focal_point_movement(self, prev_center, prev_zoom, prev_barrel):
        # todo: move the byte safety code to ints_to_2d
        try:
            center0 = self._pre_crop_callback.center[0] + self._pre_crop_callback.input_size[0]
            center1 = self._pre_crop_callback.center[1] + self._pre_crop_callback.input_size[1]
            encode_center = ints_to_2d(
                center0,
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[0]) / 8) * 8),  # todo: move this to ints_to_2d
                self.movement_encoding_widths.CENTER_X,
                center1,
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[1]) / 8) * 8),
                self.movement_encoding_widths.CENTER_Y,
            )
            if self.center_x_y[0] == 0.001 or self.center_x_y[1] == 0.001 or self.center_x_y[
                0] == 1 or self.center_x_y[1] == 1:
                self.bad_actions += 1

            d_center0 = self._pre_crop_callback.center[0] - prev_center[0] + self._pre_crop_callback.input_size[0]
            d_center1 = self._pre_crop_callback.center[1] - prev_center[1] + self._pre_crop_callback.input_size[0]
            d_center0 = max(0, d_center0)
            d_center1 = max(0, d_center1)

            encode_change_in_center = ints_to_2d(
                d_center0,
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[0] * 2) / 8) * 8),
                self.movement_encoding_widths.CENTER_DX,
                d_center1,
                int(m.ceil(m.log2(self._pre_crop_callback.input_size[1] * 2) / 8) * 8),
                self.movement_encoding_widths.CENTER_DY,
            )
            if d_center1 == 0 or d_center0 == 0 or d_center0 >= self._pre_crop_callback.input_size[
                0] * 2 or d_center1 >= self._pre_crop_callback.input_size[1] * 2:
                self.bad_actions += 1

            encode_zoom = int_to_1d(
                self._lens_callback.zoom * (2 ** 8),
                16,
                self.movement_encoding_widths.ZOOM,
            )
            if self.zoom == 0.1 or self.zoom == 10:
                self.bad_actions += 1

            d_zoom = max((self._lens_callback.zoom - prev_zoom + 1), 0)
            encode_change_in_zoom = int_to_1d(
                d_zoom * (2 ** 8),
                16,
                self.movement_encoding_widths.DZOOM,
            )
            if d_zoom == 0 or d_zoom > 1:
                # going faster than the encoder can encode is cheating, so set a bad_action
                self.bad_actions += 1

            encode_barrel = int_to_1d(
                self._lens_callback.barrel_power * (2 ** 8),
                16,
                self.movement_encoding_widths.BARREL,
            )
            if self.barrel == 0.3 or self.barrel == 10:
                self.bad_actions += 1

            d_barrel = max((self._lens_callback.barrel_power - prev_barrel + 1), 0)
            encode_change_in_barrel = int_to_1d(
                d_barrel * (2 ** 8),
                16,
                self.movement_encoding_widths.DBARREL,
            )
            if d_barrel == 0 or d_barrel > 1:
                self.bad_actions += 1

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
            print("How the hell do you keep doing this?")
