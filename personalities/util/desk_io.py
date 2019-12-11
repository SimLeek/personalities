import mss
import numpy as np
from PIL import Image
from displayarray import display
import pyautogui
from personalities.util.cam_eye import CamYield, RecognitionSystem, MovementEncodingWidth, CropSettings
from typing import Optional, List
import torch
from personalities.util.cv_helpers import cv_image_to_pytorch, vector_to_2d_encoding, pytorch_image_to_cv
from coordencode import ints_to_2d, int_to_1d
import math as m
from personalities.util.simple_auto_encoder_256 import AutoEncoder
import pyautogui


class ScreenWatcher(object):
    ZOOM_MIN = 0.5
    ZOOM_MAX = 4.0
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

    def __init__(self,
                 recognition_system=RecognitionSystem(model=AutoEncoder(782)),
                 yields: CamYield = CamYield(),
                 movement_encoding_widths: MovementEncodingWidth = MovementEncodingWidth(),
                 crop_settings: CropSettings = CropSettings(),
                 ):
        """Create a virtual eye that can run on your desktop and control your computer and mouse."""

        self.yields = yields
        self.recognition_system = recognition_system
        self.movement_encoding_widths = movement_encoding_widths
        self.crop_settings = crop_settings

        self.x, self.y, self.w, self.h = 1, 1, 1, 1
        self.input_size = np.asarray([1, 1, 3])

        self.bad_actions = 0

        self.state = ScreenWatcher.State()
        self.state.center = np.asarray([1, 1])
        self.state.zoom = 1

        self._move_data_len = 0

        self.pred_img = None

        self.loss = 0
        self.reward = 0
        self.mouse = False

        if torch.cuda.is_available():
            self.recognition_system.model.cuda()
            self.recognition_system.loss_criteria.cuda()

    def run(self, control_mouse=False, controllable_keys=(), csv_reward=False):
        from personalities.base.actor_critic import ContinualProximalActorCritic
        import pyautogui

        if control_mouse:
            self.mouse = True
            pyautogui.FAILSAFE = False
        pac = None

        outputs = 10 + bool(control_mouse) * 2

        for encoding, loss in self:
            if pac is None:
                pac = ContinualProximalActorCritic(encoding.numel(), outputs, 64, memory_len=2)
                pac.cuda()
            else:
                reward = (loss * 10 / (1 + self.bad_actions * 10)) - self.bad_actions * 10
                print(loss.item(), reward.item(), end=', ')
                if csv_reward:
                    with open("reward_hist.csv", "a+") as reward_file:
                        reward_file.write(f"{reward},\n")
                pac.memory.update(reward=reward, done=[0])
                self.loss = loss.detach().item()
                self.reward = reward.detach().item()
            action = pac.get_action(encoding).squeeze()
            if action[0] > 1:
                self.set_focal_point(
                    0.5 + (action[2:4].cpu().numpy() * action[8].cpu().numpy()),
                    0.5 + (action[4].cpu().item()),
                )
            if action[1] > 1:
                self.move_focal_point(
                    action[5:7].cpu().numpy() * action[9].cpu().numpy(),
                    action[7].cpu().item(),
                )
            if control_mouse:
                if action[10] > 1:
                    pyautogui.mouseDown(*pyautogui.position(), pyautogui.PRIMARY)
                if action[11] > 1:
                    pyautogui.mouseDown(*pyautogui.position(), pyautogui.SECONDARY)

            pac.update_ppo()

    def set_focal_point(self, center_x_y: np.ndarray, zoom):
        """Set where we'll focus in the next frame once we can move."""
        self.state.unclipped_center = center_x_y.copy()
        self.state.unclipped_zoom = zoom
        self.state.center = center_x_y.clip(self.CENTER_MIN, self.CENTER_MAX)
        self.state.zoom = min(max(zoom, self.ZOOM_MIN), self.ZOOM_MAX)

    def move_focal_point(self, center_x_y: np.ndarray, zoom):
        center = self.state.center + center_x_y.copy().astype(self.state.unclipped_center.dtype)
        zoom = self.state.zoom + zoom
        self.state.unclipped_center = center
        self.state.unclipped_zoom = zoom
        self.state.center = center.clip(self.CENTER_MIN, self.CENTER_MAX)
        self.state.zoom = min(max(zoom, self.ZOOM_MIN), self.ZOOM_MAX)

    def _update_prev(self, frame: np.ndarray, prev: State):
        """Update our memory."""
        prev.center = self.state.center.copy()
        prev.zoom = self.state.zoom

        if prev.movement is None:
            prev_prev = ScreenWatcher.State()
            prev_prev.center = (0, 0)
            prev_prev.zoom = 0
            self.set_focal_point(np.asarray([0, 0]), 0)
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
        eye = cls(RecognitionSystem.deserialize(ser['recognition_system']),
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
        return ScreenWatcher.deserialize_full(state)

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

        self.pred_img = guess_current_frame

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
        prev = ScreenWatcher.State()
        with display() as d:
            while d:
                if self.pred_img is not None:
                    d.update_specific(pytorch_image_to_cv(self.pred_img) / 256.0, "pred")
                d.update_specific(self.get_screen(self.x, self.y, self.w, self.h), "grab")
                if len(d.frames) > 0:
                    frame = d.frames[0]

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

    def _update_focal_point(self):
        """Set the focal point to update in the next iteration."""

        self.x = self.state.center[0]
        self.y = self.state.center[1]
        if self.mouse:
            pyautogui.moveTo(self.x * self.input_size[0], self.y * self.input_size[1])
        self.w = self.h = self.state.zoom * self.crop_settings.POST_LENS[0]

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

    def _punish_overly_rapid_actions(self, diff: State):
        if diff.center[0] == 0:
            self.bad_actions += (
                                        diff.center[0] - diff.unclipped_center[0]
                                ) / self.input_size[0]
        if diff.center[1] == 0:
            self.bad_actions += (
                                        diff.center[1] - diff.unclipped_center[1]
                                ) / self.input_size[1]
        if diff.center[0] >= self.input_size[0] * 2:
            self.bad_actions += (
                                        diff.center[0] - self.input_size[0] * 2
                                ) / self.input_size[0]
        if diff.center[1] >= self.input_size[1] * 2:
            self.bad_actions += (
                                        diff.center[1] - self.input_size[0] * 2
                                ) / self.input_size[1]

        if diff.zoom == 0:
            self.bad_actions += diff.zoom - diff.unclipped_zoom
        if diff.zoom > 1:
            self.bad_actions += diff.zoom - 1

    def _setup_movement_values_for_encoding(self, prev: State):
        """The encoder currently only supports positive integers, so wen need to convert some of our values."""
        diff = ScreenWatcher.State()

        center = [None, None]
        center[0] = (
                self.state.center[0]
                + self.input_size[0]
        )
        center[1] = (
                self.state.center[1]
                + self.input_size[1]
        )

        diff.unclipped_center = [None, None]
        diff.unclipped_center[0] = (
                self.state.center[0]
                - prev.center[0]
                + self.input_size[0]
        )
        diff.unclipped_center[1] = (
                self.state.center[1]
                - prev.center[1]
                + self.input_size[1]
        )
        diff.center = [None, None]
        diff.center[0] = max(0, diff.unclipped_center[0])
        diff.center[1] = max(0, diff.unclipped_center[1])

        diff.unclipped_zoom = self.state.zoom - prev.zoom + 1
        diff.zoom = max(diff.unclipped_zoom, 0)

        return center, diff

    def _encode_focal_point_movement(self, prev: State):
        """Encode our motion in a way neural networks will be able to learn from it."""

        # todo: move the byte safety code to ints_to_2d
        try:
            center, diff = self._setup_movement_values_for_encoding(prev)

            encode_center = ints_to_2d(
                center[0],
                int(
                    m.ceil(m.log2(self.input_size[0]) / 8) * 8
                ),  # todo: move this to ints_to_2d
                self.movement_encoding_widths.CENTER_X,
                center[1],
                int(m.ceil(m.log2(self.input_size[1]) / 8) * 8),
                self.movement_encoding_widths.CENTER_Y,
            )

            encode_change_in_center = ints_to_2d(
                diff.center[0],
                int(m.ceil(m.log2(self.input_size[0] * 2) / 8) * 8),
                self.movement_encoding_widths.CENTER_DX,
                diff.center[1],
                int(m.ceil(m.log2(self.input_size[1] * 2) / 8) * 8),
                self.movement_encoding_widths.CENTER_DY,
            )

            encode_zoom = int_to_1d(
                self.state.zoom * (2 ** 8),
                16,
                self.movement_encoding_widths.ZOOM,
            )

            encode_change_in_zoom = int_to_1d(
                diff.zoom * (2 ** 8), 16, self.movement_encoding_widths.DZOOM
            )

            oob = np.asarray(
                [
                    max(0, self.state.center[0] - self.state.unclipped_center[0]),
                    max(0, self.state.center[1] - self.state.unclipped_center[1]),
                    max(0, self.state.unclipped_center[0] - self.state.center[0]),
                    max(0, self.state.unclipped_center[1] - self.state.center[1]),
                    max(0, self.state.zoom - self.state.unclipped_zoom),
                    max(0, self.state.unclipped_zoom - self.state.zoom),
                ]
            )

            oos = np.asarray(
                [
                    max(0, (
                            diff.center[0] - diff.unclipped_center[0]
                    ) / self.input_size[0]),
                    max(0, (
                            diff.center[1] - diff.unclipped_center[1]
                    ) / self.input_size[1]),
                    max(0, (
                            diff.center[0] - self.input_size[0] * 2
                    ) / self.input_size[0]),
                    max(0, (
                            diff.center[1] - self.input_size[0] * 2
                    ) / self.input_size[1]),
                    max(0, diff.zoom - diff.unclipped_zoom),
                    max(0, diff.zoom - 1),
                ]
            )

            lr = np.asarray([self.loss, self.reward])

            prev_movement = np.concatenate(
                (
                    encode_center,
                    encode_change_in_center,
                    encode_zoom,
                    encode_change_in_zoom,
                    oob, oos, lr
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        pass

    def get_screen(self, center_x, center_y, w, h):
        with mss.mss() as sct:
            top = center_y * self.input_size[1] - h / 2.0
            left = center_x * self.input_size[0] - w / 2.0
            box = {"top": int(top), "left": int(left), "width": int(w), "height": int(h)}
            self.input_size = np.asarray([sct.monitors[0]['width'], sct.monitors[0]['height']])
            screen_image = np.array(sct.grab(box))
            out_image = np.array(
                Image.fromarray(screen_image).resize(self.crop_settings.POST_LENS[:2], Image.BICUBIC)
            )
            return out_image[..., :3]


if __name__ == "__main__":
    eye = ScreenWatcher()
    eye.run()
