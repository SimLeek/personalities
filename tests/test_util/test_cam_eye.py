from personalities.util.cam_eye import *
import mock
import os
import pytest


def test_recognition_system_object():
    r = RecognitionSystem(1, 2, 3)

    assert r.model == 1
    assert r.loss_criteria == 2
    assert r.optimizer == 3

    r = RecognitionSystem()

    assert r.model.is_same_at_start(AutoEncoder(1024))
    assert isinstance(r.loss_criteria, nn.SmoothL1Loss)
    assert isinstance(r.optimizer, optim.Adam)


def test_recognition_system_serialization():
    rs = RecognitionSystem()

    assert rs.model.is_same_at_start(AutoEncoder(1024))
    assert isinstance(rs.loss_criteria, nn.SmoothL1Loss)
    assert isinstance(rs.optimizer, optim.Adam)

    ser = rs.serialize()

    rs2 = RecognitionSystem.deserialize(ser)

    assert rs2.model.is_same_at_start(AutoEncoder(1024))
    assert isinstance(rs2.loss_criteria, nn.SmoothL1Loss)
    assert isinstance(rs2.optimizer, optim.Adam)


def test_cam_eye_init():
    with mock.patch(
            "personalities.util.cam_eye.display"
    ) as mock_cam, mock.patch.object(VirtualEyeWithLens, "_init_cam") as mock_mouse_loop:
        ceye = VirtualEyeWithLens()

        assert ceye.yields.LOSS
        assert ceye.yields.ENCODING

        assert ceye.recognition_system.model.is_same_at_start(AutoEncoder(1024))
        assert isinstance(ceye.recognition_system.loss_criteria, nn.SmoothL1Loss)
        assert isinstance(ceye.recognition_system.optimizer, optim.Adam)

        assert ceye.movement_encoding_widths.CENTER_X == 4
        assert ceye.movement_encoding_widths.CENTER_Y == 4
        assert ceye.movement_encoding_widths.CENTER_DX == 4
        assert ceye.movement_encoding_widths.CENTER_DY == 4
        assert ceye.movement_encoding_widths.ZOOM == 4
        assert ceye.movement_encoding_widths.DZOOM == 4
        assert ceye.movement_encoding_widths.BARREL == 4
        assert ceye.movement_encoding_widths.DBARREL == 4

        assert ceye.crop_settings.CAM_SIZE_REQUEST == (99999, 99999)
        assert ceye.crop_settings.PRE_LENS == (480, 640, 3)
        assert ceye.crop_settings.POST_LENS == (256, 256, 3)


def test_cam_eye_serialization():
    with mock.patch.object(VirtualEyeWithLens, "_init_cam", autospec=True) as mock_cam:
        def new_cam(self, cam):
            cam_mock = mock.MagicMock()
            if not isinstance(cam, list):
                cam = [cam]
            cam_mock.source_names = cam
            self.cam = cam_mock

        mock_cam.side_effect = new_cam

        ceye = VirtualEyeWithLens("Hi!")

        cserial = ceye.serialize_full()
        ceye2 = VirtualEyeWithLens.deserialize_full(cserial)

        assert ceye2.cam.source_names == ["Hi!"]

        assert ceye2.yields.LOSS
        assert ceye2.yields.ENCODING

        assert ceye2.recognition_system.model.is_same_at_start(AutoEncoder(1024))
        assert isinstance(ceye2.recognition_system.loss_criteria, nn.SmoothL1Loss)
        assert isinstance(ceye2.recognition_system.optimizer, optim.Adam)

        assert ceye2.movement_encoding_widths.CENTER_X == 4
        assert ceye2.movement_encoding_widths.CENTER_Y == 4
        assert ceye2.movement_encoding_widths.CENTER_DX == 4
        assert ceye2.movement_encoding_widths.CENTER_DY == 4
        assert ceye2.movement_encoding_widths.ZOOM == 4
        assert ceye2.movement_encoding_widths.DZOOM == 4
        assert ceye2.movement_encoding_widths.BARREL == 4
        assert ceye2.movement_encoding_widths.DBARREL == 4

        assert ceye2.crop_settings.CAM_SIZE_REQUEST == (99999, 99999)
        assert ceye2.crop_settings.PRE_LENS == (480, 640, 3)
        assert ceye2.crop_settings.POST_LENS == (256, 256, 3)


# @pytest.mark.skip("Not saving to the file system.")
def test_cam_eye_save_and_load():
    with mock.patch.object(VirtualEyeWithLens, "_init_cam", autospec=True) as mock_cam:
        def new_cam(self, cam):
            cam_mock = mock.MagicMock()
            if not isinstance(cam, list):
                cam = [cam]
            cam_mock.source_names = cam
            self.cam = cam_mock

        try:
            mock_cam.side_effect = new_cam

            ceye = VirtualEyeWithLens("Hi!")

            ceye.save("test_cam_eye_save_and_load.torch")
            ceye2 = VirtualEyeWithLens.load("test_cam_eye_save_and_load.torch")

            assert ceye2.cam.source_names == ["Hi!"]

            assert ceye2.yields.LOSS
            assert ceye2.yields.ENCODING

            assert ceye2.recognition_system.model.is_same_at_start(AutoEncoder(1024))
            assert isinstance(ceye2.recognition_system.loss_criteria, nn.SmoothL1Loss)
            assert isinstance(ceye2.recognition_system.optimizer, optim.Adam)

            assert ceye2.movement_encoding_widths.CENTER_X == 4
            assert ceye2.movement_encoding_widths.CENTER_Y == 4
            assert ceye2.movement_encoding_widths.CENTER_DX == 4
            assert ceye2.movement_encoding_widths.CENTER_DY == 4
            assert ceye2.movement_encoding_widths.ZOOM == 4
            assert ceye2.movement_encoding_widths.DZOOM == 4
            assert ceye2.movement_encoding_widths.BARREL == 4
            assert ceye2.movement_encoding_widths.DBARREL == 4

            assert ceye2.crop_settings.CAM_SIZE_REQUEST == (99999, 99999)
            assert ceye2.crop_settings.PRE_LENS == (480, 640, 3)
            assert ceye2.crop_settings.POST_LENS == (256, 256, 3)
        finally:
            os.remove("test_cam_eye_save_and_load.torch")


def test_set_focal_point():
    with mock.patch.object(VirtualEyeWithLens, "_init_cam"):
        ceye = VirtualEyeWithLens()

        assert ceye.CENTER_MIN == 0.001
        assert ceye.CENTER_MAX == 1
        assert ceye.ZOOM_MIN == 0.5
        assert ceye.ZOOM_MAX == 2.0
        assert ceye.BARREL_MIN == 0.5
        assert ceye.BARREL_MAX == 2.0

        ceye.set_focal_point(np.asarray([-1, -1]), 0, 0)

        assert all(ceye.unclipped_center_x_y == np.asarray([-1, -1]))
        assert all(ceye.center_x_y == np.asarray([ceye.CENTER_MIN, ceye.CENTER_MIN]))
        assert ceye.unclipped_zoom == 0
        assert ceye.zoom == ceye.ZOOM_MIN
        assert ceye.unclipped_barrel == 0
        assert ceye.barrel == ceye.BARREL_MIN

        ceye.set_focal_point(np.asarray([2, 3]), 4, 5)

        assert all(ceye.unclipped_center_x_y == np.asarray([2, 3]))
        assert all(ceye.center_x_y == np.asarray([ceye.CENTER_MAX, ceye.CENTER_MAX]))
        assert ceye.unclipped_zoom == 4
        assert ceye.zoom == ceye.ZOOM_MAX
        assert ceye.unclipped_barrel == 5
        assert ceye.barrel == ceye.BARREL_MAX

        ceye.set_focal_point(np.asarray([0.3, 0.4]), 0.6, 0.7)

        assert all(ceye.unclipped_center_x_y == np.asarray([0.3, 0.4]))
        assert all(ceye.center_x_y == np.asarray([0.3, 0.4]))
        assert ceye.unclipped_zoom == 0.6
        assert ceye.zoom == 0.6
        assert ceye.unclipped_barrel == 0.7
        assert ceye.barrel == 0.7
