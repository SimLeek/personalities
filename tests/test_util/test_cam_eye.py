from personalities.util.cam_eye import *
import mock


def test_cam_eye_init():
    with mock.patch("personalities.util.cam_eye.display") as mock_cam:
        ceye = CamEye()

        assert ceye.yields.LOSS
        assert ceye.yields.ENCODING

        assert ceye.recognition_system.model == AutoEncoder(1024)
        assert ceye.recognition_system.loss_criteria == nn.MSELoss()
        assert ceye.recognition_system.optimizer == optim.Adam(ceye.recognition_system.model.parameters())

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
