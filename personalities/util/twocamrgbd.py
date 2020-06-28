from personalities.base.twocamrgbd import PSMNet
import torch
from torch import optim
from torch.autograd import Variable
from torch.functional import F
import math

class TwoCamRgbd(object):
    def __init__(self, load_file="", save_file=None, max_disparity=255):
        self.model = PSMNet(max_disparity)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1, betas=(0.9, 0.999))

        if torch.cuda.is_available():
            self.model.cuda()

        if load_file:
            state_dict = torch.load(load_file)
            self.model.load_state_dict(state_dict["state_dict"])
            self._train_calls = state_dict["train_calls"]
            self.optimizer.load_state_dict(state_dict["optim_dict"])

        if save_file is None:
            self.save_file = load_file

        self._train_calls = 0

    def _adjust_learning_rate(self, train_call):
        lr = 0.001*math.e**(-0.000359778*train_call)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, left_img, right_img, depth_img):
        """

        :param left_img: left rgb camera image
        :param right_img: right rgb camera image
        :param depth_img: reverse depth format.
                          0 means infinite distance,
                          255 (or 1.0) may either be 255 pixels or ~0 cm from both cameras.
                          If it's reversed, the depth mask must be changed.
        :return:
        """
        self.model.train()
        self._adjust_learning_rate(self._train_calls)
        self._train_calls += 1

        # todo: auto convert 0-255 to 0-1

        img_l = Variable(torch.FloatTensor(left_img))
        img_r = Variable(torch.FloatTensor(right_img))
        true_disparity = Variable(torch.FloatTensor(depth_img))

        if torch.cuda.is_available():
            img_l, img_r, true_disparity = img_l.cuda(), img_r.cuda(), true_disparity.cuda()

        finite_depth_mask = true_disparity > 0
        finite_depth_mask.detach_()

        output1, output2, output3 = self.model(img_l, img_r)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = (
                0.5 * F.smooth_l1_loss(output1[finite_depth_mask],
                                       true_disparity[finite_depth_mask],
                                       size_average=True)
                + 0.7 * F.smooth_l1_loss(output2[finite_depth_mask],
                                         true_disparity[finite_depth_mask],
                                         size_average=True)
                + F.smooth_l1_loss(output3[finite_depth_mask],
                                   true_disparity[finite_depth_mask],
                                   size_average=True)
        )

        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def run(self, left_img, right_img):
        self.model.eval()

        img_l = Variable(torch.FloatTensor(left_img))
        img_r = Variable(torch.FloatTensor(right_img))

        if torch.cuda.is_available():
            img_l, img_r = img_l.cuda(), img_r.cuda()

        with torch.no_grad():
            reverse_depth = self.model(img_l, img_r)

        reverse_depth = reverse_depth.data.cpu()

        torch.cuda.empty_cache()

        return reverse_depth

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save()

    def __del__(self):
        self._save()

    def __delete__(self, instance):
        self._save()

    def _save(self):
        if self.save_file:
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "optim_dict": self.optimizer.state_dict(),
                    "train_calls": self._train_calls
                },
                self.save_file,
            )


