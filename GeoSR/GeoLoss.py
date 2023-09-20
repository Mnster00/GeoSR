import torch
import torch.nn as nn
import cv2
import numpy as np

class GeometricErrorLoss(nn.Module):
    def __init__(self):
        super(GeometricErrorLoss, self).__init__()

    def hough_transform(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        return lines

    def zoom_lines(self, lines, zoom_factor):
        return lines * zoom_factor

    def mse_error(self, lines1, lines2):
        # Assuming lines1 and lines2 are of the same length
        diff = lines1 - lines2
        mse = torch.mean(torch.pow(diff, 2))
        return mse

    def forward(self, low_res_img, high_res_img):
        # Convert PyTorch tensors to NumPy arrays and then to OpenCV format
        low_res_img_cv = (low_res_img.permute(0, 2, 3, 1).cpu().detach().numpy() * 255).astype(np.uint8)
        high_res_img_cv = (high_res_img.permute(0, 2, 3, 1).cpu().detach().numpy() * 255).astype(np.uint8)

        # Extract lines
        low_res_lines = self.hough_transform(low_res_img_cv[0])
        high_res_lines = self.hough_transform(high_res_img_cv[0])

        # Zoom in low-res lines
        zoom_factor = high_res_img.shape[-1] / low_res_img.shape[-1]
        low_res_lines_zoomed = self.zoom_lines(low_res_lines, zoom_factor)

        # Calculate MSE
        mse = self.mse_error(torch.from_numpy(low_res_lines_zoomed), torch.from_numpy(high_res_lines))

        return mse