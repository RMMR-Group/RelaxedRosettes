from pathlib import Path
import numpy as np
import nibabel as nib
from twixtools import read_twix, map_twix
import matplotlib.pyplot as plt
from pynufft import NUFFT


class Gradient:
    def __init__(self, grad_path):
        header_type = np.dtype([("num_points", "u4"), ("num_grad_points", "u4")])
        header_data = np.fromfile(grad_path, dtype=header_type, count=1)
        self.num_points = header_data["num_points"][0]
        self.num_grad_points = header_data["num_grad_points"][0]
        grad_data = np.fromfile(grad_path, dtype="f4", offset=8)
        self.grad_x = grad_data[0 : self.num_points]
        self.grad_y = grad_data[self.num_points : 2 * self.num_points]
        self.grad_z = grad_data[2 * self.num_points : 3 * self.num_points]

    def get_kspace_trajectory(self):
        return np.stack((np.cumsum(self.grad_x), np.cumsum(self.grad_y), np.cumsum(self.grad_z)), axis=1)[
            : self.num_grad_points, :
        ]

    def get_normalized_kspace_trajectory(self):
        return self.get_kspace_trajectory() * np.pi


if __name__ == "__main__":
    data_path = Path(__file__).parent / "data"
    # read in the dat file
    dat_path = next(data_path.glob("*.dat"))
    twix_data = read_twix(str(dat_path))
    mapped_data = map_twix(twix_data[1])

    # get data
    mapped_data["image"].flags["remove_os"] = False
    mapped_data["image"].flags["average"]["Seg"] = False
    img_data = mapped_data["image"][:, :, :, :, :, :, :, :].squeeze().T

    # read in gradient file
    grad_path = next(data_path.glob("*.bin"))
    grad_data = Gradient(grad_path)

    # linspace the time points to the gradient points
    t = np.round(np.linspace(0, grad_data.num_grad_points - 1, img_data.shape[0])).astype(int)
    print(grad_data.num_grad_points)
    print(t.shape)

    # # plot kspace trajectory
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(
    #     grad_data.get_normalized_kspace_trajectory()[:, 0],
    #     grad_data.get_normalized_kspace_trajectory()[:, 1],
    #     grad_data.get_normalized_kspace_trajectory()[:, 2],
    # )
    # plt.show()

    # get subsampled kspace trajectory
    kspace_trajectory = grad_data.get_normalized_kspace_trajectory()[t, :]

    # get where the kspace trajectory is zero
    zero_idx = np.where(np.all(kspace_trajectory == 0, axis=1))[0]

    # plot subsampled kspace trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(kspace_trajectory[:, 0], kspace_trajectory[:, 1], kspace_trajectory[:, 2])
    plt.show()

    # Create NuFFT object
    nufft_obj = NUFFT()
    Nd = 128
    Kd = 256
    Jd = 8
    nufft_obj.plan(kspace_trajectory, (Nd, Nd, Nd), (Kd, Kd, Kd), (Jd, Jd, Jd))
    recon_list = []
    recon_list2 = []
    for i in range(img_data.shape[1]):
        print(f"Reconstructing {i}")
        recon_list.append(nufft_obj.adjoint(img_data[:, i, 0]))
    recon = np.stack(recon_list, axis=-1)
    mean_recon = np.sqrt(np.sum(recon ** 2, axis=-1))
    mag_recon = np.abs(mean_recon)
    nib.Nifti1Image(mag_recon, np.eye(4)).to_filename(data_path / "recon.nii.gz")
