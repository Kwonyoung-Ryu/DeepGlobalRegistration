import argparse
import torch
import numpy as np
import open3d as o3d

from dataloader.common import make_point_cloud


def backend_ns(backend):
    backend_namespaces = {'numpy': np, 'torch': torch}
    return backend_namespaces[backend]


def constructor_fn(backend, device='cpu'):
    constructor_fns = {
        'numpy': lambda x: np.array(x),
        'torch': lambda x: torch.tensor(x, device=device)
    }
    return constructor_fns[backend]


def unsqueeze_fn(backend):
    unsqueeze_fns = {
        'numpy': lambda x, axis: np.expand_dims(x, axis=axis),
        'torch': lambda x, axis: torch.unsqueeze(x, dim=axis)
    }
    return unsqueeze_fns[backend]


def atan2_fn(backend):
    atan2_fns = {
        'numpy': lambda y, x: np.arctan2(y, x),
        'torch': lambda y, x: torch.atan2(y, x)
    }
    return atan2_fns[backend]


def norm_fn(backend):
    norm_fns = {
        'numpy': lambda x, axis: np.linalg.norm(x, axis=axis),
        'torch': lambda x, axis: torch.linalg.norm(x, dim=axis)
    }
    return norm_fns[backend]


def permute_fn(backend):
    permute_fns = {
        'numpy': lambda x, dims: x.transpose(dims),
        'torch': lambda x, dims: x.permute(*dims)
    }
    return permute_fns[backend]


def to_fn(backend, device):
    to_fns = {
        'numpy': lambda x, t: x.astype(t),
        'torch': lambda x, t: x.to(device, t)
    }
    return to_fns[backend]


def linspace_fn(backend, device):
    linspace_fns = {
        'numpy':
        np.linspace,
        'torch':
        lambda start, end, step: torch.linspace(
            start, end, step, device=device)
    }
    return linspace_fns[backend]


def copy_fn(backend):
    copy_fns = {'numpy': np.copy, 'torch': torch.clone}
    return copy_fns[backend]


class OusterCalib:
    def __init__(self, backend='numpy', device='cpu'):
        print('OusterCalib __init__')
        self.h = 128
        self.w = 1024
        self.n = 27.67
        self.range_unit = 0.001

        self.device = device
        self.constructor = constructor_fn(backend, device)
        self.linspace = linspace_fn(backend, device)
        self.to = to_fn(backend, device)

        self.backend = backend_ns(backend)
        self.unsqueeze = unsqueeze_fn(backend)
        self.atan2 = atan2_fn(backend)
        self.norm = norm_fn(backend)
        self.permute = permute_fn(backend)

        self.lidar_to_sensor_transform = self.constructor(
            [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 36.18, 0, 0, 0, 1]).reshape(
                (4, 4))

        self.altitude_table = self.constructor([
            45.92, 44.91, 44.21, 43.76, 42.93, 41.95, 41.25, 40.81, 39.97,
            39.02, 38.31, 37.85, 37.02, 36.08, 35.38, 34.9, 34.09, 33.16,
            32.46, 31.97, 31.16, 30.25, 29.56, 29.06, 28.25, 27.36, 26.67,
            26.16, 25.34, 24.48, 23.79, 23.26, 22.45, 21.62, 20.93, 20.39,
            19.58, 18.77, 18.09, 17.54, 16.72, 15.94, 15.26, 14.69, 13.88,
            13.11, 12.44, 11.85, 11.03, 10.3, 9.63, 9.04, 8.2, 7.49, 6.83,
            6.22, 5.37, 4.69, 4.04, 3.4, 2.55, 1.89, 1.24, 0.58, -0.27, -0.91,
            -1.56, -2.23, -3.09, -3.71, -4.36, -5.05, -5.91, -6.51, -7.17,
            -7.87, -8.74, -9.32, -9.98, -10.71, -11.57, -12.14, -12.79, -13.54,
            -14.41, -14.96, -15.62, -16.41, -17.26, -17.8, -18.46, -19.26,
            -20.12, -20.64, -21.31, -22.14, -22.98, -23.49, -24.16, -25.01,
            -25.87, -26.36, -27.03, -27.9, -28.76, -29.24, -29.91, -30.81,
            -31.67, -32.13, -32.82, -33.73, -34.61, -35.06, -35.75, -36.69,
            -37.56, -38, -38.7, -39.66, -40.54, -40.97, -41.68, -42.65, -43.57,
            -43.97, -44.68, -45.68
        ])

        self.azimuth_table = self.constructor([
            11.47, 4.15, -3.05, -10.13, 11, 3.95, -2.97, -9.83, 10.59, 3.8,
            -2.9, -9.55, 10.23, 3.65, -2.86, -9.31, 9.93, 3.52, -2.82, -9.12,
            9.66, 3.41, -2.79, -8.94, 9.42, 3.31, -2.77, -8.8, 9.21, 3.22,
            -2.74, -8.69, 9.03, 3.13, -2.73, -8.59, 8.88, 3.06, -2.73, -8.51,
            8.74, 3.01, -2.71, -8.45, 8.64, 2.95, -2.74, -8.4, 8.54, 2.9,
            -2.73, -8.38, 8.45, 2.86, -2.76, -8.37, 8.4, 2.82, -2.78, -8.36,
            8.36, 2.8, -2.79, -8.37, 8.34, 2.78, -2.81, -8.4, 8.32, 2.75,
            -2.86, -8.43, 8.32, 2.72, -2.89, -8.5, 8.34, 2.72, -2.93, -8.58,
            8.36, 2.71, -2.98, -8.67, 8.41, 2.72, -3.03, -8.79, 8.47, 2.72,
            -3.07, -8.9, 8.55, 2.73, -3.13, -9.06, 8.65, 2.74, -3.2, -9.21,
            8.77, 2.77, -3.28, -9.41, 8.92, 2.81, -3.39, -9.63, 9.1, 2.84,
            -3.48, -9.87, 9.3, 2.89, -3.6, -10.16, 9.53, 2.95, -3.73, -10.51,
            9.82, 3.04, -3.89, -10.89, 10.17, 3.12, -4.05, -11.33
        ])

        self.shift_table = self.constructor([
            65, 44, 23, 3, 63, 43, 24, 4, 62, 43, 24, 5, 61, 42, 24, 6, 60, 42,
            24, 6, 59, 42, 24, 7, 59, 41, 24, 7, 58, 41, 24, 7, 58, 41, 24, 8,
            57, 41, 24, 8, 57, 41, 24, 8, 57, 40, 24, 8, 56, 40, 24, 8, 56, 40,
            24, 8, 56, 40, 24, 8, 56, 40, 24, 8, 56, 40, 24, 8, 56, 40, 24, 8,
            56, 40, 24, 8, 56, 40, 24, 8, 56, 40, 24, 7, 56, 40, 23, 7, 56, 40,
            23, 7, 56, 40, 23, 6, 57, 40, 23, 6, 57, 40, 23, 5, 57, 40, 22, 5,
            58, 40, 22, 4, 58, 40, 22, 3, 59, 40, 21, 2, 60, 41, 21, 1, 61, 41,
            20, 0
        ])

        assert self.h == len(self.azimuth_table)
        assert self.h == len(self.altitude_table)

        self.lut_dir, self.lut_offset = self._xyz2lut()
        self.inv_altitude_lut, self.altitude_lut_resolution = self._philut()

    def unproject(self, range_im, factor=1):
        '''
        Unproject a range image in the shape of (H, W)
        Return a xyz image in the shape of (H, W, 3)
        xyz corresponding to 0 range will be assigned 0
        '''
        h, w = range_im.shape
        assert h == self.h // factor
        assert w == self.w // factor

        mask_im = range_im > 0
        range_im = self.unsqueeze(range_im, -1)
        xyz_im = range_im * self.lut_dir[::factor, ::factor] \
            + self.lut_offset[::factor, ::factor]

        # Filter no-value points --> (0, 0, 0)
        xyz_im[~mask_im] = 0

        return xyz_im, mask_im

    def project(self, xyz, factor=1):
        '''
        \param xyz Input point cloud (N, 3).

        \param w Width of the range image. Default: 1024
        \param h Height of the range image. Default: 128c = Calib()
        \param altitude_table Table of the altitude per row. (128, ) array
        \param n Lidar origin to beam origin offset in mm
        \param lidar_to_sensor_transform Lidar to scan coordinate transformation (4, 4)
        \param shift_table Shift per row in pixels. (128, ) array
        '''
        # A scalar
        alpha = np.deg2rad(360 / self.w)

        # Rigid transformation
        sensor_to_lidar_transform = self.backend.linalg.inv(
            self.lidar_to_sensor_transform)
        R = sensor_to_lidar_transform[:3, :3]
        t = sensor_to_lidar_transform[:3, 3:]

        xyz = xyz @ R.T + t.T * self.range_unit

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        range_pt = self.norm(xyz, axis=1)

        # Azimuth
        u_hat = self.atan2(y, x)
        u_hat[u_hat < 0] += 2 * np.pi
        u_hat = self.backend.round((2 * np.pi - u_hat) / alpha)
        u = u_hat

        # Altitude
        phi = self.backend.arcsin(z / range_pt)
        v, mask = self.lookup_v(self.backend.rad2deg(phi))

        # Update, works when normals are fixed Jul-22
        u -= (self.azimuth_table[v] * self.w / 360.0)

        u = self.to(self.backend.round(u), int)
        u[u < 0] += self.w
        u[u >= self.w] -= self.w

        return u // factor, v // factor, range_pt, mask

    def shift(self, im, factor=1):
        shifted = self.backend.zeros_like(im)
        h = shifted.shape[0]
        w = shifted.shape[1]

        assert h == self.h // factor
        assert w == self.w // factor

        for i in range(h):
            s = np.round(self.azimuth_table[i * factor] * self.w /
                         360.0).astype(int)
            if s < 0:
                s += self.w

            shifted[i, s:] = im[i, :(w - s)]
            shifted[i, :s] = im[i, (w - s):]

        return shifted

    def lookup_v(self, phi):
        phi_int = self.to(
            self.backend.round(phi - self.altitude_table[-1]) /
            self.altitude_lut_resolution, int)
        mask = (phi_int >= 0) & (phi_int < len(self.inv_altitude_lut))
        phi_int[~mask] = -1

        v1 = self.h - 1 - self.inv_altitude_lut[phi_int]
        v2 = self.backend.maximum(v1 - 1, self.constructor([0]))
        v3 = self.backend.minimum(v1 + 1, self.constructor([self.h - 1]))
        vs = self.backend.stack((v1, v2, v3))

        diff1 = self.backend.abs(self.altitude_table[v1] - phi)
        diff2 = self.backend.abs(self.altitude_table[v2] - phi)
        diff3 = self.backend.abs(self.altitude_table[v3] - phi)
        diff = self.backend.stack((diff1, diff2, diff3))
        v_sel = self.backend.argmin(diff, axis=0)
        v = vs[v_sel, self.backend.arange(len(phi))]

        return v, mask

    def _philut(self):
        altitude_table = self.backend.flipud(self.altitude_table)
        reversed_table = altitude_table - altitude_table[0]

        resolution = 0.4
        reversed_table_int = self.to((reversed_table / resolution).round(),
                                     int)
        padding = np.round(1.0 / resolution).astype(int)
        inv_table = self.to(
            self.backend.zeros((reversed_table_int[-1] + padding)), int)
        j = 0
        for i in range(len(reversed_table_int) - 1):
            delta = reversed_table_int[i + 1] - reversed_table_int[i]
            inv_table[j:j + delta] = i
            j += delta
        inv_table[-padding:] = len(reversed_table_int) - 1

        return inv_table, resolution

    def _xyz2lut(self):
        '''
        Ref: https://github.com/ouster-lidar/ouster_example/blob/master/ouster_client/src/lidar_scan.cpp#L11
        \brief Generate two (H, W, 3) pixel-wise lookup tables for xyz.
        \param w Width of the range image. Default: 1024
        \param h Height of the range image. Default: 128
        \param azimuth_table Table of the azimuth offset per column. (128, ) array
        \param altitude_table Table of the altitude offset per column. (128, ) array
        \param n Lidar origin to beam origin offset in mm
        \param lidar_to_sensor_transform Lidar to scan coordinate transformation (4, 4)
        \return lut_dir Directional look up table (H, W, 3)
        \return lut_offset Offset look up table (H, W, 3)
        '''

        # column 0: 2pi ==> column w-1: 2pi - (w-1)/w*2pi = 2pi/w
        theta_encoder = self.linspace(2 * np.pi, 2 * np.pi / self.w, self.w)
        theta_encoder = self.backend.tile(theta_encoder, (self.h, 1))

        # unroll azimuth table
        theta_azimuth = -self.backend.deg2rad(self.azimuth_table)
        theta_azimuth = self.backend.tile(self.unsqueeze(theta_azimuth, 1),
                                          (1, self.w))

        # unroll altitude table
        phi = self.backend.deg2rad(self.altitude_table)
        phi = self.backend.tile(self.unsqueeze(phi, 1), (1, self.w))

        x_dir = self.backend.cos(theta_encoder +
                                 theta_azimuth) * self.backend.cos(phi)
        y_dir = self.backend.sin(theta_encoder +
                                 theta_azimuth) * self.backend.cos(phi)
        z_dir = self.backend.sin(phi)
        lut_dir = self.permute(self.backend.stack((x_dir, y_dir, z_dir)),
                               (1, 2, 0))

        x_offset = self.n * (self.backend.cos(theta_encoder) - x_dir)
        y_offset = self.n * (self.backend.sin(theta_encoder) - y_dir)
        z_offset = self.n * (-z_dir)

        lut_offset = self.permute(
            self.backend.stack((x_offset, y_offset, z_offset)), (1, 2, 0))

        R = self.lidar_to_sensor_transform[:3, :3]
        t = self.lidar_to_sensor_transform[:3, 3:]

        lut_dir = (lut_dir.reshape((-1, 3)) @ R.T).reshape((self.h, self.w, 3))
        lut_offset += t.T

        return lut_dir * self.range_unit, lut_offset * self.range_unit


ouster_calib_singleton = None


def get_ouster_calib(backend='numpy', device='cpu'):
    global ouster_calib_singleton
    if not ouster_calib_singleton:
        ouster_calib_singleton = OusterCalib(backend, device)
    return ouster_calib_singleton


def load_range(filename):
    if filename.endswith('png'):
        import cv2
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(float)
    elif filename.endswith('csv'):
        return np.loadtxt(filename, delimiter=',')
    else:
        print('Unrecognized extension {}'.format(filename))


'''
Note: only use the torch + cuda backend for integration, since adaptation of icp is not done.
'''


class OusterData:
    def __init__(self, input=None, backend='numpy', device='cpu'):
        self.backend = backend
        self.device = device

        c = get_ouster_calib(backend, device)

        if isinstance(input, str):
            self.range_im = load_range(input)  # numpy
        else:
            self.range_im = input  # numpy or torch

        if isinstance(self.range_im, np.ndarray):
            # numpy
            if backend == 'torch':
                self.range_im = torch.from_numpy(self.range_im).to(device)

        elif isinstance(self.range_im, torch.Tensor):
            if backend == 'numpy':
                self.range_im = self.range_im.detach().cpu().numpy()

        else:
            self.range_im = None

        if self.range_im is not None:
            assert c.h == self.range_im.shape[0]
            assert c.w == self.range_im.shape[1]

            self.xyz_im, self.mask_im = c.unproject(self.range_im)
        else:
            self.xyz_im = None
            self.mask_im = None

    def downsample(self, factor=2):
        copy = copy_fn(self.backend)

        output = OusterData(None, self.backend, self.device)

        output.range_im = copy(self.range_im[::factor, ::factor])
        output.mask_im = copy(self.mask_im[::factor, ::factor])
        output.xyz_im = copy(self.xyz_im[::factor, ::factor])

        return output

    def get_pcd_and_normal_im(self):
        xyz_im = self.xyz_im
        mask_im = self.mask_im
        if self.backend == 'torch':
            xyz_im = xyz_im.detach().cpu().numpy()
            mask_im = mask_im.detach().cpu().numpy()

        pcd_tmp = make_point_cloud(xyz_im.reshape((-1, 3)))
        pcd_tmp.estimate_normals()
        normal_im = np.asarray(pcd_tmp.normals).reshape(xyz_im.shape)

        pcd = make_point_cloud(xyz_im[mask_im], normal_im[mask_im])
        return pcd, normal_im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='numpy')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    ouster = OusterCalib(args.backend, args.device)

    np.savez('ouster_calib.npz',
             altitude_table=ouster.altitude_table,
             inv_altitude_table=ouster.inv_altitude_lut,
             azimuth_table=ouster.azimuth_table,
             lidar_to_sensor=ouster.lidar_to_sensor_transform,
             lut_dir=ouster.lut_dir,
             lut_offset=ouster.lut_offset)

    constructor = constructor_fn(args.backend, args.device)

    phi = constructor([-300, 31.5, 42.5, -3.6, -20.7, -22, 500, 45.79])
    v, mask = ouster.lookup_v(phi)
    print(phi[mask])
    print(ouster.altitude_table[v[mask]])

    phi = np.linspace(-50.0, 50.0, 200)
    v, mask = ouster.lookup_v(phi)
    diff = phi[mask] - ouster.altitude_table[v[mask]]
    print(diff)
    ind = np.argmax(np.abs(diff))
    print(ind, v[mask][ind])
    print(phi[mask][ind])
    print(ouster.altitude_table[v[mask]][ind])

    print(np.abs(diff).min())
