# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import os
import glob

from dataloader.base_loader import *
from dataloader.transforms import *
from dataloader.ouster import OusterData

from util.pointcloud import get_matching_indices, make_open3d_point_cloud



postech_cache = {}
postech_icp_cache = {}

class POSTECHPairDataset(PairDataset):
  AUGMENT = None
  DATA_FILES = {
      'train': './dataloader/split/train_postech.txt',
      'val': './dataloader/split/val_postech.txt',
      'test':'./dataloader/split/test_postech.txt'
  }
  TEST_RANDOM_ROTATION = False
  IS_ODOMETRY = True

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    # For evaluation, use the odometry dataset training following the 3DFeat eval method
    self.root = root = config.postech_dir
    random_rotation = self.TEST_RANDOM_ROTATION
    self.icp_path = config.icp_cache_path
    try:
      os.mkdir(self.icp_path)
    except OSError as error:
      pass
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")
    # Use the postech root
    self.max_time_diff = max_time_diff = config.postech_max_time_diff

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for dirname in subset_names:
      drive_id = dirname
      inames = self.get_all_scan_ids(drive_id)
      for start_time in inames:
        for time_diff in range(2, max_time_diff):
          pair_time = time_diff + start_time
          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

  def get_all_scan_ids(self, drive_id):
    fnames = glob.glob(self.root + drive_id+'/depth/*.png')
    assert len(
        fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
    inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
    return inames

  def odometry_to_positions(self, odometry):
    T_w_cam0 = odometry.reshape(3, 4)
    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
    return T_w_cam0

  def rot3d(self, axis, angle):
    ei = np.ones(3, dtype='bool')
    ei[axis] = 0
    i = np.nonzero(ei)[0]
    m = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    m[i[0], i[0]] = c
    m[i[0], i[1]] = -s
    m[i[1], i[0]] = s
    m[i[1], i[1]] = c
    return m

  def pos_transform(self, pos):
    x, y, z, rx, ry, rz, _ = pos[0]
    RT = np.eye(4)
    RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
    RT[:3, 3] = [x, y, z]
    return RT

  def get_position_transform(self, pos0, pos1, invert=False):
    T0 = self.pos_transform(pos0)
    T1 = self.pos_transform(pos1)
    return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
        np.linalg.inv(T1), T0).T)

  def get_ouster_fn(self, drive, t):
    fname = self.root + drive +'/depth/%06d.png' % t
    return fname

  def get_pseudo_gt_pose(self, drive, indices=None, return_all=False):
    dirname = self.root + drive
    pose_graph = o3d.io.read_pose_graph(os.path.join(dirname, 'pose_graph.json'))
    
    if dirname not in postech_cache:
      postech_cache[dirname] = list(pose_graph.nodes)
    if return_all:
      return postech_cache[dirname]
    else:
      return [postech_cache[dirname][i].pose for i in indices]

  def __getitem__(self, idx):
    
    drive = self.files[idx][0]
    t0, t1 = self.files[idx][1], self.files[idx][2]
    pose = self.get_pseudo_gt_pose(drive, [t0, t1])
    fname0 = self.get_ouster_fn(drive, t0)
    fname1 = self.get_ouster_fn(drive, t1)

    # XYZ and reflectance
    xyz0, _ = OusterData(fname0).get_pcd_and_normal_im()
    xyz1, _ = OusterData(fname1).get_pcd_and_normal_im()

    xyz0, xyz1 = np.array(xyz0.points), np.array(xyz1.points)

    key = '%s_%d_%d' % (drive, t0, t1)
    filename = self.icp_path + '/' + key + '.npy'
    if key not in postech_icp_cache:
      if not os.path.exists(filename):
        # work on the downsampled xyzs, 0.05m == 5cm

        sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

        M = (pose[0].T @ np.linalg.inv(pose[1].T)).T #@ np.linalg.inv(self.velo2cam)).T
        
        xyz0_t = self.apply_transform(xyz0[sel0[1]], M)
        pcd0 = make_open3d_point_cloud(xyz0_t)
        pcd1 = make_open3d_point_cloud(xyz1[sel1[1]])
        reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                   o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                   o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        pcd0.transform(reg.transformation)
        # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
        M2 = M @ reg.transformation
        # o3d.draw_geometries([pcd0, pcd1])
        # write to a file
        np.save(filename, M2)
      else:
        M2 = np.load(filename)
      postech_icp_cache[key] = M2
    else:
      M2 = postech_icp_cache[key]

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
      T1 = sample_random_trans(xyz1, self.randg, np.pi / 4)
      trans = T1 @ M2 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = M2

    matching_search_voxel_size = self.matching_search_voxel_size
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    # Voxelization
    xyz0_th = torch.from_numpy(xyz0)
    xyz1_th = torch.from_numpy(xyz1)

    sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0[sel0[1]])
    pcd1 = make_open3d_point_cloud(xyz1[sel1[1]])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    if len(matches) < 1000:
      raise ValueError(f"Insufficient matches in {drive}, {t0}, {t1}")

    # Get features
    npts0 = len(sel0[1])
    npts1 = len(sel1[1])

    feats_train0, feats_train1 = [], []

    unique_xyz0_th = xyz0_th[sel0[1]]
    unique_xyz1_th = xyz1_th[sel1[1]]

    feats_train0.append(torch.ones((npts0, 1)))
    feats_train1.append(torch.ones((npts1, 1)))

    feats0 = torch.cat(feats_train0, 1)
    feats1 = torch.cat(feats_train1, 1)

    coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
    coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    extra_package = {'drive': drive, 't0': t0, 't1': t1}

    return (unique_xyz0_th.float(),
            unique_xyz1_th.float(), coords0.int(), coords1.int(), feats0.float(),
            feats1.float(), matches, trans, extra_package)


class POSTECHNMPairDataset(POSTECHPairDataset):
  r"""
  Generate postech pairs within N meter distance
  """
  MIN_DIST = 10

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.root = root = os.path.join(config.postech_dir, 'dataset')
    self.icp_path = os.path.join(config.postech_dir, config.icp_cache_path)
    try:
      os.mkdir(self.icp_path)
    except OSError as error:
      pass
    random_rotation = self.TEST_RANDOM_ROTATION
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for dirname in subset_names:
      drive_id = int(dirname)
      fnames = glob.glob(root + drive_id + '/depth/*.png')
      assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
      inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

      all_pos = self.get_pseudo_gt_pose(drive_id, return_all=True)
      Ts = all_pos[:, :3, 3]
      pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
      pdist = np.sqrt(pdist.sum(-1))
      more_than_10 = pdist > self.MIN_DIST
      curr_time = inames[0]
      while curr_time in inames:
        # Find the min index
        next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
        if len(next_time) == 0:
          curr_time += 1
        else:
          next_time = next_time[0] + curr_time - 1

        if next_time in inames:
          self.files.append((drive_id, curr_time, next_time))
          curr_time = next_time + 1

    # Remove problematic sequence
    for item in [
        (8, 15, 58),
    ]:
      if item in self.files:
        self.files.pop(self.files.index(item))
