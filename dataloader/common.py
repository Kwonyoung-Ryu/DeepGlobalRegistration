import os
import numpy as np
import open3d as o3d
import glob

flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


def robust_kernel_fn(alpha, c):
    if alpha == None:
        # print('Binary truncation loss')
        return lambda x: (np.abs(x) < c).astype(float)
    if alpha == 2:
        # print('L2 loss')
        return lambda x: np.ones_like(x) / c**2
    elif alpha == 0:
        # print('Cauchy loss')
        return lambda x: 2 / (x**2 + 2 * c**2)
    elif alpha < -1e5:
        # print('Welsch loss')
        return lambda x: 1 / c**2 * np.exp(-0.5 * (x / c)**2)
    else:
        #if alpha == -2:
            # print('Geman-McClure loss')
        #elif alpha == 1:
            # print('Charbonnier / Pseudo-Huber loss')
        #else:
            # print('General loss with alpha = ', alpha)
        return lambda x: 1 / c**2 * np.float_power(
            (x / c)**2 / np.abs(alpha - 2) + 1, alpha / 2 - 1)


def lineset_from_pose_graph(pose_graph, show_loops=True, edge_density=0.1, l = 0.1):

    POINTS_PER_FRUSTUM = 5
    EDGES_PER_FRUSTUM = 8

    points = []
    colors = []
    lines = []

    cnt = 0
    for i, node in enumerate(pose_graph.nodes):
        pose = np.array(node.pose)

        #l = 0.1
        points.append((pose @ np.array([0, 0, 0, 1]).T)[:3])
        points.append((pose @ np.array([l, l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([l, -l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([-l, -l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([-l, l, 2 * l, 1]).T)[:3])

        lines.append([cnt + 0, cnt + 1])
        lines.append([cnt + 0, cnt + 2])
        lines.append([cnt + 0, cnt + 3])
        lines.append([cnt + 0, cnt + 4])
        lines.append([cnt + 1, cnt + 2])
        lines.append([cnt + 2, cnt + 3])
        lines.append([cnt + 3, cnt + 4])
        lines.append([cnt + 4, cnt + 1])

        for i in range(0, EDGES_PER_FRUSTUM):
            colors.append(np.array([0, 0, 1]))

        cnt += POINTS_PER_FRUSTUM

    print('nodes: {}'.format(len(pose_graph.nodes)))
    loops = 0
    random_index = np.random.choice(len(pose_graph.edges), int(len(pose_graph.edges)*edge_density), replace=False)
    
    if show_loops:
        for i, edge in enumerate(pose_graph.edges):
            switch = np.random.rand(1) < edge_density
            s = edge.source_node_id
            t = edge.target_node_id
            
            if (edge.uncertain & switch):
                lines.append([POINTS_PER_FRUSTUM * s, POINTS_PER_FRUSTUM * t])
                colors.append(np.array([0, 1, 0])) 
                loops += 1
            elif not edge.uncertain:
                lines.append([POINTS_PER_FRUSTUM * s, POINTS_PER_FRUSTUM * t])
                colors.append(np.array([0, 0, 1]))
                
    print('loops: {}'.format(loops))

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.vstack(points))
    lineset.lines = o3d.utility.Vector2iVector(np.vstack(lines).astype(int))
    lineset.colors = o3d.utility.Vector3dVector(np.vstack(colors))

    return lineset


def get_normal_map_o3d(data):
    pcd = data.to_o3d_pointcloud()
    pcd.estimate_normals()
    normal_map = np.asarray(pcd.normals).reshape(data.xyz_im.shape)
    normal_map = np.squeeze(normal_map)

    return normal_map, pcd


def make_point_cloud(points, normals=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_icp(pcd_source, pcd_target, T):
    import copy
    pcd_source = copy.deepcopy(pcd_source)
    # pcd_source.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    pcd_source.paint_uniform_color([87.0 / 255.0, 144.0 / 255.0, 252.0 / 255.0])

    pcd_target = copy.deepcopy(pcd_target)
    pcd_target.paint_uniform_color([248.0 / 255.0, 156.0 / 255.0, 32.0 / 255.0])

    pcd_source.transform(T)
    o3d.visualization.draw([
        pcd_source, pcd_target
        # pcd_source.transform(flip_transform),
        # pcd_target.transform(flip_transform)
    ])


def visualize_correspondences(source_points, target_points, T):
    if len(source_points) != len(target_points):
        print(
            'Error! source points and target points has different length {} vs {}'
            .format(len(source_points), len(target_points)))
        return

    pcd_source = make_point_cloud(source_points)
    pcd_source.paint_uniform_color([1, 0, 0])
    pcd_source.transform(T)
    pcd_source.transform(flip_transform)

    pcd_target = make_point_cloud(target_points)
    pcd_target.paint_uniform_color([0, 1, 0])
    pcd_target.transform(flip_transform)

    corres = []
    for k in range(len(source_points)):
        corres.append((k, k))

    lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        pcd_source, pcd_target, corres)

    o3d.visualization.draw_geometries([pcd_source, pcd_target, lineset])


def make_point_cloud(points, normals=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def load_range_file_names(config):
    if not os.path.exists(config.path_dataset):
        print(
            'Path \'{}\' not found.'.format(config.path_dataset),
            'Please provide --path_dataset in the command line or the config file.'
        )
        return [], []

    range_folder = os.path.join(config.path_dataset, config.range_folder)

    range_names = glob.glob(os.path.join(range_folder, '*.csv'))
    n_range = len(range_names)
    if n_range == 0:
        print('Range files not found in {}, abort!'.format(range_folder))
        return []

    return sorted(range_names,
                  key=lambda x: int(x.split('/')[-1].split('.')[0]))


def load_depth_file_names(config):
    if not os.path.exists(config.path_dataset):
        print(
            'Path \'{}\' not found.'.format(config.path_dataset),
            'Please provide --path_dataset in the command line or the config file.'
        )
        return [], []

    depth_folder = os.path.join(config.path_dataset, config.depth_folder)

    range_names = glob.glob(os.path.join(depth_folder, '*.png'))
    n_range = len(range_names)
    if n_range == 0:
        print('Range files not found in {}, abort!'.format(depth_folder))
        return []

    return sorted(range_names,
                  key=lambda x: int(x.split('/')[-1].split('.')[0]))


def load_fragment_file_names(config):
    if not os.path.exists(config.path_dataset):
        print(
            'Path \'{}\' not found.'.format(config.path_dataset),
            'Please provide --path_dataset in the command line or the config file.'
        )
        return [], []

    fragment_folder = os.path.join(config.path_dataset, config.fragment_folder)

    fragment_names = glob.glob(os.path.join(fragment_folder, '*.ply'))
    n_fragments = len(fragment_names)
    if n_fragments == 0:
        print('Fragment point clouds not found in {}, abort!'.format(
            depth_folder))
        return []

    return sorted(fragment_names)


def load_image_file_names(config):
    if not os.path.exists(config.path_dataset):
        print(
            'Path \'{}\' not found.'.format(config.path_dataset),
            'Please provide --path_dataset in the command line or the config file.'
        )
        return [], []

    depth_folder = os.path.join(config.path_dataset, config.depth_folder)
    color_folder = os.path.join(config.path_dataset, config.color_folder)

    # Only 16-bit png depth is supported
    depth_file_names = glob.glob(os.path.join(depth_folder, '*.png'))
    n_depth = len(depth_file_names)
    if n_depth == 0:
        print('Depth image not found in {}, abort!'.format(depth_folder))
        return [], []

    # Try png
    extensions = ['*.png', '*.jpg']
    for ext in extensions:
        color_file_names = glob.glob(os.path.join(color_folder, ext))
        if len(color_file_names) == n_depth:
            return sorted(depth_file_names), sorted(color_file_names)

    print('Found {} depth images in {}, but cannot find matched number of '
          'color images in {} with extensions {}, abort!'.format(
              n_depth, depth_folder, color_folder, extensions))
    return [], []


def load_intrinsic(config):
    if config.path_intrinsic is None or config.path_intrinsic == '':
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    else:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(config.path_intrinsic)

    if config.engine == 'legacy':
        return intrinsic
    elif config.engine == 'tensor':
        return o3d.core.Tensor(intrinsic.intrinsic_matrix,
                               o3d.core.Dtype.Float32)
    else:
        print('Unsupported engine {}'.format(config.engine))


def load_extrinsics(path_trajectory, config):
    extrinsics = []

    # For either a fragment or a scene
    if path_trajectory.endswith('log'):
        data = o3d.io.read_pinhole_camera_trajectory(path_trajectory)
        for param in data.parameters:
            extrinsics.append(param.extrinsic)

    # Only for a fragment
    elif path_trajectory.endswith('json'):
        data = o3d.io.read_pose_graph(path_trajectory)
        for node in data.nodes:
            extrinsics.append(np.linalg.inv(node.pose))

    if config.engine == 'legacy':
        return extrinsics
    elif config.engine == 'tensor':
        return list(
            map(lambda x: o3d.core.Tensor(x, o3d.core.Dtype.Float64),
                extrinsics))
    else:
        print('Unsupported engine {}'.format(config.engine))


def save_poses(
    path_trajectory,
    poses,
    intrinsic=o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)):
    if path_trajectory.endswith('log'):
        traj = o3d.camera.PinholeCameraTrajectory()
        params = []
        for pose in poses:
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = intrinsic
            param.extrinsic = np.linalg.inv(pose)
            params.append(param)
        traj.parameters = params
        o3d.io.write_pinhole_camera_trajectory(path_trajectory, traj)

    elif path_trajectory.endswith('json'):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        for pose in poses:
            node = o3d.pipelines.registration.PoseGraphNode()
            node.pose = pose
            pose_graph.nodes.append(node)
        o3d.io.write_pose_graph(path_trajectory, pose_graph)


def init_volume(mode, config):
    if config.engine == 'legacy':
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config.voxel_size,
            sdf_trunc=config.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    elif config.engine == 'tensor':
        if mode == 'scene':
            block_count = config.block_count
        else:
            block_count = config.block_count
        return o3d.t.geometry.TSDFVoxelGrid(
            {
                'tsdf': o3d.core.Dtype.Float32,
                'weight': o3d.core.Dtype.UInt16,
                'color': o3d.core.Dtype.UInt16
            },
            voxel_size=config.voxel_size,
            sdf_trunc=config.sdf_trunc,
            block_resolution=16,
            block_count=block_count,
            device=o3d.core.Device(config.device))
    else:
        print('Unsupported engine {}'.format(config.engine))


def extract_pointcloud(volume, config, file_name=None):
    if config.engine == 'legacy':
        mesh = volume.extract_triangle_mesh()

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors

        if file_name is not None:
            o3d.io.write_point_cloud(file_name, pcd)

    elif config.engine == 'tensor':
        pcd = volume.extract_surface_points(
            weight_threshold=config.surface_weight_thr)

        if file_name is not None:
            o3d.io.write_point_cloud(file_name, pcd.to_legacy())

    return pcd


def extract_trianglemesh(volume, config, file_name=None):
    if config.engine == 'legacy':
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        if file_name is not None:
            o3d.io.write_triangle_mesh(file_name, mesh)

    elif config.engine == 'tensor':
        mesh = volume.extract_surface_mesh(
            weight_threshold=config.surface_weight_thr)
        mesh = mesh.to_legacy()

        if file_name is not None:
            o3d.io.write_triangle_mesh(file_name, mesh)

    return mesh
