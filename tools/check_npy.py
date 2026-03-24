import numpy as np
import matplotlib.pyplot as plt


# ===============================
# H36M 17-joint 定义
# ===============================
H36M_NAMES = [
    "pelvis",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "neck", "head",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist"
]

# 骨架连接关系
H36M_BONES = [
    (0,7),(7,8),(8,9),(9,10),
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16)
]


# ===============================
# 读取 npy
# ===============================
def load_h36m_npy(path):
    data = np.load(path, allow_pickle=True)

    print("========== H36M DATA INFO ==========")
    print("Type:", type(data))
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)

    if len(data.shape) != 3:
        raise ValueError("Expected shape [T, J, C]")

    T, J, C = data.shape
    print(f"Frames (T): {T}")
    print(f"Joints (J): {J}")
    print(f"Coords (C): {C}")

    if J == 17:
        print("\nJoint mapping:")
        for i, name in enumerate(H36M_NAMES):
            print(f"{i:02d}: {name}")

    return data


# ===============================
# 可视化单帧（2D/3D自动判断）
# ===============================
def visualize_frame(pose, frame_id=0):

    frame = pose[frame_id]
    C = frame.shape[-1]

    if C == 2:
        plt.figure()
        for b in H36M_BONES:
            x = [frame[b[0],0], frame[b[1],0]]
            y = [frame[b[0],1], frame[b[1],1]]
            plt.plot(x, y)

        plt.scatter(frame[:,0], frame[:,1])
        plt.gca().invert_yaxis()
        plt.title(f"Frame {frame_id} (2D)")
        plt.show()

    elif C == 3:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for b in H36M_BONES:
            xs = [frame[b[0],0], frame[b[1],0]]
            ys = [frame[b[0],1], frame[b[1],1]]
            zs = [frame[b[0],2], frame[b[1],2]]
            ax.plot(xs, ys, zs)

        ax.scatter(frame[:,0], frame[:,1], frame[:,2])
        ax.set_title(f"Frame {frame_id} (3D)")
        plt.show()

    else:
        raise ValueError("Coordinate dim must be 2 or 3")


# ===============================
# 主函数
# ===============================
if __name__ == "__main__":

    file_path =r"D:\Python项目\KeyFrame\AthletePose3D\data\train_set\S3\Running_55_cam_1_h36m.npy"   # 修改为你的路径

    pose = load_h36m_npy(file_path)

    # 查看第一帧
    visualize_frame(pose, frame_id=0)