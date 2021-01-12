import numpy as np
import matplotlib.pyplot as plt
import glob

COLOR = ['k', 'b', 'g', 'r', 'y', 'm', 'c']


def read_from_file(file_name):
    data_file = open(file_name, "r")
    data_array = np.loadtxt(data_file, dtype=np.float)
    data_file.close()
    return data_array


def search_data_file(dirpath, filename_prefix):
    pathname = dirpath+"/"+filename_prefix+"_*.txt"
    file_names = glob.glob(pathname)
    file_names.sort()
    for fn in file_names:
        yield read_from_file(fn)


if __name__ == "__main__":
    dirpath = "/home/huangjp/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04/programming/RL_Snake_Robot/algorithm/RL_algorithm/obs_record"
    fn_prefix = "obs_record"
    tx = []
    ty = []
    SAMPLE_NUM = 500
    id = 0
    cnt = 10
    for data in search_data_file(dirpath, fn_prefix):
        color = COLOR[id % len(COLOR)]
        if cnt == 0:
            id += 1
            cnt = 10
        cnt -= 1
        size = data.shape[0]
        idx = np.linspace(0, size-1, SAMPLE_NUM)
        idx = idx.astype(np.int)

        rob_pos = data[0]
        tar_pos = data[-1]
        k = (tar_pos[1]-rob_pos[1]) / (tar_pos[0]-rob_pos[0] + 1e-9)
        b = rob_pos[1] - k * rob_pos[0]
        opt_x = np.linspace(rob_pos[0], tar_pos[0], SAMPLE_NUM)
        opt_y = k * opt_x + b

        # plt.figure(1)
        # plt.plot(data[idx, 0], data[idx, 1])
        # plt.plot(opt_x, opt_y, color='k')

        opt_xy = np.array(list(zip(opt_x, opt_y)))  # 最优路径
        xnoise = data[idx, 0] - opt_xy[:, 0]
        ynoise = data[idx, 1] - opt_xy[:, 1]
        # xnoise_id = np.where(np.abs(xnoise) < 0.1)
        # ynoise_id = np.where(np.abs(ynoise) < 0.25)
        # plt.figure(2)
        # plt.xlim(0, SAMPLE_NUM)
        # plt.ylim(-0.3, 0.5)
        # plt.plot(xnoise, color=color)
        # plt.figure(3)
        # plt.xlim(0, SAMPLE_NUM)
        # plt.ylim(-1.5, 0.6)
        # plt.plot(ynoise, color=color)

        # print(np.std(ynoise))
        # if xnoise_id[0].shape[0] > SAMPLE_NUM/2 and np.all(np.abs(xnoise) < 0.2) and ynoise_id[0].shape[0] > SAMPLE_NUM/3 and np.all(np.abs(ynoise) < 0.5) :
        if np.std(ynoise) < 0.148 and np.std(xnoise) < 0.02:
            print('target_pos: (', opt_x[-1], ',', opt_y[-1], ')')
            plt.figure(4)
            plt.plot(xnoise, color=color)
            plt.xlim(0, SAMPLE_NUM)
            plt.ylim(-0.3, 0.5)
            plt.figure(5)
            plt.plot(ynoise, color=color)
            plt.xlim(0, SAMPLE_NUM)
            plt.ylim(-1.5, 0.6)
        # elif np.std(ynoise) < 0.148:
        #     print('target_pos: (', opt_x[-1], ',', opt_y[-1], ')')
        #     plt.figure(6)
        #     plt.plot(xnoise, color=color)
        #     plt.xlim(0, SAMPLE_NUM)
        #     plt.ylim(-0.3, 0.5)
        #     plt.figure(7)
        #     plt.plot(ynoise, color=color)
        #     plt.xlim(0, SAMPLE_NUM)
        #     plt.ylim(-1.5, 0.6)
        # tx.append(xnoise[10])
        # ty.append(ynoise[10])
        # plt.figure(4)
        # plt.plot(opt_xy[:, 0]+xnoise, opt_xy[:, 1]+ynoise)

    # plt.figure(4)
    # plt.scatter(tx, list(range(0, len(tx))))
    # plt.figure(5)
    # plt.scatter(ty, list(range(0, len(tx))))
    plt.show()