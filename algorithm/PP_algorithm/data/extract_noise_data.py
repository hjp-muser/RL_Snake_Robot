import numpy as np
import glob
import matplotlib.pyplot as plt
import os

def read_from_file(file_name):
    data_file = open(file_name, "r")
    data_array = np.loadtxt(data_file, dtype=np.float)
    data_file.close()
    return data_array


def search_data_file(dirpath, filename_prefix, start=0):
    pathname = dirpath+"/"+filename_prefix+"_*.txt"
    file_names = glob.glob(pathname)
    file_names.sort()
    for fn in file_names[start:]:
        yield read_from_file(fn)


if __name__ == "__main__":
    dirpath = "/home/huangjp/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04/programming/RL_Snake_Robot/algorithm/RL_algorithm/obs_record"
    yndatafile_name = "ynoise_data.txt"
    yndatafile = open(yndatafile_name, 'a+')
    yndatafile.seek(0)
    xndatafile_name = "xnoise_data.txt"
    xndatafile = open(xndatafile_name, 'a+')
    xndatafile.seek(0)
    data_num = len(xndatafile.readlines())
    start_data_id = data_num

    fn_prefix = "obs_record"
    SAMPLE_NUM = 500
    for data in search_data_file(dirpath, fn_prefix, start_data_id):
        size = data.shape[0]
        idx = np.linspace(0, size - 1, SAMPLE_NUM)
        idx = idx.astype(np.int)
        rob_pos = data[0]
        tar_pos = data[-1]
        k = (tar_pos[1] - rob_pos[1]) / (tar_pos[0] - rob_pos[0] + 1e-9)
        b = rob_pos[1] - k * rob_pos[0]
        opt_x = np.linspace(rob_pos[0], tar_pos[0], size)
        opt_y = k * opt_x + b
        opt_xy = np.array(list(zip(opt_x, opt_y)))  # 最优路径
        # xnoise = data[idx, 0] - opt_xy[:, 0]
        # ynoise = data[idx, 1] - opt_xy[:, 1]
        # xnoise = data[:, 0] - opt_xy[:, 0]
        ynoise = data[:, 1][:500] - opt_xy[:, 1][:500]
        plt.plot(data[:, 1][:500], c='r')
        # plt.plot(data[:, 0])
        # plt.plot(opt_xy[:, 0])
        plt.plot(opt_xy[:, 1][:500], c='g')
        plt.plot(ynoise, c='b')
        plt.xlim(0, 1400)
        plt.ylim(-1.0, 1.5)
        # plt.show()
    plt.show()
        # xnoise = np.concatenate((rob_pos, tar_pos, xnoise))
        # ynoise = np.concatenate((rob_pos, tar_pos, ynoise))
        # np.savetxt(xndatafile, xnoise[np.newaxis, :], "%.6f")
        # np.savetxt(yndatafile, ynoise[np.newaxis, :], "%.6f")
    # xndatafile.seek(0)
    # data = np.loadtxt(xndatafile_name)
    # print(data)