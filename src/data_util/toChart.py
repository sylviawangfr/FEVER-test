import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import config
from utils.file_loader import get_current_time_str

def draw_loss_epoch_detailed(epoches_loss, save_name):
    # ite_loss = np.array(epoches_loss).flatten() //not wok for list in different shape
    ite_loss = []
    for i in epoches_loss:
        ite_loss.extend(i)

    ep_len = [len(i) for i in epoches_loss]
    major = [0]
    formater = []
    for idx, i in enumerate(ep_len):
        major.append(int(major[len(major)-1]) + int(i))
        formater.append(f"e{idx}")
    formater.append("end")

    plt.ioff()
    fig, ax = plt.subplots(1, 1)
    ax.xaxis.set_major_locator(ticker.FixedLocator(major))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(formater))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.plot(ite_loss)
    # ax.set_xticklabels(ticks)
    plt.title("Loss changes")
    # plt.show()
    plt.savefig(config.LOG_PATH / save_name)


def draw_loss_epoches(epoch_loss, save_name):
    plt.ioff()
    fig, ax = plt.subplots(1, 1)
    # ax.xaxis.set_major_locator(ticker.FixedLocator(major))
    # ax.xaxis.set_major_formatter(ticker.FixedFormatter(formater))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.plot(epoch_loss)
    # ax.set_xticklabels(ticks)
    plt.title("Loss changes")
    # plt.show()
    plt.savefig(config.LOG_PATH / save_name)



if __name__ == "__main__":
    # epoches_loss = [[0.6, 0.5, 0.4, 0.41, 0.38], [0.36, 0.35, 0.34, 0.241, 0.238],
    #                 [0.236, 0.235, 0.1934, 0.18241, 0.17238, 0.15, 0.143, 0.143, 0.134]]
    # draw_loss_epoch_detailed(epoches_loss, 'test_loss_chart')

    eval_loss = [0.236, 0.235, 0.1934, 0.18241, 0.17238, 0.15, 0.143, 0.143, 0.134]
    draw_loss_epoches(eval_loss, f"eval_loss_0.001_{get_current_time_str()}.png")