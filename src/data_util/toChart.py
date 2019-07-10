import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker
import numpy as np
import config

def drawLoss(epoches_loss, save_name):
    ite_loss = np.array(epoches_loss).flatten()
    ep_len = [len(i) for i in epoches_loss]
    major = [0]
    for i in ep_len:
        major.append(int(major[len(major)-1]) + int(i))

    ticks = []
    for idx, i in enumerate(epoches_loss):
        e_ticks = f"epoch{idx}"
        ticks.extend([e_ticks for j in i])
    # plt.xticks(np.arange(len(ite_loss)), ticks)
    plt.ioff()
    fig, ax = plt.subplots(1, 1)
    ax.xaxis.set_major_locator(ticker.FixedLocator(major))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(['e0', 'e1', 'e2', 'end']))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.plot(ite_loss)
    # ax.set_xticklabels(ticks)
    plt.title("Loss changes")
    # plt.show()
    plt.savefig(config.LOG_PATH / save_name)



if __name__ == "__main__":
    epoches_loss = [[0.6, 0.5, 0.4, 0.41, 0.38], [0.36, 0.35, 0.34, 0.241, 0.238],
                    [0.236, 0.235, 0.1934, 0.18241, 0.17238]]
    drawLoss(epoches_loss, 'test_loss_chart')