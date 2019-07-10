import matplotlib.pyplot as plt

def drawLoss(epoches_loss):
    epoches_loss = [[0.6, 0.5, 0.4, 0.41, 0.38], [0.36, 0.35, 0.34, 0.241, 0.238], [0.236, 0.235, 0.1934, 0.18241, 0.17238]]
    fig, ax = plt.subplot()
    ax.plot(epoches_loss)
    plt.show()


if __name__ == "__main__":
    drawLoss()