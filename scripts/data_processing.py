import matplotlib.pyplot as plt

def point_graph(data):
    """
    input data is a nested list [[[x,y]]...[[x1,y1]]]
    """
    new_x = []
    new_y = []
    for i in data:
        for j in i:
            new_x.append(j[0])
            new_y.append(j[1])
    interval = (max(new_x) - min(new_x))/len(new_x)
    print new_x, new_y, interval
    plt.plot(new_x, new_y, 'ro')
    plt.axis([min(new_x) - interval, max(new_x) + interval,
                min(new_y) - interval, max(new_y)+ interval], interval)
    plt.show()


if __name__ == "__main__":
    data = [[[1,2]], [[2,3]], [[4,5]]]
    point_graph(data)
