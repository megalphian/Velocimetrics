import matplotlib.pyplot as plt
import numpy

class plotting:
    def __init__(self):
        self.figure = None
        self.lines = None
        self.ax = None
        plt.ion()
        self.xdata = []
        self.ydata = []


    def point_graph(self,data):
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

    def realtime_plotting(self, h1, g_plt, data):
        h1.set_xdata(numpy.append(h1.get_xdata(), data[1]))
        h1.set_ydata(numpy.append(h1.get_ydata(), data[0]))
        g_plt.draw()

    def launch_plt(self):
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], '-')
        self.ax.set_autoscaley_on(True)
        #self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.grid()

    def on_running(self):
        self.lines.set_xdata(self.xdata)
        self.lines.set_ydata(self.ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

if __name__ == "__main__":
    data = [[[1,2]], [[2,3]], [[4,5]]]
    plotting.point_graph(data)
