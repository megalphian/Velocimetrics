import matplotlib.pyplot as plt
import numpy

class plotting:
    def __init__(self):
        self.figure = None
        self.lines = None
        self.ax = None
        self.xdata = []
        self.ydata = []
        self.graph_on = 1

        if self.graph_on:
            plt.ion()

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
