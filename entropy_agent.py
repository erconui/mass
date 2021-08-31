class Agent(object):
    def __init__(self, center, width, height, period, dt, fov=30.):
        self._c = center
        self._w = width
        self._h = height
        self.pos = np.array(self._c)
        self._dt = dt
        self._t = 0
        self._period = period
        self._sc_agent = None
        self._sc_fov = None
        self.fov = fov

    def update(self):
        self._t += self._dt
        self.pos = np.array((self._w*np.cos(2*np.pi*self._t/self._period) + self._c[0],
                             self._h*np.sin(2*2*np.pi*self._t/self._period) + self._c[1]))
        if self._sc_agent is not None:
            self.update_plot()

    def init_plot(self, ax):
        self._sc_agent = ax.scatter([self.pos[0]], [self.pos[1]], s=50, marker='D', facecolor='red', label='agent')
#         self._sc_fov = ax.scatter([self.pos[0]], [self.pos[0]], s=500, marker='o', facecolor='None', edgecolor='orange', label='fov')
        self._sc_fov = plt.Circle((self.pos[0], self.pos[1]), self.fov, facecolor='None', edgecolor='orange')
        ax.add_patch(self._sc_fov)

    def update_plot(self):
        self._sc_agent.set_offsets([self.pos])
        self._sc_fov.center = self.pos[0], self.pos[1]
        
