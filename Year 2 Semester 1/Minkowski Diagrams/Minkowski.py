"""MAIN MINKOWSKI CLASS WITH USER FUNCTIONS"""

from MinkowskiHelpers import *

class Minkowski(MinkowskiHelpers):
        
    def worldline(
        self, name, beta, colour, grid, label, _built=False
    ):
        """
        Add a worldline (reference frame), specifying relative speed, beta, to 
        current ground frame as a fraction of c. _built ensure worldlines have 
        unique names or aren't defined multiple times.
        """
        
        if not _built:
            self.worldlines.append({
                "name":  name,   # identifier
                "beta":  beta,   # v / c
                "col" :  colour, # colour of axes/gridlines
                "grid":  grid,   # show gridlines
                "label": label   # show x and ct labels on axes
            })

        # offset factor to help plot axes/gridlines
        offset = self.lim * beta
        
        # plot ct and x axes as quivers
        for data in [
            [-offset, -self.lim, 2*offset, 2*self.lim],
            [-self.lim, -offset, 2*self.lim, 2*offset]
        ]:      
            self.axes.append(self.ax.quiver(
                *data, color=colour, angles="xy", scale_units="xy",
                scale=1, width=0.01, zorder=10
            ))

        if grid:
            # rest extent of gridline container
            L0 = 12
        
            # rest grid coordinates
            divisions = [-12, -9, -6, -3, 3, 6, 9, 12]
        
            # constant position and time gridlines
            for d0 in divisions:
                x, ct = self._transform(
                    np.array([d0, d0]), np.array([-L0, L0]), beta
                )
                for data in [[x,ct], [ct,x]]:
                    self.gridlines.append(self.ax.plot(
                        *data, color=colour, alpha=0.4*(2-abs(beta))
                    )[0])

        if label:
            # x label
            self.labels.append(self.ax.text(
                self.lim+0.1, offset, s=r"$x$", 
                color=colour, fontsize=12, va="center"
            ))

            # ct label
            self.labels.append(self.ax.text(
                offset, self.lim+0.1, s=r"$ct$", 
                color=colour, fontsize=12, ha="center"
            ))

    def boost(self, target, beta2=None):
        """
        Transform (boost) from current ground reference frame to target 
        reference frame.
        """

        # find speed of target reference frame
        if beta2 == None:
            for worldline in self.worldlines:
                if worldline["name"] == target:
                    beta2 = worldline["beta"]

        # update all worldline velocities
        for worldline in self.worldlines:
            worldline["beta"] = self._velocity_transform(
                worldline["beta"], beta2
            )

        # redraw diagram with boosted worldlines
        self._redraw()

        # update ground beta text
        if self.show_data:
            self.beta_text.set_text(
                rf"$\beta = {self._sign(beta2)}{abs(beta2):.3f}$"
            )

    def save(self, title):
        """Save current diagram as png."""
        self.fig.savefig(
            f"{title}.png", dpi=300, bbox_inches="tight"
        )

    def animate(self, boosts, frames, save=False, title="Minkowski-diagram"):
        """Animate any sequence of Lorentz boosts."""

        # frames to allocate to each boost animation
        frames_per_boost = frames // len(boosts)

        # path to animate
        trajectory = []

        # current speeds relative to ground frame
        ground_betas = []

        # current speeds of all reference frames
        base_snapshot = self._snapshot()

        # base rapidity to kepp track of ground beta
        base_eta = 0

        # animate each boost sequence
        for target in boosts:

            # find boost target worldline index
            for i, worldline in enumerate(self.worldlines):
                if worldline["name"] == target:
                    target_index = i

            # find rapidity change
            target_beta = base_snapshot[target_index]
            eta2 = np.arctanh(target_beta)

            # interpolate rapidity
            for i in self._curve_out(np.linspace(0, 1, frames_per_boost)):

                eta_step = i * eta2
                beta_step = np.tanh(eta_step)

                # build all velocity snapshots for sequence
                state = [
                    self._velocity_transform(beta, beta_step)
                    for beta in base_snapshot
                ]
                
                trajectory.append(state)

                # keep track of ground beta values
                ground_beta = np.tanh(base_eta + eta_step)
                ground_betas.append(ground_beta)

            # animation starting point for next boost sequence
            base_snapshot = trajectory[-1].copy()

            base_eta += eta2

        # functions for FuncAnimation
        def init():
            self._restore(base_snapshot)
            self._redraw()
            return self.axes + self.gridlines
            
        def update(frame):
            self._restore(trajectory[frame])
            self._redraw()
            beta = ground_betas[frame]
            if self.show_data:
                self.beta_text.set_text(
                    rf"$\beta = {self._sign(beta)}{abs(beta):.3f}$"
                )

        # animate boosts sequence
        anim = FuncAnimation(
            self.fig, update, frames=frames, init_func=init, blit=False
        )

        if save:
            anim.save(f"{title}.gif", writer=PillowWriter(fps=frames/10))

        plt.close(self.fig)

        if not save:
            return HTML(anim.to_jshtml())