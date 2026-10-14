"""
HELPER FUNCTION CLASS FOR MAIN MINKOWSKI DIAGRAM CLASS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML

class MinkowskiHelpers:

    def __init__(
        self, lightcone=False, hyperbolas=False, S0=True, show_data=True
    ):
        """
        Initialise with lightcone, hyperbolas (to show path of Lorentz 
        boosts), and ground reference frames as optional plot elements. The
        show_data parameter determines if plot title, axis labels, and beta 
        value are displayed.
        """

        # worldlines (or reference frames)
        self.worldlines = []

        # plot artists data
        self.axes = []
        self.gridlines = []
        self.labels = []
        
        # extent of plot units
        self.lim = 10

        # show external plot data or not
        self.show_data = show_data

        # plot formatting
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        top = 0.9 if self.show_data else 0.95
        plt.subplots_adjust(left=0.05, right=0.95, top=top, bottom=0.05)
        if self.show_data:
            self.fig.suptitle(
                "Lorentz Boosts in Minkowski Spacetime", fontsize=20
            )
        self.ax.set_xlim(-self.lim, self.lim)
        self.ax.set_ylim(-self.lim, self.lim)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        # speed relative to ground reference frame
        if self.show_data:
            self.beta_text = self.ax.text(
                -self.lim, -self.lim-0.1, s=r"$\beta =  0.000$",
                fontsize=15, va="top"
            )

        # "present" dot to hide intersecting axes 
        self.ax.scatter(0, 0, s=50, color="grey", zorder=100)

        # format static plot elements
        if lightcone:
            # add 2 lightcones (45 degree lines)
            self.ax.plot(
                [-self.lim, self.lim], [-self.lim, self.lim], 
                linestyle="--", color="dimgrey", alpha=0.5
            )
            self.ax.plot(
                [-self.lim, self.lim], [self.lim, -self.lim],
                linestyle="--", color="dimgrey", alpha=0.5
            )
        
        if hyperbolas:
            # add 3 hyperbola branches per quarter
            x = np.linspace(-self.lim, self.lim, 200)
            for d in [3, 6, 9]:
                h = np.sqrt(d**2 + x**2)
                for data in [[x,h], [x,-h], [h,x], [-h,x]]:
                    self.ax.plot(
                        *data, linestyle="--", color="dimgrey", alpha=0.5
                    )
                
        if S0:
            # add ground reference frame S0
            self.worldline("S0", 0, "grey", True, True)

    def _redraw(self):
        """Clear and redraw diagram with new axes and gridlines."""

        # clear artists
        for artist in self.axes + self.gridlines + self.labels:
            artist.remove()
        self.axes.clear()
        self.gridlines.clear()
        self.labels.clear()

        # redraw artists
        for worldline in self.worldlines:
            self.worldline(*worldline.values(), _built=True)

    def _snapshot(self):
        """Store all current beta values."""
        return [worldline["beta"] for worldline in self.worldlines]

    def _restore(self, snapshot):
        """Restore all beta values to a given snapshot."""
        for worldline, beta in zip(self.worldlines, snapshot):
            worldline["beta"] = beta

    @staticmethod
    def _transform(xp, ctp, beta):
        """Perform Loretnz transformation"""
        gamma = 1 / np.sqrt(1 - beta**2)
        x = gamma * (xp + beta * ctp)
        ct = gamma * (ctp + beta * xp)
        return x, ct

    @staticmethod
    def _velocity_transform(beta1, beta2):
        """Perform velocity transformation from beta1 to beta2."""
        return (beta1 - beta2) / (1 - beta1 * beta2)

    @staticmethod
    def _sign(beta):
        """Format sign of beta for plotting."""
        if round(beta, 3) >= 0:        
            return "+"    
        return "-"

    @staticmethod
    def _curve_out(vals):
        """
        Converts evenly-distributed values between 0 and 1 to be denser near the
        edges and sparser near the centre of the range.
        """
        return vals**2 / (2*vals**2 - 2*vals + 1)