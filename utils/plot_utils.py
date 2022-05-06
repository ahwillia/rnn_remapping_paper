import os
import subprocess
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Line3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.linalg


class Easy3DAxes:

    def __init__(self, ax):
        self.ax = ax
        self.objects = []
        self.shadows = []
        self.xline = None
        self.yline = None
        self.zline = None

    def delete_shadows(self):
        while len(self.shadows) > 0:
            self.shadows.pop().remove()

    def delete_axlines(self):
        self.xline.remove()
        self.yline.remove()
        self.zline.remove()
        self.xline = None
        self.yline = None
        self.zline = None

    def simplify(
            self, lw=1, color="k", frac=0.5, pad=0.0, **kwargs):
        """
        Makes 3D axes less busy.
        """
        
        x_min, x_max = self.get_xlim()
        y_min, y_max = self.get_ylim()
        z_min, z_max = self.get_zlim()

        xm = 0.5 * (x_max + x_min)
        xd = 0.5 * (x_max - x_min)
        ym = 0.5 * (y_max + y_min)
        yd = 0.5 * (y_max - y_min)
        zm = 0.5 * (z_max + z_min)
        zd = 0.5 * (z_max - z_min)

        self.set_xlim(xm - xd * (1 + pad), xm + xd * (1 + pad))
        self.set_ylim(ym - yd * (1 + pad), ym + yd * (1 + pad))
        self.set_zlim(zm - zd * (1 + pad), zm + zd * (1 + pad))

        x_min, x_max = self.get_xlim()
        y_min, y_max = self.get_ylim()
        z_min, z_max = self.get_zlim()

        self.ax.axis("off")

        self.xline = self.ax.plot(
            [x_min, x_min + frac * (x_max - x_min)],
            [y_min, y_min],
            [z_min, z_min],
            lw=lw, color=color,
            **kwargs
        )
        self.yline = self.ax.plot(
            [x_min, x_min],
            [y_min, y_min + frac * (y_max - y_min)],
            [z_min, z_min],
            lw=lw, color=color,
            **kwargs
        )
        self.zline = self.ax.plot(
            [x_min, x_min],
            [y_min, y_min],
            [z_min, z_min + frac * (z_max - z_min)],
            lw=lw, color=color,
            **kwargs
        )

        self.set_xlim([x_min, x_max])
        self.set_ylim([y_min, y_max])
        self.set_zlim([z_min, z_max])

    def plot(self, *args, **kwargs):
        self.objects += self.ax.plot(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        self.objects.append(self.ax.scatter(*args, **kwargs))

    def plot_shadow(self, line, lw=3, color="k", s=5, alpha=.1):

        if isinstance(line, Line3D):
            x, y, _ = line.get_data_3d()
        elif isinstance(line, Path3DCollection):
            x, y, _ = line._offsets3d
            x, y = x.data, y.data
        else:
            raise ValueError("did not recognize line object...")

        z = np.full(x.size, self.ax.get_zlim()[0])
        if isinstance(line, Line3D):
            self.shadows += self.ax.plot(x, y, z, lw=lw, color=color, s=s, alpha=alpha)
        else:            
            self.shadows += [self.ax.scatter(x, y, z, lw=lw, color=color, s=s, alpha=alpha)]

    def plot_all_shadows(self, **kwargs):
        for line in self.objects:
            if isinstance(line, (Path3DCollection, Line3D)):
                self.plot_shadow(line, **kwargs)

    def __getattr__(self, item):
        return getattr(self.ax, item)


class Poly3DPlane:

    def __init__(self, ax, x, y, z):

        self.ax = ax
        self.triangles = [
            Poly3DCollection(verts=[
                [x[0], y[0], z[0]],
                [x[1], y[1], z[1]],
                [x[2], y[2], z[2]],
            ]),
            Poly3DCollection([
                [x[-2], y[-2], z[-2]],
                [x[-1], y[-1], z[-1]],
                [x[0], y[0], z[0]],
            ])
        ]
        ax.add_collection3d(self.triangles[0])
        ax.add_collection3d(self.triangles[1])
        self.triangles[0].set_linewidth(0.0)
        self.triangles[1].set_linewidth(0.0)

    def set_alpha(self, alpha):
        self.triangles[0].set_alpha(alpha)
        self.triangles[1].set_alpha(alpha)

    def set_color(self, color):
        self.triangles[0].set_color(color)
        self.triangles[1].set_color(color)


def set_rcParams(rcParams):
    rcParams["axes.titlesize"] = 8
    rcParams["axes.labelsize"] = 8
    rcParams["xtick.major.size"] = 2
    rcParams["ytick.major.size"] = 2
    rcParams["xtick.major.pad"] = 2
    rcParams["ytick.major.pad"] = 2
    rcParams["xtick.labelsize"] = 6
    rcParams["ytick.labelsize"] = 6
    rcParams["savefig.transparent"] = True


def square_lims(ax):

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    s = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    )

    xm = (x_max + x_min) / 2
    ym = (y_max + y_min) / 2
    zm = (z_max + z_min) / 2

    ax.set_xlim(
        xm - (s / 2),
        xm + (s / 2)
    )
    ax.set_ylim(
        ym - (s / 2),
        ym + (s / 2)
    )
    ax.set_zlim(
        zm - (s / 2),
        zm + (s / 2)
    )



def plot_shadow(ax, line, lw=3, color="k", alpha=.1):

    if isinstance(line, Line3D):
        x, y, _ = line.get_data_3d()
    elif isinstance(line, Path3DCollection):
        x, y, _ = line._offsets3d
        x, y = x.data, y.data
    else:
        raise ValueError("did not recognize line object...")

    z = np.full(x.size, ax.get_zlim()[0])
    return ax.plot(x, y, z, lw=lw, color="k", alpha=alpha)[0]


def simplify_3d_axes(
        ax, lw=1, color="k", frac=0.5,
        pad=0.5, **kwargs
    ):
    """
    Makes 3D axes less busy.
    """
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    ax.axis("off")

    xline = ax.plot(
        [x_min, x_min + frac * (x_max - x_min)],
        [y_min, y_min],
        [z_min, z_min],
        lw=lw, color=color,
        **kwargs
    )
    yline = ax.plot(
        [x_min, x_min],
        [y_min, y_min + frac * (y_max - y_min)],
        [z_min, z_min],
        lw=lw, color=color,
        **kwargs
    )
    zline = ax.plot(
        [x_min, x_min],
        [y_min, y_min],
        [z_min, z_min + frac * (z_max - z_min)],
        lw=lw, color=color,
        **kwargs
    )

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ax.view_init(azim=30, elev=30)


def plot_plane(ax, basis, offset=None, npts=2):
    """
    Plots a 2D affine set given orthonormal basis and offset.
    """

    # Default to plane through the origin.
    if offset is None:
        offset = np.zeros(3)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Check that origin is inside plot limits.
    assert np.clip(offset[0], *xlim) == offset[0]
    assert np.clip(offset[1], *ylim) == offset[1]
    assert np.clip(offset[2], *zlim) == offset[2]

    # Make sure basis is orthonormal.
    basis = scipy.linalg.orth(basis)

    directions = [
         1.0 * basis[:, 0]  + 1.0 * basis[:, 1],
         1.0 * basis[:, 0]  - 1.0 * basis[:, 1],
        -1.0 *  basis[:, 0] - 1.0 * basis[:, 1],
        -1.0 *  basis[:, 0] + 1.0 * basis[:, 1],
    ]

    def compute_scale(x, x0, lims):
        if np.abs(x) < 1e-8:
            return np.inf
        else:
            return max(
                (lims[0] - x0) / x,
                (lims[1] - x0) / x,
            )

    corners = []
    for v in directions:
        scale = min(
            compute_scale(v[0], offset[0], xlim),
            compute_scale(v[1], offset[1], ylim),
            compute_scale(v[2], offset[2], zlim),
        )
        corners.append(offset + scale * v)

    x, y, z = np.column_stack(corners)
    return Poly3DPlane(ax, x, y, z)


def simple_cmap(*colors, name='none'):
    """Create a colormap from a sequence of rgb values.
    cmap = simple_cmap((1,1,1), (1,0,0)) # white to red colormap
    cmap = simple_cmap('w', 'r')         # white to red colormap
    """

    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')

    # make sure colors are specified as rgb
    colors = [colorConverter.to_rgb(c) for c in colors]

    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    return LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})


def ring_colormap():
    return simple_cmap(
        [.5,0,.5],
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [.5,0,.5],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [.5,0,.5],
    )


def pdf_grid(nrows, ncols, width, height, files, outfile):
    """
    Uses latex to stitch together pdfs into a grid.
    """
    outdir = os.path.split(outfile)[0]
    total_height = nrows * height
    total_width = ncols * width

    preamble = (
        "\\documentclass{article}\n" +
        "\\usepackage{graphicx}\n" +
        "\\usepackage[" +
        ("paperheight={}in,".format(total_height)) +
        ("paperwidth={}in,".format(total_width)) +
        "margin=0in]{geometry}\n\n"
    )

    document = ["\\begin{document}", "\\begin{figure}"]
    for fn in files:
        document.append(
            "\\includegraphics" +
            ("[width={}in]".format(width)) +
            "{" + fn + "}"
        )
        document.append("\\hspace{-1em}")

    document += ["\\end{figure}", "\\end{document}"]
    document = "\n".join(document)
    
    with open(outfile + ".tex", 'w') as f:
        f.write(preamble + document)

    subprocess.call([
        'pdflatex',
        '-interaction=nonstopmode',
        '-output-directory=./{}'.format(outdir),
        outfile + ".tex"]
    )

