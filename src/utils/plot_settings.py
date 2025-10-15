import matplotlib

marker_legend_params = {'facecolor':'Navy','edgecolor':'k', 'framealpha':0.05 , 'frameon':True}
histogram_facecolor = (0,1,0,0.25)
# fig_width
fig_width = 5.7*0.75
fig_height = fig_width/1.618


COLORS = ["Navy", "Crimson", "mediumseagreen", "darkorchid"]

def to_grey_color(named_color):
	rgb = matplotlib.colors.to_rgb(named_color)
	Grayscale = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
	rgb = (Grayscale,Grayscale,Grayscale)
	return rgb

def to_grey(COLORS):
	colors = COLORS.copy()
	for i in range(len(colors)):
		colors[i] = to_grey_color(colors[i])
	return colors

GREYCOLORS = to_grey(COLORS)

linewidths=0.75
tickwidths=0.75*linewidths
tickminor=3
tickmajor=2*tickminor
params = {
			'font.size':11,
			'axes.titlesize': 11,
			'axes.labelsize': 11, 
			'xtick.labelsize': 10,
			'ytick.labelsize': 10,
			'legend.fontsize': 10,
			
			'font.family': 'sans-serif',
			'mathtext.fontset': 'dejavuserif',
			'mathtext.rm': 'serif',
			'mathtext.bf': 'serif:bold',
			'mathtext.it': 'serif:italic',
			'font.weight': 'medium',

			'xtick.minor.visible': True,
			# 'ytick.minor.visible': True,
			'xtick.direction': 'in',
			'ytick.direction': 'in',
			'ytick.major.width': tickwidths,
			'xtick.major.width': tickwidths,
			'ytick.minor.width': tickwidths,
			'xtick.minor.width': tickwidths,
			'ytick.minor.size': tickminor,
			'xtick.minor.size': tickminor,
			'ytick.major.size': tickmajor,
			'xtick.major.size': tickmajor,
			# 'ytick.right': True,
			# 'xtick.top': True,
# 			'text.usetex': True,
# 			"text.latex.preamble": "\n".join([
# 		        r"\usepackage[utf8]{inputenc}",
# 		        r"\usepackage[T1]{fontenc}",
# 		        r"\usepackage[detect-all, per-mode=symbol]{siunitx}",
# 		        r"\usepackage{amsmath, amssymb}",
# 		        ]),


			'figure.figsize': [fig_width,fig_height],
			'figure.constrained_layout.use': False,

			'axes.linewidth': linewidths,
			'lines.linewidth': linewidths,
			'lines.markeredgewidth': linewidths,
			'lines.markersize': 7,
			'legend.handlelength': 1.7,
			'legend.frameon': False,
}
matplotlib.rcParams.update(params)
