import numpy as np
from matplotlib.animation import PillowWriter, FuncAnimation

def save_gif(animation, savefile, **writer_kwargs):
    """
    Save animation as gif using Pillow

    Based on this example: https://matplotlib.org/stable/gallery/animation/simple_scatter
    """
    writer = PillowWriter(**writer_kwargs)
    animation.save(savefile, writer=writer)
    print(f'Saved gif as {savefile}')
    return

def make_fading_overlay_gif(fig, overlay, savefile, transit=30, pause=5, custom_alpha=None, fps=15, bitrate=1800):
    """
    Make a gif where the transparancy of the overlay image changes.

    Example:
        fig, ax = plt.subplots()
        image = np.random.rand(10,10)
        ax.pcolormesh(image)
        overlay = ax.pcolormesh(image, cmap='gray')
        make_fading_overlay_gif(fig, overlay, 'test.gif')
    """

    if custom_alpha is None:
        # Overlay goes from fully transparent to fully opaque and back again
        alpha = np.concatenate([np.zeros(pause),
                                np.linspace(0,1,transit),
                                np.ones(pause),
                                np.linspace(1,0,transit)]) 
    else:
        alpha = custom_alpha

    def animate(i):
        overlay.set_alpha(alpha[i])

    animation = FuncAnimation(fig, animate, frames=len(alpha)-1)
    save_gif(animation, savefile, fps=fps, bitrate=bitrate)
    return
