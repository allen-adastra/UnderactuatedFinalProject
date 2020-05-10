import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.set_xlim(-5, 25)
ax.set_ylim(5, -25)
points = f.ginput(6)