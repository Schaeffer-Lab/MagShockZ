import matplotlib.pyplot as plt
import numpy as np


# Not exactly sure if there is any more work required from me here.
# t_3305ns = plt.imread('/mnt/cellar/shared/data/ZMachine_MagShockZ1_07-2024/LIONZ/z4013_SIMX/shot/LIONZ_3305ns.tif')
# t_3310ns = plt.imread('/mnt/cellar/shared/data/ZMachine_MagShockZ1_07-2024/LIONZ/z4013_SIMX/shot/LIONZ_3310ns.tif')
# t_3315ns = plt.imread('/mnt/cellar/shared/data/ZMachine_MagShockZ1_07-2024/LIONZ/z4013_SIMX/shot/LIONZ_3315ns.tif')
# t_3320ns = plt.imread('/mnt/cellar/shared/data/ZMachine_MagShockZ1_07-2024/LIONZ/z4013_SIMX/shot/LIONZ_3320ns.tif')
t_raw = plt.imread("/mnt/cellar/shared/data/Z_MagShockZ-A1340A_07-2024/LIONZ/z4013_SIMX/shot/z4013-simx-shot_rotated2.77.tif")
plt.imshow(t_3305ns, cmap='gray') 
plt.axis('off')
plt.savefig('LIONZ_3305ns.png', bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(t_3310ns, cmap='gray')
plt.axis('off')
plt.savefig('LIONZ_3310ns.png', bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(t_3315ns, cmap='gray')
plt.axis('off')
plt.savefig('LIONZ_3315ns.png', bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(t_3320ns, cmap='gray')
plt.axis('off')
plt.savefig('LIONZ_3320ns.png', bbox_inches='tight', pad_inches=0)
plt.show()
