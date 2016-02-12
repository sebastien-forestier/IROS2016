import numpy as np
import matplotlib.pylab as plt

from environment import CogSci2016Environment



env_cfg = dict(
                max_params=[500, 500, 500, 500, 500, 500, 500, 500, 1, 1, 1, 1],
                gui=True)

env = CogSci2016Environment(**env_cfg)

#m = [0.02]*12
m = [ 0.48981662, -0.23193498,  0.26313222, -0.29216035,  0.28578779, -0.63969026,
  0.82305975,  0.86425687, -0.05611152, -0.34211432, -0.60495534, -0.93206953]
####1   1   2   2   3   3   4   4

print env.update(m)

m_traj = env.compute_motor_command(m)

m_traj = np.clip(m_traj, -1, 1)


fig, ax = plt.subplots(figsize=(20,3))
ax.plot(m_traj, lw=3)


# x_track = env.motor_dmp.dmp.cs.rollout()
# psi_track = env.motor_dmp.dmp.gen_psi(x_track)
# print "centers", env.motor_dmp.dmp.c
# print "std", 1. / np.sqrt(env.motor_dmp.dmp.h)
# print x_track, psi_track
# fig, ax = plt.subplots()
# ax.plot(psi_track)


plt.xlim([0, 49])
plt.ylim([-1.1, 1.1])
plt.tick_params(labelsize=18)

plt.legend(["Joint 1", "Joint 2", "Joint 3", "Gripper"],fontsize=18)
plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/dmp.pdf", format='pdf', dpi=1000, bbox_inches='tight')

plt.show()