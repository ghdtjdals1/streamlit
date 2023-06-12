import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

num = [100]
den = [1,2,3]
sys = signal.TransferFunction(num, den)

num_nonunity = [100]
den_nonunity = [1,2,3]
sys_nonunity = signal.TransferFunction(num_nonunity, den_nonunity)

w, mag, phase = signal.bode(sys)
w_nonunity, mag_nonunity, phase_nonunity = signal.bode(sys_nonunity)

plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.semilogx(w, mag, label='Unity feedback')
plt.semilogx(w_nonunity, mag_nonunity, label='Non-unity feedback')
plt.title('bode magnitude plot')
plt.xlabel('frequency [rad/s]')
plt.ylabel('magnitude [db]')
plt.grid(which='both', axis='both')
plt.legend

plt.subplot(2, 1, 2)
plt.semilogx(w, phase, label='Unity feedback')
plt.semilogx(w_nonunity, phase_nonunity, label='non-unity feedback')
plt.title('bode phase plot')
plt.xlabel('frequency [rad/s]')
plt.ylabel('phase [degrees]')
plt.grid(which='both', axis='both')
plt.legend()

plt.tight_layout()
plt.show()