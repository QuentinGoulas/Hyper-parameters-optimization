import os
import subprocess as process

n_workers = 4

process.Popen(['python','bohb_master.py','--n_workers',str(4)])
# process.Popen(['lxterminal','-e','python','-i','PyTorchWorker.py'])

# process.Popen(['lxterminal','-e','python','-i','PyTorchWorker.py','--worker'])
# process.Popen(['lxterminal','-e','python','-i','PyTorchWorker.py','--worker'])
for i in range(n_workers):
    process.Popen(['python','bohb_master.py','--n_workers',str(4),'--worker'])

