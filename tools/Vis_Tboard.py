import matplotlib
matplotlib.use('Agg')
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def vis_tboard(log_file, key_names):
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    all_key = ea.scalars.Keys()
    
    plt.rcParams['savefig.dpi'] = 300 #图片像素
    plt.rcParams['figure.dpi'] = 300 #分辨率

    for key in all_key:
        plt.figure()
        val = ea.scalars.Items(key)
        plt.plot([i.step for i in val], [j.value for j in val], label=key)
        plt.xlabel('step')
        plt.ylabel(f'{key}')
        plt.savefig(f'{key}.jpg')
    
if __name__ == "__main__":
    tensorboard_path = 'events.out.tfevents.1593936281.gpu-128'
    vis_tboard(tensorboard_path, None)

