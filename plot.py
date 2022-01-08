import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history_path, monitor_class):
    history = pd.read_csv(history_path)

    history['train_mean_loss'] = history[[col for col in history if col.startswith('train') and col.endswith('loss')]].mean(axis=1)
    history['valid_mean_loss'] = history[[col for col in history if col.startswith('valid') and col.endswith('loss')]].mean(axis=1)
    history['train_mean_metric'] = history[[col for col in history if col.startswith('train') and col.endswith('metric')]].mean(axis=1)
    history['valid_mean_metric'] = history[[col for col in history if col.startswith('valid') and col.endswith('metric')]].mean(axis=1)


    fig = plt.figure(figsize=(20,30))
    plt.suptitle("History", fontsize=30)
    brightness="ee"
    high_sat="00"
    low_sat="cc"

    for i, yo in enumerate(['loss', 'metric']):
      ax1 = fig.add_subplot(2,1, i+1)
      ax2 = ax1.twinx()
      ax1.tick_params(axis='x', labelsize=20)
      ax1.tick_params(axis='y', labelsize=20)
      ax2.tick_params(axis='y', labelsize=20)

      p4 = ax1.plot(history['epoch'], history[f'valid_{monitor_class}_{yo}'],color=f"#{brightness}{high_sat}{high_sat}", label=f"valid class {monitor_class} {yo}")
      p3 = ax1.plot(history['epoch'], history[f'train_{monitor_class}_{yo}'],color=f"#{high_sat}{brightness}{high_sat}", label=f"train class {monitor_class} {yo}")
      p2 = ax1.plot(history['epoch'], history[f'valid_mean_{yo}'],color=f"#{brightness}{low_sat}{low_sat}", label=f"valid mean {yo}")
      p1 = ax1.plot(history['epoch'], history[f'train_mean_{yo}'],color=f"#{low_sat}{brightness}{low_sat}", label=f"train mean {yo}")

      p5 = ax2.plot(history['epoch'], np.log2(history['lr']),color=f"#5555ff", label="log₂ lr",)

      ps = p5+p4+p3+p2+p1

      ax1.set_xlabel("epoch", fontsize=30)
      ax1.set_ylabel(f"{yo}", fontsize=30)
      ax2.set_ylabel("log₂ lr", fontsize=30)
      labs = [l.get_label() for l in ps]
      ax1.legend(ps, labs, fontsize=20)

    plt.show()

def plot_single_data(
    # input images and mask as numpy
    image,
    gt=None,
    pred=None,
    opacity=0.2,
    figsize=(30, 10),
    monitor_class=1,
    suptitle="suptitle"
  ):
  '''
    Expect input in range [0, 1], gt and pred should be in shape HW
  '''
  plt.figure(figsize=figsize)
  plt.suptitle(suptitle)

  # Show image
  plt.subplot(1, 3, 1)
  plt.imshow(image)
  plt.title('image')

  # Calculate mask
  mask = np.zeros((image.shape[0], image.shape[1], 3))

  if gt is not None:
    mask[(gt==monitor_class)] = [255, 0, 0]

  if pred is not None:
    mask[(pred==monitor_class)] = [0, 0, 255]

  if gt is not None and pred is not None:
    mask[(gt==monitor_class) & (pred==monitor_class)] = [0, 255, 0]

  mask /= 255

  # Show image with mask
  out_img = image
  out_img[(mask[:, :, 0] != 0) | (mask[:, :, 1] != 0) | (mask[:, :, 2] != 0)] *= 1 - opacity
  out_img += mask * opacity


  plt.subplot(1, 3, 2)
  plt.imshow(out_img)
  plt.title('Image with mask')

  # Show mask
  plt.subplot(1, 3, 3)
  plt.imshow(mask)
  plt.title('mask')

  # Show figure
  plt.show()

if __name__ == "__main__":
  pass