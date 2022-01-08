from tqdm.notebook import tqdm
import numpy as np

def predict(
    loader,
    model,
    loss_fn,
    metric_fn,
    n_classes,
    device
):
  '''
  Return np losses, np metrics
  '''
  losses = [[] for i in range(n_classes)]
  metrics = [[] for i in range(n_classes)]

  model.eval()

  for image, gt, _ in tqdm(loader):
    image = image.to(device)
    gt = gt.to(device)
    pred = model(image)


    for i in range(image.shape[0]):
      loss = loss_fn(gt[i:i+1], pred[i:i+1])
      metric = metric_fn(gt[i:i+1], pred[i:i+1])

      for i in range(n_classes):
        losses[i].append(loss[i].item())
        metrics[i].append(metric[i].item())

    del loss, metric, pred

  return np.array(losses), np.array(metrics)