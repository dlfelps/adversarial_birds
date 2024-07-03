from art.attacks import EvasionAttack
import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image
from art.estimators.classification import PyTorchClassifier

def undo_normalization(normalized_image: torch.Tensor) -> Image.Image:
  if len(normalized_image.shape) == 4:
    normalized_image = torch.squeeze(normalized_image)

  transform = v2.Compose([v2.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), v2.Normalize(mean = [ -0.485, -0.456, -0.406 ], std=[1.,1.,1.]), v2.ToPILImage(), ])
  return transform(normalized_image)

def redo_normalization(original_image: Image.Image) -> torch.Tensor:
  transform = v2.Compose([
              v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
              v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
              v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  return transform(original_image)

def undo_then_redo_normalization(normalized_image: torch.Tensor) -> torch.Tensor:
  original_image = undo_normalization(normalized_image)
  new_image = redo_normalization(original_image)
  return new_image

class AdvImageGen():

  def __init__(self, model: torch.nn.Module, attack: EvasionAttack, dataset, undo_redo=True):
    self.model = self._wrapmodel(model)
    self.dataset = dataset
    self.attack = attack
    self.undo_redo = undo_redo

  def _wrapmodel(self, model):
    # Step 2a: Define the loss function and the optimizer

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Step 3: Create the ART classifier

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(-3, 3), # for normally distributed data, +-3 includes 99.7% of pixel values
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=200,
    )

    return classifier

  def __getitem__(self, idx):
    def get_pred(img):
      if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
      probs = self.model.predict(img)
      return np.argmax(probs)

    image = np.expand_dims(self.dataset[idx][0], axis=0)
    x_test_adv = self.attack.generate(x=image)
    if x_test_adv.dtype == 'float64':
      x_test_adv = x_test_adv.astype('float32')
    x_test_adv = torch.tensor(x_test_adv)

    # evaluate success of attack
    original_class = get_pred(image)
    if self.undo_redo:
      temp = undo_then_redo_normalization(x_test_adv) # this simulates saving to file and back
      new_class = get_pred(temp)
    else:
      new_class = get_pred(x_test_adv)
    if original_class != new_class:
      return undo_normalization(x_test_adv)
    else:
      return None

