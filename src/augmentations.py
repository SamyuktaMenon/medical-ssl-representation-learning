"""Data augmentations for self-supervised contrastive learning and supervised baselines.

Medical Histopathology Context & Clinical Rationale:
---------------------------------------------------
Hematoxylin and Eosin (H&E) stained histology images possess distinct biomedical properties:

1. Rotational and reflectional invariance:
   - Tissue biopsies mounted on glass slides have no intrinsic orientation or gravitational
     top/bottom. Both random horizontal and vertical flips (p=0.5) are strictly label-preserving
     and reflect true microscopic invariance.

2. Biochemical Stain Semantics vs Color Jitter:
   - Hematoxylin: Basic dye binding acidic polyanions (nuclear DNA/RNA), producing deep purple/blue staining.
   - Eosin: Acidic counterstain binding basic protein residues in cytoplasm and extracellular collagen,
     producing bright pink/red staining.
   - Clinical pathologists distinguish benign from malignant tissue based on the nucleo-cytoplasmic ratio
     (nuclear crowding and hyperchromasia vs cytoplasmic area).
   - In standard natural image vision, large hue shifts (e.g. +/- 0.5) are common. In histopathology,
     a large hue shift would artificially turn eosin blue and hematoxylin yellow, corrupting diagnostic
     distinctions.
   - Therefore, our color jitter introduces moderate brightness, contrast, and saturation variations
     (+/- 0.2 to 0.4) to model slide preparation, light source differences, and scanner exposure, while
     strictly constraining hue shift (+/- 0.05 to 0.08) to preserve H&E chromatic separation.

3. Role of Grayscale Augmentation:
   - In natural image contrastive learning (SimCLR on ImageNet), grayscale (p=0.2) was introduced to prevent
     models from relying on color histograms to match views.
   - In histopathology, color is not an arbitrary background shortcut—it directly reflects tissue biochemistry.
   - Blindly applying grayscale removes the fundamental differential contrast between nucleic acids and cytoplasm.
   - In our primary pipeline, grayscale is disabled (random_gray_p = 0.0) to maintain diagnostic integrity,
     and is evaluated as an explicit ablation study.

4. Random Resized Crop:
   - Scale (0.6, 1.0): Simulates varying optical magnification (e.g. 15x vs 20x objective lens field of view).

5. Gaussian Blur:
   - Simulates minor optical defocus across varying slide focal planes.
"""

from typing import Tuple, List, Union
from PIL import Image
import torch
import torchvision.transforms as T


class SimCLRAugmentation:
    """Produces two independently augmented stochastic views of the same image.

    Given an image x, applies transforms t1 ~ T and t2 ~ T to yield (x_i, x_j).
    """

    def __init__(
        self,
        image_size: int = 28,
        crop_scale: Tuple[float, float] = (0.6, 1.0),
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        color_jitter_p: float = 0.8,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.08,
        random_gray_p: float = 0.0,  # Default 0.0 for medical validity; evaluated as ablation
        gaussian_blur_p: float = 0.5,
        kernel_size: int = 3,
        mean: List[float] = (0.7405, 0.5330, 0.7058),
        std: List[float] = (0.1235, 0.1768, 0.1244),
    ):
        transforms_list = [
            T.RandomResizedCrop(
                size=(image_size, image_size),
                scale=crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=horizontal_flip_p),
            T.RandomVerticalFlip(p=vertical_flip_p),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue,
                    )
                ],
                p=color_jitter_p,
            ),
        ]

        if random_gray_p > 0.0:
            transforms_list.append(T.RandomGrayscale(p=random_gray_p))

        if gaussian_blur_p > 0:
            transforms_list.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))],
                    p=gaussian_blur_p,
                )
            )

        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        self.transform = T.Compose(transforms_list)

    def __call__(self, x: Union[Image.Image, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


def get_supervised_transforms(
    image_size: int = 28,
    is_train: bool = True,
    mean: List[float] = (0.7405, 0.5330, 0.7058),
    std: List[float] = (0.1235, 0.1768, 0.1244),
) -> T.Compose:
    """Returns data transforms for supervised training or deterministic evaluation."""
    if is_train:
        return T.Compose([
            T.RandomCrop(size=(image_size, image_size), padding=2, padding_mode="reflect"),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)],
                p=0.5,
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
