from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src import utils
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator


class DataLoading:
    def __init__(self,
                 input_path,
                 output_path,
                 transform=None):

        self.input_path = input_path
        self.output_path = output_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def get_data_loader(self, batch_size, shuffle=False):
        dataset = ImageFolder(root=self.input_path, transform=self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader, dataset


class StimulusRendering:
    """
   Generate phosphene stimuli from different image representations.

   This class wraps the phosphene simulator and provides methods to:
   - generate optimized phosphene stimuli from a trained model,
   - extract contour representations,
   - derive luminance-based control images,
   - render phosphene images from placement maps,
   - sort conditions by number of activated electrodes,
   - reduce placement maps to a target number of electrodes.

   Parameters
   ----------
   args : argparse.Namespace
       Configuration object containing runtime arguments. Must include at
       least `phos_density`.
   batch_size : int
       Batch size used by the phosphene simulator.
   model : torch.nn.Module
       Trained phosphene optimization model.
   simulator_params : dict
       Parameters used to initialize the phosphene simulator.
   phosphene_coordinates : array-like
       Visual-field coordinates of the simulated phosphenes.

   Attributes
   ----------
   args : argparse.Namespace
       Runtime configuration.
   batch_size : int
       Batch size used by the simulator.
   model : torch.nn.Module
       Trained phosphene optimization model.
   simulator_params : dict
       Simulator configuration parameters.
   simulator : dynaphos.simulator.GaussianSimulator
       Simulator used to render phosphene images.
   gray_scale : torchvision.transforms.Grayscale
       Grayscale transform used for luminance-based control stimuli.
   """

    def __init__(self,
                 args,
                 batch_size,
                 model,
                 simulator_params,
                 phosphene_coordinates):

        self.args = args
        self.batch_size = batch_size
        self.model = model
        self.simulator_params = simulator_params
        self.simulator = PhospheneSimulator(self.simulator_params,
                                            phosphene_coordinates,
                                            batch_size=self.batch_size,
                                            phos_density=self.args.phos_density,
                                            rng=np.random.default_rng(self.simulator_params['run']['seed']))
        self.gray_scale = transforms.Grayscale(num_output_channels=1)

    @torch.no_grad()
    def get_optimized_stimuli(self, input_img):
        output_img, intensity, placement_map = self.model(input_img)
        num_elecs = torch.count_nonzero(intensity).item()
        return output_img, num_elecs, placement_map

    @torch.no_grad()
    def get_contours(self,input_img):
        contours = self.model.contours_extraction(input_img)
        return contours

    def get_luminance(self, input_img, background='black'):
        gray_img = self.gray_scale(input_img)
        # Rescale img and set background as black for the phosphene simulator
        gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min())
        if background == 'black':
            gray_img[gray_img == 1] = 0
        return gray_img

    def phosphenize(self, cond_placement_map, reset_thresholds=False):
        if reset_thresholds:
            placement_map = cond_placement_map.clone()
            self.simulator.reset()
            self.simulator.threshold.reinitialize(torch.zeros_like(self.simulator.threshold.get()))
        else:
            placement_map = utils.normalized_rescaling(cond_placement_map,
                                                       max_stimulation_intensity=self.simulator_params['sampling'][
                                                           'stimulus_scale'])
            placement_map = self.simulator.sample_stimulus(placement_map)
            self.simulator.reset()
        phos_im, intensities = self.simulator(placement_map)
        num_elecs = torch.count_nonzero(intensities).item()
        return phos_im, num_elecs, placement_map

    @staticmethod
    def sort_num_elecs(stimulus_dict):
        sorted_cond = sorted(stimulus_dict.keys(),
                             key=lambda c: stimulus_dict[c]["num_elecs"])
        return sorted_cond

    def filter_num_elecs(self, current_placement_map, target_num_elecs):
        topk_activations, topk_indices = torch.topk(current_placement_map, k=target_num_elecs, dim=2)
        reduced_placement_map = torch.zeros_like(current_placement_map)
        reduced_placement_map.scatter_(2, topk_indices, topk_activations)
        self.simulator.reset()
        print(torch.count_nonzero(reduced_placement_map).item(), target_num_elecs)
        return reduced_placement_map


class SavingStimuli:
    def __init__(self,
                 output_path):
        self.output_path = output_path
        self.num_phosphenes = pd.DataFrame(columns=['filename', 'num_phosphenes'])

    def get_relative_output_dir(self, relative_path, dataset):
        subfolder = Path(relative_path)
        output_img_dir = Path(self.output_path) / subfolder.relative_to(dataset.root)
        output_img_dir.mkdir(parents=True, exist_ok=True)
        return output_img_dir

    def save_num_phos(self, relative_path, phos_density, target_num_elec):
        self.num_phosphenes.loc[len(self.num_phosphenes)] = {'filename': relative_path.name,
                                                             'num_phosphenes': target_num_elec}
        self.num_phosphenes.to_csv(self.output_path / Path(f"number_phosphenes_{phos_density}.csv"),
                                   index=False)

    def save_stimuli(self, output_imgs, output_filename, output_img_dir):
        if output_imgs.ndim == 3 and output_imgs.shape[0] == 3:
            output_imgs = output_imgs.permute(1, 2, 0)
        output_imgs = (output_imgs - output_imgs.min()) / (output_imgs.max() - output_imgs.min())
        plt.imsave(f"{output_img_dir}/{output_filename}", output_imgs.numpy(), cmap='gray')