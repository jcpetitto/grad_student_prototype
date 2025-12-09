"""
Title: Simulation of Noisy PSF Data with Perlin Noise Background and GLRT Analysis

Description:
This script simulates point spread function (PSF) data with various noise characteristics, including Perlin noise backgrounds, and performs a Generalized Likelihood Ratio Test (GLRT) for signal detection. The main components of the script are:

1. **Perlin Noise Generation**:
   - Implements functions to generate 3D Perlin noise using PyTorch.
   - Used to simulate spatially correlated background noise in images.

2. **Synthetic Data Generation**:
   - The `generate_rois_withnoise` function creates synthetic regions of interest (ROIs) containing simulated PSFs with added noise.
   - Incorporates various noise types, including Poisson noise, Perlin noise, and Gaussian noise, to mimic real-world imaging conditions.
   - Simulates time series data if `numchannels` is greater than 1.

3. **Generalized Likelihood Ratio Test (GLRT)**:
   - The `glrtfunction` performs a GLRT to distinguish between the presence and absence of a PSF in the noisy data.
   - Calculates test statistics that can be used to evaluate detection performance.

4. **Data Processing and Visualization**:
   - Functions like `generate_data_prog_meeting` and `procces_data_prog_meeting` are used to generate and process data for analysis.
   - Includes functions to plot the results of the GLRT and bias analysis, allowing for visual assessment of detection performance under different noise conditions.

Dependencies:
- PyTorch
- NumPy
- SciPy
- Matplotlib
- torchvision (for transforms)
- napari (optional, for visualization)
- Custom modules:
  - `utils.psf_fit_utils`
  - `utils.Neural_networks`

Usage:
- Ensure all dependencies are installed.
- Adjust simulation parameters as needed (e.g., number of spots, ROI size, noise characteristics).
- Run the script to generate synthetic data and perform GLRT analysis.
- Use the provided plotting functions to visualize the results.

Notes:
- The script relies on custom modules `utils.psf_fit_utils` and `utils.Neural_networks`. Ensure these modules are available in your environment.
- Some functions are annotated with `@torch.jit.script` for potential Just-In-Time (JIT) compilation, although this may require additional adjustments depending on your PyTorch version.

"""

from utils.psf_fit_utils import gauss_psf_2D, gauss_psf_2D_Bg
import torch
from utils.Neural_networks import Unet_pp_timeseries
import torchvision.transforms as T
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2
import scipy
from torch.utils.data import DataLoader
#@torch.jit.script
def show_tensor(image):
    import napari
    napari.imshow(image.detach().cpu().numpy())

def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


#@torch.jit.script
def generate_perlin_noise_3d_torch(shape: tuple[int,int,int], res: tuple[int,int,int],dev: str):
    """Generate a 3D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of three ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of three ints). Note shape must be a multiple
            of res.
        tileable: If the noise should be tileable along each axis
            (tuple of three bools). Defaults to (False, False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """

    if shape[0] % res[0] != 0 or shape[1] % res[1] != 0 or shape[2] % res[2] != 0:
        raise ValueError("shape must be a multiple of res")

    dev = dev

    delta_v1 = torch.tensor([res[0] / shape[0], res[1] / shape[1], res[2] / shape[2]]).to(dev)
    # weird thing to prevent bug:
    div_0 = shape[0] / res[0]
    div_1 = shape[1] / res[1]
    div_2 = shape[2] / res[2]
    d_v1 = torch.empty(3).to(dev)
    d_v1[0] = div_0
    d_v1[1] = div_1
    d_v1[2] = div_2
    d_v1 = torch.floor(d_v1)
    gridv1= torch.meshgrid(torch.arange(0,res[0],delta_v1[0]),torch.arange(0,res[1],delta_v1[1]),torch.arange(0,res[2],delta_v1[2]),
                          indexing='xy')
    gridv1 = torch.stack(list(gridv1), dim=0)[:,:,0:shape[0],:]
    gridv1 = torch.permute(gridv1,[2,1,3,0]).to(dev) # maybe swap 1 and 3.

    thetav1 = 2*torch.pi*torch.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    phiv1 = thetav1*1

    gradients = torch.stack((torch.sin(phiv1)*torch.cos(thetav1), torch.sin(phiv1)*torch.sin(thetav1), torch.cos(phiv1)), dim=3).to(dev)
    gradients = gradients.repeat_interleave(d_v1[0].type(torch.int) , 0).repeat_interleave(d_v1[1].type(torch.int) ,
                                                                             1).repeat_interleave(d_v1[2].type(torch.int) , 2)
    d_v1_0 = d_v1[0].type(torch.int)
    d_v1_1 = d_v1[1].type(torch.int)
    d_v1_2 = d_v1[2].type(torch.int)
    g000 = gradients[    :-d_v1_0,    :-d_v1_1,    :-d_v1_2]
    g100 = gradients[d_v1_0:     ,    :-d_v1_1,    :-d_v1_2]
    g010 = gradients[    :-d_v1_0,d_v1_1:     ,    :-d_v1_2]
    g110 = gradients[d_v1_0:     ,d_v1_1:     ,    :-d_v1_2]
    g001 = gradients[    :-d_v1_0,    :-d_v1_1,d_v1_2:     ]
    g101 = gradients[d_v1_0:     ,    :-d_v1_1,d_v1_2:     ]
    g011 = gradients[    :-d_v1_0,d_v1_1:     ,d_v1_2:     ]
    g111 = gradients[d_v1_0:     ,d_v1_1:     ,d_v1_2:     ]
    # g000 = gradients[    ::,    :-d_v1[1],    :-d_v1[2]]
    # g100 = gradients[0:     ,    :-d_v1[1],    :-d_v1[2]]
    # g010 = gradients[    ::,d_v1[1]:     ,    :-d_v1[2]]
    # g110 = gradients[0:     ,d_v1[1]:     ,    :-d_v1[2]]
    # g001 = gradients[    ::,    :-d_v1[1],d_v1[2]:     ]
    # g101 = gradients[0:     ,    :-d_v1[1],d_v1[2]:     ]
    # g011 = gradients[    ::,d_v1[1]:     ,d_v1[2]:     ]
    # g111 = gradients[0:     ,d_v1[1]:     ,d_v1[2]:     ]
    #

    # Ramps
    n000 = torch.sum(torch.stack((gridv1[:,:,:,0]  , gridv1[:,:,:,1]  , gridv1[:,:,:,2]  ), dim=3) * g000, 3)
    n100 = torch.sum(torch.stack((gridv1[:,:,:,0]-1, gridv1[:,:,:,1]  , gridv1[:,:,:,2]  ), dim=3) * g100, 3)
    n010 = torch.sum(torch.stack((gridv1[:,:,:,0]  , gridv1[:,:,:,1]-1, gridv1[:,:,:,2]  ), dim=3) * g010, 3)
    n110 = torch.sum(torch.stack((gridv1[:,:,:,0]-1, gridv1[:,:,:,1]-1, gridv1[:,:,:,2]  ), dim=3) * g110, 3)
    n001 = torch.sum(torch.stack((gridv1[:,:,:,0]  , gridv1[:,:,:,1]  , gridv1[:,:,:,2]-1), dim=3) * g001, 3)
    n101 = torch.sum(torch.stack((gridv1[:,:,:,0]-1, gridv1[:,:,:,1]  , gridv1[:,:,:,2]-1), dim=3) * g101, 3)
    n011 = torch.sum(torch.stack((gridv1[:,:,:,0]  , gridv1[:,:,:,1]-1, gridv1[:,:,:,2]-1), dim=3) * g011, 3)
    n111 = torch.sum(torch.stack((gridv1[:,:,:,0]-1, gridv1[:,:,:,1]-1, gridv1[:,:,:,2]-1), dim=3) * g111, 3)
    t=  gridv1 * gridv1 * gridv1 * (gridv1 * (gridv1 * 6 - 15) + 10)
    #t = interpolant(gridv1)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)

#@torch.jit.script
def generate_fractal_noise_3d_torch(shape:tuple[int,int,int], res:tuple[int,int,int], octaves:int =1 , persistence:float=0.5, lacunarity:int=2, dev: str='cpu'):
    """Generate a 3D tensor of fractal noise using PyTorch.

        Args:
            shape (tuple of int): The shape of the generated tensor.
            res (tuple of int): The number of periods of noise to generate along each axis.
            octaves (int, optional): The number of octaves in the noise. Defaults to 1.
            persistence (float, optional): The scaling factor between two octaves.
            lacunarity (int, optional): The frequency factor between two octaves.

        Returns:
            A tensor of fractal noise and of shape `shape` generated by combining several octaves of Perlin noise.

        Raises:
            ValueError: If shape is not a multiple of (lacunarity**(octaves-1)*res).
        """


    noise = torch.zeros(shape[0], shape[1], shape[2]).to(dev)
    frequency = 1.0
    amplitude = 1.0
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d_torch(
            shape,
            res, dev
        )
        frequency *= lacunarity
        amplitude *= persistence

    return noise


#@torch.jit.script
def sig(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return e/(1 + torch.exp(-a*(x.flatten(start_dim=1)-c) -b*(y.flatten(start_dim=1)-d)))

#@torch.jit.script
def gaussian_kernel(n:int, std, normalised:bool=False):
    '''
    Generates a n x n matrix with a centered Gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    k = (n - 1) // 2
    x = torch.linspace(-k, k, n, device=std.device)
    xv, yv = torch.meshgrid(x, x,indexing='xy')
    d = xv * xv + yv * yv
    sigma = std ** 2
    g = torch.exp(-d / (2 * sigma.unsqueeze(-1).unsqueeze(-1)))
    if normalised:
        g /= 2 * torch.pi * sigma.unsqueeze(-1).unsqueeze(-1)
    return g

def generate_rois_noTIRF(VPSF, dataloader, returnjac=False):
    ## simulate data
    psf_list = []
    jaclist = []
    for batch in tqdm(dataloader, desc="Processing batches", leave=False):
        psf_tot, jac_tot = VPSF.poissonrate(batch[:, :5])
        psf_list.append(psf_tot)
        jaclist.append(jac_tot)
    psf_tot = torch.concatenate(psf_list, dim=0)
    jac_tot = torch.concatenate(jaclist, dim=0)
    if returnjac:
        return psf_tot, jac_tot
    else:
        return psf_tot

#@torch.jit.script
def generate_rois_withnoise(numspots: int, roisize: int, factorx: tuple[float, float], factory: tuple[float, float], posx:
    tuple[float, float], posy: tuple[float, float], delta_noise: tuple[float, float], offset_noise:tuple[float, float],
                            offset_perlin:tuple[float,float], border_from_roi: float, photons: tuple[float, float],
                            bg_psf: tuple[float, float],  sigma: tuple[float, float], dev: str, number_of_psf:int,
                            numchannels:int=1, beaddim:float=0, vector=True, random=True):
    """
      Generates synthetic ROIs with noise.

      Args:
          numspots (int): The number of ROIs to generate.
          roisize (int): The size of each ROI (in pixels).
          factorx (tuple[float, float]): A tuple of two floats representing the range of values for the factor x.
          factory (tuple[float, float]): A tuple of two floats representing the range of values for the factor y.
          posx (tuple[float, float]): A tuple of two floats representing the range of values for the x position.
          posy (tuple[float, float]): A tuple of two floats representing the range of values for the y position.
          delta_noise (tuple[float, float]): A tuple of two floats representing the range of values for the delta noise.
          offset_noise (tuple[float, float]): A tuple of two floats representing the range of values for the offset noise.
          border_from_roi (float): A float representing the distance of the PSF from the edge of the ROI.
          photons (tuple[float, float]): A tuple of two floats representing the range of values for the number of photons.
          bg_psf (tuple[float, float]): A tuple of two floats representing the range of values for the background PSF.
          sigma (tuple[float, float]): A tuple of two floats representing the range of values for the sigma of the PSF.
          dev (str): A string representing the device to use (e.g. "cpu" or "cuda").

      Returns:
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three tensors:
              - A tensor of shape (numspots, roisize, roisize) containing the normalized samples.
              - A tensor of shape (numspots, roisize, roisize) containing the normalized background values.
              - A tensor of shape (numspots, roisize, roisize) containing the normalized PSF values.
      """
    # function body



    # create noise
    x = torch.linspace(1, roisize, roisize).to(dev)
    y = torch.linspace(1, roisize, roisize).to(dev)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    X = torch.repeat_interleave(X.unsqueeze(0), numspots, dim=0)
    Y = torch.repeat_interleave(Y.unsqueeze(0), numspots, dim=0)

    # Create tensors for random values
    factorx_m = torch.empty((numspots, 1), device=dev).uniform_(factorx[0], factorx[1])
    factory_m = torch.empty(numspots, 1).uniform_(factory[0], factory[1]).to(dev)
    posx_m = torch.empty(numspots, 1).uniform_(posx[0], posx[1]).to(dev)
    posy_m = torch.empty(numspots, 1).uniform_(posy[0], posy[1]).to(dev)
    delta_noise_m = torch.empty(numspots, 1).uniform_(delta_noise[0], delta_noise[1]).to(dev)
    normal_dist = torch.distributions.Normal(offset_noise[0], offset_noise[1])

    # Draw samples from the Normal distribution
    offset_noise_m = normal_dist.sample((numspots, 1)).to(dev)
    offset_noise_m = torch.clamp(offset_noise_m, min=0)

    offset_perlin_m = torch.empty(numspots, 1).uniform_(offset_perlin[0], offset_perlin[1]).to(dev)
    #photons_m = (torch.empty(numspots, 1).uniform_(photons[0], photons[1])).to(dev)

    # Define the parameters for the Log-Normal distribution
    shape = 0.3838630427153042
    loc = -65.971015291529
    scale = 279.9681508233608

    # Number of samples to draw
    dev = 'cuda'  # Example device, replace with your actual device

    # Convert the given parameters to the format required by PyTorch
    # Ensure that the parameters result in positive values
    mu_smp = torch.tensor(np.log(scale), device=dev)
    sigma_smp = torch.tensor(shape, device=dev)

    # Create a LogNormal distribution object
    lognormal_dist = torch.distributions.LogNormal(mu_smp, sigma_smp)

    # Draw samples from the LogNormal distribution
    photons_m = lognormal_dist.sample((numspots, 1))

    # Adjust the samples by the loc parameter
    photons_m = photons_m + loc

    # Ensure all values are positive
    photons_m = torch.clamp(photons_m, min=0)

    print(photons_m)

    sigma_m = torch.empty(numspots, 1).uniform_(sigma[0], sigma[1]).to(dev)
    bg_psf_m = torch.empty(numspots, 1).uniform_(bg_psf[0], bg_psf[1]).to(dev)

    # perlin noise
    shape_perlin = (numspots, roisize, roisize)
    res_perlin = (4, 1, 1)
    perlin_noise = generate_fractal_noise_3d_torch(shape_perlin, res_perlin, octaves=4, dev=dev)
    perlin_noise = perlin_noise - torch.min(torch.min(perlin_noise, dim=-1, keepdim=True).values, dim=-2,
                                            keepdim=True).values
    perlin_noise = perlin_noise / torch.max(torch.max(perlin_noise, dim=-1, keepdim=True).values, dim=-2,
                                            keepdim=True).values

    # Compute bg tensor
    bg = sig(X.to(dev), Y.to(dev), factorx_m, factory_m, posx_m, posy_m, delta_noise_m)
    bg = torch.reshape(bg, (numspots, roisize, roisize)) + (offset_perlin_m[...,None] * perlin_noise)
    bg = bg +  offset_noise_m[...,None]*torch.ones(bg.size()).to(dev)
    bg = torch.clip(bg, 0, torch.inf)

    # Create psf
    thetax = torch.empty(numspots, 1).uniform_(border_from_roi, roisize - border_from_roi).to(dev)
    thetay = torch.empty(numspots, 1).uniform_(border_from_roi, roisize - border_from_roi).to(dev)
    theta_ = torch.cat(
        (thetax,
         thetay,
         photons_m,
         bg_psf_m,
         sigma_m
         ),
        dim=1)

    ####


    if vector==True:
        VPSF = create_psfclass(beaddiam=beaddim)
        theta_permuted = theta_[:,[0,1,4,2,3]]
        theta_permuted[:,2] = 0
        theta_permuted[:,[0,1]] = (theta_permuted[:,[0,1]] -(roisize/2)) * VPSF.pixelsize
        ground_truth_set = GroundTruthDataset(theta_permuted)
        dataloader = DataLoader(ground_truth_set, batch_size=2000, shuffle=False)
        mu_ = generate_rois_noTIRF(VPSF, dataloader)
    else:
        mu_, _ = gauss_psf_2D(theta_, roisize)

    # add psf number_of_psf of the time
    #rand_array = torch.randint(2, size=(mu_.size(0),), dtype=torch.bool).to(dev)

    if number_of_psf == 10:
        __=1 # do nothing
    elif number_of_psf ==0:
        mu_[:] = mu_[:] * 0
    else:
        rand_array = torch.randint(number_of_psf, size=(mu_.size(0),)).to(dev) < 1
        mu_[rand_array] = mu_[rand_array] * 0
    # add values and create noise
    smp = torch.poisson(mu_ + bg)

    # normalize
    norm = torch.max(torch.max(torch.max(smp, dim=-1).values, dim=-1).values, dim=-1).values
    if numchannels !=1:
        rand_array = torch.randint(number_of_psf, size=(mu_.size(0),)).to(dev) < 1
        mu_temp =  mu_.unsqueeze(1).repeat(1, numchannels, 1, 1)
        rand_array = torch.randint(2, size=(mu_temp.size(0),mu_temp.size(1))).to(dev) < 1

        photons_m_time = torch.empty(numspots*numchannels, 1).uniform_(photons[0], photons[1]).to(dev)
        # Generate random values between 0 and 1
        random_values = torch.rand(photons_m_time.shape).to(dev)
        random_values2 = torch.rand(photons_m_time.shape).to(dev)

        # Scale and shift the random values to the range -2 to 2
        if random==True:
            random_values = (random_values * 4) - 2
            random_values2 = (random_values2 * 4) - 2
        else:
            random_values = (random_values * 0)
            random_values2 = (random_values2 * 0)
        # random_values = (random_values * 2) -1
        # random_values2 = (random_values2 * 2) -1
        thetax_time = torch.ones(numspots*numchannels, 1).to(dev)*thetax.repeat_interleave(numchannels)[...,None].to(dev)+random_values
        thetay_time = torch.ones(numspots*numchannels, 1).to(dev)*thetay.repeat_interleave(numchannels)[...,None].to(dev)+random_values2
        sigma_m_time = torch.empty(numspots*numchannels, 1).uniform_(sigma[0], sigma[1]).to(dev)
        bg_psf_m_time = torch.empty(numspots*numchannels, 1).uniform_(bg_psf[0], bg_psf[1]).to(dev)
        theta_time = torch.cat(
            (thetax_time,
             thetay_time,
             photons_m_time,
             bg_psf_m_time,
             sigma_m_time
             ),
            dim=1)
        mu_temp2,_ = gauss_psf_2D(theta_time, roisize)

        mu_temp3 = torch.reshape(mu_temp2,(numspots,numchannels,roisize,roisize))
        bg = bg.unsqueeze(1).repeat(1, numchannels, 1, 1)
        if random==True:
            mu_temp3[~rand_array,:,:] = mu_temp3[~rand_array,:,:] * 0
        else:
            pass
        smp = mu_temp3 + bg
        #smp = smp.unsqueeze(1).repeat(1, numchannels, 1, 1)

        smp = torch.poisson(smp)




    smp_bgonly = torch.poisson(bg)/norm
    norm_smp = smp / norm
    norm_bg_notrand = bg / norm
    norm_mu = mu_ / norm

    # Poisson noise was then applied. The signal-to-noise ratio (SNR) of the single molecules was defined as
    # SNR= (S-B)/σ, where S is the peak single molecule pixel intensity and B and σ are the average
    # and standard deviation of background pixel intensity, respectively. (i use S as Signal/sigma^2)
    # compute SNR (Evaluating single molecule detection methods for microarrays with high dynamic range for quantitative single cell analysis)
    if numchannels != 1:
        mean_bg = (smp_bgonly[:,0, 3:13, 3:13] * norm).mean(-1).mean(-1)
        std_bg = (smp_bgonly[:,0, 3:13, 3:13] * norm).std(-1).std(-1)
        SNR = (photons_m[:,0] / torch.sqrt(2 * torch.pi * (sigma_m[:, 0] ** 2))) / std_bg
        photons_normalized = photons_m / norm
        # compute non-uniformity background
        non_uniformity = (norm_bg_notrand[:,0, 3:13, 3:13] * norm).std(-1).std(
            -1)  # /(photons_m[:,0]/torch.sqrt(2*torch.pi*(sigma_m[:,0]**2)))

    else:
        mean_bg = (smp_bgonly[:,3:13,3:13]*norm).mean(-1).mean(-1)
        std_bg = (smp_bgonly[:, 3:13, 3:13]*norm).std(-1).std(-1)
        SNR = (photons_m[:,0]/torch.sqrt(2*torch.pi*(sigma_m[:,0]**2)))/std_bg
        photons_normalized = photons_m/norm
        # compute non-uniformity background
        non_uniformity = (norm_bg_notrand[:, 3:13, 3:13]*norm).std(-1).std(-1)#/(photons_m[:,0]/torch.sqrt(2*torch.pi*(sigma_m[:,0]**2)))

    target = norm_bg_notrand #-(torch.min(torch.min(norm_bg_notrand,dim=-1)[0],dim=-1)[0])[...,None,None]

    if numchannels != 1:
        return norm_smp, target[:,0:1,...], norm_mu, smp_bgonly[:,0:1,...], SNR, non_uniformity, norm,photons_m
    else:
        return norm_smp, target, norm_mu, smp_bgonly, SNR, non_uniformity, norm, photons_m

from utils.psf_fit_utils import LM_MLE_with_iter, Gaussian2D_IandBg, Gaussian2D_Bg, gauss_psf_2D_I_Bg, gauss_psf_2D_Bg, compute_crlb
from tqdm import tqdm
import math
from typing import Optional
#@torch.jit.script
def glrtfunction(smp_arr, batch_size:int, bounds, initial_arr, roisize:int,sigma:float, tol, lambda_:float=1e-5,
                 iterations:int=30, bg_constant:Optional[torch.Tensor]=None,roi_small:Optional[int] = None):
    n_iterations = smp_arr.size(0) // batch_size + int(smp_arr.size(0) % batch_size > 0)

    if roi_small is not None:
        roi_ori = smp_arr.size(-1)
        smp_arr = smp_arr[:,int(math.ceil((roi_ori-roi_small)/2)):roi_ori-int((roi_ori-roi_small)/2),int(math.ceil((roi_ori-roi_small)/2)):roi_ori-int((roi_ori-roi_small)/2)]
        if bg_constant is not None:
            bg_constant = bg_constant[:,int((math.ceil((roi_ori-roi_small)/2))):roi_ori-int((roi_ori-roi_small)/2),int(math.ceil((roi_ori-roi_small)/2)):roi_ori-int((roi_ori-roi_small)/2)]

    loglik_bg_all = torch.zeros(smp_arr.size(0), device=smp_arr.device)
    loglik_int_all = torch.zeros(smp_arr.size(0), device=smp_arr.device)
    traces_bg_all = torch.zeros((smp_arr.size(0),iterations+1,1), device=smp_arr.device)
    traces_int_all = torch.zeros((smp_arr.size(0),iterations+1,2), device=smp_arr.device)
    for batch in range(n_iterations):
        smp_ = smp_arr[batch * batch_size:min(batch * batch_size + batch_size, smp_arr.size(0)), :, :]
        initial_ = initial_arr[batch * batch_size:min(batch * batch_size + batch_size, smp_arr.size(0)), :]

        with torch.no_grad():  # when no tensor.backward() is used

            # setup model and compute Likelhood for hypothesis I and Bg
            model = Gaussian2D_IandBg(roisize, sigma)
            mle = LM_MLE_with_iter(model, lambda_=lambda_, iterations=iterations,
                                    param_range_min_max=bounds[[2, 3], :], tol=tol)
            # mle = LM_MLE(model, lambda_=1e-3, iterations=40,
            #              param_range_min_max=param_range[[2, 3], :], traces=True)

            mle = torch.jit.script(mle)  # select if single gpus

            params_, loglik_I_andbg, traces_iandbg = mle.forward(smp_, initial_[:, [2, 3]], bg_constant)
            mu_iandbg, jac_iandbg = gauss_psf_2D_I_Bg(params_, roisize, sigma, bg_constant)

            # fisher = compute_crlb(mu_iandbg,jac_iandbg, full_matrix=True)
            # fisher_check = fisher[:,0,0] - (fisher[:,1,0] * (1/fisher[:,1,1])* fisher[:,0,1])
            # non_unifority = (fisher_check)*params_[:,0]**2
            # setup model and compute Likelhood for hypothesis Bg
            model = Gaussian2D_Bg(roisize, sigma)
            bg_params = bounds[3, :]
            bg_params = bg_params[None, ...]
            mle = LM_MLE_with_iter(model, lambda_=lambda_, iterations=iterations, param_range_min_max=bg_params, tol=tol[:1])

            mle = torch.jit.script(mle)
            bg = initial_[:, 3]
            bg = bg[..., None]
            params_bg_, loglik_bgonly, traces_bgonly = mle.forward(smp_[:, :, :], bg, bg_constant)
            mu_bg,_ = gauss_psf_2D_Bg(params_bg_, roisize,sigma, bg_constant)
            loglik_bg_all[int(batch * batch_size):int(batch * batch_size + len(loglik_bgonly))] = loglik_bgonly
            loglik_int_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg))] = loglik_I_andbg
            traces_bg_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg)),:] = torch.permute(traces_bgonly,[1,0,2])
            traces_int_all[int(batch * batch_size):int(batch * batch_size + len(loglik_I_andbg)),:] = torch.permute(traces_iandbg,[1,0,2])
    ratio = 2 * (loglik_int_all - loglik_bg_all)

    ratio_np = ratio.cpu().detach().numpy()


    return ratio, loglik_int_all, loglik_bg_all, mu_iandbg,mu_bg, traces_bg_all, traces_int_all,smp_arr, bg_constant



def generate_data_prog_meeting(numspots=80000, dev='cuda', roisize=16,
                     photons=(100, 100), sigma=(1.2, 1.2), border_from_roi=8, mode='GLRT',bin_width = 200, numchannels=1):


    # General parameters

    if mode == 'GLRT':
        number_of_psf = 10
    elif mode == 'GLRTuniformbg':
        number_of_psf = 10
    else:
        number_of_psf = 2

    factorx = (-1.0, 1.0)
    factory = (-1.0, 1.0)

    if mode == 'training':
        posx = (roisize / 2 - 4, roisize / 2 + 4)
        posy = (roisize / 2 - 4, roisize / 2 + 4)
    else:
        posx = (roisize / 2 - 8, roisize / 2 + 8)
        posy = (roisize / 2 - 8, roisize / 2 + 8)

    delta_noise = (0, 40)
    offset_noise = (1, 15)
    offset_perlin = (0, 40)

    # Perform likelihood ratio test
    with torch.no_grad():
        # Replace 'generate_rois_withnoise' with actual function call
        smp_withPSF, bg_without_noise, _, target, _, non_uniformity, norm, _ = generate_rois_withnoise(numspots,
                                                                                                       roisize,
                                                                                                       factorx,
                                                                                                       factory,
                                                                                                       posx, posy,
                                                                                                       delta_noise,
                                                                                                       offset_noise,
                                                                                                       offset_perlin,
                                                                                                       border_from_roi,
                                                                                                       photons,
                                                                                                       (0, 0),
                                                                                                       sigma, dev,
                                                                                                       number_of_psf,
                                                                                                       numchannels)

        # delta_noise1 = (0, 0)
        # offset_noise1 = (1, 15)
        # offset_perlin1 = (0, 0)
        # smp_withPSF1, bg_without_noise1, _, target1, _, non_uniformity1, norm1, _ = generate_rois_withnoise(
        #     bin_width,
        #     roisize,
        #     factorx,
        #     factory, posx,
        #     posy,
        #     delta_noise1,
        #     offset_noise1,
        #     offset_perlin1,
        #     border_from_roi,
        #     photons, (0, 0),
        #     sigma, dev,
        #     number_of_psf,numchannels)

        #bg_without_noise = torch.cat((bg_without_noise, bg_without_noise1), dim=0)
        initial_estimate_bg = torch.amax(target, dim=(-1, -2))
        # target = torch.cat((target * norm, target1 * norm1), dim=0).to(dev)
        # smp_withPSF = torch.cat((smp_withPSF * norm, smp_withPSF1 * norm1), dim=0).to(dev)
        # non_uniformity = torch.cat((non_uniformity, non_uniformity1), dim=0)

    return bg_without_noise, initial_estimate_bg, smp_withPSF, target, non_uniformity, norm

def procces_data_prog_meeting(bg_without_noise, initial_estimate_bg, smp_withPSF, background_withnoise, non_uniformity,norm, numspots=80000, dev='cuda', roisize=16, modelpath='noleakyv2_singlechannel.pth',
                     photons=(100, 100), sigma=(1.2, 1.2),plot='no_cor', roi_small=4,bin_width = 200,slope=False
                              , numchannels=1):
    bg_psf = (0, 0)
    # Target is background without Poissonian noise
    photons_m = torch.empty(numspots , 1).uniform_(*photons).to(dev)
    sigma_m = torch.empty(numspots , 1).uniform_(*sigma).to(dev)
    bg_psf_m = torch.empty(numspots , 1).uniform_(*bg_psf).to(dev)

    if roi_small is not None:
        thetax = torch.empty(numspots , 1).uniform_(roi_small / 2, roi_small / 2).to(dev)
        thetay = torch.empty(numspots , 1).uniform_(roi_small / 2, roi_small / 2).to(dev)
    else:
        thetax = torch.empty(numspots , 1).uniform_(roisize / 2, roisize / 2).to(dev)
        thetay = torch.empty(numspots , 1).uniform_(roisize / 2, roisize / 2).to(dev)

    # Replace the following code with actual function calls

    theta_ = torch.cat((thetax, thetay, photons_m, bg_psf_m, sigma_m), dim=1)
    theta_[:, 3] = initial_estimate_bg[:,0]

    if slope==True:
        model = SlopedGaussianFixedSigmaPSFonlybg(16,1.2)
        initial_ = theta_[:,-2][...,None]
        initial_ = torch.concatenate((initial_, torch.zeros(initial_.size(0),2).to(dev)), dim=-1)
        bounds = torch.tensor([[-1e5, 1e5]])
        bounds = torch.concatenate((bounds, torch.tensor([[-1e5, 1e5], [-1e5, 1e5]])), dim=0)
        mle = LM_MLE_with_iter(model, lambda_=1e-5, iterations=100, param_range_min_max=bounds,
                               tol=torch.tensor([1e-3, 1e-3, 1e-2]).to(dev))

        params_, loglik_, traces = mle.forward(target.type(torch.float32),
                                               initial_.type(torch.float32),
                                               None)
        mu_slope,_ = gauss_psf_slope_onlybg(params_,target.size(-1),1.2)

        bg_estimated =mu_slope*1

    elif plot == 'no_cor':
        bg_estimated = None
        bg_estimated2 = None
    elif plot == 'GT':
        bg_estimated = bg_without_noise
        bg_estimated2 = bg_without_noise
    elif plot == 'DL':
        Unet_pp_inst = Unet_pp_timeseries(numchannels).to(dev)

        Unet_pp_inst.load_state_dict(torch.load(modelpath)['model_state_dict'])
        Unet_pp_inst.eval()
        bg_estimated = Unet_pp_inst(smp_withPSF / norm).to(dev)[:, 0, ...] * norm
        # transform = T.GaussianBlur(kernel_size=(7, 7), sigma=(1.2, 1.2))
        # blurred_img = transform(bg_estimated)
        # bg_estimated = blurred_img * 1


    return bg_estimated, non_uniformity,smp_withPSF,theta_, bg_without_noise

def normcdf(x, sigma=1.0, mu=0.0):
    import scipy
    return 0.5 * (1 + scipy.special.erf((x-mu) / (sigma*np.sqrt(2))))

def calculate_error(vals, xedges):
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    chi_vals = chi2.pdf(xcenters, df=1)
    chi_vals = torch.from_numpy(chi_vals)
    vals = torch.from_numpy(vals)
    mse = torch.sum((vals - chi_vals) ** 2) / len(vals)

    return mse.item()


def plot_lrt_results(number_of_bins, bin_width,ratio_psf,hist_bins, ratio, non_uniformity, dev='cpu',roi_small = None):
    percentage_list = []
    overlap_list = []
    xtick_list = []
    chi_noncentral_ = ratio_psf.detach().cpu().numpy()

    # plt.hist(chi_noncentral_)
    # plt.show()
    # sort the ratio tensor to match sorted std from least to greatest
    ratio = ratio[torch.argsort(non_uniformity)]
    ratio_psf = ratio_psf[torch.argsort(non_uniformity)]
    # sort standard deviations from least to greatest
    sorted_std = non_uniformity[torch.argsort(non_uniformity)]

    # list that holds tensors for min and max std of each bin
    max_min_of_bins = []

    # fill list with tensors of size 2
    for i in range(number_of_bins):
        bin = torch.zeros(2)
        max_min_of_bins.append(bin)

    # fill in max and min of each bin into the list
    index = 0
    for i in range(number_of_bins):
        current_bin = max_min_of_bins[i]
        for j in range(2):
            current_bin[j] = sorted_std[index]
            if j == 0:
                index += bin_width - 1
            else:
                index += 1

    # list that holds tensors for each LRT bin
    bin_list = []
    bin_list_psf = []

    # fill bin_list with needed tensors of zeros
    for i in range(number_of_bins):
        bin = torch.zeros(bin_width)
        bin1 = torch.zeros(bin_width)
        bin_list.append(bin)
        bin_list_psf.append(bin1)
    import copy
    # populate bin_list with LRT values
    tensor_index = 0
    for i in range(number_of_bins):
        # set current bin as one of the bins (of zeros) in bin list
        current_bin = bin_list[i]
        current_binpsf = bin_list_psf[i]
        for j in range(bin_width):
            # fill the current bin with the values from the ratio tensor
            current_bin[j] = ratio[tensor_index]
            current_binpsf[j] = ratio_psf[tensor_index]
            tensor_index += 1
        current_bin[torch.isnan(current_bin)] = 0
        current_binpsf[torch.isnan(current_binpsf)] = 0

    for i in range(number_of_bins):
        LRT_bin = bin_list[i]
        LRT_bin = np.clip(LRT_bin, 0, np.inf)
        LRT_bin_psf = bin_list_psf[i]
        max_min_bin = max_min_of_bins[i]
        min_val = torch.min(LRT_bin).item()
        max_val_psf = torch.max(LRT_bin_psf).item()

        plt.figure()

        if dev != 'cpu':
            hist_bins = np.linspace(min(min(LRT_bin), min(LRT_bin_psf)),
                                    max(max(LRT_bin), max(LRT_bin_psf)), num=100)
            hist_bg, _ = np.histogram(LRT_bin.detach().cpu().numpy(), bins=hist_bins, density=True)
            hist_psf, _ = np.histogram(LRT_bin_psf.detach().cpu().numpy(), bins=hist_bins, density=True)

            vals, xedges, _ = plt.hist(LRT_bin.detach().cpu().numpy(), bins=hist_bins, density=True, label='Only BG',
                                       alpha=0.5)
            valspsf, xedgespsf, _ = plt.hist(LRT_bin_psf.detach().cpu().numpy(), bins=hist_bins, alpha=0.5,
                                             density=True, label='Only PSF')
            plt.fill_between(hist_bins[:-1], np.minimum(hist_bg, hist_psf), color='gray', label='Overlap')
            overlap = np.trapz(np.minimum(hist_bg, hist_psf), dx=(hist_bins[1] - hist_bins[0]))
        else:
            vals, xedges, _ = plt.hist(LRT_bin, bins=hist_bins, density=True, label='Only BG')
            valspsf, xedgespsf, _ = plt.hist(LRT_bin_psf, bins=hist_bins, density=True, label='Only PSF')

        mse = calculate_error(vals, xedges)

        x = np.linspace(0.1, max_val_psf, 1000)
        plt.plot(x, chi2.pdf(x, df=1))
        pfa = 0.05
        gamma = (1 / normcdf(pfa / 2)) ** 2
        Pd = 1 - 0.5 * (scipy.special.erf((-np.sqrt(LRT_bin) + np.sqrt(gamma / 2)) / np.sqrt(2)) - scipy.special.erf(
            (-np.sqrt(LRT_bin) - np.sqrt(gamma / 2)) / np.sqrt(2)))
        Pd = Pd.detach().cpu().numpy()
        percentage = len(LRT_bin[LRT_bin > gamma]) / len(LRT_bin)
        percentage_list.append(percentage)
        overlap_list.append(overlap)
        xtick_list.append(f"{max_min_bin[0]:.2f}-{max_min_bin[1]:.2f}")


        x1, y1 = [gamma, gamma], [0, 0.15]
        if False:
            plt.plot(x1, y1, linestyle='dashed', color="red")

            plt.suptitle('Bin ' + str(i + 1), fontsize=18)
            plt.title("STD (" + str(np.round(max_min_bin[0].item(), 4)) + " -> " + str(np.round(max_min_bin[1].item(), 4))
                      + ")     MSE: " + str(np.round(mse, 7)), fontsize=10)
            lineOne, = plt.plot([2, 4, 6], label='Chi-Squared Distribution, df= 1')
            lineTwo, = plt.plot([1, 2, 3], label='5% FP Threshold')
            plt.legend(loc='upper right')
            plt.xlabel('Test statistic LRT')
            plt.ylabel('Probability density')
            plt.close('all')


    # Create the figures for Pfa and Overlap
    plt.figure()
    x1, y1 = [0, 6.3], [5, 5]
    list = np.arange(1, number_of_bins + 1)
    if roi_small is not None:
        plt.title('ROI size = ' + str(roi_small))
    plt.plot(x1, y1, linestyle='dashed', label="0.05")
    plt.ylim(0, 80)
    plt.xlabel('Standard Dev. Noise', fontsize=20)
    plt.xticks(list, xtick_list, rotation=45)
    plt.ylabel('Pfa [%]', fontsize=20)
    plt.tight_layout(pad=0.3)
    plt.scatter(np.array(list), np.array(percentage_list) * 100)
    for x, y in zip(list, percentage_list):
        plt.text(x, y * 100, str(np.round(y * 100, 2)) + "%")

    fig_pfa = plt.gcf()

    plt.figure()
    x1, y1 = [0, 6.3], [5, 5]
    plt.plot(x1, y1, linestyle='dashed', label="0.05")
    plt.ylim(0, 50)
    if roi_small is not None:
        plt.title('ROI size = ' + str(roi_small))
    plt.xlabel('Standard Dev. Noise', fontsize=20)
    plt.xticks(list, xtick_list, rotation=45)
    plt.ylabel('Overlap [%]', fontsize=20)
    plt.tight_layout(pad=0.3)
    plt.scatter(np.array(list), np.array(overlap_list) * 100)
    for x, y in zip(list, overlap_list):
        plt.text(x, y * 100, str(np.round(y * 100, 2)) + "%")

    fig_overlap = plt.gcf()
    return fig_pfa, fig_overlap, np.array(list), xtick_list, np.array(percentage_list) * 100, np.array(overlap_list) * 100

def plot_bias_results(number_of_bins, bin_width,ratio_psf,hist_bins, ratio, non_uniformity, dev='cpu',roi_small = None):
    mean_list = []
    std_list = []
    xtick_list = []
    ratio = torch.tensor(ratio).to(dev)
    # plt.hist(chi_noncentral_)
    # plt.show()
    # sort the ratio tensor to match sorted std from least to greatest
    ratio = ratio[torch.argsort(non_uniformity)]

    # sort standard deviations from least to greatest
    sorted_std = non_uniformity[torch.argsort(non_uniformity)]

    # list that holds tensors for min and max std of each bin
    max_min_of_bins = []

    # fill list with tensors of size 2
    for i in range(number_of_bins):
        bin = torch.zeros(2)
        max_min_of_bins.append(bin)

    # fill in max and min of each bin into the list
    index = 0
    for i in range(number_of_bins):
        current_bin = max_min_of_bins[i]
        for j in range(2):
            current_bin[j] = sorted_std[index]
            if j == 0:
                index += bin_width - 1
            else:
                index += 1

    # list that holds tensors for each LRT bin
    bin_list = []


    # fill bin_list with needed tensors of zeros
    for i in range(number_of_bins):
        bin = torch.zeros(bin_width)

        bin_list.append(bin)

    import copy
    # populate bin_list with LRT values
    tensor_index = 0
    for i in range(number_of_bins):
        # set current bin as one of the bins (of zeros) in bin list
        current_bin = bin_list[i]

        for j in range(bin_width):
            # fill the current bin with the values from the ratio tensor
            current_bin[j] = ratio[tensor_index]

            tensor_index += 1
        current_bin[torch.isnan(current_bin)] = 0


    for i in range(number_of_bins):
        LRT_bin = bin_list[i].detach().cpu().numpy()
        LRT_bin = np.clip(LRT_bin, -np.inf, np.inf)
        from scipy.optimize import curve_fit
        if dev != 'cpu':
            def gaussian(x, mu, sigma):
                return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

            n, bins, patches = plt.hist(LRT_bin, bins=50, density=True, alpha=0.7, edgecolor='black', label='Data')

            bin_centers = (bins[:-1] + bins[1:]) / 2
            params, params_covariance = curve_fit(gaussian, bin_centers, n,
                                                  p0=[np.mean(LRT_bin), np.std(LRT_bin)])
            mean = params[0]
            std = params[1]
            plt.plot(bin_centers, gaussian(bin_centers, mean, std), 'k', linewidth=2, label='Gaussian Fit')

            plt.close('all')
        else:
            assert(ValueError, 'wrong device')

        mean_list.append(mean)
        std_list.append(std)




    return mean_list,std_list