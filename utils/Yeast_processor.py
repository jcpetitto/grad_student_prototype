"""
Yeast Image Processing Module

This module provides tools for processing yeast imaging data, including calibration,
registration, NPC (Nuclear Pore Complex) detection, drift computation, and spline
refinement for analyzing yeast cell structures. It leverages various libraries such
as Matplotlib, TrackPy, Scipy, Skimage, Torch, and others for image processing and
neural network-based segmentation.


Classes:
    Yeast_processor:
        A class for processing yeast imaging data with methods for calibration,
        registration, NPC detection, drift computation, and spline refinement.
"""


import matplotlib as mpl
import trackpy as tp
import glob
from utils.psf_fit_utils import Gaussian2DFixedSigmaPSF, LM_MLE_with_iter, Gaussian2DFixedSigmaPSFFixedPos
import copy
from datetime import datetime
from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.interpolate import UnivariateSpline
from skimage.draw import polygon2mask
from skimage import measure
from scipy import ndimage
from utils.Neural_networks import Segment_NE
import torch.nn.functional as F
import skimage
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.feature import  peak_local_max
from skimage.morphology import skeletonize
import tqdm as tqdm
import pandas as pd
import imageio
import imreg_dft as ird
import math
import pickle
from utils.Neural_networks import glrtfunction
from utils.Neural_networks import Unet_pp_timeseries as Unet_pp
from utils.psf_fit_utils import compute_crlb
from scipy.spatial.distance import cdist
import napari
import os
from scipy.ndimage import zoom
import torch
from utils.ne_fit_utils import LM_MLE_forspline_new, npcfit_class
from scipy import  stats
import scipy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def fast_harmonic(n):
    """Returns an approximate value of n-th harmonic number.
       http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + np.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)
def show_napari(img):
    viewer = napari.imshow(img)

def show_tensor(img):
    viewer = napari.imshow(img.detach().cpu().numpy())

class Yeast_processor:

    """
    Yeast processing.
    """
    def __init__(self, cfg):
        self.path = cfg['path']
        self.fn_reg_npc1 = cfg['fn_reg_npc1']
        self.fn_reg_rnp1 = cfg['fn_reg_rnp1']
        self.fn_reg_npc2 = cfg['fn_reg_npc2']
        self.fn_reg_rnp2 = cfg['fn_reg_rnp2']
        self.fn_track_rnp = cfg['fn_track_rnp']
        self.fn_track_npc = cfg['fn_track_npc']
        self.fn_dark_image = cfg['offset']
        self.fn_bright_image = cfg['gain']
        dir, fn = os.path.split(self.path)
        self.resultsdir = dir + cfg['resultdir'] + os.path.splitext(fn)[0] + "/"
        if os.path.isfile(self.resultsdir):
            os.remove(self.resultsdir)
        if not os.path.isdir(self.resultsdir):
            os.makedirs(self.resultsdir, exist_ok=True)
        self.resultprefix = self.resultsdir
        self.roisize = cfg['roisize']
        self.sigma = float(cfg['sigma'])
        self.frames = cfg['frames']
        self.frames_npcfit = cfg['frames_npcfit']
        self.NE_model = cfg['model_NE']
        self.bg_model = cfg['model_bg']
        self.driftbins = cfg['drift_bins']
        self.dev = 'cuda'
        self.pixelsize = cfg['pixelsize']
    def check_files(self, dircheck = False):

        err = False
        if not os.path.exists(self.path + self.fn_reg_npc1):
            err = True
        if not os.path.exists(self.path + self.fn_reg_rnp1):
            err = True
        if not os.path.exists(self.path + self.fn_reg_npc2):
            err = True
        if not os.path.exists(self.path + self.fn_reg_rnp2):
            err = True
        if not os.path.exists(self.path + self.fn_track_rnp):
            err = True
        if not os.path.exists(self.path + self.fn_track_npc):
            err = True

        if dircheck==True:
            if len(os.listdir(self.resultsdir)) != 0:
                err = True

        return err

    def calibrate(self, savefig=True):
        if type(self.fn_bright_image) == str and type(self.fn_dark_image) == str:

            dark_image = tifffile.imread(self.fn_dark_image)
            track_rnp = tifffile.imread(self.fn_bright_image)
            varbg = np.var(dark_image)
            dark_mean = np.mean(dark_image, 0)
            bg_corrected = track_rnp[:, :, :] - np.mean(dark_mean)

            # Normalization of data (bleaching correction)
            firstplane_avg = np.mean(bg_corrected[0, :, :])
            mean_array = np.mean(bg_corrected, (1, 2))

            dim_array = np.ones((1, bg_corrected.ndim), int).ravel()
            dim_array[0] = -1
            b_reshaped = (firstplane_avg / mean_array).reshape(dim_array)

            bg_corrected = bg_corrected * b_reshaped

            variance = np.var(bg_corrected, 0)
            mean = np.mean(bg_corrected, 0)

            meanvarplot = scipy.stats.binned_statistic(mean.flatten(), variance.flatten(), bins=100, statistic='mean')
            weights, _ = np.histogram(mean.flatten(), bins=meanvarplot.bin_edges)
            weights = weights / np.sum(weights)

            center = (meanvarplot.bin_edges[1:] + meanvarplot.bin_edges[:-1]) / 2
            # remove nans and fit
            nanvalues = np.isnan(meanvarplot.statistic)
            fit = np.polyfit(center[~nanvalues], meanvarplot.statistic[~nanvalues], 1, w=weights[~nanvalues])
            if savefig:
                fig, ax = plt.subplots()
                ax.scatter(center, meanvarplot.statistic, marker='x', label='mean variance in bin')
                ax.plot(center, np.polyval(fit, center), color='red',
                        label='fit = ' + str(np.around(fit[0], 3)) + 'x + ' + str(np.around(fit[1], 3)))
                ax.set_xlabel('mean ADU')
                ax.set_ylabel('Variance ADU')
                ax2 = ax.twinx()
                ax2.plot(center, weights, label='weights')
                ax2.set_ylabel("Weights (prop to pixels)", color="blue")
                ax2.set_yscale('log')
                fig.legend()
                fig.savefig(self.resultprefix + 'calibration.png',
                            format='png',
                            dpi=100,
                            bbox_inches='tight')
                plt.close('all')
            self.gain = 1 / fit[0]
            self.offset = np.mean(dark_mean)
            print('gain is set at ' + str(self.gain) + ' and offset is set at ' + str(self.offset))
            return 1 / fit[0], np.mean(dark_mean), np.sqrt(varbg) * self.gain
        elif (type(self.fn_bright_image) == float or type(self.fn_bright_image) == int) and (
                type(self.fn_dark_image) == float or type(self.fn_dark_image) == int):
            self.gain = self.fn_bright_image
            self.offset = self.fn_dark_image
            print('gain is set at ' +str(self.gain) + ' and offset is set at '+ str(self.offset))
            return self.gain, self.offset, 0
        else:
            raise ValueError('Error in calibration, set gain and offset both as filenames or numbers')

    def compute_registration(self, plotfig=True, regmode =1,print_update=True, scale = None,angle=None):

        if regmode == 1:
            bf_green = tifffile.imread(self.path + self.fn_reg_rnp1)
            bf_red = tifffile.imread(self.path+self.fn_reg_npc1)
        if regmode==2:
            bf_green = tifffile.imread(self.path + self.fn_reg_rnp2)
            bf_red = tifffile.imread(self.path+self.fn_reg_npc2)

        mean_green = (np.mean(bf_green, axis=0)[None, ...] / np.max(np.mean(bf_green, axis=0)) )[0,:,:]
        mean_red = (np.mean(bf_red, axis=0)[None, ...] / np.max(np.mean(bf_red, axis=0)))[0,:,:]
        mean_green = skimage.filters.difference_of_gaussians(mean_green, 0.5, 1.5)
        mean_red = skimage.filters.difference_of_gaussians(mean_red, 0.5, 1.5)
        #mean_green = skimage.filters.gaussian(mean_green,sigma=1.5)
        #mean_red= skimage.filters.gaussian(mean_red,sigma=1.5)

        ii = 1
        translation_old = np.array([10, 10])
        mean_red_t_trans= mean_red * 1
        #scale = 1
        #angle = 0
        trans_tot = np.array([0.0,0.0])

        #while ii <10:
        upsample_angle = 1
        upsampled_im0 = zoom(mean_green, upsample_angle, order=1)  # Bilinear interpolation
        upsampled_im1 = zoom(mean_red_t_trans, upsample_angle, order=1)
        if scale is None and angle is None:
            scale,angle = ird.imreg._get_ang_scale([upsampled_im0, upsampled_im1],None)

            angle = angle*upsample_angle

        mean_red_angle= ird.transform_img(mean_red, angle=angle, scale=scale)
        translation_new, _, _ = skimage.registration.phase_cross_correlation(mean_green, mean_red_angle,
                                                                      upsample_factor=1000)
        trans_tot += translation_new
        mean_red_t_trans = ird.transform_img(mean_red, tvec=translation_new)
        # print(translation_new)
        # print(translation_old)
        # print(max(abs(translation_new - translation_old)/abs(trans_tot)))
        #if max(abs(translation_new - translation_old)/abs(trans_tot)) <1e-5:
            #break
        translation_old = translation_new*1

        ii+=1


        mean_red_t = ird.transform_img(mean_red, scale=scale, angle=angle,
                               tvec=translation_new)

        if plotfig:
            # Adjust the colormap and alpha levels
            cmap_green = 'Blues'
            cmap_red = 'Reds'
            alpha = 1.0  # Increase alpha to make the image more opaque

            # Plot and save the first image
            fig, ax = plt.subplots()

            # Display the 'mean_green' image with 'cmap_green' colormap
            ax.imshow(mean_green, cmap='gray')
            ax.axis('off')
            plt.savefig(self.resultprefix + 'green_original.svg',
                        format='svg', )

            plt.close()
            # Display the 'mean_red_t' image with 'cmap_red' colormap and 0.5 alpha
            # Plot and save the first image
            fig, ax = plt.subplots()

            # Display the 'mean_green' image with 'cmap_green' colormap
            ax.axis('off')
            ax.imshow(mean_red, cmap='gray')
            plt.savefig(self.resultprefix + 'red_original.svg',
                        format='svg', )

            plt.close()

            # Plot and save the second image
            fig, ax = plt.subplots()
            alpha = 0.5  # Adjust this value as needed

            # Create a new image by blending the two images with the same transparency
            overlay_image = alpha * mean_green + (1 - alpha) * mean_red

            # Plot the overlay image
            ax.imshow(overlay_image, cmap='gray', alpha=1)

            # Remove x and y axes
            ax.axis('off')

            plt.savefig(self.resultprefix + 'registration_original.svg',
                        format='svg',)
            plt.close()
            plt.close('all')
            # Plot and save the second image
            fig, ax = plt.subplots()
            alpha = 0.5  # Adjust this value as needed

            # Create a new image by blending the two images with the same transparency
            overlay_image = alpha * mean_green + (1 - alpha) * mean_red_t

            # Plot the overlay image
            ax.imshow(overlay_image, cmap='gray', alpha=1)

            # Remove x and y axes
            ax.axis('off')

            plt.savefig(self.resultprefix + 'registration_finished.svg',
                        format='svg',)

            plt.close('all')


        result = {}
        result['angle'] = angle
        result['tvec'] = translation_new
        result['scale'] = scale

        self.registration = result
        if print_update:
            print('translation = '+str(result['tvec']) +'\n')
            print('rotation = ' + str(result['angle']) + '\n')
            print('scale = ' + str(result['scale']) + '\n')
        return scale,angle, mean_red



    def detect_npc(self,save_fig=False, init_spline_sampling = 1000,count_good_label=75,
                   gap_closing_distance=100,threshold=0.5,oldmethod=False, usegreen=False):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def nearest_neighbor(coordinates, distances):
            num_coordinates = len(coordinates)
            visited = [False] * num_coordinates
            order = [0]  # Starting from the first coordinate
            visited[0] = True  # Mark the starting point as visited

            for _ in range(num_coordinates - 1):
                current_point = order[-1]
                min_distance = float('inf')
                next_point = None

                for i in range(num_coordinates):
                    if not visited[i] and distances[current_point, i] < min_distance:
                        min_distance = distances[current_point, i]
                        next_point = i

                visited[next_point] = True
                order.append(next_point)

            return order

        # define device
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if usegreen:
            image = tifffile.imread(self.path + self.fn_track_rnp)[self.frames_npcfit[0]:self.frames_npcfit[1]]
        else:
            image = tifffile.imread(self.path+self.fn_track_npc)[self.frames_npcfit[0]:self.frames_npcfit[1]]
        npc_mean = np.mean(image,axis=0)
        npc_mean_plot = np.mean(image[0:250,...], axis=0)
        # define model parameters


        architecture = "FPN"
        encoder = "resnet34"

        # define padding value
        padsize = int((256 - np.shape(npc_mean)[0])/2)

        if padsize < 0:
            assert ValueError('wrong image size, needs to be smaller than 256')
        model = Segment_NE(architecture, encoder, in_channels=3, out_classes=1)
        with torch.no_grad():
            state_dict = torch.load(self.NE_model)
            model.load_state_dict(state_dict)
            (model.eval()).to(dev)
            npc_mean = npc_mean/np.max(npc_mean)
            npc_mean_ = torch.tensor(npc_mean).to(dev)[None,None,...]

            npc_mean_ = F.pad(npc_mean_, (padsize, padsize, padsize, padsize), mode='reflect').type(torch.float32)
            logits = model(npc_mean_)
            logits = sigmoid(logits.detach().cpu().numpy()[:,0,:,:])
            logits[logits <threshold] = 0
            logits[logits >= threshold] = 1

            if padsize !=0:
                logits = logits[0,padsize:-padsize,padsize:-padsize]
            else:
                logits = logits[0,:,:]
            if save_fig:
                # plt.figure(dpi=400)
                # plt.imshow(np.concatenate((logits, npc_mean)), cmap='gray')
                # plt.axis('off')  # Turn off the axis
                # plt.tight_layout()
                # plt.savefig(self.resultprefix + 'segmentation.svg', format='svg', dpi=400, bbox_inches='tight')
                plt.close('all')

                plt.figure(dpi=400)
                plt.imshow(npc_mean_plot, cmap='gray')
                plt.axis('off')  # Turn off the axis
                plt.tight_layout()
                plt.savefig(self.resultprefix + 'npc_mean.svg', format='svg', dpi=400, bbox_inches='tight')
                plt.close('all')
                plt.figure(dpi=400)
                plt.imshow(logits, cmap='gray')
                plt.axis('off')  # Turn off the axis
                plt.tight_layout()
                plt.savefig(self.resultprefix + 'logits.svg', format='svg', dpi=400, bbox_inches='tight')
                plt.close('all')
            cleared = clear_border(logits)
        label_image = label(cleared)

        max_labels = np.max(label_image)
        label_set = np.arange(max_labels) + 1
        # remove poor labels
        for i in range(max_labels):
            lv = i + 1

            count = np.sum(label_image == lv)

            # remove labels, which has less than 150 points
            if count < count_good_label:
                label_image[label_image == lv] = 0
                label_set = label_set[label_set != lv]

        spline_points = []
        spline_deriv = []
        spline_points_good = []
        spline_deriv_good = []
        groups = []

        for i in range(len(label_set)):
            lv = label_set[i]
            image = label_image == lv
            image = skimage.morphology.binary_dilation(image, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
            prop = regionprops(image.astype(int))[0]

            npc_label = npc_mean * 1
            npc_label[~image] = 0
            if oldmethod==False:
                skeleton = skeletonize(npc_label, method='lee')
                peaks = np.column_stack(np.where(skeleton))
            else:
                peaks = peak_local_max(npc_label, 1, exclude_border=False, p_norm=2)
            # Step 1 & 2: Create a mask of the area enclosed by peaks (assuming peaks form a simple polygon)
            image_shape = npc_label.shape
            enclosed_area_mask = polygon2mask(image_shape, peaks)

            # Step 3: Count zero values within the enclosed area on the original mask
            enclosed_area_values = npc_label[enclosed_area_mask]
            zero_count = np.sum(enclosed_area_values == 0)

            # Check if there are more than 5 zero values
            if zero_count > 2:
                print("More than 5 zero values found within the enclosed area.")
            else:
                continue
            sortarr = np.argsort(peaks[:, 0])
            if len(sortarr) < 4:
                continue
            peaks = peaks[sortarr, :]

            hull = scipy.spatial.ConvexHull(peaks)
            if len(peaks[hull.vertices, 1]) < 4:
                continue
            tck, u = scipy.interpolate.splprep([peaks[hull.vertices, 1], peaks[hull.vertices, 0]], s=0, per=True,
                                               quiet=3)

            # evaluate the spline fits for 1000 evenly spaced distance values
            xi, yi = scipy.interpolate.splev(np.linspace(0, 1, init_spline_sampling), tck)
            dxi, dyi = scipy.interpolate.splev(np.linspace(0, 1, init_spline_sampling), tck, der=1)

            rounded_coords = (np.round(np.array([xi, yi]))).astype(int)
            # spline = scipy.interpolate.UnivariateSpline(peaks[:,1],peaks[:,0] ,k=3, s=30)
            good_values = image[np.clip(rounded_coords[1],0,255).astype(int), np.clip(rounded_coords[0],0,255).astype(int)]
            if sum(good_values) != init_spline_sampling:

                unsortedx = peaks[hull.vertices, 1]
                unsortedy = peaks[hull.vertices, 0]

                # Combine x and y arrays into (x, y) pairs
                coordinates = np.column_stack((unsortedx, unsortedy))

                # Calculate pairwise Euclidean distances
                distances = cdist(coordinates, coordinates, 'euclidean')

                # Initialize variables to store the best order and minimum distance
                best_order = None
                min_distance = float('inf')

                best_order= nearest_neighbor(coordinates, distances)

                # Find the optimal order using brute force (permutations)
                # for order in permutations(range(len(coordinates))):
                #     total_distance = 0
                #     for i in range(1, len(order)):
                #         total_distance += distances[order[i - 1], order[i]]
                #     if total_distance < min_distance:
                #         min_distance = total_distance
                #         best_order = order

                # Reorder the x and y arrays based on the best order
                peakx_sorted = unsortedx[np.array(best_order)]
                peaky_sorted = unsortedy[np.array(best_order)]

                tck, u = scipy.interpolate.splprep([peakx_sorted, peaky_sorted], s=0, per=False,
                                                   quiet=3)
                # evaluate the spline fits for 1000 evenly spaced distance values
                xi, yi = scipy.interpolate.splev(np.linspace(0, 1, sum(good_values)), tck)
                dxi, dyi = scipy.interpolate.splev(np.linspace(0, 1, sum(good_values)), tck, der=1)

                rounded_coords = (np.round(np.array([xi, yi]))).astype(int)
                # spline = scipy.interpolate.UnivariateSpline(peaks[:,1],peaks[:,0] ,k=3, s=30)
                good_values = image[
                    np.clip(rounded_coords[1], 0, 255).astype(int), np.clip(rounded_coords[0], 0, 255).astype(int)]


            list_groups = self.islandinfo(good_values, trigger_val=True)[0]
            # only keep longest section
            interval_durations = [end - start for start, end in list_groups]
            longest_interval_index = interval_durations.index(max(interval_durations))
            list_groups = [list_groups[longest_interval_index]]

            new_list = list_groups
            new_list = []
            k = 0

            while k <= len(list_groups) - 1:
                if k == len(list_groups) - 1:
                    cur = list_groups[k]
                    new_list.append(cur)
                    k = k + 1

                else:

                    next = list_groups[k + 1]
                    cur = list_groups[k]
                    if next[0] - cur[1] > gap_closing_distance:
                        new_list.append(cur)
                        # new_list.append(next)
                        k = k + 1
                    else:
                        q = 1

                        while next[0] - list_groups[k + q - 1][1] <= gap_closing_distance:

                            next = list_groups[k + q]
                            q = q + 1
                            if q + k == len(list_groups):
                                q = q + 1
                                break

                        new_list.append((cur[0], list_groups[k + q - 2][1]))
                        k = k + q

            if new_list[-1][1] >= init_spline_sampling - gap_closing_distance and new_list[0][0] == 0 and len(
                    new_list) > 1:
                roll_value = new_list[-1][1] - new_list[-1][0] + 1
                xi, yi = scipy.interpolate.splev(np.linspace(0, 1, init_spline_sampling), tck)
                xi = np.roll(xi, roll_value)
                yi = np.roll(yi, roll_value)

                dxi, dyi = scipy.interpolate.splev(np.linspace(0, 1, init_spline_sampling), tck, der=1)
                dxi = np.roll(dxi, roll_value)
                dyi = np.roll(dyi, roll_value)

                del new_list[-1]

                new_list = list(np.array(new_list) + roll_value)
                new_list[0][0] = 0

            list_deletes = np.ones(len(new_list)).astype(bool)
            for listvalue in range(len(new_list)):
                group_temp = new_list[listvalue]
                if group_temp[1] - group_temp[0] < init_spline_sampling / 10:
                    list_deletes[listvalue] = False
                    good_values[group_temp[0]:group_temp[1]] = 0
            filtered_list = [i for indx, i in enumerate(new_list) if list_deletes[indx] == True]
            new_list = filtered_list * 1

            for xx in range(len(new_list)):

                plt.plot(xi[new_list[xx][0]:new_list[xx][1]], yi[new_list[xx][0]:new_list[xx][1]], linewidth=0.5,color='green')

                spline_points.append([np.array([xi, yi])])
                spline_deriv.append([np.array([dxi, dyi])])

                spline_points_good.append([np.array([xi[new_list[xx][0]:new_list[xx][1]], yi[new_list[xx][0]:new_list[xx][1]]])])
                spline_deriv_good.append([np.array([dxi[new_list[xx][0]:new_list[xx][1]], dyi[new_list[xx][0]:new_list[xx][1]]])])
                groups.append([0,len(xi[new_list[xx][0]:new_list[xx][1]])])

        plt.imshow(npc_mean, cmap='gray')
        plt.axis('off')
        if save_fig == True:
            plt.savefig(self.resultprefix + 'initial_spline.svg',
                        format='svg',
                        dpi=400,
                        bbox_inches='tight')
        #plt.show()
        plt.close('all')
        self.initial_spline_points = spline_points_good
        self.initial_spline_derivative = spline_deriv_good
        self.initial_spline_groups = groups
        self.initial_spline_sampling = init_spline_sampling
        return logits


    def compute_drift(self, save_fig=False, estimate_precision=False):
        full_image = tifffile.imread(self.path+self.fn_track_npc)
        size = np.shape(full_image)[0]
        bins = np.linspace(0,size,self.driftbins+1)
        reference_image = np.mean(full_image[int(bins[0]):int(bins[1])],axis=0)
        translation_array = np.zeros((self.driftbins,2))
        translation_array1 = np.zeros((self.driftbins, 2))
        translation_array2 = np.zeros((self.driftbins, 2))
        for i in range(self.driftbins):
            tempim = np.mean(full_image[int(bins[i]):int(bins[i+1])],axis=0)
            # plt.imshow(tempim)
            # plt.show()
            translation, _, _ = skimage.registration.phase_cross_correlation(reference_image, tempim,
                                                                    upsample_factor=1000)
            # if i==0:
            #     translation_array[i,:] = translation
            # else:
            translation_array[i, :] = translation + translation_array[i-1, :]
            reference_image = tempim*1

        bincenters = (bins[0:-1]+ bins[1::])/2
        smoothx = scipy.interpolate.CubicSpline(bincenters,translation_array[:,1],extrapolate=True,bc_type = 'natural' )
        smoothy = scipy.interpolate.CubicSpline(bincenters,translation_array[:,0],extrapolate=True,bc_type = 'natural')
        frames = np.arange(0,size)
        xdrift = smoothx(frames)
        ydrift = smoothy(frames)
        self.xdrift = xdrift
        self.ydrift = ydrift
        if save_fig:
            plt.figure(figsize=[2,2])
            plt.scatter(bincenters, translation_array[:,1]*self.pixelsize,marker='x')
            plt.scatter(bincenters, translation_array[:, 0]*self.pixelsize, marker='x')
            plt.plot(frames, xdrift*self.pixelsize,label=r'$D_x$')
            plt.plot(frames, ydrift*self.pixelsize, label=r'$D_y$')
            plt.legend()
            plt.ylabel('Drift $D$ [nm]')
            plt.xlabel('Frame')
            plt.savefig(self.resultprefix + 'drift.svg',
                        format='svg')
            plt.close('all')

        if estimate_precision:

            set_1 = full_image[::2,:,:] # even
            set_2 = full_image[1::2,:,:] # odd
            size_prec = np.shape(set_2)[0]
            bins_prec = np.linspace(0, size_prec, self.driftbins + 1)
            reference_image1 = np.mean(set_1[int(bins_prec[0]):int(bins_prec[1])], axis=0)
            reference_image2= np.mean(set_2[int(bins_prec[0]):int(bins_prec[1])], axis=0)

            # odd
            for i in range(self.driftbins):
                tempim1  = np.mean(set_1[int(bins_prec[i]):int(bins_prec[i + 1])], axis=0)
                tempim2 = np.mean(set_2[int(bins_prec[i]):int(bins_prec[i + 1])], axis=0)
                # tempim1 = skimage.filters.difference_of_gaussians(tempim1, 0.5, 1.5)
                # tempim2 = skimage.filters.difference_of_gaussians(tempim2, 0.5, 1.5)
                translation1, _, _ = skimage.registration.phase_cross_correlation(reference_image1, tempim1,
                                                                                 upsample_factor=1000)
                translation2, _, _ = skimage.registration.phase_cross_correlation(reference_image2, tempim2,
                                                                 upsample_factor=1000)
                translation_array1[i, :] = translation1 + translation_array1[i - 1, :]
                translation_array2[i, :] = translation2 + translation_array2[i - 1, :]

                reference_image1 = tempim1 * 1
                reference_image2 = tempim2 * 1

        return translation_array1, translation_array2
    def islandinfo(self, y, trigger_val, stopind_inclusive=True):
        # Setup "sentients" on either sides to make sure we have setup
        # "ramps" to catch the start and stop for the edge islands
        # (left-most and right-most islands) respectively
        y_ext = np.r_[False, y == trigger_val, False]

        # Get indices of shifts, which represent the start and stop indices
        idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

        # Lengths of islands if needed
        lens = idx[1::2] - idx[:-1:2]

        # Using a stepsize of 2 would get us start and stop indices for each island
        return list(zip(idx[:-1:2], idx[1::2] - int(stopind_inclusive))), lens

    def refinement_npcfit_movie_new(self, number_points=5, length_line=12, sampling=1000,
                                     movie=True, sampling_normal=100,
                                    registration=True, smoothness=1, Lambda=1e-3, estimate_prec=False,
                                    number_mean=250,dual_strain=False, save_fig=True,iterations=300,makefig=False,max_signs=8,return_data=False):

        """
        :param number_points: number of points from inital spline to be considered (1 mean all points)-> np.arange(x,x,number_points)
        :param length_line: length of the normal line in pixels
        :param sampling: sampling points of the final spline
          :param movie: if true: movie will be saved in results folder
        :param sampling_normal: sampling of normal line
        :return:
        """
        if makefig==True:
            points = self.pointsnpc[3:5]
            deriv = self.derivnpc[3:5]
            groups = self.initial_spline_groups[3:5]
        if dual_strain==True:
            groups = []
            points = self.pointsnpc
            deriv = self.derivnpc
            for i in range(len(points)):
                groups.append([0,np.size(points[i][0],axis=1)])


        else:
            points = self.initial_spline_points
            deriv = self.initial_spline_derivative
            groups = self.initial_spline_groups
        npc = tifffile.imread(self.path + self.fn_track_npc)
        if dual_strain:
            npc = tifffile.imread(self.path + self.fn_track_rnp)
        npc = npc[self.frames_npcfit[0]:self.frames_npcfit[1], :, :]


        self.imgshape = np.shape(npc)


        npc_mean = np.mean(npc[:, :, :], 0)

        length_line = length_line / 2
        bounds = [[-1e4, 1e4], [-1e4, 1e4], [-100, 100], [-1e4, 1e4], [-100, 100],
                  [-1e4, 1e4], [length_line -2, length_line + 2], [length_line -4, length_line + 4], [-1e4, 1e4],
                  [-1e4, 1e4],[-1000, 1000]]
        if estimate_prec == False:
            if makefig:
                points_allcells, params, closed_allcells, \
                    deriv_allcells, errors, values, amplitude_allcells, sigma_allcells,\
                    start_arr, end_arr = self.fit_per_meanv2(npc_mean,
                                                                                 deriv,
                                                                                 length_line,
                                                                                 bounds,
                                                                                 number_points,
                                                                                 sampling,
                                                                                 groups, points,
                                                                                 movie,
                                                                                 sampling_normal,
                                                                                 registration,
                                                                                 0, smoothness,
                                                                                 Lambda,dual_strain=dual_strain,
                                                                                    iterations=iterations,
                                                                                                              makefig=makefig, max_signs=max_signs)
            else:
                points_allcells, params, closed_allcells,\
                    deriv_allcells, errors, values,  amplitude_allcells, sigma_allcells = self.fit_per_meanv2(npc_mean,
                                                                                 deriv,
                                                                                 length_line,
                                                                                 bounds,
                                                                                 number_points,
                                                                                 sampling,
                                                                                 groups, points,
                                                                                 movie,
                                                                                 sampling_normal,
                                                                                 registration,
                                                                                 0, smoothness,
                                                                                 Lambda,dual_strain=dual_strain,
                                                                                    iterations=iterations
                                                                                                             )
            # make image
            figimg, ax = plt.subplots()
            # figimg.set_size_inches(10, 10)
            ax.imshow(npc_mean, cmap='gray')  # crap for image david
            ax.axis('off')

            for cell in range(len(points_allcells)):
                groupsplot = points_allcells[cell]

                for group in range(len(groupsplot)):

                    points_refined = groupsplot[group]

                    xi, yi = points_refined[0, :], points_refined[1, :]

                    ax.plot(xi, yi, label='Refined estimation', linewidth=0.5, color='green')
                    if registration == True:
                        points_refined_morphed = self.transform_coordinates(points_refined.T,
                                                                            self.registration['scale'],
                                                                            self.registration['angle'],
                                                                            self.registration['tvec'][::-1],
                                                                            [np.shape(npc_mean)[1] / 2,
                                                                             np.shape(npc_mean)[0] / 2])

                        points_allcells[cell][group] = points_refined_morphed.T

                        # ax.plot(self.initial_spline_points[cell][0][0,:]-30,self.initial_spline_points[cell][0][1,:]-100, color='red',label ='Initial guess' )
            # ax.legend()
            if save_fig:
                figimg.savefig(self.resultprefix + 'refined_spline' + '.svg',
                               format='svg',
                               dpi=300,
                               bbox_inches='tight')
                # figimg.savefig('refined_spline' +str(np.random.randint(0,10000))+ '.svg',
                #                format='svg',
                #                dpi=300,
                #                bbox_inches='tight')

            # plt.show()
            plt.close('all')
            if makefig:
                print('do nothing')
            else:
                self.amplitude_array_tot = amplitude_allcells
                self.sigma_array_tot = sigma_allcells
                self.pointsnpc = points_allcells
                self.closednpc = closed_allcells
                self.derivnpc = deriv_allcells
                self.errors_npc = errors
                self.values_npc = values
            # Create bounding box

            boundingbox = []

            for cell in range(len(points_allcells)):
                minrow, maxrow, mincol, maxcol = 1e6, 0, 1e6, 0
                groupsplot = points_allcells[cell]
                for group in range(len(groupsplot)):
                    points_refined = groupsplot[group]
                    minrow = min(np.min(points_refined[1, :]), minrow)
                    maxrow = max(np.max(points_refined[1, :]), maxrow)
                    mincol = min(np.min(points_refined[0, :]), mincol)
                    maxcol = max(np.max(points_refined[0, :]), maxcol)

                boundingbox.append([minrow, maxrow, mincol, maxcol])

            self.bbox_NE = np.array(boundingbox)

            track_rnp_full = tifffile.imread(self.path + self.fn_track_rnp)
            if len(np.shape(track_rnp_full)) == 2:  # 2D
                track_rnp_full = track_rnp_full[None, ...]  # add dimension
            self.bbnum = np.shape(self.bbox_NE)[0]
            track_rnp_full = track_rnp_full[self.frames[0]:self.frames[1], :, :]

            for qq in range(np.shape(self.bbox_NE)[0]):
                boundingbox = self.bbox_NE[qq, :] * 1
                boundingbox[[0, 2]] = boundingbox[[0, 2]] - self.roisize / 2 - 16
                boundingbox[[1, 3]] = boundingbox[[1, 3]] + self.roisize / 2 + 16

                boundingbox[0] = int(np.ceil(int(max(boundingbox[0], 0)) / 2) * 2)
                boundingbox[2] = int(np.ceil(int(max(boundingbox[2], 0)) / 2) * 2)
                boundingbox[1] = int(np.ceil(int(min(boundingbox[1], np.shape(track_rnp_full)[1])) / 2) * 2)
                boundingbox[3] = int(np.ceil(int(min(boundingbox[3], np.shape(track_rnp_full)[2])) / 2) * 2)
                self.bbox_NE[qq, :] = boundingbox * 1

            if dual_strain:
                return points_allcells, errors, values, deriv_allcells
            else:
                if makefig:
                    return points_allcells, errors, values, start_arr,end_arr
                else:

                    return points_allcells, errors, values
        else:
            npc = tifffile.imread(self.path + self.fn_track_npc)
            # Select the first 250 even frames for npc1
            npc1 = npc[::2, :, :][:number_mean, :, :]
            # Select the first 250 odd frames for npc2
            npc2 = npc[1::2, :, :][:number_mean, :, :]

            npc_mean1 = np.mean(npc1[:, :, :], 0)
            npc_mean2 = np.mean(npc2[:, :, :], 0)

            # plt.imshow(abs(npc_mean1-npc_mean2))
            #
            # test = np.std(npc_mean1 - npc_mean2)
            # testmax = np.max(npc_mean1 - npc_mean2)
            #
            # plt.imshow(npc_mean2)
            # plt.show()
            params1,params2 = self.fit_per_meanforprecision(npc_mean1,
                                                                                                           deriv,
                                                                                                           length_line,
                                                                                                           bounds,
                                                                                                           number_points,
                                                                                                           sampling,
                                                                                                           groups,
                                                                                                           points,
                                                                                                           movie,
                                                                                                           sampling_normal,
                                                                                                           registration,
                                                                                                           0,
                                                                                                           smoothness,
                                                                                                           Lambda,npc_mean2=npc_mean2)

            return params1, params2

    def fit_per_meanv2(self, npc_mean, deriv, length_line, bounds, number_points, sampling, groups, points, movie,
                     sampling_normal, registration, index, smoothness, Lambda, iterations=300,max_gap_good_fits = 100,dual_strain=False, makefig=False, max_signs=8):

        def intensity_func_sig(t, A, K, B, C, nu, Q, M, mu, sigma, amplitude,offset):

            exponent_richards = -B * (t - M)
            denominator_richards = C + Q * np.exp(exponent_richards)
            power_richards = 1 / nu
            richards_curve = A + (K - A) / denominator_richards ** power_richards

            exponent_gaussian = -(t - mu) ** 2 / (2 * sigma ** 2)
            gaussian_curve = amplitude * np.exp(exponent_gaussian)

            combined_curve = richards_curve + gaussian_curve+ offset

            return combined_curve



        points_allcells = []
        deriv_allcells = []
        end_allcells =[]
        start_allcells = []
        closed_condition_allcells = []
        params_allcells = []
        errors_allcells = []
        values_allcells = []
        amplitude_allcells = []
        sigma_allcells = []
        for i in tqdm.tqdm(range(len(groups)), desc='refined fitting nuclei'):

            tangent_slopes = deriv[i][0]
            normal_slopes = np.array([-tangent_slopes[1, :], tangent_slopes[0, :]])
            points_it = points[i][0]
            group = groups[i]
            selection = np.array([])
            points_new_percell = []
            deriv_percell = []
            closed_condition_percell = []
            params_percell = []
            errors_percell = []
            values_percell = []
            amplitude_percell = []
            sigma_percell = []
            errorflag = 0

            # how much points we need to consider (1 means all points)

            selection = np.arange(group[0], group[1], number_points).astype(int)
            # selection = np.array([1])#########################################

            num = sampling_normal  # sampling of normal line
            zi_array = np.zeros((len(selection), num))
            dist_array = np.zeros((len(selection), num))
            initial = np.zeros((len(selection), 7))
            #temp = bounds[0] * 1
            #bounds.append(temp)
            for k in range(len(selection)):
                select = selection[k]
                point = points_it[:, select]
                normal_slope = normal_slopes[:, select] / np.linalg.norm(normal_slopes[:, select])

                start = point - length_line * normal_slope
                end = point + length_line * normal_slope
                # -- Extract the line...
                # Make a line with "num" points...
                x0, y0 = start[0], start[1]
                x1, y1 = end[0], end[1]
                if np.round(x0) < 0 or np.round(y0) < 0 or np.round(x1) >= self.imgshape[1] - 2 or np.round(y1) >= self.imgshape[2] - 2 or np.round(x0) >= self.imgshape[1] - 2\
                        or errorflag == 1 or np.round(y0) >= self.imgshape[2] - 2:
                    errorflag = 1
                    zi_array = np.empty((1, 1))
                    dist_array = np.empty((1, 1))

                else:

                    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
                    # Extract the values along the line

                    zi = npc_mean[np.round(y).astype(int), np.round(x).astype(int)]
                    dist_alongline = np.cumsum(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))
                    dist_alongline = np.append(0, dist_alongline)
                    zi_array[k, :] = zi
                    dist_array[k, :] = dist_alongline


                        # initial_guess = [length_line, 1.5, max(zi) - min(zi), min(zi), -0.6, zi[0] - zi[-1],length_line]
                        # boundscurvefit = (np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                        # try:
                        #     initial[k,:], _=scipy.optimize.curve_fit(intensity_func_sig,dist_array[k, :], zi_array[k, :] ,p0=initial_guess, bounds=boundscurvefit )
                        # except:
                        #     initial[k, :], _ = scipy.optimize.curve_fit(intensity_func_sig, dist_array[k, :],
                        #                                                 zi_array[k, :], p0=initial_guess)

                        #initial[k,:] = scipy.optimize.minimize(LL_intensity_func_sig, dist_array[k, :],initial_guess, method=('L-BFGS-B'))


            if errorflag == 0:

                dev = torch.device('cuda')

                # with torch.no_grad():  # when no tensor.backward() is used
                # convert image to tensor
                smp = torch.from_numpy(zi_array).to(dev)

                def construct_initial_guess(x_line, zi_init):
                    def line(x_line, zi_init, range):
                        a = (zi_init[:, -1] - zi_init[:, 0]) / (range[:, -1] - range[:, 0])
                        b = zi_init[:, 0]
                        return a[..., None] * x_line + b[..., None]

                    area = torch.sum(zi_init - line(x_line, zi_init, x_line), dim=-1) * (x_line[:, 1] - x_line[:, 0])
                    amp = zi_init[:, int(x_line.shape[1] / 2)] - line(x_line[:, int(x_line.shape[1] / 2)], zi_init, x_line)[:, 0]

                    initial_guess = torch.zeros((x_line.shape[0], len(bounds))).to(dev)
                    initial_guess[:, 0] = zi_init[:, 0] #A
                    #initial_guess[:, 1] = ((zi_init[:, -1] - zi_init[:, 0]) + torch.min(zi_init, dim=-1)[0]) #K
                    initial_guess[:, 1] = zi_init[:, -1]#K

                    initial_guess[:, 2] = 0.4 #B
                    initial_guess[:, 3] = 1# C
                    initial_guess[:, 4] = 1 #nu
                    initial_guess[:, 5] = 0.5 #Q
                    # for init_it in range(np.shape(zi_array)[0]):
                    #     row_vals = zi_init[init_it,:]
                    #     row_line = x_line[init_it,:]
                    #     maxval = torch.amax(row_vals)
                    #     indices = row_vals==maxval
                    #     true_indices = torch.where(indices)[0]
                    #     median_index = torch.median(true_indices)
                    #     #median_index = true_indices[-1]
                    #     #initial_guess[init_it,6] = dist_alongline[int(median_index)]
                    #     initial_guess[init_it, 7] = dist_alongline[int(median_index)]
                    initial_guess[:, 6] = length_line
                    initial_guess[:, 7] = length_line
                    if makefig:
                        initial_guess[:, 8] = area / amp * 0.28 # sigma
                    else:
                        initial_guess[:, 8] = area / amp * 0.35 # sigma
                    initial_guess[:, 9] = amp # amp
                    initial_guess[:, 10] = 0 # offset
                    return initial_guess
                initial_guess = construct_initial_guess(torch.from_numpy(dist_array).to(dev),smp)
                init_guess_np = initial_guess.detach().cpu().numpy()
                if dual_strain:
                    testtttt = 0
                #initial = np.concatenate((initial, initial[:, 0][..., None]), axis=1)
                model = npcfit_class(torch.from_numpy(dist_array).to(dev))
                # mu, _ = model.forward(initial_guess, torch.ones_like(torch.from_numpy(dist_array).to(dev), dtype=bool)[:, 0])
                # real = smp.detach().cpu().numpy()
                # mu = mu.detach().cpu().numpy()
                # plt.figure(dpi=400)
                # number = 1
                # plt.plot(dist_array[number,:],real[number,:])
                #
                # plt.plot(dist_array[number,:],mu[number, :])
                # plt.xlabel(r'Distance along normal $n$ [px]')
                # plt.ylabel(r'Intensity $I$ [au]')
                # plt.tight_layout()
                # plt.show()

                bounds_ext = bounds * 1
                #temp = bounds[0] * 1
                #bounds_ext.append(temp)
                param_range = torch.Tensor(bounds).to(dev)





                mle = LM_MLE_forspline_new(model)


                # mle = torch.jit.script(mle)
                params_, _, traces = mle.forward(initial_guess.type(torch.cuda.FloatTensor),
                                                 smp[..., None].type(torch.cuda.FloatTensor),
                                                 param_range, iterations, Lambda)
                ##
                # real = smp.detach().cpu().numpy()
                #
                # mu, _ = model.forward(params_,
                #                       torch.ones_like(torch.from_numpy(dist_array).to(dev), dtype=bool)[:, 0])
                # mu = mu.detach().cpu().numpy()
                # plt.figure(dpi=400)
                # number = 0
                # plt.plot(dist_array[number,:],real[number,:])
                #
                # plt.plot(dist_array[number,:],mu[number, :])
                # plt.xlabel(r'Distance along normal $n$ [px]')
                # plt.ylabel(r'Intensity $I$ [au]')
                # plt.tight_layout()
                # plt.show()
                # ####
                traces = torch.permute(traces,(1,0,2))
                traces = traces.cpu().detach().numpy()

                params = params_.cpu().detach().numpy()
                # filter based on iterations
                iterations_vector = np.zeros(np.size(params, 0))
                for loc in range(np.size(params, 0)):
                    try:
                        iterations_vector[loc] = np.where(traces[:, loc, 0] == 0)[0][0]
                    except:
                        iterations_vector[loc] = iterations

                filter_border = np.zeros(np.size(params, 0), dtype=bool)  # Initialize as a boolean array
                for coll in range(np.size(params, 1)):
                    test1 = np.logical_or(params[:, coll] < bounds_ext[coll][0] * 0.95,
                                          params[:, coll] > bounds_ext[coll][1] * 0.95)
                    if coll > 0:
                        filter_border = np.logical_or(filter_border, test1)
                    else:
                        filter_border = test1  # For the first column, directly assign test1 to filter_border

                filter_border = np.logical_or(filter_border, iterations_vector==iterations)
                print('\nfiltering NPC on position = ', np.sum(test1), '/', len(test1))
                print('\nfiltering NPC on iterations = ', np.sum(iterations_vector==iterations), '/', len(filter_border))
                params = params[np.invert(filter_border), :]
                selection = selection[np.invert(filter_border),]
                dist_array = dist_array[np.invert(filter_border),]
                zi_array = zi_array[np.invert(filter_border),]

                if len(selection) == 0:
                    errorflag=1
                torch.cuda.empty_cache()
                if errorflag == 0:
                    # only select longest sequence:
                    current_sequence = [selection[0]]  # Initialize the current sequence with the first element
                    longest_sequence = []  # Initialize the longest sequence with an empty list

                    for seq in range(1, len(selection)):
                        if selection[seq] - current_sequence[-1] < max_gap_good_fits:
                            current_sequence.append(selection[seq])
                        else:
                            if len(current_sequence) > len(longest_sequence):
                                longest_sequence = current_sequence.copy()
                            current_sequence = [selection[seq]]

                    # Check if the last sequence is longer than the longest found so far
                    if len(current_sequence) > len(longest_sequence):
                        longest_sequence = current_sequence

                    # Create a filter array with boolean values using NumPy
                    filter_array = np.isin(selection, longest_sequence)




                    params = params[filter_array, :]
                    selection = selection[filter_array,]
                    dist_array = dist_array[filter_array,]
                    zi_array = zi_array[filter_array,]


                    arr = []

                    fig, axes = plt.subplots(1, 2)
                    fig.set_size_inches(6, 3)
                    fig.tight_layout(w_pad=5, h_pad=5, pad=3)
                    # print('\n Print videos for tracking NE ' + str(i + 1) + '/' + str(self.bbnum) + '\n')
                    weight_arr = []
                    error_arr = []
                    values_arr = []
                    values = np.empty((100, 0))
                    error = np.empty((100, 0))

                filter_error = np.ones(len(selection))
                points_refined = np.zeros((np.size(selection), 2))
                pos_along_normal = np.zeros(np.shape(selection))
                start_arr = []
                end_arr = []
                if errorflag == 0:
                    error_alarm = False
                    if len(selection) == np.size(params, 0):
                        # selection = selection[good_chisq]
                        # params = params[good_chisq]

                        for qq in range(len(selection)):

                            select = selection[qq]
                            point = points_it[:, select]
                            normal_slope = normal_slopes[:, select] / np.linalg.norm(normal_slopes[:, select])

                            start = point - length_line * normal_slope
                            end = point + length_line * normal_slope
                            start_arr.append(start)
                            end_arr.append(end)
                            # -- Extract the line...
                            # Make a line with "num" points...
                            x0, y0 = start[0], start[1]
                            x1, y1 = end[0], end[1]
                            # compute test statistic
                            r = (zi_array[qq, :] - intensity_func_sig(dist_alongline, *params[qq,
                                                                                                         :])) ** 2 / intensity_func_sig(
                                dist_alongline, *params[qq, :])
                            error_non_squared = (zi_array[qq, :] - intensity_func_sig(dist_alongline, *params[qq,
                                                                                                         :]))  / intensity_func_sig(
                                dist_alongline, *params[qq, :])
                            # plt.plot(np.linspace(0,12,len(zi_array[qq,:])), zi_array[qq, :])
                            # plt.plot(np.linspace(0, 12, len(zi_array[qq, :])), intensity_func_sig(dist_alongline, *params[qq,
                            #                                                                              :]))
                            # plt.show()
                            error_over_line = r * 1
                            error = np.concatenate((error, error_non_squared[..., None]), axis=1)
                            if np.max(abs(error_non_squared))>0.50:
                                filter_error[qq] = 0
                            values = np.concatenate(
                                (values, intensity_func_sig(dist_alongline, *params[qq, :])[..., None]), axis=1)

                            x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
                            if movie == True and filter_error[qq]==1:
                                zoom_factor = 40  # You can change this to control the zoom level
                                if qq == 0:
                                    x_min, x_max = x0 - zoom_factor, x0 + zoom_factor
                                    y_min, y_max = y0 - zoom_factor, y0 + zoom_factor
                                axes[0].imshow(npc_mean, cmap='gray')
                                axes[0].plot(x, y, '-', label='Normal line')
                                axes[0].scatter(x0, y0, marker="x", label='Start')
                                axes[0].axis('image')
                                #axes[0].legend()
                                axes[0].axis('off')  # Hide the axes
                                axes[0].legend(frameon=True, facecolor='white')
                                axes[0].set_xlim(x_min, x_max)
                                axes[0].set_ylim(y_min, y_max)
                                axes[1].plot(dist_array[qq, :], (zi_array[qq, :]-self.offset)*self.gain, label='Intensity along line')
                                # axes[1].axvline(x=length_line, color='red', label='initial guess')
                                axes[1].plot(dist_array[qq, :], (intensity_func_sig(dist_array[qq, :], *params[qq, :])-self.offset)*self.gain,
                                             label='Fit')
                                axes[1].set_xlabel('Distance [pixels]')
                                axes[1].set_ylabel('Intensity [photons]')
                                # axes[1].set_title( 'Pvalue: '+str(r))

                                axes[1].legend(loc='lower left')

                                arr.append(self.cvtFig2Numpy(fig))
                                if qq%30 == 0:
                                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"{self.resultprefix}_figure_{current_time}.svg"
                                    fig.savefig(filename, format='svg')
                                axes[0].cla()
                                axes[1].cla()

                                plt.close('all')
                                ##

                            pos_along_normal[qq] = params[qq, 7]
                            pos_along_normal[qq] -= length_line
                            pos_change = pos_along_normal[qq] * normal_slope
                            points_refined[qq] = point + pos_change

                        filter_error = filter_error.astype(bool)
                        points_refined = points_refined[filter_error]*1
                        params = params[filter_error, :]
                        selection = selection[filter_error,]

                        values = values[:,filter_error]
                        error = error[:,filter_error]

                        # filter again
                        current_sequence = [selection[0]]  # Initialize the current sequence with the first element
                        longest_sequence = []  # Initialize the longest sequence with an empty list

                        for seq in range(1, len(selection)):
                            if selection[seq] - current_sequence[-1] < max_gap_good_fits:
                                current_sequence.append(selection[seq])
                            else:
                                if len(current_sequence) > len(longest_sequence):
                                    longest_sequence = current_sequence.copy()
                                current_sequence = [selection[seq]]

                        # Check if the last sequence is longer than the longest found so far
                        if len(current_sequence) > len(longest_sequence):
                            longest_sequence = current_sequence

                        # Create a filter array with boolean values using NumPy
                        filter_array_v2 = np.isin(selection, longest_sequence)
                        points_refined = points_refined[filter_array_v2]*1
                        params = params[filter_array_v2, :]
                        selection = selection[filter_array_v2,]

                        values = values[:,filter_array_v2]
                        error = error[:,filter_array_v2]

                        points_refined = np.reshape(points_refined[points_refined != 0],
                                                    (int(np.size(
                                                        points_refined[points_refined != 0]) / 2),
                                                     2)) * 1


                        if np.size(points_refined) != 0 and len(points_refined[:, 0])>4:

                            error_arr.append(error)
                            values_arr.append(values)
                            weights = 1 / np.array(weight_arr)
                            closed = 0
                            distances_betweenpoints = np.sqrt(np.sum(np.diff(points_refined, axis=0) ** 2, axis=1))
                            cumulative_distances = np.cumsum(np.insert(distances_betweenpoints, 0, 0))
                            tck_amplitude, _ = scipy.interpolate.splprep([cumulative_distances,params[:,9]],
                                                               per=False,
                                                               k=3,
                                                               quiet=3, s=smoothness)# w=weights/max(weights)+0.5,
                            tck_sigma, _ = scipy.interpolate.splprep(
                                [cumulative_distances, params[:, 8]],
                                per=False,
                                k=3,
                                quiet=3, s=smoothness)  # w=weights/max(weights)+0.5,

                            if len(selection) > (
                            sampling_normal) * 0.98 and group[1]>self.initial_spline_sampling*0.98:  # periodic condition
                                tck, u = scipy.interpolate.splprep([points_refined[:, 0], points_refined[:, 1]],
                                                                   per=True,
                                                                   k=3,
                                                                   quiet=3, s=smoothness)  # w=weights/max(weights)+0.5,
                                closed = 1
                            else:

                                tck, u = scipy.interpolate.splprep([points_refined[:, 0], points_refined[:, 1]],
                                                                   per=False, k=3,
                                                                   s=smoothness)  # w=weights/max(weights)+0.5,

                            cut = 100  # cut points at edges
                            amplitude_array = scipy.interpolate.splev(np.linspace(0, 1, max(len(points_refined), sampling)), tck_amplitude)[1]

                            sigma_array = scipy.interpolate.splev(np.linspace(0, 1, max(len(points_refined), sampling)),
                                                                tck_sigma)[1]
                            xi, yi = scipy.interpolate.splev(np.linspace(0, 1, max(len(points_refined), sampling)), tck)

                            dxi, dyi = scipy.interpolate.splev(np.linspace(0, 1, max(len(points_refined), sampling)),
                                                               tck, der=1)
                            ddxi, ddyi = scipy.interpolate.splev(np.linspace(0, 1, max(len(points_refined), sampling)),
                                                                 tck,
                                                                 der=2)
                            if closed == 0:
                                amplitude_array = amplitude_array[min(cut, len(ddyi)):max(len(ddyi) - cut, cut)]
                                sigma_array = sigma_array[min(cut, len(ddyi)):max(len(ddyi) - cut, cut)]
                                xi = xi[min(cut, len(xi)):max(len(xi) - cut, cut)]
                                yi = yi[min(cut, len(yi)):max(len(yi) - cut, cut)]
                                dxi = dxi[min(cut, len(dxi)):max(len(dxi) - cut, cut)]
                                dyi = dyi[min(cut, len(dyi)):max(len(dyi) - cut, cut)]
                                ddxi = ddxi[min(cut, len(ddxi)):max(len(ddxi) - cut, cut)]
                                ddyi = ddyi[min(cut, len(ddyi)):max(len(ddyi) - cut, cut)]

                            # filter based on change in radius of curvature

                            # Calculate angles between consecutive tangents
                            angles = np.arctan2(dyi, dxi)  # Angle of each vector

                            # Calculate raw differences
                            angle_diffs = np.diff(angles)

                            # Normalize differences to the range [-pi, pi]
                            angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

                            # Convert to degrees and take the absolute value
                            angle_diffs = np.abs(np.rad2deg(angle_diffs))
                            rad = (dxi ** 2 + dyi ** 2) ** (3 / 2) * 1 / (dxi * ddyi - dyi * ddxi)

                            # plt.hist(rad,bins=50)
                            # plt.show()
                            # plt.plot(xi,yi)
                            # plt.show()
                            normal_slope = np.array([rad * dyi, -rad * dxi]) / np.linalg.norm(
                                np.array([rad * dyi, -rad * dxi]), axis=0)
                            # plot second derivative vectors:
                            # V = normal_slope
                            # origin = np.array([xi[np.arange(1,len(xi)-2,10)],yi[np.arange(1,len(xi)-2,10)]])# origin point
                            #
                            # plt.quiver(*origin, V[0, [np.arange(1,len(xi)-2,10)]], V[1, [np.arange(1,len(xi)-2,10)]])
                            # plt.plot(xi[np.arange(1,len(xi)-2,10)],yi[[np.arange(1,len(xi)-2,10)]])
                            # plt.show()

                            normal_slope_rolled = np.roll(normal_slope, 1, axis=1)
                            #
                            dotproduct = (normal_slope[:, 1:-1] * normal_slope_rolled[:, 1:-1]).sum(0)
                            asign = np.sign(dotproduct)
                            signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

                            if sum(signchange) < max_signs and error_alarm==False and np.amax(abs(angle_diffs))<1:
                                points_new_percell.append(np.array([xi, yi]))
                                deriv_percell.append(np.array([dxi, dyi]))
                                closed_condition_percell.append(closed)
                                errors_percell.append(error_arr)
                                params_percell.append(params)
                                values_percell.append(values_arr)
                                amplitude_percell.append(amplitude_array)
                                sigma_percell.append(sigma_array)
                                end_allcells.append(end_arr)
                                start_allcells.append(start_arr)
                                # plt.plot(xi,yi)
                                # plt.title(str(np.amax(abs(angle_diffs))))
                                # plt.tight_layout()
                                # plt.savefig('test' + str(np.random.randint(0,1000)))

                    if movie:
                        self.makevideoFromArray(self.resultprefix + 'npcfit' + str(i) + str(index) + '.mp4', arr)

            if np.size(points_new_percell) != 0:
                points_allcells.append(points_new_percell)
                params_allcells.append(params_percell)
                closed_condition_allcells.append(closed_condition_percell)
                deriv_allcells.append(deriv_percell)
                errors_allcells.append(errors_percell)
                values_allcells.append(values_percell)
                amplitude_allcells.append(amplitude_percell)
                sigma_allcells.append(sigma_percell)
        if makefig:
            return points_allcells, params_allcells, closed_condition_allcells, \
                deriv_allcells, errors_allcells, values_allcells, amplitude_allcells, sigma_allcells,start_allcells, end_allcells
        else:

            return points_allcells, params_allcells, closed_condition_allcells, deriv_allcells, errors_allcells, values_allcells, amplitude_allcells, sigma_allcells

    def transform_coordinates(self,coordinates, scale_factor, angle_degrees, translation_vector, center):
        # Scaling matrix
        scale_matrix = np.array([[scale_factor, 0],
                                 [0, scale_factor]])

        # Rotation matrix
        angle_radians = math.radians(angle_degrees)
        rotation_matrix = np.array([[math.cos(angle_radians), -math.sin(angle_radians)],
                                    [math.sin(angle_radians), math.cos(angle_radians)]])

        # Combine scaling and rotation matrices
        transformation_matrix = np.dot(rotation_matrix, scale_matrix)

        # Apply the scaling and rotation
        scaled_rotated_coords = np.dot(coordinates - center, transformation_matrix) + center

        # Apply the translation
        transformed_coords = scaled_rotated_coords + translation_vector

        return transformed_coords
    def cvtFig2Numpy(self,fig):

        canvas = FigureCanvas(fig)
        canvas.draw()

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height.astype(np.uint32),
                                                                            width.astype(np.uint32), 3)

        return image


    def makevideoFromArray(self, movieName, array, fps=25):


        imageio.mimwrite(movieName, array,fps=fps)



    def prepare_mle(self, minsize=15, max_eccentricity=0.3, use_old_mask=False,alfa=0.1, min_dist_centroid = 1):


        poslist = []
        for box in range(np.shape(self.bbox_NE)[0]):
            pos = np.empty(shape=[0, 3])
            if not os.path.exists(self.path + '/mask' + str(box) + '.npy'):
                print('mask not found')
            else:
                if use_old_mask:
                    mask = np.load(self.path + '/pfa_adj_arr' + str(box) + '.npy')
                    mask[mask==0]=1
                    mask = mask<alfa
                else:
                    mask = np.load(self.path + '/mask' + str(box) + '.npy')
                image = mask.astype(int) * 1
                distance = ndimage.distance_transform_edt(image)
                local_maxi = np.zeros(np.shape(image))
                filtered_mask = np.zeros(np.shape(image))

                for i in range(np.size(image, 0)):
                    mask[i, :, :] = ndimage.binary_fill_holes(mask[i, :, :]).astype(int)
                    labels_ws = measure.label(mask[i, :, :])
                    table = measure.regionprops_table(
                        labels_ws,
                        properties=('label', 'area', 'centroid', 'eccentricity'),
                    )
                    condition = np.logical_and(table['area'] > minsize, table['eccentricity'] < max_eccentricity)
                    labels_filt = table['label'][condition]
                    new_mask = np.zeros(np.shape(labels_ws))
                    for q in range(len(labels_filt)):
                        temp = labels_ws == labels_filt[q]
                        new_mask = np.logical_or(new_mask, temp)

                    filtered_mask[i, :, :] = new_mask

                    temp = np.array([table['centroid-0'][condition], table['centroid-1'][condition]]).T
                    temp = np.hstack((np.ones((np.shape(temp)[0], 1)) * i, temp))

                    # ------------------filter pos if they are too close to eachother-----------------------------
                    #
                    # # Convert the data into a NumPy array
                    # data_array = np.array(pos)
                    #
                    # # Group data by frame
                    # frame_dict = {}
                    # for row in data_array:
                    #     frame = row[0]
                    #     if frame not in frame_dict:
                    #         frame_dict[frame] = []
                    #     frame_dict[frame].append(row[1:])
                    #
                    # # Filter spots within each frame
                    # filtered_spots = set()  # Use a set to ensure unique spots
                    #
                    # for frame, spots in frame_dict.items():
                    #
                    #     if len(spots) > 1:
                    #         spots = np.array(spots)
                    #         num_spots = len(spots)
                    #         for i in range(num_spots):
                    #             for j in range(i + 1, num_spots):
                    #                 x1, y1 = spots[i]
                    #                 x2, y2 = spots[j]
                    #                 distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    #                 if distance > 4:
                    #                     # Convert the spots to tuples for hashing
                    #                     spot1 = (frame, x1, y1)
                    #                     spot2 = (frame, x2, y2)
                    #                     filtered_spots.add(spot1)
                    #                     filtered_spots.add(spot2)
                    #     else:
                    #         x1, y1 = spots[0]
                    #
                    #         filtered_spots.add((frame, x1, y1))
                    # # Convert the set of unique spots back to a list
                    # filtered_spots = list(filtered_spots)
                    # filtered_spots = np.array(filtered_spots)
                    # # ----------    --------filter pos if they are too close to eachother-----------------------------
                    pos = np.append(pos, temp, axis=0)
            poslist.append(pos)

            np.save(self.path + '/poslist' + str(box), filtered_mask)

        return poslist



    def extract_patches(self,input, roisize):
        """
        Extracts patches of size [roisize, roisize] from the input tensor,
        excluding edges to avoid incomplete patches.
        """
        # Ensure input is (H, W), add a batch and channel dimension: becomes (1, 1, H, W)
        input_unsqueezed = input.unsqueeze(0).unsqueeze(0)

        # Calculate the valid area size to apply unfold
        valid_H = input.shape[0] - (roisize - 1)
        valid_W = input.shape[1] - (roisize - 1)

        # Use unfold to extract the patches
        patches = input_unsqueezed.unfold(2, roisize, 1).unfold(3, roisize, 1)

        # Adjust patches tensor to exclude the padding, focusing on the valid area
        # The result is of shape (1, 1, valid_H, valid_W, roisize, roisize)
        # Since we didn't actually pad, we directly move to extracting the valid patches

        # Reshape to put patches in the expected order: (valid_H*valid_W, roisize, roisize)
        patches = patches.contiguous().view(-1, roisize, roisize)

        return patches
    def GLRT_detector_DL_multichannel(self, initial_guess, bounds, lmlambda=100, use_cuda=True,
                                      iterations=100, alfa=0.05, batch_size = 20000*8, full_image = False,
                                      number_channel=20, plot_background =False, own_area = None):
        all_masks = []
        all_bbs = []
        def batch_frames(input_array, batch_size=100):
            num_frames = np.size(input_array, 0)
            num_batches = int(np.ceil(num_frames / batch_size))
            batched_frames = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_frames)
                batched_frames.append(input_array[start_idx:end_idx])
            return batched_frames
        def concatenate_images(images_list):
            return np.concatenate(images_list, axis=0)
        self.roisize=16



        roi_small = [4, 12]
        traces_i, traces_bg = [],[]
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialize model
        Unet_pp_model = Unet_pp(number_channel).to(dev)
        Unet_pp_model.load_state_dict(torch.load(self.bg_model)['model_state_dict'])
        Unet_pp_model.eval()

        if use_cuda == True:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')

        batched_frames_list = []
        track_rnp_full = tifffile.imread(self.path + self.fn_track_rnp)
        if len(np.shape(track_rnp_full)) == 2:  # 2D
            track_rnp_full = track_rnp_full[None, ...]  # add dimension
        if len(np.shape(track_rnp_full)) != 2 and len(np.shape(track_rnp_full)) != 3:
            raise (ValueError('Input image must be 2D set or 3D (first axis is time)'))
        if full_image == True:
            self.bbox_NE = np.array([0,np.size(track_rnp_full,1),0,np.size(track_rnp_full,2)])
            self.bbox_NE = self.bbox_NE[None,...]
        if own_area is not None:
            self.bbox_NE = np.array([own_area[0],own_area[1],own_area[2],own_area[3]])
            self.bbox_NE = self.bbox_NE[None,...]
        for qq in range(np.shape(self.bbox_NE)[0]):
            print('\n boundingbox ' + str(qq + 1) + '/' + str(np.shape(self.bbox_NE)[0]) + '\n')
            batched_frames_list = batch_frames(track_rnp_full, batch_size=number_channel)

            for batch_idx, track_rnp_batch in enumerate(batched_frames_list):


                track_rnp = track_rnp_batch[:, int(self.bbox_NE[qq, 0]):int(self.bbox_NE[qq, 1]),
                            int(self.bbox_NE[qq, 2]):int(self.bbox_NE[qq, 3])]

                mask = np.zeros((int(np.size(track_rnp, 0)), int(np.size(track_rnp, 1) - self.roisize),
                                int(np.size(track_rnp, 2) - self.roisize)))
                pfa_arr = np.zeros((int(np.size(track_rnp, 0)), int(np.size(track_rnp, 1) - self.roisize),
                                   int(np.size(track_rnp, 2) - self.roisize)))
                pfa_adj_arr = np.zeros((int(np.size(track_rnp, 0)), int(np.size(track_rnp, 1) - self.roisize),
                                       int(np.size(track_rnp, 2) - self.roisize)))

                flag = 0
                flag_bg = 0
                border = np.ceil(self.roisize / 2)
                rows = np.arange(border, (int(np.size(track_rnp, 1)) - border))
                collumns = np.arange(border, (int(np.size(track_rnp, 2)) - border))

                # smp_arr = np.zeros((len(rows) * len(collumns) * np.size(track_rnp,0), self.roisize, self.roisize),dtype=np.ushort)

                mean_bg = np.zeros((len(rows) * len(collumns) * np.size(track_rnp,0)))
                image = track_rnp
                image = np.clip(image, 1e-1, np.inf)
                bg_image = torch.tensor(np.zeros_like(image)).to(self.dev)
                smp_list = []
                # for i in range(len(rows)):
                #
                #     row = rows[i]
                #     for j in range(len(collumns)):
                #         col = collumns[j]
                #         for frame_iter in np.arange(self.frames[0], min(self.frames[1], np.size(track_rnp,0))):
                #             #slice = track_rnp
                #              # clip image at zero
                #
                #             # remove borders form image
                #             # original
                #
                #
                #                     #mean_bg[flag] = np.mean(image[int(row), int(col)]) + np.std(image)
                #                     smp_arr[flag, :, :] = image[frame_iter,
                #                                       int(row - np.ceil(self.roisize  / 2)): int(row + np.ceil(self.roisize  / 2)),
                #                                       int(col - np.ceil(self.roisize / 2)): int(col + np.ceil(self.roisize  / 2))]
                #                     flag = flag + 1
                for frame_iter in np.arange(self.frames[0], min(self.frames[1], np.size(track_rnp, 0))):
                    slice = torch.tensor(image[frame_iter]).to(self.dev)
                    smp_arrv2_ = self.extract_patches(slice[0:-1,0:-1], self.roisize)
                    smp_arrv2 = smp_arrv2_.detach().cpu().numpy()

                    smp_list.append(smp_arrv2)
                smp_arr_ = np.array(smp_list)
                smp_arr = smp_arr_.reshape(smp_arr_.shape[0]*smp_arr_.shape[1],smp_arr_.shape[2],smp_arr_.shape[3],order='F')
                smp_arr = ((smp_arr)  - self.offset)*self.gain
                smp_arr = np.clip(smp_arr,0,np.inf)
                # initial guess

                initial_arr = np.zeros((np.size(smp_arr, 0), 4))
                initial_arr[:, :2] = np.array([initial_guess[0], initial_guess[1]])  # position
                initial_arr[:, 2] = initial_guess[2]  # photons
                initial_arr[:, 3] = np.mean(smp_arr,axis=(-2,-1)) # bg


                n_iterations = np.ceil(np.size(smp_arr,0)/batch_size)
                torch.cuda.empty_cache()

                ratio_all = np.zeros(np.size(smp_arr,0))


                for batch in (range(int(n_iterations))):
                    scaling_factor_map = torch.ones_like(bg_image)

                    smp = smp_arr[batch*batch_size:min(batch*batch_size + batch_size, np.size(smp_arr,0)),:,:]
                    initial = initial_arr[batch*batch_size:min(batch*batch_size + batch_size, np.size(smp_arr,0)),:]
                    smp = smp.astype(np.float16)
                    initial = initial.astype(np.float16)
                    with torch.no_grad():  # when no tensor.backward() is used

                        # convert image to tensor

                        smp_ = torch.Tensor(smp).to(dev)
                        norm, _ = torch.max(torch.max(smp_, 1)[0], -1)
                        smp_arr_forNN = smp_ / norm[..., None, None]
                        smp_arr_forNN = smp_arr_forNN.type(torch.float32)
                        if True: # nn with channels!
                            smp_arr_forNN = torch.reshape(smp_arr_forNN,(int(smp_.size(0)/number_channel),number_channel,self.roisize,self.roisize))

                        predicted_bg = Unet_pp_model(smp_arr_forNN).repeat(1, number_channel, 1, 1) #[:, 0, roi_small[0]:roi_small[1],
                        predicted_bg = predicted_bg * torch.reshape(norm, predicted_bg.shape[0:2])[...,None,None]
                                       #roi_small[0]:roi_small[1]]

                        # smp_= smp_arr_forNN[:, roi_small[0]:roi_small[1],
                        #           roi_small[0]:roi_small[1]]
                        self.roisize=16
                        # bounds estimator
                        param_range = torch.Tensor(bounds).to(dev)

                        predicted_bg = predicted_bg.view(smp_.size())

                        if plot_background == True:

                            # Initialize an empty tensor for the scaling map
                            scaling_map = torch.zeros_like(bg_image)

                            # Kernel dimensions
                            kernel_size = self.roisize
                            half_kernel = kernel_size // 2
                            # The number of positions the kernel can be in horizontally and vertically
                            vertical_positions = scaling_map.shape[1] - kernel_size-1
                            horizontal_positions = scaling_map.shape[2] - kernel_size-1
                            # Iterate over the image and simulate kernel coverage
                            for i_i in range(scaling_map.shape[1]):
                                for j_j in range(scaling_map.shape[2]):
                                    min_i = max(0, i_i - kernel_size+1 )
                                    max_i = min(vertical_positions, i_i )
                                    min_j = max(0, j_j - kernel_size+1)
                                    max_j = min(horizontal_positions, j_j)

                                    scaling_map[:,i_i, j_j] = (max_i - min_i+1) * (max_j - min_j+1)

                            #scaling_map = abs(scaling_map-self.roisize/2*self.roisize/2) +1

                            for i_ii in range(len(rows)):

                                row = rows[i_ii]
                                for j_jj in range(len(collumns)):
                                    col = collumns[j_jj]
                                    for frame_iter in np.arange(self.frames[0], min(self.frames[1], np.size(track_rnp, 0))):
                                        # Calculate the ROI boundaries, ensuring they stay within the image boundaries

                                        bg_image[frame_iter,
                                        int(row - np.ceil(self.roisize / 2)): int(row + np.ceil(self.roisize / 2)),
                                        int(col - np.ceil(self.roisize / 2)): int(col + np.ceil(self.roisize / 2))] += predicted_bg[flag_bg]

                                        flag_bg = flag_bg + 1

                            bg_image2 = bg_image/scaling_map
                            # Your PyTorch operations here (make sure all tensors are properly defined and operations are valid)
                            concatenated_image = torch.concatenate((((torch.tensor(
                                track_rnp.astype(float)) - self.offset) * self.gain).to(self.dev), bg_image2),
                                                                   dim=1).detach().cpu().numpy()[2]

                            # Plotting
                            plt.figure(dpi=1000)
                            plt.imshow(concatenated_image, cmap='gray')  # Use the 'gray' colormap for grayscale
                            plt.axis('off')  # Remove all axes

                            # Save the figure as an SVG file
                            plt.savefig('image_background.svg', format='svg', bbox_inches='tight', pad_inches=0)
                            plt.close()  # Show the plot
                            show_napari(concatenated_image)
                            tifffile.imwrite('background_fullcell.tiff',concatenated_image)
                            # show_tensor(torch.concatenate((((torch.tensor(
                            #     track_rnp.astype(float)) - self.offset) * self.gain).to(self.dev), bg_image2), dim=1))


                        initial_ = torch.Tensor(initial).to(dev)
                        #initial_[:,3] = 1
                        initial_[:, 0:2] = 10/2 #roisize/2


                        ratio, _, _, mu_iandbg, _, traces_bg, traces_i = \
                            glrtfunction(smp_[:,3:13,3:13], batch_size, param_range, initial_, 10,
                                         self.sigma, tol = torch.Tensor([1e-3, 1e-3]).to('cuda'), lambda_=1e-4,
                                         iterations=iterations,bg_constant=predicted_bg[:,3:13,3:13], vector=False)
                        traces_bg = traces_i.detach().cpu().numpy()
                        ratio = ratio.detach().cpu().numpy()

                        ratio_all[int(batch*batch_size):int(batch*batch_size+len(ratio))] = ratio
                def normcdf(x):
                    return 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))

                def fast_harmonic(n):
                    """Returns an approximate value of n-th harmonic number.
                       http://en.wikipedia.org/wiki/Harmonic_number
                    """
                    # Euler-Mascheroni constant
                    gamma = 0.57721566490153286060651209008240243104215933593992
                    return gamma + np.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)

                #ratio = 2 * (loglik_int_all - loglik_bg_all)
                # ratio = np.reshape(ratio, ( (np.size(track_rnp, 0)),
                #     (np.size(track_rnp, 1) - self.roisize), (np.size(track_rnp, 2) - self.roisize)))
                N = len(ratio_all)
                num_groups = (N - 1) // number_channel

                original_array = np.arange(0, N)
                indices_per_group = [np.arange(i, N - (number_channel-1) + i, number_channel) for i in range(num_groups)]




                for frame_iter in np.arange(self.frames[0], min(self.frames[1], np.size(track_rnp,0)-1)):
                    ratio_flat = ratio_all[indices_per_group[frame_iter]]

                    pfa = 2 * normcdf(-np.sqrt(np.clip(ratio_flat, 0, np.inf)))


                    # weird trick for sorting

                    argsort = np.argsort(pfa)
                    aargsort = np.argsort(argsort)
                    pfa_sorted = np.sort(pfa)
                    numtests = len(pfa_sorted)
                    c_m = fast_harmonic(numtests)
                    thres_arr = np.arange(1, numtests + 1, 1)/(numtests*c_m) * alfa
                    #thres_arr = np.arange(1, numtests + 1, 1) / (numtests ) * alfa
                    good_arr_sorted = pfa_sorted <= thres_arr
                    pfa_sorted_adj = numtests * c_m / np.arange(1, numtests + 1, 1) * pfa_sorted


                    pfa_adj = pfa_sorted_adj[aargsort]  # undo sort
                    good_arr = good_arr_sorted[aargsort]

                    mask[frame_iter,:,:] = np.reshape(good_arr, ((np.size(track_rnp,1) - self.roisize), (np.size(track_rnp,2) - self.roisize)))
                    pfa_arr[frame_iter,:,:] = np.reshape(pfa, ((np.size(track_rnp,1) - self.roisize), (np.size(track_rnp,2) - self.roisize)))
                    pfa_adj_arr[frame_iter,:,:] = np.reshape(pfa_adj, ((np.size(track_rnp,1) - self.roisize), (np.size(track_rnp,2) - self.roisize)))

                mask = np.pad(mask, [(0, 0), (int(self.roisize / 2), int(self.roisize / 2)),
                                     (int(self.roisize / 2), int(self.roisize / 2))])
                pfa_arr = np.pad(pfa_arr, [(0, 0), (int(self.roisize / 2), int(self.roisize / 2)),
                                           (int(self.roisize / 2), int(self.roisize / 2))])
                pfa_adj_arr = np.pad(pfa_adj_arr, [(0, 0), (int(self.roisize / 2), int(self.roisize / 2)),
                                                   (int(self.roisize / 2), int(self.roisize / 2))])
                # Save the processed images for this batch
                np.save(self.path + '/mask_batch_' + str(batch_idx), mask)
                # #np.save(self.path + '/pfa_arr_batch_' + str(batch_idx), pfa_arr)
                # np.save(self.path + '/pfa_adj_arr_batch_' + str(batch_idx), pfa_adj_arr)
                # np.save(self.path + '/boundingbox_batch_' + str(batch_idx), track_rnp)
            if batched_frames_list:
                concatenated_mask = concatenate_images(
                    [np.load(self.path + '/mask_batch_' + str(i) + '.npy') for i in range(len(batched_frames_list))])
                # concatenated_pfa_arr = concatenate_images(
                #     [np.load(self.path + '/pfa_arr_batch_' + str(i) + '.npy') for i in range(len(batched_frames_list))])
                # concatenated_pfa_adj_arr = concatenate_images(
                #     [np.load(self.path + '/pfa_adj_arr_batch_' + str(i) + '.npy') for i in range(len(batched_frames_list))])
                # concatenated_boundingbox = concatenate_images(
                #     [np.load(self.path + '/boundingbox_batch_' + str(i) + '.npy') for i in range(len(batched_frames_list))])
                # # Save the concatenated images
                np.save(self.path + '/mask'+ str(qq), concatenated_mask.astype(bool))
                # np.save(self.path + '/boundingbox' + str(qq), concatenated_boundingbox)
                # # np.save(self.path + '/pfa_arr'+ str(qq), concatenated_pfa_arr)
                # np.save(self.path + '/pfa_adj_arr'+ str(qq), concatenated_pfa_adj_arr)
                # # np.save(self.path + '/boundingbox'+ str(qq), concatenated_boundingbox)

                # Delete intermediate arrays
                for batch_idx in range(len(batched_frames_list)):
                    os.remove(self.path + '/mask_batch_' + str(batch_idx) + '.npy')
                    # os.remove(self.path + '/pfa_arr_batch_' + str(batch_idx) + '.npy')
                #     os.remove(self.path + '/pfa_adj_arr_batch_' + str(batch_idx) + '.npy')
                #     os.remove(self.path + '/boundingbox_batch_' + str(batch_idx) + '.npy')
                # all_bbs.append(concatenated_boundingbox)
                all_masks.append(concatenated_mask)
        if not 'concatenated_mask' in locals():
            concatenated_mask = []
            concatenated_boundingbox=[]

        return traces_i, traces_bg, all_masks, all_bbs



    def gauss_mle_fixed_sigma(self, pos_list, lmlambda=1e-4, iterations=100,
                              init_photons=250, init_bg=3,num_channels =20,pfa_check=0.05, interpolated = False, vector=False):

        def custom_slice(i, array_length=1000, slice_length = 100):
            half_length = int(slice_length/2)
            if i - half_length < 0:
                start_index = 0
            elif i + half_length >= array_length:
                start_index = array_length - slice_length
            else:
                start_index = i - half_length

            end_index = start_index + slice_length

            return slice(start_index, end_index)


        Unet_pp_model = Unet_pp(num_channels).to(self.dev)
        Unet_pp_model.load_state_dict(torch.load(self.bg_model)['model_state_dict'])
        Unet_pp_model.eval()
        params_list = []
        roi_pos_list = []
        filterarray_list = []
        crlb_list = []
        pfa_spot_list = []
        smpconcat = np.empty((0, self.roisize, self.roisize))
        muconcat = np.empty((0, self.roisize, self.roisize))
        bg_smp_concat = np.empty((0, self.roisize, self.roisize))
        mu = torch.empty((0, self.roisize, self.roisize)).to(self.dev)
        bg_smp_= torch.empty((0, self.roisize, self.roisize)).to(self.dev)
        pred_bg_concat = np.empty((0, self.roisize, self.roisize))




        for box in range(len(pos_list)):
            pfa_spot = []
            crlb = []
            filterarray = []
            smp = np.empty((0, self.roisize, self.roisize))
            pos = pos_list[box]
            pos_ori = copy.copy(pos)*1

            if np.size(pos) == 0:
                params = np.empty((0, 5))
                roipos = np.empty((0, 3))
            else:
                pos[:, 1] = pos[:, 1] + self.bbox_NE[box, 0]
                pos[:, 2] = pos[:, 2] + self.bbox_NE[box, 2]
                pos_ori[:, 1] = pos_ori[:, 1] + self.bbox_NE[box, 0]
                pos_ori[:, 2] = pos_ori[:, 2] + self.bbox_NE[box, 2]


                track_rnp = tifffile.imread(self.path + self.fn_track_rnp)
                track_rnp = (track_rnp - self.offset) * self.gain

                roisize = self.roisize

                pos = np.round(np.float64(pos))

                smp = np.zeros((np.shape(pos)[0], roisize, roisize))
                bg_smp = np.zeros((np.shape(pos)[0], roisize, roisize))
                roipos = np.zeros(np.shape(pos))
                flag = 0

                for i in range(np.shape(pos)[0]):
                    time, row, col = pos[i, :]

                    smp[flag, :, :] = track_rnp[int(time),
                                      int(row - np.ceil(roisize / 2)): int(row + np.ceil(roisize / 2)),
                                      int(col - np.ceil(roisize / 2)): int(col + np.ceil(roisize / 2))]
                    bg = torch.tensor(track_rnp[custom_slice(int(time),np.shape(track_rnp)[0],num_channels),
                                      int(row - np.ceil(roisize / 2)): int(row + np.ceil(roisize / 2)),
                                      int(col - np.ceil(roisize / 2)): int(col + np.ceil(roisize / 2))]).to(self.dev)
                    norm, _ = torch.max(torch.max(bg, 1)[0], -1)
                    smp_arr_forNN = bg / norm[..., None, None]
                    smp_arr_forNN = smp_arr_forNN.type(torch.float32)[None,...]

                    predicted_bg = Unet_pp_model(smp_arr_forNN)
                    predicted_bg = predicted_bg * torch.mean(norm)
                    bg_smp[flag, :, :] = predicted_bg.detach().cpu().numpy()
                    roipos[flag, :] = np.array([time, int(row - np.ceil(roisize / 2)), int(col - np.ceil(roisize / 2))])
                    flag = flag + 1



                dev = self.dev
                bounds_mle = [[4, roisize - 4],
                              [4, roisize - 4],
                              [1, 1e9],
                              [1, 1e6]
                              ]

                initial_guess_mle = [roisize / 2, roisize / 2, init_photons, init_bg]
                pos_ori[:,[1,2]]= pos_ori[:,[1,2]]-roipos[:,[1,2]]

                smp_ = torch.from_numpy(smp).to(dev)

                # bounds estimator
                param_range = torch.Tensor(bounds_mle).to(dev)

                # initial guess
                initial = np.zeros((np.size(smp, 0), 4))
                if interpolated:
                    initial[:, :2] = pos_ori[:,[1,2]] # position
                else:

                    initial[:, :2] = np.array([initial_guess_mle[0], initial_guess_mle[1]])  # position
                initial[:, 2] = initial_guess_mle[2]  # photons
                initial[:, 3] = np.mean(bg_smp,axis=(-1,-2))  # bg

                initial_ = torch.Tensor(initial).to(self.dev)
                if interpolated:
                    model = Gaussian2DFixedSigmaPSFFixedPos(self.roisize, sigma=self.sigma)

                else:
                    model = Gaussian2DFixedSigmaPSF(self.roisize, sigma=self.sigma)

                if interpolated:
                    mle = LM_MLE_with_iter(model, lambda_=lmlambda, iterations=iterations,
                                           param_range_min_max=param_range[[2,3],:],
                                           tol=torch.tensor([ 1e-2, 1e-2]).to(self.dev))

                else:
                    mle = LM_MLE_with_iter(model, lambda_=lmlambda, iterations=iterations,
                                           param_range_min_max=param_range,
                                           tol=torch.tensor([1e-3, 1e-3, 1e-2, 1e-2]).to(self.dev))
                #mle = torch.jit.script(mle)
                bg_smp_ = torch.tensor(bg_smp).to(self.dev)
                smp_ = smp_.to(self.dev)

                if interpolated:
                    params_, loglik_, traces = mle.forward(smp_.type(torch.float32),
                                                           initial_.type(torch.float32)[:,2::], bg_smp_.type(torch.float32), pos=initial_[:, :2])
                    params_ = torch.concatenate((initial_[:, :2], params_), dim=1)
                    mu, jac = model.forward(params_, bg_smp_.type(torch.float32))

                else:
                    params_, loglik_, traces = mle.forward(smp_.type(torch.float32),
                                                           initial_.type(torch.float32), bg_smp_.type(torch.float32)
                                                           )
                    trac_test = traces.detach().cpu().numpy()
                    mu,jac= model.forward(params_,bg_smp_)



                # show_tensor(torch.concatenate((smp_, mu,mu2 ), dim=-1))
                # show_tensor(torch.concatenate((jac[...,1], jacvec[...,1]), dim=-2))

                # test_params = params_.detach().cpu().numpy()
                # traces = traces.detach().cpu().numpy()
                # if pos_fixed:
                #     pos_ori[:, 1] = pos_ori[:, 1]+ self.bbox_NE[box, 0]
                #     pos_ori[:, 2] = pos_ori[:, 2] + self.bbox_NE[box, 2]
                #     pos_ori[:, 1] = pos_ori[:, 1] -roipos[:,1]
                #     pos_ori[:, 2] = pos_ori[:, 2] -roipos[:,2]
                #     params_[:,0:2]= torch.tensor(np.array(pos_ori[:,1:3],dtype=float), dtype=params_.dtype, device=params_.device)
                #mu,_ = gauss_psf_2D_fixed_sigma(params_,self.roisize,sigma, bg_smp_.type(torch.float32))

                #show_tensor(torch.concatenate((smp_,mu),dim=-1))
                # else:
                #     params_, loglik_, chi = mle.forward(smp_, initial_, None)
                #     filterarray = np.ones(np.size(smp,0))
                #     filterarray = filterarray != 0

                ratio, _, _, mu_iandbg, _, traces_bg, traces_i = \
                    glrtfunction(smp_[:, :, :].type(torch.float32), 1000, param_range, params_, 16,
                                 self.sigma, tol=torch.Tensor([1e-3, 1e-3]).to('cuda'), lambda_=1e-4,
                                 iterations=iterations, bg_constant=bg_smp_.type(torch.float32),use_pos=True,vector=vector)
                # ensure crlb make sense restimate with all parameters
                if interpolated:
                    initial = np.zeros((np.size(smp, 0), 4))
                    initial[:, :2] = np.array([initial_guess_mle[0], initial_guess_mle[1]])  # position
                    initial[:, 2] = initial_guess_mle[2]  # photons
                    initial[:, 3] = np.mean(bg_smp, axis=(-1, -2))  # bg
                    initial_ = torch.Tensor(initial).to(self.dev)
                    model = Gaussian2DFixedSigmaPSF(self.roisize, sigma=self.sigma)
                    mle = LM_MLE_with_iter(model, lambda_=lmlambda, iterations=iterations,
                                           param_range_min_max=param_range,
                                           tol=torch.tensor([1e-3, 1e-3, 1e-2, 1e-2]).to(self.dev))
                    params_crlbonly, _, _ = mle.forward(smp_.type(torch.float32),
                                                           initial_.type(torch.float32), bg_smp_.type(torch.float32)
                                                           )

                    mu_crlbonly, jac_crlbonly = model.forward(params_crlbonly, bg_smp_)
                    crlb = compute_crlb(mu_crlbonly.type(torch.float32), jac_crlbonly.type(torch.float32))
                    crlb = crlb.detach().cpu().numpy()
                else:
                    crlb = compute_crlb(mu.type(torch.float32), jac.type(torch.float32))
                    crlb = crlb.detach().cpu().numpy()
                #tracesbg = traces_i.detach().cpu().numpy()
                def normcdf(x):
                    return 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))


                pfa_spot = 2 * normcdf(-np.sqrt(np.clip(ratio.detach().cpu().numpy(), 0, np.inf)))
                argsort = np.argsort(pfa_spot)
                aargsort = np.argsort(argsort)
                pfa_sorted = np.sort(pfa_spot)
                numtests = len(pfa_sorted)
                c_m = fast_harmonic(numtests)
                thres_arr = np.arange(1, numtests + 1, 1) / (numtests * c_m) * pfa_check
                # thres_arr = np.arange(1, numtests + 1, 1) / (numtests ) * alfa
                good_arr_sorted = pfa_sorted <= thres_arr
                pfa_sorted_adj = numtests * c_m / np.arange(1, numtests + 1, 1) * pfa_sorted

                pfa_adj = pfa_sorted_adj[aargsort]  # undo sort
                good_arr = good_arr_sorted[aargsort]

                # Pd = 1 - 0.5 * (scipy.special.erf(
                #     (-np.sqrt(LRT_bin) + np.sqrt(gamma / 2)) / np.sqrt(2)) - scipy.special.erf(
                #     (-np.sqrt(LRT_bin) - np.sqrt(gamma / 2)) / np.sqrt(2)))




                params = params_.cpu().detach().numpy()
                ## filter crap
                #filterarray0 = chisqr < chi_thresh

                if interpolated:
                    filterarray1 = params[:, 2] > 0
                    filterarray2 = params[:, 0] > roisize/2-2
                    filterarray3 = params[:, 0] < roisize/2+2
                    filterarray4 = params[:, 1] > roisize/2-2
                    filterarray5 = params[:, 1] < roisize/2+2

                    # filterarray6 = params[:, 3] < max_bg
                    filterarray7 = good_arr
                    filterarray = np.logical_and.reduce(
                        (filterarray1, filterarray2, filterarray3, filterarray4, filterarray5, filterarray7))
                else:
                    filterarray1 = params[:, 2] > 0
                    filterarray2 = params[:, 0] > bounds_mle[0][0] * 1.2
                    filterarray3 = params[:, 0] < bounds_mle[0][1] * 0.8
                    filterarray4 = params[:, 1] > bounds_mle[1][0] * 1.2
                    filterarray5 = params[:, 1] < bounds_mle[1][1] * 0.8
                    filterarray6 = crlb[:, 0] < 1e5
                    filterarray7 = good_arr
                    filterarray = np.logical_and.reduce(
                        ( filterarray1, filterarray2, filterarray3, filterarray4, filterarray5,filterarray6,filterarray7))



                #loglik_spots = loglik_.cpu().detach().numpy()
                torch.cuda.empty_cache()
            filterarray_list.append(filterarray)
            params_list.append(params[filterarray, :])
            roi_pos_list.append(roipos[filterarray, :])

            if len(pfa_spot)>0:
                pfa_spot_list.append(pfa_spot[filterarray])
                crlb_list.append(crlb[filterarray, :])
            else:
                pfa_spot_list.append(pfa_spot)
                crlb_list.append(crlb)
            smpconcat = np.concatenate((smpconcat, smp[filterarray, :, :]), axis=0)
            #show_napari(smp_.cpu().detach().numpy()[filterarray,...])
            muconcat = np.concatenate((muconcat, mu.detach().cpu().numpy()[filterarray, :, :]), axis=0)
            bg_smp_concat = np.concatenate((bg_smp_concat, bg_smp_.detach().cpu().numpy()[filterarray, :, :]))
        self.results = pd.DataFrame()
        self.results['params'] = params_list
        self.results['roi_pos_list'] = roi_pos_list
        self.results['crlb_list'] = crlb_list

        return muconcat, roi_pos_list,pfa_spot_list, smpconcat, bg_smp_concat

    def tracking(self, linking=3, memory=5, movie=True, min_dist=5, drift_correct=True, pfa_arr = None):
        # Define a function to filter tracks by keeping the longer one when they are close
        # def filter_close_tracks_in_frame(tracks, min_distance):
        #     track_groups = tracks.groupby('frame')
        #     filtered_tracks = []
        #
        #     for name, group in track_groups:
        #         if len(group) == 1:
        #             filtered_tracks.append(group)
        #         else:
        #             distances = cdist(group[['x', 'y']], group[['x', 'y']])
        #             np.fill_diagonal(distances, np.inf)
        #             close_tracks = np.any(distances < min_distance, axis=1)
        #             if np.any(close_tracks):
        #             # Calculate the duration of each track
        #                 durations = group['frame'].max() - group['frame'].min()
        #                 # Find the index of the longest track
        #                 longest_track_index = np.argmax(durations)
        #                 # Keep the longest track
        #                 filtered_tracks.append(group[group.index == longest_track_index])
        #             else:
        #                 # If no tracks are close, keep all of them
        #                 filtered_tracks.append(group)
        #
        #     return pd.concat(filtered_tracks)
        def insert_rows(df):
            # if 'frame' in df.index.names:
            #     df = df.reset_index()
            df['interpolated'] = 0
            df_add = pd.DataFrame(columns=df.columns)
            for particle_id, group in df.groupby('particle'):
                min_frame, max_frame = group['frame'].min(), group['frame'].max()

                for frame in range(min_frame, max_frame):
                      if np.sum(group['frame'] == frame) == 0:

                          #df_add.append({'frame': frame}, ignore_index=True)
                          new_row = pd.DataFrame({
    'frame': [frame],
    'NE': [group['NE'].values[0]],  # Replace NE_value with the actual value
    'scale':  [group['scale'].values[0]],  # Replace scale_value with the actual value
    'angle':  [group['angle'].values[0]],  # Replace angle_value with the actual value
    'trans_x':  [group['trans_x'].values[0]],  # Replace trans_x_value with the actual value
    'trans_y':  [group['trans_y'].values[0]],  # Replace trans_y_value with the actual value
    'y': np.NaN,  # Replace y_value with the actual value
    'x': np.NaN,  # Replace x_value with the actual value
    'signal':  [group['signal'].values[0]],  # Replace signal_value with the actual value
     'bg': [group['bg'].values[0]],
    'particle': [particle_id],
    'interpolated': [1],
  'crlbx': [group['crlbx'].values[0]],  # Replace x_value with the actual value
  'crlby': [group['crlby'].values[0]],  # Replace signal_value with the actual value
  'crlbsignal':[group['crlbsignal'].values[0]],
  'crlbbg': [group['crlbbg'].values[0]]

})

                          df_add = pd.concat([df_add, new_row], ignore_index=True)

                #         group.loc[insert]['frame'] = frame
                # new_group += [group]
            df.reset_index(drop=True, inplace=True)
            df_add.reset_index(drop=True, inplace=True)

            new_df = pd.concat([df, df_add], ignore_index=True)
            new_df = new_df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
            #new_df_inter  =interpolate_df(new_df)
            return new_df

        # Function to interpolate between frames
        def interpolate_df(df):
            # Check if 'frame' is in the index, and if so, reset it
            # if 'frame' in df.index.names:
            #     df = df.reset_index()

            df_new = insert_rows(df)
            df_new = df_new.sort_values(by=['particle', 'frame'])

            interpolated_data = []

            for particle_id, particle_group in df_new.groupby('particle'):
                particle_group = particle_group.interpolate(method='linear', axis=0)
                interpolated_data.append(particle_group)

            interpolated_df = pd.concat(interpolated_data, ignore_index=True)

            return interpolated_df

        mpl.use('Agg')
        np.save(self.resultprefix + 'bbox.npy', self.bbox_NE)

        params_list = self.results['params']
        roi_pos_list = self.results['roi_pos_list']
        trackdata_list = []


        for i in range(len(params_list)):
            trackdata = []

            if np.size(params_list[i]) != 0:
                trackdata = pd.DataFrame()
                boundingbox_NE = self.bbox_NE[i, :]
                param = params_list[i]
                roipos = roi_pos_list[i]

                trackdata['frame'] = roipos[:, 0]
                if pfa_arr is not None:
                    pfa_singlecell = pfa_arr[i]
                    trackdata['pfa'] = pfa_singlecell
                trackdata['NE'] = np.ones_like(roipos[:, 0])*i
                trackdata['scale'] = np.ones_like(roipos[:, 0]) * self.registration['scale']
                trackdata['angle'] = np.ones_like(roipos[:, 0]) * self.registration['angle']
                trackdata['trans_x'] = np.ones_like(roipos[:, 0]) * self.registration['tvec'][1]
                trackdata['trans_y'] = np.ones_like(roipos[:, 0]) * self.registration['tvec'][0]
                trackdata['y'] = param[:, 0] + roipos[:, 1] - boundingbox_NE[0]
                trackdata['x'] = param[:, 1] + roipos[:, 2] - boundingbox_NE[2]

                trackdata['signal'] = param[:, 2]
                trackdata['bg'] = param[:, 3]
                trackdata['crlbx'] = self.results['crlb_list'][i][:, 1]
                trackdata['crlby'] = self.results['crlb_list'][i][:, 0]
                trackdata['crlbsignal'] = self.results['crlb_list'][i][:, 2]
                trackdata['crlbbg'] = self.results['crlb_list'][i][:, 3]

                tp.quiet()
                # test
                # trackdata = pd.DataFrame()
                # trackdata['frame'] =np.array([1,3,4,5])
                # trackdata['y'] = np.array([10,10,10,10])
                # trackdata['x'] = np.array([11,13,14,15])
                # linking = 2.01
                # menory = 2
                # end - test
                trackdata = tp.link(trackdata, search_range=linking, memory=memory,link_strategy= 'auto')

                trackdata = tp.filtering.filter_stubs(trackdata, threshold=min_dist)
                if pfa_arr is None:
                    try:
                        trackdata = interpolate_df(trackdata)
                    except:
                        print('nothing to interpolate')

                with open(self.resultprefix + 'npc_points' + str(i) + '.data', "wb") as filename:
                    pickle.dump(self.pointsnpc[i], filename)
                with open(self.resultprefix + 'amplitude_points' + str(i) + '.data', "wb") as filename:
                    pickle.dump(self.amplitude_array_tot[i], filename)
                with open(self.resultprefix + 'sigma_points' + str(i) + '.data', "wb") as filename:
                    pickle.dump(self.sigma_array_tot[i], filename)

                # np.save(self.resultprefix + 'timepoints.npy', self.time_list_npc)




                if movie and not trackdata.empty:
                    imgs = glob.glob(self.path + self.fn_track_rnp)

                    t1 = trackdata * 1
                    frames = tifffile.imread(imgs)
                    frames = frames[self.frames[0]: self.frames[1], int(boundingbox_NE[0]):int(boundingbox_NE[1]),
                             int(boundingbox_NE[2]):int(boundingbox_NE[3])]
                    arr = []

                    print('\n Print videos for tracking NE ' + str(i + 1) + '/' + str(self.bbnum) + '\n')
                    time = 0

                    for qq in range(np.shape(frames)[0]):
                        frame = frames[qq, :, :]

                        plt.ioff()
                        plt.cla()
                        plt.clf()
                        plt.close('all')
                        fig = plt.figure(1, figsize=(8, 8))
                        if drift_correct:
                            plt.scatter(np.concatenate(self.pointsnpc[i])[0, :] - self.xdrift[qq] - self.bbox_NE[i][2],
                                        np.concatenate(self.pointsnpc[i])[1, :] - self.ydrift[qq]- self.bbox_NE[i][0], linewidth=3,
                                        color='red')
                        else:
                            plt.scatter(np.concatenate(self.pointsnpc[i])[0, :] - self.bbox_NE[i][2],
                                        np.concatenate(self.pointsnpc[i])[1, :] - self.bbox_NE[i][0], linewidth=3,
                                        color='red')
                        plt.imshow(frame)

                        if np.size(t1.query('frame<={0}'.format(qq) + '& frame>={0}'.format(max(0, qq - 5)))) != 0:
                            axes = tp.plot_traj(
                                t1.query('frame<={0}'.format(qq) + '& frame>={0}'.format(max(0, qq - 5))),
                                plot_style={'linewidth': 5})

                        arr.append(self.cvtFig2Numpy(fig))
                        plt.close('all')

                    self.makevideoFromArray(self.resultprefix + 'tracks' + str(i) + '.mp4', arr, fps=50)
                if drift_correct:
                    trackdata['y'] = trackdata['y'].values +  self.ydrift[
                        np.int32(trackdata['frame'].values)]
                    trackdata['ydrift'] = self.ydrift[
                        np.int32(trackdata['frame'].values)]
                    trackdata['x'] = trackdata['x'].values + self.xdrift[
                        np.int32(trackdata['frame'].values)]
                    trackdata['xdrift']=self.xdrift[
                        np.int32(trackdata['frame'].values)]


            trackdata_list.append(trackdata)
        self.track_results = trackdata_list

        return trackdata_list



    def compute_distancev2(self,smooth_factor=1.6):


        all_tracks = []

        for cell in range(len(self.track_results)):
            trackresults = self.track_results[cell] * 1
            tracks_percell = []
            if np.size(trackresults) != 0:
                numtracks = max(trackresults['particle'].values) + 1  # particle start at 0
                for track in range(numtracks):
                    temp = trackresults[trackresults['particle'] == track]
                    if np.size(temp, 0) <= 3:
                        filter = 1  # Do nothing
                    else:
                        tracks_percell.append(temp)

            all_tracks.append(tracks_percell)

        pd.options.mode.chained_assignment = None

        amplitude_cells = self.amplitude_array_tot
        sigma_cells = self.sigma_array_tot
        cells = self.pointsnpc
        closed_condition = self.closednpc
        deriv_cells = self.derivnpc

        for cell in range(len(cells)):
            groups = cells[cell].copy()
            amplitude_percell = amplitude_cells[cell].copy()
            sigma_percell = sigma_cells[cell].copy()
            closed_groups = closed_condition[cell].copy()
            deriv_groups = deriv_cells[cell].copy()

            npc_points = np.hstack(groups)
            deriv_points = np.hstack(deriv_groups)
            closed_points = np.hstack(closed_groups)
            amplitude_percell = np.hstack(amplitude_percell)
            sigma_percell = np.hstack(sigma_percell)

            npc_points[0, :] = npc_points[0, :] - self.bbox_NE[cell][2]
            npc_points[1, :] = npc_points[1, :] - self.bbox_NE[cell][0]

            tracks = all_tracks[cell]

            npc_points_torch = torch.tensor(npc_points, dtype=torch.float32, device=self.dev)
            deriv_points_torch = torch.tensor(deriv_points, dtype=torch.float32, device=self.dev)
            closed_points_torch = torch.tensor(closed_points, dtype=torch.float32, device=self.dev)

            for qq in range(len(tracks)):

                track = tracks[qq]
                track.loc[:, 'def_inout'] = np.nan

                track.loc[:, 'may_inout'] = np.nan
                track.loc[:, 'dist'] = np.nan
                track.loc[:, 'intersect'] = np.nan

                track.loc[:, 'def_inout_spline'] = np.nan

                track.loc[:, 'may_inout_spline'] = np.nan
                track.loc[:, 'dist_spline'] = np.nan
                track.loc[:, 'intersect_spline'] = np.nan
                frame = track['frame'].values
                x = track['x'].values
                y = track['y'].values

                # Fit splines for x and y coordinates with respect to the frame
                spline_x = UnivariateSpline(frame, x, s=smooth_factor)
                spline_y = UnivariateSpline(frame, y, s=smooth_factor)

                # Generate spline-fitted x and y values for each frame
                x_fitted = spline_x(frame)
                y_fitted = spline_y(frame)
                track.loc[:, 'x_spline'] = x_fitted
                track.loc[:, 'y_spline'] = y_fitted

                pos = np.array([track['x'], track['y']])
                pos_spline = np.array([track['x_spline'], track['y_spline']])
                timepoints = np.array([track['frame']])

                pos_torch = torch.tensor(pos).to(self.dev)
                pos_spline_torch = torch.tensor(pos_spline).to(self.dev)

                timepoints_torch = torch.tensor(timepoints, dtype=torch.int32, device=self.dev)

                # Calculate distances for all pairs of points using broadcasting
                distances = torch.norm(npc_points_torch[:, None, :] - pos_torch[:, :, None], dim=0)
                distances_spline = torch.norm(npc_points_torch[:, None, :] - pos_spline_torch[:, :, None], dim=0)

                # Find the indices of the minimum distances
                indices = torch.argmin(distances, dim=1)
                indices_spline = torch.argmin(distances_spline, dim=1)
                closest_amplitude = amplitude_percell[indices_spline.detach().cpu().numpy()]
                closest_sigma = sigma_percell[indices_spline.detach().cpu().numpy()]
                average_amplitude = np.ones_like(closest_amplitude)*np.mean(amplitude_percell)
                average_sigma = np.ones_like(closest_amplitude) * np.mean(sigma_percell)

                # Use indices to get the closest distances
                closest_distances = distances[torch.arange(0,len(indices)),indices]
                closest_distances_spline = distances_spline[torch.arange(0, len(indices_spline)), indices_spline]
                # testpoints = np.array([[25,35],[22,24]])
                # plt.close('all')
                # plt.scatter(npc_points[1,:], npc_points[0,:])
                # plt.scatter(pos[0, :], pos[1, :])
                # plt.savefig('aaatest.png')
                # in_out_array = skimage.measure.points_in_poly(testpoints.T, np.array([npc_points[1, :],
                #                                           npc_points[0, :]]).T).astype(int)
                if closed_points_torch[0] == 1:
                    in_out_array = skimage.measure.points_in_poly(pos.T, np.array([npc_points[0, :],
                                                          npc_points[1, :]]).T).astype(int)
                    in_out_array_spline = skimage.measure.points_in_poly(pos_spline.T, np.array([npc_points[0, :],
                                                                                   npc_points[1, :]]).T).astype(int)

                    track.loc[track['frame'] == timepoints[0, :], 'def_inout'] = in_out_array
                    track.loc[track['frame'] == timepoints[0, :], 'def_inout_spline'] = in_out_array_spline
                else:
                    tck, u = splprep([npc_points[0, :], npc_points[1, :]], s=0, per=True,
                                     quiet=3)  # z=None because we don't have z-values
                    spl_values = splev(np.linspace(0, 1, 2000), tck)
                    in_out_array = skimage.measure.points_in_poly(pos.T, np.array([spl_values[0],
                                                                                   spl_values[1]]).T).astype(int)
                    in_out_array_spline = skimage.measure.points_in_poly(pos_spline.T, np.array([spl_values[0],
                                                                                   spl_values[1]]).T).astype(int)

                    track.loc[track['frame'] == timepoints[0, :], 'may_inout'] = in_out_array
                    track.loc[track['frame'] == timepoints[0, :], 'may_inout_spline'] = in_out_array_spline
                    track.loc[track['frame'] == timepoints[0, :], 'closest_index'] = (indices/(distances.size(1)-1)).detach().cpu().numpy()
                track.loc[track['frame'] == timepoints[0, :], 'dist'] = closest_distances.detach().cpu().numpy()
                track.loc[track['frame'] == timepoints[0, :], 'dist_spline'] = closest_distances_spline.detach().cpu().numpy()
                if np.sum(np.isnan(average_amplitude))>0:
                    testtttt=0
                track.loc[track['frame'] == timepoints[0, :], 'avg_amplitude'] = average_amplitude
                track.loc[track['frame'] == timepoints[0, :], 'closest_amplitude'] = closest_amplitude
                track.loc[track['frame'] == timepoints[0, :], 'avg_sigma'] = average_sigma
                track.loc[track['frame'] == timepoints[0, :], 'closest_sigma'] = closest_sigma

                if pos.shape[1] > 3:
                    A = (npc_points_torch[:, :-1])[:,1:-1].t()
                    B = (npc_points_torch[:, 1:])[:,1:-1].t()

                    C = pos_torch[:, :-1].t()
                    D = pos_torch[:, 1:].t()
                    # Assuming you have A, B, C, D tensors as mentioned before

                    for i in range(C.size(0)):

                        bool_inter = intersect(A, B, C[i], D[i])
                        track.loc[
                            track['frame'] == timepoints[0, i], 'intersect'] = bool_inter
                if pos_spline.shape[1] > 3:
                    A = (npc_points_torch[:, :-1])[:,1:-1].t()
                    B = (npc_points_torch[:, 1:])[:,1:-1].t()

                    C = pos_spline_torch[:, :-1].t()
                    D = pos_spline_torch[:, 1:].t()
                    # Assuming you have A, B, C, D tensors as mentioned before

                    for i in range(C.size(0)):

                        bool_inter = intersect(A, B, C[i], D[i])
                        track.loc[
                            track['frame'] == timepoints[0, i], 'intersect_spline'] = bool_inter

                all_tracks[cell][qq] = track
        self.all_tracks = all_tracks

        with open(self.resultprefix + 'tracks.data', 'wb') as filehandle:
            pickle.dump(all_tracks, filehandle)

        return all_tracks

    # Function to calculate curvature direction

    def fit_per_meanforprecision(self, npc_mean, deriv, length_line, bounds, number_points, sampling, groups, points, movie,
                     sampling_normal, registration, index, smoothness, Lambda, iterations=300,max_gap_good_fits = 100, npc_mean2 =None ):

        def intensity_func_sig(t, A, K, B, C, nu, Q, M, mu, sigma, amplitude,offset):
            exponent_richards = -B * (t - M)
            denominator_richards = C + Q * np.exp(exponent_richards)
            power_richards = 1 / nu
            richards_curve = A + (K - A) / denominator_richards ** power_richards

            exponent_gaussian = -(t - mu) ** 2 / (2 * sigma ** 2)
            gaussian_curve = amplitude * np.exp(exponent_gaussian)

            combined_curve = richards_curve + gaussian_curve+ offset

            return combined_curve


        points_allcells = []
        deriv_allcells = []
        closed_condition_allcells = []
        params_allcells = []
        params2_allcells = []
        errors_allcells = []
        values_allcells = []
        for i in tqdm.tqdm(range(len(groups)), desc='refined fitting nuclei'):

            tangent_slopes = deriv[i][0]
            normal_slopes = np.array([-tangent_slopes[1, :], tangent_slopes[0, :]])
            points_it = points[i][0]
            group = groups[i]
            selection = np.array([])
            points_new_percell = []
            deriv_percell = []
            closed_condition_percell = []
            params_percell = []
            params2_percell = []

            errors_percell = []
            values_percell = []
            errorflag = 0

            # how much points we need to consider (1 means all points)

            selection = np.arange(group[0], group[1], number_points).astype(int)
            # selection = np.array([1])#########################################

            num = sampling_normal  # sampling of normal line
            zi_array = np.zeros((len(selection), num))
            zi_array2 = np.zeros((len(selection), num))
            dist_array = np.zeros((len(selection), num))
            initial = np.zeros((len(selection), 7))
            #temp = bounds[0] * 1
            #bounds.append(temp)
            for k in range(len(selection)):
                select = selection[k]
                point = points_it[:, select]
                normal_slope = normal_slopes[:, select] / np.linalg.norm(normal_slopes[:, select])

                start = point - length_line * normal_slope
                end = point + length_line * normal_slope
                # -- Extract the line...
                # Make a line with "num" points...
                x0, y0 = start[0], start[1]
                x1, y1 = end[0], end[1]
                if x0 < 0 or y0 < 0 or x1 >= self.imgshape[1] - 1 or y1 >= self.imgshape[2] - 1 or errorflag == 1:
                    errorflag = 1
                    zi_array = np.empty((1, 1))
                    zi_array2 = np.empty((1, 1))
                    dist_array = np.empty((1, 1))

                else:

                    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
                    # Extract the values along the line

                    zi = npc_mean[np.round(y).astype(int), np.round(x).astype(int)]#########################
                    zi2 = npc_mean2[np.round(y).astype(int), np.round(x).astype(int)]
                    dist_alongline = np.cumsum(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))
                    dist_alongline = np.append(0, dist_alongline)
                    zi_array[k, :] = zi
                    zi_array2[k, :] = zi2
                    dist_array[k, :] = dist_alongline


                        # initial_guess = [length_line, 1.5, max(zi) - min(zi), min(zi), -0.6, zi[0] - zi[-1],length_line]
                        # boundscurvefit = (np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
                        # try:
                        #     initial[k,:], _=scipy.optimize.curve_fit(intensity_func_sig,dist_array[k, :], zi_array[k, :] ,p0=initial_guess, bounds=boundscurvefit )
                        # except:
                        #     initial[k, :], _ = scipy.optimize.curve_fit(intensity_func_sig, dist_array[k, :],
                        #                                                 zi_array[k, :], p0=initial_guess)

                        #initial[k,:] = scipy.optimize.minimize(LL_intensity_func_sig, dist_array[k, :],initial_guess, method=('L-BFGS-B'))


            if errorflag == 0:

                dev = torch.device('cuda')

                # with torch.no_grad():  # when no tensor.backward() is used
                # convert image to tensor
                smp = torch.from_numpy(zi_array).to(dev)
                smp2 = torch.from_numpy(zi_array2).to(dev)

                def construct_initial_guess(x, zi):
                    def line(x, zi, range):
                        a = (zi[:, -1] - zi[:, 0]) / (range[:, -1] - range[:, 0])
                        b = zi[:, 0]
                        return a[..., None] * x + b[..., None]

                    area = torch.sum(zi - line(x, zi, x), dim=-1) * (x[:, 1] - x[:, 0])
                    amp = zi[:, int(x.shape[1] / 2)] - line(x[:, int(x.shape[1] / 2)], zi, x)[:, 0]

                    initial_guess = torch.zeros((x.shape[0], len(bounds))).to(dev)
                    initial_guess[:, 0] = zi[:, 0] #A
                    #initial_guess[:, 1] = ((zi[:, -1] - zi[:, 0]) + torch.min(zi, dim=-1)[0]) #K
                    initial_guess[:, 1] = zi[:, -1]

                    initial_guess[:, 2] = 0.5 #B
                    initial_guess[:, 3] = 1# C
                    initial_guess[:, 4] = 1 #nu
                    initial_guess[:, 5] = 1 #Q
                    initial_guess[:, 6] = length_line #M
                    initial_guess[:, 7] = length_line # mu
                    initial_guess[:, 8] = area / amp * 0.39 # sigma
                    initial_guess[:, 9] = amp # amp
                    initial_guess[:, 10] = 0 # offset
                    return initial_guess
                initial_guess = construct_initial_guess(torch.from_numpy(dist_array).to(dev),smp)
                initial_guess2 = construct_initial_guess(torch.from_numpy(dist_array).to(dev), smp2)
                #initial = np.concatenate((initial, initial[:, 0][..., None]), axis=1)
                model = npcfit_class(torch.from_numpy(dist_array).to(dev))
                # mu, _ = model.forward(initial_guess, torch.ones_like(torch.from_numpy(dist_array).to(dev), dtype=bool)[:, 0])
                # real = smp.detach().cpu().numpy()
                # mu = mu.detach().cpu().numpy()
                # plt.figure(dpi=400)
                # plt.plot(dist_array[10,:],real[10,:])
                #
                # plt.plot(dist_array[10,:],mu[10, :])
                # plt.xlabel(r'Distance along normal $n$ [px]')
                # plt.ylabel(r'Intensity $I$ [au]')
                # plt.tight_layout()
                # plt.show()

                bounds_ext = bounds * 1
                #temp = bounds[0] * 1
                #bounds_ext.append(temp)
                param_range = torch.Tensor(bounds).to(dev)





                mle = LM_MLE_forspline_new(model)


                # mle = torch.jit.script(mle)
                params_, _, traces = mle.forward(initial_guess.type(torch.cuda.FloatTensor),
                                                 smp[..., None].type(torch.cuda.FloatTensor),
                                                 param_range, iterations, Lambda)
                params2_, _, traces2 = mle.forward(initial_guess2.type(torch.cuda.FloatTensor),
                                                 smp2[..., None].type(torch.cuda.FloatTensor),
                                                 param_range, iterations, Lambda)
                traces = traces.cpu().detach().numpy()
                traces2 = traces2.cpu().detach().numpy()
                params = params_.cpu().detach().numpy()
                params2 = params2_.cpu().detach().numpy() ###############


                # --------------------filter based on iterations for params 1----------------------------------
                iterations_vector = np.zeros(np.size(params, 0))
                for loc in range(np.size(params, 0)):
                    try:
                        iterations_vector[loc] = np.where(traces[:, loc, 0] == 0)[0][0]
                    except:
                        iterations_vector[loc] = iterations
                # --------------------filter based on iterations for params 2----------------------------------
                iterations_vector2 = np.zeros(np.size(params2, 0))
                for loc in range(np.size(params, 0)):
                    try:
                        iterations_vector2[loc] = np.where(traces2[:, loc, 0] == 0)[0][0]
                    except:
                        iterations_vector2[loc] = iterations

                # --------------------filter based on border for params 1----------------------------------
                filter_border1 = False
                for coll in range(np.size(params, 1)):
                    test1 = np.logical_or(params[:, coll] < bounds_ext[coll][0] * 0.95,
                                          params[:, coll] > bounds_ext[coll][1] * 0.95)
                    if coll > 0:
                        filter_border1 = np.logical_or(filter_border1, test1)
                filter_border1 = np.logical_or(filter_border1, iterations_vector==iterations)

                # --------------------filter based on border for params 2----------------------------------
                filter_border2 = False
                for coll in range(np.size(params2, 1)):
                    test2 = np.logical_or(params2[:, coll] < bounds_ext[coll][0] * 0.95,
                                          params2[:, coll] > bounds_ext[coll][1] * 0.95)
                    if coll > 0:
                        filter_border2 = np.logical_or(filter_border2, test2)
                filter_border2 = np.logical_or(filter_border2, iterations_vector2==iterations)

                filter_border = np.logical_or(filter_border1, filter_border2)

                params = params[np.invert(filter_border), :]
                selection = selection[np.invert(filter_border),]
                dist_array = dist_array[np.invert(filter_border),]
                zi_array = zi_array[np.invert(filter_border),]
                params2 = params2[np.invert(filter_border), :]
                zi_array2 = zi_array2[np.invert(filter_border),]


                if len(selection) == 0:
                    errorflag=1
                torch.cuda.empty_cache()
                if errorflag == 0:
                    # only select longest sequence:
                    current_sequence = [selection[0]]  # Initialize the current sequence with the first element
                    longest_sequence = []  # Initialize the longest sequence with an empty list

                    for seq in range(1, len(selection)):
                        if selection[seq] - current_sequence[-1] < max_gap_good_fits:
                            current_sequence.append(selection[seq])
                        else:
                            if len(current_sequence) > len(longest_sequence):
                                longest_sequence = current_sequence.copy()
                            current_sequence = [selection[seq]]

                    # Check if the last sequence is longer than the longest found so far
                    if len(current_sequence) > len(longest_sequence):
                        longest_sequence = current_sequence

                    # Create a filter array with boolean values using NumPy
                    filter_array = np.isin(selection, longest_sequence)




                    params = params[filter_array, :]
                    params2 = params2[filter_array, :]
                    selection = selection[filter_array,]
                    dist_array = dist_array[filter_array,]
                    zi_array = zi_array[filter_array,]
                    zi_array2 = zi_array2[filter_array,]

                    arr = []

                    fig, axes = plt.subplots(1, 2)
                    fig.set_size_inches(10.08, 7.04)
                    fig.tight_layout(w_pad=5, h_pad=5, pad=3)
                    # print('\n Print videos for tracking NE ' + str(i + 1) + '/' + str(self.bbnum) + '\n')
                    weight_arr = []
                    error_arr = []
                    values_arr = []
                    values = np.empty((100, 0))
                    error = np.empty((100, 0))

                filter_error = np.ones(len(selection))
                points_refined = np.zeros((np.size(selection), 2))
                pos_along_normal = np.zeros(np.shape(selection))
                points_refined2 = np.zeros((np.size(selection), 2))
                pos_along_normal2 = np.zeros(np.shape(selection))

                if errorflag == 0:
                    error_alarm = False
                    if len(selection) == np.size(params, 0):
                        # selection = selection[good_chisq]
                        # params = params[good_chisq]

                        for qq in range(len(selection)):

                            select = selection[qq]
                            point = points_it[:, select]
                            normal_slope = normal_slopes[:, select] / np.linalg.norm(normal_slopes[:, select])

                            start = point - length_line * normal_slope
                            end = point + length_line * normal_slope
                            # -- Extract the line...
                            # Make a line with "num" points...
                            x0, y0 = start[0], start[1]
                            x1, y1 = end[0], end[1]
                            # compute test statistic
                            r = (zi_array[qq, :] - intensity_func_sig(dist_alongline, *params[qq,
                                                                                                         :])) ** 2 / intensity_func_sig(
                                dist_alongline, *params[qq, :])
                            error_non_squared = (zi_array[qq, :] - intensity_func_sig(dist_alongline, *params[qq,
                                                                                                         :]))  / intensity_func_sig(
                                dist_alongline, *params[qq, :])
                            error_non_squared2 = (zi_array2[qq, :] - intensity_func_sig(dist_alongline,
                                                                                                        *params2[qq,
                                                                                                         :])) / intensity_func_sig(
                                dist_alongline, *params2[qq, :])
                            error_over_line = r * 1
                            error = np.concatenate((error, error_non_squared[..., None]), axis=1)
                            if np.max(abs(error_non_squared))>0.1 or np.max(abs(error_non_squared2))>0.1:
                                filter_error[qq] = 0
                            values = np.concatenate(
                                (values, intensity_func_sig(dist_alongline, *params[qq, :])[..., None]), axis=1)

                            x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
                            if movie == True and filter_error[qq]==1:
                                zoom_factor = 40  # You can change this to control the zoom level
                                if qq == 0:
                                    x_min, x_max = x0 - zoom_factor, x0 + zoom_factor
                                    y_min, y_max = y0 - zoom_factor, y0 + zoom_factor
                                axes[0].imshow(npc_mean, cmap='gray')
                                axes[0].plot(x, y, '-', label='Normal line')
                                axes[0].scatter(x0, y0, marker="x", label='Start')
                                axes[0].axis('image')
                                #axes[0].legend()
                                axes[0].axis('off')  # Hide the axes
                                axes[0].legend(frameon=True, facecolor='white')
                                axes[0].set_xlim(x_min, x_max)
                                axes[0].set_ylim(y_min, y_max)
                                axes[1].plot(dist_array[qq, :], zi_array[qq, :], label='Intensity along line')
                                # axes[1].axvline(x=length_line, color='red', label='initial guess')
                                axes[1].plot(dist_array[qq, :], intensity_func_sig(dist_array[qq, :], *params[qq, :]),
                                             label='Fit')

                                axes[1].set_xlabel('Distance [pixels]')
                                axes[1].set_ylabel('Intensity [ADU]')

                                # axes[1].set_title( 'Pvalue: '+str(r))

                                axes[1].legend(loc='lower left')
                                arr.append(self.cvtFig2Numpy(fig))
                                fig.savefig(self.resultprefix + 'npcfitline.svg', format='svg')
                                axes[0].cla()
                                axes[1].cla()

                                plt.close('all')
                                ##

                            pos_along_normal[qq] = params[qq, 7]
                            pos_along_normal[qq] -= length_line
                            pos_change = pos_along_normal[qq] * normal_slope
                            points_refined[qq] = point + pos_change

                        filter_error = filter_error.astype(bool)
                        points_refined = points_refined[filter_error]*1
                        params = params[filter_error, :]
                        selection = selection[filter_error,]
                        params2 = params2[filter_error, :]
                        values = values[:,filter_error]
                        error = error[:,filter_error]

                        # filter again
                        current_sequence = [selection[0]]  # Initialize the current sequence with the first element
                        longest_sequence = []  # Initialize the longest sequence with an empty list

                        for seq in range(1, len(selection)):
                            if selection[seq] - current_sequence[-1] < max_gap_good_fits:
                                current_sequence.append(selection[seq])
                            else:
                                if len(current_sequence) > len(longest_sequence):
                                    longest_sequence = current_sequence.copy()
                                current_sequence = [selection[seq]]

                        # Check if the last sequence is longer than the longest found so far
                        if len(current_sequence) > len(longest_sequence):
                            longest_sequence = current_sequence

                        # Create a filter array with boolean values using NumPy
                        filter_array_v2 = np.isin(selection, longest_sequence)
                        points_refined = points_refined[filter_array_v2]*1
                        #points_refined2 = points_refined2[filter_array_v2]*1
                        params = params[filter_array_v2, :]
                        params2 = params2[filter_array_v2, :]
                        selection = selection[filter_array_v2,]

                        values = values[:,filter_array_v2]
                        error = error[:,filter_array_v2]

                        points_refined = np.reshape(points_refined[points_refined != 0],
                                                    (int(np.size(
                                                        points_refined[points_refined != 0]) / 2),
                                                     2)) * 1


                        if np.size(points_refined) != 0 and len(points_refined[:, 0])>4:

                            error_arr.append(error)
                            values_arr.append(values)
                            weights = 1 / np.array(weight_arr)
                            closed = 0
                            if len(selection) > (
                            sampling_normal) * 0.98 and group[1]>self.initial_spline_sampling*0.98:  # periodic condition
                                tck, u = scipy.interpolate.splprep([points_refined[:, 0], points_refined[:, 1]],
                                                                   per=True,
                                                                   k=3,
                                                                   quiet=3, s=smoothness)  # w=weights/max(weights)+0.5,
                                closed = 1
                            else:

                                tck, u = scipy.interpolate.splprep([points_refined[:, 0], points_refined[:, 1]],
                                                                   per=False, k=3,
                                                                   s=smoothness)  # w=weights/max(weights)+0.5,
                            cut = 120  # cut points at edges
                            xi, yi = scipy.interpolate.splev(np.linspace(0, 1, max(len(points_refined), sampling)), tck)

                            dxi, dyi = scipy.interpolate.splev(np.linspace(0, 1, max(len(points_refined), sampling)),
                                                               tck, der=1)
                            ddxi, ddyi = scipy.interpolate.splev(np.linspace(0, 1, max(len(points_refined), sampling)),
                                                                 tck,
                                                                 der=2)
                            if closed == 0:
                                xi = xi[min(cut, len(xi)):max(len(xi) - cut, cut)]
                                yi = yi[min(cut, len(yi)):max(len(yi) - cut, cut)]
                                dxi = dxi[min(cut, len(dxi)):max(len(dxi) - cut, cut)]
                                dyi = dyi[min(cut, len(dyi)):max(len(dyi) - cut, cut)]
                                ddxi = ddxi[min(cut, len(ddxi)):max(len(ddxi) - cut, cut)]
                                ddyi = ddyi[min(cut, len(ddyi)):max(len(ddyi) - cut, cut)]

                            # filter based on change in radius of curvature

                            # Calculate angles between consecutive tangents
                            angles = np.arctan2(dyi, dxi)  # Angle of each vector

                            # Calculate raw differences
                            angle_diffs = np.diff(angles)

                            # Normalize differences to the range [-pi, pi]
                            angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

                            # Convert to degrees and take the absolute value
                            angle_diffs = np.abs(np.rad2deg(angle_diffs))
                            rad = (dxi ** 2 + dyi ** 2) ** (3 / 2) * 1 / (dxi * ddyi - dyi * ddxi)

                            # plt.hist(rad,bins=50)
                            # plt.show()
                            # plt.plot(xi,yi)
                            # plt.show()
                            normal_slope = np.array([rad * dyi, -rad * dxi]) / np.linalg.norm(
                                np.array([rad * dyi, -rad * dxi]), axis=0)
                            # plot second derivative vectors:
                            # V = normal_slope
                            # origin = np.array([xi[np.arange(1,len(xi)-2,10)],yi[np.arange(1,len(xi)-2,10)]])# origin point
                            #
                            # plt.quiver(*origin, V[0, [np.arange(1,len(xi)-2,10)]], V[1, [np.arange(1,len(xi)-2,10)]])
                            # plt.plot(xi[np.arange(1,len(xi)-2,10)],yi[[np.arange(1,len(xi)-2,10)]])
                            # plt.show()

                            normal_slope_rolled = np.roll(normal_slope, 1, axis=1)
                            #
                            dotproduct = (normal_slope[:, 1:-1] * normal_slope_rolled[:, 1:-1]).sum(0)
                            asign = np.sign(dotproduct)
                            signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

                            if sum(signchange) < np.inf and error_alarm == False and np.amax(
                                    abs(angle_diffs)) < 1:
                                points_new_percell.append(np.array([xi, yi]))
                                deriv_percell.append(np.array([dxi, dyi]))
                                closed_condition_percell.append(closed)
                                errors_percell.append(error_arr)
                                params_percell.append(params)
                                params2_percell.append(params2)
                                values_percell.append(values_arr)

                                # plt.plot(xi,yi)
                                # plt.title(str(np.amax(abs(angle_diffs))))
                                # plt.tight_layout()
                                # plt.savefig('test' + str(np.random.randint(0,1000)))


                    if movie:
                        self.makevideoFromArray(self.resultprefix + 'npcfit' + str(i) + str(index) + '.mp4', arr)

            if np.size(points_new_percell) != 0:
                points_allcells.append(points_new_percell)
                params_allcells.append(params_percell)
                params2_allcells.append(params2_percell)
                closed_condition_allcells.append(closed_condition_percell)
                deriv_allcells.append(deriv_percell)
                errors_allcells.append(errors_percell)
                values_allcells.append(values_percell)

        return params_allcells,params2_allcells



def ccw1(A,B,C):
    return (C[1]-A[:,1]) * (B[:,0]-A[:,0]) > (B[:,1]-A[:,1]) * (C[0]-A[:,0])
def ccw2(A,B,C):
    return (C[1]-A[:,1]) * (B[0]-A[:,0]) > (B[1]-A[:,1]) * (C[0]-A[:,0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    full_arr = torch.logical_and(ccw2(A,C,D) != ccw2(B,C,D), ccw1(A,B,C) != ccw1(A,B,D))
    if torch.sum(full_arr) == 0:
        return 0
    elif torch.sum(full_arr) == 1:
        return 1
    else:
        assert ValueError('check intersect~!!')
