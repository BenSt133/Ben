"""
Pulse shaper code for the PICe version of the SLM

As usual write it in a functional programming manner so that much of the code is refractorable
"""
#Following are hardcoded parameters to make the code easier to write (could be made as variables)
resX = 1920
resY = 1152
PeriodWidth = 24

import os
import numpy
from ctypes import *
from scipy import misc

dir_sdk = "C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK"
os.environ['PATH'] = dir_sdk + os.pathsep + os.environ['PATH']

#Basic parameters for calling Create_SDK
bit_depth = c_uint(12)
num_boards_found = c_uint(0)
constructed_okay = c_bool(0)
is_nematic_type = c_bool(1)
RAM_write_enable = c_bool(1)
use_GPU = c_bool(1)
max_transients = c_uint(20)
board_number = c_uint(1)
wait_For_Trigger = c_uint(0)
output_Pulse = c_uint(0)
timeout_ms = c_uint(5000)
center_x = c_float(256)
center_y = c_float(256)
VortexCharge = c_uint(3)

def slm_init():
    cdll.LoadLibrary("Blink_C_wrapper")
    slm = CDLL("Blink_C_wrapper")
    slm.Create_SDK(bit_depth, byref(num_boards_found), byref(constructed_okay), is_nematic_type, RAM_write_enable, use_GPU, max_transients, 0)
    return slm

def get_wl(wl_cal_file):
    W=np.load(wl_cal_file)
    lambdap=W['arr_4'] #polynomial fit of pixels to wavelength

    x=np.linspace(0,resX,resX)
    lambda_mod=np.polyval(lambdap,x)
    return lambda_mod

class PulseShaper():
    def __init__(s, amp_cal_file, wl_cal_file):
        #ampcal stuff
        B=np.load(amp_cal_file)
        A=B['A']
        L=B['L']
        s.A = A
        s.L = L

        wl = get_wl(wl_cal_file)
        s.wl = wl

        s.slm = slm_init()
        s.phase_mask = None

    def set_phase(s, phi):
        """
        Set phi to the slm - fast since the phase mask does not change!
        """
        s.phase_mask = generate_phase_mask(resX, resY, PeriodWidth, phi)

    def get_image(s, amp):
        """
        return the c level image!
        """
        phase_mask = s.phase_mask
        A = s.A
        L = s.L

        img = blazedgrating(resX, resY, PeriodWidth, 255, 0, amp, phase_mask=phase_mask, bias='B',A=A,L=L,amp=1,apodizationwidth=4)
        img = img.ctypes.data_as(POINTER(c_ubyte))
        return img

    def set_image(s, img):
        s.slm.Write_image(board_number, img, resX*resY, wait_For_Trigger, output_Pulse, timeout_ms)

    def set_amp_phase(s, amp, phi):
        """
        This function is "slow" if phi is not changed! 
        """
        A = s.A
        L = s.L

        img = blazedgrating(resX, resY, PeriodWidth, 255, 0, amp, phi=phi, bias='B',A=A,L=L,amp=1,apodizationwidth=4)
        img = img.ctypes.data_as(POINTER(c_ubyte))
        s.set_image(img)

    def set_amp(s, amp):
        """
        Recommended to do all the images first but this function is available for the lazy...
        """
        phase_mask = s.phase_mask
        A = s.A
        L = s.L

        img = blazedgrating(resX, resY, PeriodWidth, 255, 0, amp, phase_mask=phase_mask, bias='B',A=A,L=L,amp=1,apodizationwidth=4)
        img = img.ctypes.data_as(POINTER(c_ubyte))
        s.set_image(img)

# functions below are for constructing pattern for slm
import numpy as np
from scipy.interpolate import interp1d
from copy import copy

def sawtooth_inplace(t):
    """
    Sawtooth function that acts in place, in math f(t) = t - np.floor(t)
    """
    buffer = copy(t) #2ms - if pass buffer can save 0.3ms...
    np.floor(t, out=t) #4ms
    t *= -1 #1ms
    t += buffer #3ms...

def generate_phase_mask(SLMwidth, SLMheight, PeriodWidth, phi, pat=None):
    ilist = np.array(range(SLMheight))
    temp = -ilist/PeriodWidth
    phi_temp = phi/(2*np.pi)

    if pat is None:
        pat = np.zeros((SLMheight,SLMwidth))

    pat_t = pat.T
    pat_t[:, :] = temp #0.7ms here if we pass in matrix we can save this time... broadcasting saves about 1ms...
    pat += phi_temp
    sawtooth_inplace(pat)
    return pat
    
def blazedgrating(SLMwidth, SLMheight, PeriodWidth, vmax, vmin, amplitudes, phi=None, phase_mask=None, bias = 'B', A=np.zeros(100),L=0,amp=0,apodizationwidth=0, pat=None):
    """
    Generate a blazed grating with the intended phase and amplitude modulations    
    Aligned period is 24 pixels so PeriodWidth should be 24 currently.   
    Bias determines which direction the sawtooth is directed (which direction there will be enhanced 1st order) 
    Amplitudes is expected to be a vector of the spectral amplitudes desired at each pixel. 
    phi is the vector of spectral phases at each pixel
    The latter two should be input according to the wavelength calibration of wavelength to pixel. The amplitude is expected
    to be from 0 to 1, and is automatically adjusted according to the amplitude calibration, if it is provided
    
    A and L are the curves for the amplitude calibration
    
    apodizationwidth is the width, in pixels, of the apodization gaussian
    pat is a buffer that can be passed
    """ 
    pat_flag = pat is not None #flag for whether pat is passed in...
    
    if not pat_flag: #can save 0.7ms!
        pat = np.zeros((SLMheight,SLMwidth)) 
    if bias == 'A':
        width=1
    elif bias == 'B':
        width=0
    
    #note this interp section takes 0.5ms...
    x=np.linspace(0,len(A)-1,100)
    xfull=np.arange(len(A))
    Asmall=np.interp(x,xfull,A)
    Lsmall=np.interp(x,xfull,L)

    if amp==1:
        ampcal=interp1d(Lsmall,Asmall,fill_value='extrapolate',bounds_error=False)
        amplitudes=ampcal(amplitudes)

    amp_factors = amplitudes*((vmax-vmin))
    
    if phase_mask is None:
        if phi is not None:
            pat = generate_phase_mask(SLMwidth, SLMheight, PeriodWidth, phi, pat=pat)
        else:
            raise Exception("neither phi or phase_mask was supplied.")
    else:
        if phi is None:
            pat[:] = phase_mask
        else:
            raise Exception("If phase_mask is supplied, then phi has to be None.")
    
    pat *= amp_factors
    if np.abs(vmin) > 1e-2: #if zero don't do this...
        pat += vmin
    
    return pat.astype('uint8') #wreaked my head - could not get around this allocation... about 2ms here...

