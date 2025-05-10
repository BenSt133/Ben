from time import sleep
from ctypes import cast, POINTER, c_double, c_ushort, c_ulong

from mcculw import ul
from mcculw.enums import ScanOptions, FunctionType, Status
from mcculw.device_info import DaqDeviceInfo

try:
    from console_examples_util import config_first_detected_device
except ImportError:
    from .console_examples_util import config_first_detected_device

import matplotlib.pyplot as plt 
import numpy as np
import time    
from scipy.interpolate import interp1d    
    

def digitizex(data, xlevels, fine_factor=20):
    #First do downsampling in x
    x=np.linspace(0,len(data),fine_factor*xlevels)
    datafine=np.interp(x,np.arange(0,len(data)),data)
    #now here get a new set of data that is mean over each thing
    datad = np.zeros(xlevels)
    for i in range(xlevels):
        datad[i] = np.sum(datafine[i*fine_factor:(i+1)*fine_factor])/fine_factor
    return datad

def digitizey(x, levels, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()
    x = (levels-1)*(x - min_val)/(max_val-min_val)
    x = x.round()
    x = torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x)*(levels-1))
    x = x*(max_val-min_val)/(levels-1) + min_val
    return x

    
def setup_daq():
    use_device_detection = True
    dev_id_list = []
    board_num = 0
    
    if use_device_detection:
        config_first_detected_device(board_num, dev_id_list)

    daq_dev_info = DaqDeviceInfo(board_num)
    
    ai_info = daq_dev_info.get_ai_info()
    ai_range = ai_info.supported_ranges[0]
    
    ao_info = daq_dev_info.get_ao_info()
    ao_range = ao_info.supported_ranges[0]
    
    return board_num,ai_info,ai_range,ao_info,ao_range

def release_daq(board_num):
    ul.release_daq_device(board_num)
    
def record_daq(board_num,channel,ai_range):
    return ul.v_in_32(board_num, channel, ai_range)
    
def write_daq(data_value,board_num,channel,ao_range):
    ul.v_out(board_num, channel, ao_range, data_value)    
    
def write_read_daq_fast(data_in,rate,board_num,ai_info,ai_range,ao_info,ao_range,low_chan=0,high_chan=0,low_chani=0,high_chani=0,outpad=100,num_channels_out=1,num_channels_in=2): #only 1 channel now
    
    if num_channels_in >1:
        points_per_channel = data_in.shape[1]
    else:
        points_per_channel = data_in.shape[0]
        
    total_count= points_per_channel*num_channels_out
    
    memhandle=None
    memhandlei=None
    
    memhandle = ul.win_buf_alloc(total_count)
    data_array = cast(memhandle, POINTER(c_ushort))
    
    if num_channels_out>1:
        data_index = 0
        for point_num in range(points_per_channel):
            for channel_num in range(num_channels_out):
                value = data_in[channel_num,point_num]
                raw_value = ul.from_eng_units(board_num, ao_range, value)
                data_array[data_index] = raw_value
                data_index += 1
    else:
        data_index = 0
        for point_num in range(points_per_channel):
            value = data_in[point_num]
            raw_value = ul.from_eng_units(board_num, ao_range, value)
            data_array[data_index] = raw_value
            data_index += 1        
        
        
        
    #Set up read    
    
    total_count_in=(points_per_channel+outpad)*num_channels_in
    scan_options = ScanOptions.BACKGROUND

    if ScanOptions.SCALEDATA in ai_info.supported_scan_options:
        scan_options |= ScanOptions.SCALEDATA

        memhandlei = ul.scaled_win_buf_alloc(total_count_in)
            # Convert the memhandle to a ctypes array.
        data_arrayi = cast(memhandlei, POINTER(c_double))
    elif ai_info.resolution <= 16:
            # Use the win_buf_alloc method for devices with a resolution <= 16
        memhandlei = ul.win_buf_alloc(total_count_in)
            # Convert the memhandle to a ctypes array.
        data_arrayi = cast(memhandlei, POINTER(c_ushort))
    else:
            # Use the win_buf_alloc_32 method for devices with a resolution > 16
        memhandlei = ul.win_buf_alloc_32(total_count_in)
            # Convert the memhandle to a ctypes array.
        data_arrayi = cast(memhandlei, POINTER(c_ulong))
        
        
    ul.a_in_scan(board_num, low_chani, high_chani, total_count_in,rate, ai_range, memhandlei, scan_options)    
        
    
    ul.a_out_scan(board_num, low_chan, high_chan, total_count, rate,ao_range, memhandle, ScanOptions.BACKGROUND)
    
    #print('Waiting for output scan to complete...', end='')
    status = Status.RUNNING
    while status != Status.IDLE:
        #print('.', end='')

            # Slow down the status check so as not to flood the CPU
        sleep(0.002)

        status, _, _ = ul.get_status(board_num, FunctionType.AOFUNCTION)
    #print('')

    #print('Scan completed successfully')
    
    ul.stop_background(board_num, FunctionType.AIFUNCTION)
    ul.stop_background(board_num, FunctionType.AOFUNCTION)
    data_read=list()
    for data_index in range(total_count_in):
        if ScanOptions.SCALEDATA in scan_options:
                        # If the SCALEDATA ScanOption was used, the values
                        # in the array are already in engineering units.
            eng_value = data_arrayi[data_index]
        else:
                        # If the SCALEDATA ScanOption was NOT used, the
                        # values in the array must be converted to
                        # engineering units using ul.to_eng_units().
            eng_value = ul.to_eng_units(board_num, ai_range,data_arrayi[data_index])
        data_read.append(eng_value)
  
    
    #ul.stop_background(board_num, FunctionType.AIFUNCTION)  
    
    if memhandle:
            # Free the buffer in a finally block to prevent a memory leak.
        ul.win_buf_free(memhandle)
        
    if memhandlei:
            # Free the buffer in a finally block to prevent a memory leak.
        ul.win_buf_free(memhandlei)
        
    return np.array(data_read)

def addpre(datain,tau=0.25,nzeros=0,amppre=10.):
    t=np.linspace(0,500,500)
    return np.hstack((np.zeros(nzeros),amppre*np.exp(-(t-1)**2/tau**2),datain))


def run_exp(x,board_num,ai_info,ai_range,ao_info,ao_range,A=1,B=0,OD=56,Tmax=600,st=660,Nde=450,input_pad=100,checklength=3000,trigger=0.05):
    
    ##COLLECT RAW DATA##
    rate=1000000//2 #1 MS/s
    dt=1/rate #1 us
    
    t=np.arange(Tmax)*dt
    
    batch_size=x.shape[0]
    Ni=x.shape[1]//2
    
    time_i=np.linspace(int(Tmax*0.3),int(Tmax*0.9),Ni)*dt

    Amax=(ao_range.range_max-ao_range.range_min)*1

    outs=list()
 

    for r in range(batch_size):
        fa=interp1d(time_i,x[r,:Ni], kind="nearest", fill_value=(0.0,0.0),bounds_error=False)
        if r==0:
            signal_0=np.hstack((fa(t)*Amax,np.zeros(input_pad)))
        else:
            signal_0=np.hstack((signal_0,fa(t)*Amax,np.zeros(input_pad)))
        fa=interp1d(time_i,x[r,Ni:], kind="nearest", fill_value=(0.0,0.0),bounds_error=False)
        if r==0:
            signal_1=np.hstack((fa(t)*Amax,np.zeros(input_pad)))
        else:
            signal_1=np.hstack((signal_1,fa(t)*Amax,np.zeros(input_pad)))


    data_in = np.vstack((addpre(signal_0),addpre(signal_1)))
    

    data_read=write_read_daq_fast(data_in,rate,board_num,ai_info,ai_range,ao_info,ao_range,low_chan=0,high_chan=1,low_chani=0,high_chani=1,outpad=3000,num_channels_out=2,num_channels_in=2)
    outs.append(data_read)
      
    step=int(Tmax+input_pad)
    
    Y=np.array(outs)
    
    Yp1=list()

    out0=outs[0][::2] #trigger source
    out1=outs[0][1::2] 

    outy0=out0[:checklength]
    outy1=out1

    timer=outy0/np.max(outy0)
    timerI=np.arange(len(outy0))

    trigged_timer=timerI[timer>=trigger]


    outy_start = trigged_timer[0]
    
    check = outy0[outy_start:outy_start+st]
    
    outy0=outy0[outy_start+st:]
    outy1=outy1[outy_start+st:]
    
    try:
        for r in range(batch_size):
            outy=outy1[(step)*r:(step)*(r+1)]
            if OD is not None:                  
                oy=digitizex(outy[:Nde],OD)/A-B/A #This is to renormalize 
                Yp1.append(oy)
            else:
                Yp1.append(outy[:Nde])   
        return np.array(Yp1)   
    except:
        print('PNN failed')
        return np.zeros(1)


def process_input(outs,R,step,Nde,st,batch_size,input_pad=100,checklength=3000):

    repeats=1
    trigger=0.05
    Nr=len(outs)//repeats*batch_size


    Y=list()
    X=list()
    
    idc=0   
    for idx in range(0,Nr,batch_size):
        out0=outs[idc][::2] #trigger source
        out1=outs[idc][1::2] 

        outy0=out0[:checklength]
        outy1=out1
        
        timer=outy0/np.max(outy0)
        timerI=np.arange(len(outy0))


        trigged_timer=timerI[timer>=trigger]

        outy_start = trigged_timer[0]
        
        outy0=outy0[outy_start+st:]
        outy1=outy1[outy_start+st:]

        for r in range(batch_size):
            outy=outy1[(step)*r:(step)*(r+1)]
            Y.append(outy[:Nde])
            X.append(np.hstack((R[0,r+idx,:],R[1,r+idx,:])))
            
        idc+=1
    return np.array(X),np.array(Y) 