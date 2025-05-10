import numpy as np

resX=1280
resY=800
X,Y=np.meshgrid(np.arange(resX),np.arange(resY))
Z=np.flipud(np.round(2*np.pi/np.sqrt(2)*(X+Y)).astype(int))
lam=np.flipud(np.round(1/np.sqrt(2)*(X-Y)).astype(int))
X=X.astype(int)
Y=Y.astype(int)
lamv=np.arange(np.amin(lam),np.amax(lam))
period=128
resA = len(lamv)


def gen_bin_img(amps,X,Y,Z,lam,period=128): #need to precompute grid vectors 
    #idea is that the energy into the diffracted beam is quadratically related to its duty cycle. 
    #The DMD itself is however rotated at 45 degrees, so each line corresponds to a 45 degree line through the image
    #we assume amps is between 0 and 1
    img=np.sin(2*np.pi*Z/period)+(2*amps[lam-np.min(lam)-1]-1) #need to do offset to get full range of amplitude modulation
    img[img>0]=255
    img[img<=0]=0
    return img.astype('uint8')

Z_temp = np.sin(2*np.pi*Z/period)-1
lam_temp = lam-np.min(lam)-1

lam_temp2 = lam_temp.astype('int16')
Z_temp2 = (-127*Z_temp/2).astype('uint8')

def get_amp_img(amps):
    amps = (127*(amps) + 127.5).astype('uint8')
    img = amps[lam_temp2] 
    img -= Z_temp2 #need to do offset to get full range of amplitude modulation
    mask = img > 127
    img[mask] = 255
    img[np.bitwise_not(mask)]=0
    return img