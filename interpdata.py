import numpy as np
import logging

logger = logging.getLogger("text.regression.interpdata")

def interpdata(data, oldtime, newtime):

    if not len(oldtime) == data.shape[0]:
        raise IndexError("oldtime must have same number of elements as data has rows.")

    newdata = np.empty((len(newtime), data.shape[1]))

    for ci in range(data.shape[1]):
        if (ci%100) == 0:
            logger.info("Interpolating column %d/%d.." % (ci+1, data.shape[1]))
        
        newdata[:,ci] = np.interp(newtime, oldtime, data[:,ci])

    return newdata

def sincinterp1D(data, oldtime, newtime, cutoff_mult=1.0, window=1):
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    print ("Doing sinc interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))

    newdata = np.zeros((len(newtime),1))
    for ndi in range(len(newtime)):
        for di in range(len(oldtime)):
            newdata[ndi] += sincfun(cutoff, newtime[ndi]-oldtime[di], window) * data[di]
    return newdata

def sincinterp2D(data, oldtime, newtime, cutoff_mult=1.0, window=1, causal=False, renorm=True):

    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    print ("Doing sinc interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))

    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = sincfun(cutoff, newtime[ndi]-oldtime, window, causal, renorm)

    newdata = np.dot(sincmat, data)

    return newdata

def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):

    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    print ("Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))

    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)
    
    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)), 
                            np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        newdata = np.dot(sincmat, data)

    return newdata

def sincupinterp2D(data, oldtime, newtimes, cutoff, window=1):
    #cutoff = 1/np.mean(np.diff(oldtime))
    print ("Doing sinc interpolation with cutoff=%0.3f and %d lobes."%(cutoff, window))
    
    sincmat = np.zeros((len(newtimes), len(oldtime)))
    for ndi in range(len(newtimes)):
        sincmat[ndi,:] = sincfun(cutoff, newtimes[ndi]-oldtime, window, False)

    newdata = np.dot(sincmat, data)
    return newdata

def sincfun(B, t, window=np.inf, causal=False, renorm=True):
    val = 2*B*np.sin(2*np.pi*B*t)/(2*np.pi*B*t+1e-20)
    if t.shape:
        val[np.abs(t)>window/(2*B)] = 0
        if causal:
            val[t<0] = 0
        if not np.sum(val)==0.0 and renorm:
            val = val/np.sum(val)
    elif np.abs(t)>window/(2*B):
        val = 0
        if causal and t<0:
            val = 0
    return val

def lanczosfun(cutoff, t, window=3):
    t = t * cutoff
    val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0
    return val

def expinterp2D(data, oldtime, newtime, theta):
    intmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        intmat[ndi,:] = expfun(theta, newtime[ndi]-oldtime)

    newdata = np.dot(intmat, data)
    return newdata

def expfun(theta, t):
    val = np.exp(-t*theta)
    val[t<0] = 0.0
    if not np.sum(val)==0.0:
        val = val/np.sum(val)
    return val

def gabor_xfm(data, oldtimes, newtimes, freqs, sigma):
    sinvals = np.vstack([np.sin(oldtimes*f*2*np.pi) for f in freqs])
    cosvals = np.vstack([np.cos(oldtimes*f*2*np.pi) for f in freqs])
    outvals = np.zeros((len(newtimes), len(freqs)), dtype=np.complex128)
    for ti,t in enumerate(newtimes):
        gaussvals = np.exp(-0.5*(oldtimes-t)**2/(2*sigma**2))*data
        sprod = np.dot(sinvals, gaussvals)
        cprod = np.dot(cosvals, gaussvals)
        outvals[ti,:] = cprod + 1j*sprod

    return outvals

def gabor_xfm2D(ddata, oldtimes, newtimes, freqs, sigma):
    return np.vstack([gabor_xfm(d, oldtimes, newtimes, freqs, sigma).T for d in ddata])

def test_interp(**kwargs):
    oldtime = np.linspace(0, 10, 100)
    newtime = np.linspace(0, 10, 49)
    data = np.zeros((4, 100))
    data[0,50] = 1.0
    data[1,45:55] = 1.0
    data[2,40:45] = 1.0
    data[2,55:60] = 1.0
    data[3,40:45] = 1.0
    data[3,55:60] = 2.0

    interpdata = sincinterp2D(data.T, oldtime, newtime, **kwargs).T

    from matplotlib.pyplot import figure, show
    fig = figure()
    for d in range(4):
        ax = fig.add_subplot(4,1,d+1)
        ax.plot(newtime, interpdata[d,:], 'go-')
        ax.plot(oldtime, data[d,:], 'bo-')

    show()
    return newtime, interpdata
