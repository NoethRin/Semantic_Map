import numpy as np
import itertools as itools
import scipy.io as scio
from interpdata import sincinterp2D, gabor_xfm2D, lanczosinterp2D

def make_delayed(stim, delays, circpad=False):
    nt,ndim = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0:
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else:
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def reduce_data(data):
    res = np.zeros([data.shape[0], data.shape[1], data.shape[2], 100])
    for i in range(100):
        res[:, :, :, i] = data[:, :, :, i * 3]
    return res

def get_story_wordseqs(story, chunk_len):
    trfiles = load_simulated_trfiles(story, chunk_len, 2.13, 5)
    wordseqs = make_word_ds(story, trfiles)
    return wordseqs

def load_simulated_trfiles(story, chunk_len, tr=2.13, pad=5):
    trf = TRFile(None, tr)
    trf.soundstarttime = 10.65
    trf.simulate(chunk_len //3 + 10)
    return [trf]

def make_word_ds(story, trfiles, bad_words=frozenset([])): # "。", "“", "”", "，", "、", "！", "?", "《", "》", ";", ":"
    annotation = []
    embedding = []
    embs = scio.loadmat("/home/public/public/CASdata/metadata/stimuli/annotations/embeddings/word2vec/word-level/300d/story_" + str(story) + "_word_word2vec.mat")["data"]
    res = scio.loadmat("/home/public/public/CASdata/metadata/stimuli/annotations/time_align/word-level/story_" + str(story) + "_word_time.mat")
    for j in range(res['start'].shape[1]):
        if res['word'][j].strip() not in bad_words:
            annotation.append((res['start'][0, j], res['end'][0, j], res['word'][j].strip()))
            embedding.append(embs[j, :])
    d = DataSequence.from_grid(annotation, trfiles[0])
    return d, np.vstack(embedding)

def make_semantic_model(ds, lsasm):
    newdata = []
    for w in ds.data:
        try:
            v = lsasm[w]
        except KeyError as e:
            v = np.zeros((lsasm.data.shape[0],))
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)

class DataSequence(object):
    def __init__(self, data, split_inds, data_times=None, tr_times=2.13):
        self.data = data
        self.split_inds = split_inds
        self.data_times = data_times
        self.tr_times = tr_times

    def mapdata(self, fun):
        return DataSequence(self, map(fun, self.data), self.split_inds)

    def chunks(self):
        return np.split(self.data, self.split_inds)

    def data_to_chunk_ind(self, dataind):
        zc = np.zeros((len(self.data),))
        zc[dataind] = 1.0
        ch = np.array([ch.sum() for ch in np.split(zc, self.split_inds)])
        return np.nonzero(ch)[0][0]

    def chunk_to_data_ind(self, chunkind):
        return list(np.split(np.arange(len(self.data)), self.split_inds)[chunkind])

    def chunkmeans(self):
        dsize = self.data.shape[1]
        outmat = np.zeros((len(self.split_inds)+1, dsize))
        for ci, c in enumerate(self.chunks()):
            if len(c):
                outmat[ci] = np.vstack(c).mean(0)

        return outmat

    def chunksums(self, interp="rect", **kwargs):
        if interp=="sinc":
            return sincinterp2D(self.data, self.data_times, self.tr_times, **kwargs)
        elif interp=="lanczos":
            return lanczosinterp2D(self.data, self.data_times, self.tr_times, **kwargs)
        elif interp=="gabor":
            return np.abs(gabor_xfm2D(self.data.T, self.data_times, self.tr_times, **kwargs)).T
        else:
            dsize = self.data.shape[1]
            outmat = np.zeros((len(self.split_inds)+1, dsize))
            for ci, c in enumerate(self.chunks()):
                if len(c):
                    outmat[ci] = np.vstack(c).sum(0)
                    
            return outmat

    def copy(self):
        return DataSequence(list(self.data), self.split_inds.copy(), self.data_times, self.tr_times)
    
    @classmethod
    def from_grid(cls, grid_transcript, trfile):
        data_entries = list(zip(*grid_transcript))[2]
        if isinstance(data_entries[0], str):
            data = list(map(str.lower, list(zip(*grid_transcript))[2]))
        else:
            data = data_entries
        word_starts = np.array(list(map(float, list(zip(*grid_transcript))[0])))
        word_ends = np.array(list(map(float, list(zip(*grid_transcript))[1])))
        word_starts -= trfile.soundstarttime
        word_ends -= trfile.soundstarttime
        word_avgtimes = (word_starts + word_ends)/2.0

        tr = trfile.avgtr
        trtimes = trfile.trtimes

        split_inds = [(word_starts<(t+tr)).sum() for t in trtimes][:-1]
        return cls(data, split_inds, word_avgtimes, trtimes+tr/2.0)

    @classmethod
    def from_chunks(cls, chunks):
        lens = map(len, chunks)
        split_inds = np.cumsum(lens)[:-1]
        data = list(itools.chain(*map(list, chunks)))
        return cls(data, split_inds)
    
class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.13):
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr
        
        if trfilename is not None:
            self.load_from_file(trfilename)
        

    def load_from_file(self, trfilename):
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label=="sound-start":
                self.soundstarttime = time

            elif label=="sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))
        
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes>(itrtimes.mean()*1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            newtrtime = self.trtimes[btr]+self.expectedtr
            newtrs.append((newtrtime,btr))

        for ntr,btr in newtrs:
            self.trtimes.insert(btr+1, ntr)

    def simulate(self, ntrs):
        self.trtimes = list(np.arange(ntrs)*self.expectedtr)
    
    def get_reltriggertimes(self):
        return np.array(self.trtimes)-self.soundstarttime

    @property
    def avgtr(self):
        return np.diff(self.trtimes).mean()