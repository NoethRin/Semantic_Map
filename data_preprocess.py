import torch
import numpy as np 
import nibabel as nib
import os
import cortex

from cortex import mni
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from interpdata import lanczosinterp2D
from zhsutils import get_story_wordseqs, reduce_data

def embProcess(start, end, mode):
    if os.path.exists("data/" + mode + "_x.pt"):
        return
    final_emb = []
    for story in tqdm(range(start, end + 1), desc="processing embedding"):
        word_seqs, word_vecs = get_story_wordseqs(story, 300)
        ds_vecs = lanczosinterp2D(word_vecs, word_seqs.data_times, word_seqs.tr_times)[5:-5]
        ds_mat = np.nan_to_num(np.dot((ds_vecs - ds_vecs.mean(0)), np.linalg.inv(np.diag(ds_vecs.var(0)))))
        final_emb.append(torch.from_numpy(ds_mat).to(torch.float32))
    torch.save(torch.vstack(final_emb), "data/" + mode + "_x.pt")

def fmriProcess(start, end, mode):
    if os.path.exists("data/" + mode + "_y.pt"):
        return
    final_data = []
    s1_to_mni = mni.compute_mni_transform(subject='S1', xfm='fullhead') 
    default_template = os.path.join("/home/zak/fsl", "data", "standard", "MNI152_T1_2mm_brain.nii.gz")
    subject_mask = cortex.db.get_mask('S1', 'fullhead', 'thick')
    for story in tqdm(range(start, end + 1), desc="processing fMRI data"):
        img = nib.load('/home/public/public/CASdata/metadata/preprocessed_data/sub-05/MNI/sub-05_task-RDR_run-' + str(story) + '_bold.nii.gz')
        fmri_data = reduce_data(img.get_fdata()[:, :, :, 30:30 + 300])
        for j in range(100):
            sub_img = mni.transform_mni_to_subject('S1', 'fullhead', fmri_data[:, :, :, j], s1_to_mni, template=default_template)
            sub_data = sub_img.get_fdata()
            sub_data = np.transpose(sub_data[:, :, :], (2, 1, 0))[np.newaxis, :, :, :]
            final_data.append(torch.from_numpy(sub_data[:, subject_mask == True]).to(torch.float32))
    final_data = torch.vstack(final_data)
    scaler = StandardScaler()
    final_data = scaler.fit_transform(final_data.T).T
    torch.save(torch.from_numpy(final_data).to(torch.float32), "data/" + mode + "_y.pt")

if __name__ == "__main__":
    embProcess(1, 42, "train")
    embProcess(43, 54, "val")
    embProcess(55, 60, "test")
    fmriProcess(1, 42, "train")
    fmriProcess(43, 54, "val")
    fmriProcess(55, 60, "test")
