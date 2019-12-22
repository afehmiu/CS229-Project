import spams
import h5py
import run_smaf
import numpy as np
import os

## replace relevant file path for wherever the data is stored locally
f = h5py.File('../../../../Downloads/human_matrix.h5', 'r')

## matrix with rows as samples and columns as genes
expression = f['data']['expression']
print(expression.shape)

## genes: is an array of the gene names in order
## samples: is an array of the sample titles in order
genes = f['meta']['genes']
samples = f['meta']['Sample_title']

## for testing take a smaller chunk of expression
## convert to array of floats or none of the SMAF code works
print("converting")
expression = expression[0:5000,:].astype(float)
print("converted")

## I got this chunk of code to run but didn't seem to work. The printed correlations weren't getting better
"""
d = 500
k = 15
UW = (U,W)
U,W = run_smaf.smaf(expression,d,k,0.1,maxItr=10,activity_lower=0.,module_lower=min(20,expression.shape[0]/20),UW=UW,donorm=True,mode=1,use_chol=True,mink=5,doprint=True)
"""
## saving the data as a .npy file which Cleary's code is expecting to load
np.save("expression_data2",expression)

## running various matrix factorizations
## printed output will follow this format for each model: # [overall Pearson, overall Spearman, gene Pearson, sample Pearson, sample dist Pearson, sample dist Spearman, gene dist Pearson, gene dist Spearman]
os.system('python direct_vs_distributed.py -i expression_data2.npy -m 100 -s 15 -d 500 -t .80 -g 500 -n 5000 -r 2 -b 0 -a 0')

## example output I got from running this script. I ran it on 50,000 samples. In this script we just run it on 5,000 samples.
## lots of warnings since a bunch of things are deprecated or old
"""
SMAF (testing)	0.026856899325801487	0.434485350457355	0.02479398487479507	0.34104360262457134	-0.0837588396229737	0.04676455122460656	0.6573858987229159	0.8632688575474639
SMAF (training)	0.013863020717035712	0.4365306300480728	0.009586552419625147	0.3510810719381737	-0.045939791857927854	-0.05635806162226628	0.40054396585696006	0.4481284962118773
SVD (testing)	0.7590467665196917	0.4396813273634533	0.4664347325726027	0.3196394533767107	0.8784098340532731	0.793993647189910.9556631119388912	0.9554134351023468
SVD (training)	0.7537596775798386	0.4569235072440033	0.4638834548460939	0.311870877485983	0.8912335780376546	0.7661430711565594	0.9632972662493028	0.9705856456568454
k-SVD (testing)	0.8195094745408573	0.5603476909694682	0.5948485046584291	0.3456362206617018	0.9462723745747824	0.912170962448730.9798883907927537	0.9721279554284472
k-SVD (training)	0.813474778377153	0.5161940345968984	0.5832080381540006	0.3415620543845245	0.9419033344560828	0.8357934605078347	0.9861327181620915	0.981885600198155
sparse NMF (sample_dist)	0.978057190658414	0.9446175528628966
sparse NMF (testing)	0.008114584551244142	0.4191956298544133	0.011332304088327312	0.34603617118037	-0.07124547840296663	0.05750327513549521	0.5277399685697701	0.4897014590844448
sparse NMF (training)	0.010022723805365286	0.43017743947899656	0.009228694779645013	0.34774944460520446	-0.122464948271356	0.02514697170365662	0.4777050010306525	0.6561824776742282
"""
