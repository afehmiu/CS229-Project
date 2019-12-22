import numpy as np
import spams
import sys,getopt
from dl_simulation import *
from signature_genes import get_signature_genes,build_signature_model
from analyze_predictions import *
# from union_of_transforms import random_submatrix,double_sparse_nmf,smaf
from union_of_transforms import random_submatrix,smaf


def compare_results(A,B):
	results = list(correlations(A,B,0))[:-1]
	results += list(compare_distances(A,B))
	results += list(compare_distances(A.T,B.T))
	#results += [compare_clusters(A,B)]
	#results += [compare_clusters(A.T,B.T)]
	return results

THREADS = 10

# Results will consist of the following for each method (in both training and testing):
# overall Pearson, overall Spearman, gene Pearson, sample Pearson, sample dist Pearson, sample dist Spearman, gene dist Pearson, gene dist Spearman, sample cluster MI, gene cluster MI

if __name__ == "__main__":
	biased_training = 0.
	composition_noise = 0.
	subset_size = 0
	opts, args = getopt.getopt(sys.argv[1:], 'i:m:s:d:t:g:n:r:b:a:', [])
	for opt, arg in opts:
		if opt == '-i':
			data_path = arg
		elif opt == '-m':
			measurements = int(arg)
		elif opt == '-s':
			sparsity = int(arg)
		elif opt == '-d':
			dictionary_size = float(arg)
		elif opt == '-t':
			training_dictionary_fraction = float(arg)
		elif opt == '-g':
			max_genes = int(arg)
		elif opt == '-n':
			max_samples = int(arg)
		elif opt == '-r':
			SNR = float(arg)
		elif opt == '-b':
			biased_training = float(arg)
		elif opt == '-a':
			composition_noise = float(arg)
		elif opt == '-z':
			subset_size = int(arg)
	#data_path,measurements,sparsity,dictionary_size,training_dictionary_fraction,max_genes,max_samples,SNR,biased_training = 'GTEx/data.commonGenes.npy',100,15,0.5,0.05,5000,10000,2.,0.
	X = np.load(data_path)
	X0,xo,Xobs = random_submatrix(X,max_genes,max_samples,0)
	# train bases
	training_dictionary_size = max(int(training_dictionary_fraction*X0.shape[1]),5)
	if dictionary_size < 1:
		dictionary_size = dictionary_size*training_dictionary_size
	dictionary_size = int(dictionary_size)
	xi = np.zeros(X0.shape[1],dtype=np.bool)
	if biased_training > 0:
		i = np.random.randint(len(xi))
		dist = distance.cdist([X0[:,i]],X0.T,'correlation')[0]
		didx = np.argsort(dist)[1:int(biased_training*training_dictionary_size)+1]
	else:
		didx = []
	xi[didx] = True
	if biased_training < 1:
		remaining_idx = np.setdiff1d(range(len(xi)),didx)
		xi[np.random.choice(remaining_idx,training_dictionary_size-xi.sum(),replace=False)] = True
	xa = X0[:,xi]
	xb = X0[:,np.invert(xi)]
	print 'data: %s measurements: %d, sparsity: %d, dictionary size: %d, training fraction: %.2f, genes: %d, samples: %d, SNR: %.1f, bias: %.1f, composition_noise: %.2f, subset_size: %d' % (data_path,
			measurements,sparsity,dictionary_size,training_dictionary_fraction,X0.shape[0],X0.shape[1],SNR,biased_training,composition_noise,subset_size)
	Results = {}
	ua,sa,vta = np.linalg.svd(xa,full_matrices=False)
	ua = ua[:,:min(dictionary_size,xa.shape[1])]

	# x1a,phi,y,w,d,psi = recover_system_knownBasis(xa,measurements,sparsity,Psi=ua,snr=SNR,use_ridge=False)
	# Results['SVD (training)'] = compare_results(xa,x1a)
	# x1b,phi,y,w,d,psi = recover_system_knownBasis(xb,measurements,sparsity,Psi=ua,snr=SNR,use_ridge=False)
	# Results['SVD (testing)'] = compare_results(xb,x1b)
	# ua = ua[:,:min(sparsity,xa.shape[1])]
	# x1c,phi,y,w,d,psi = recover_system_knownBasis(xa,measurements,sparsity,Psi=ua,snr=SNR,use_ridge=False)
	# Results['k-SVD (training)'] = compare_results(xa,x1c)
	# x1d,phi,y,w,d,psi = recover_system_knownBasis(xb,measurements,sparsity,Psi=ua,snr=SNR,use_ridge=False)
	# Results['k-SVD (testing)'] = compare_results(xb,x1d)
	# ua,va = spams.nmf(np.asfortranarray(xa),return_lasso=True,K=dictionary_size,clean=True,numThreads=THREADS)
	# x2a,phi,y,w,d,psi = recover_system_knownBasis(xa,measurements,sparsity,Psi=ua,snr=SNR,use_ridge=False)
	# Results['sparse NMF (training)'] = compare_results(xa,x2a)
	# x2b,phi,y,w,d,psi = recover_system_knownBasis(xb,measurements,sparsity,Psi=ua,snr=SNR,use_ridge=False)
	# Results['sparse NMF (testing)'] = compare_results(xb,x2b)
	# Results['sparse NMF (sample_dist)'] = compare_distances(xb,y)


	# Ideas for making this more sensitive to lowly expressed genes:
	#	The root of the problem may be that the dictionary is biased towards high expression
	#	(because the loss function in SMAF wasn't told to specifically care about lowly expressed genes)
	#	So we might change the loss function to be sensitive to the average loss for each gene
	#	...possibly normalized somehow so that we care the same about each gene individually
	#	This may also help with the poor performance for each gene across tissues
	#	In addition, we might weight Phi by a prior on (inverse) expression
	#	Note that applying this weight without making the dictionary sensitive to low expression did not improve the results
	k = min(int(xa.shape[1]*1.5),150)
	UW = (np.random.random((xa.shape[0],k)),np.random.random((k,xa.shape[1])))
	ua,va = smaf(xa,k,5,0.0005,maxItr=10,use_chol=True,activity_lower=0.,module_lower=xa.shape[0]/10,UW=UW,donorm=True,mode=1,mink=3.)
	x2a,phi,y,w,d,psi = recover_system_knownBasis(xa,measurements,sparsity,Psi=ua,snr=SNR,use_ridge=False,f="smaf_train_measurement")
	Results['SMAF (training)'] = compare_results(xa,x2a)
	np.save("SMAF_gene_actual_train_100",xa)
	np.save("SMAF_gene_pred_train_100",x2a)
	x2b,phi,y,w,d,psi = recover_system_knownBasis(xb,measurements,sparsity,Psi=ua,snr=SNR,use_ridge=False,nsr_pool=composition_noise,subset_size=subset_size,f="smaf_test_measurement")
	np.save("SMAF_gene_actual_test_100",xb)
	np.save("SMAF_gene_pred_test_100",x2b)
	Results['SMAF (testing)'] = compare_results(xb,x2b)
	# sg = get_signature_genes(xa,measurements,lda=1000000)
	# model = build_signature_model(xa,sg)
	# X3 = (model.predict(xa[sg].T)).T
	# Results['Signature (training)'] = compare_results(xa,X3)
	# x0 = xb[sg].T
	# noise = np.array([np.random.randn(x0.shape[1]) for _ in range(x0.shape[0])])
	# noise *= np.linalg.norm(x0)/np.linalg.norm(noise)/SNR
	# X3 = (model.predict(x0 + noise)).T
	# Results['Signature (testing)'] = compare_results(xb,X3)
	# Results['Signature (sample_dist)'] = compare_distances(xb,X3[sg])
	for k,v in sorted(Results.items()):
		print '\t'.join([k] + [str(x) for x in v])
