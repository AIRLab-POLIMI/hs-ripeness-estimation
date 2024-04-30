# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections as collections
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from collections import Counter
import hydra
from omegaconf import DictConfig
import os

def snv(input_data):
    """
    Applies Standard Normal Variate

        Parameters:
            input_data (np.array): data to correct

        Returns:
            output_data (np.array): corrected data
    """
  
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
 
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
 
    return output_data

def msc(input_data, reference=None):
    """
    Applies Multi Scattering Correction

        Parameters:
            input_data (np.array): data to correct
            reference (np.array): reference spectrum, defaults to the mean of the input data

        Returns:
            data_msc (np.array): corrected data
    """
    input_data_ = np.copy(input_data)

    # mean centre correction
    for i in range(input_data_.shape[0]):
        input_data_[i,:] -= input_data_[i,:].mean()
    
    # Get the reference spectrum. If not given, estimate it from the mean
    # Define a new array and populate it with the corrected data 
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_data_, axis=0)
    else:
        ref = reference 
       
    data_msc = np.zeros_like(input_data_)
    for i in range(input_data_.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data_[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data_[i,:] - fit[0][1]) / fit[0][0]
    return data_msc

def pls_optimise_components(X, y, npc, cv):
    """
    Performs single loop cross-validation optimization of the number of components by minimizing the mean squared error

        Parameters:
            X (np.array): independent data
            y (np.array): target data
            npc (int): maximum number of components of the PLSR model
            cv (int): number of folds for cross-validation

        Returns:
            opt_comp (int): optimal numer of components
            msecv_min (float): mean squared error obtained with the optimal number of components
    """

    msecvs = np.zeros(npc)
    # for all considered components
    for components in range(1,npc+1,1):
 
        # build a PLSR model
        pls_simple = PLSRegression(n_components=components)
 
        # Calculate the cross-validation mean squared error
        msecv = -np.mean(cross_validate(estimator=pls_simple,X=X,y=y,scoring='neg_mean_squared_error',cv=cv)['test_score'])
        msecvs[components-1] = msecv
 
    # Find the minimum of ther MSE and its location
    opt_comp, msecv_min = np.argmin(msecvs),  msecvs[np.argmin(msecvs)]
 
    return (opt_comp+1, msecv_min)

def SG_cross_val(X, y, indices, derivative, order, window_length, max_comp, cv, folder_path, data_nested_loop, data_single_loop):
    """
    Performs the nested cross-validation loop for all Savitsky-Golay preprocessing options and returns the
    best triplet of preprocessing parameters. The CV results for each preprocessing is saved in an apposite dataset. 

        Parameters:
            X (np.array): independent data
            y (np.array): target data
            indices (list): iterable object containing the training and validation indexes for each outer fold
            derivative (list): list of derivatives of the preprocessings
            order (list): list of polynomial orders of interpolation of the preprocessings
            window_length (list): list of window lenghts of the preprocessings
            max_comp (int): maximum number of components of the PLSR model

        Returns:
            der (int): optimal derivative parameter
            ord (int): optimal order parameter
            wl (int): optimal window length parameter
    """

    # list for storing the data of the preprocessings
    sg_data = []
    # matrix for storing the MSE values achieved by the preprocessings
    results = np.zeros((len(derivative),len(order), len(window_length)))
    # cycles on the Savitzky-Golay parameters
    for k, der in enumerate(tqdm(derivative)):
        for i,ord in enumerate(order):
            for j,wl in enumerate(window_length):
                # consistency check for the correct use of Savitzky-Golay
                if ord < wl:
                    # apply the preprocessing
                    X_sg = savgol_filter(X, wl, polyorder=ord, deriv=der)
                    to_exclude = wl//2
                    X_sg = X_sg[:,to_exclude:-to_exclude]

                    preprocessing = [der,ord,wl]
                    r2_mean, r2_std, mse_mean, mse_std, rmse_mean, rmse_std, n_components_cv, true_ys_cv, pred_ys_cv, data_per_fold, datum_total = nested_cv(X_sg,y,max_comp,indices,cv,preprocessing)
                    results[k,i,j] = mse_mean
                    data_nested_loop.append(datum_total)

                    best_comp, r2_mean, r2_std, mse_mean, mse_std, rmse_mean, rmse_std, datum = single_cv(X_sg,y,max_comp,indices,preprocessing)
                    data_single_loop.append(datum)

    # select the preprocessing with minimum nested CV MSE
    mink,minx,miny = np.where(results==np.min(results[np.nonzero(results)]))
    print('')
    print('Quantity: ', results[mink[0],minx[0],miny[0]])
    print('Derivative: ', derivative[mink[0]])
    print('Order: ', order[minx[0]])
    print('Window length: ', window_length[miny[0]])
    print('')

    return derivative[mink[0]], order[minx[0]], window_length[miny[0]]


def create_folds(df_original,target, n_s=10, n_grp = 5):
    """
    Creates the outer folds for the nested cross-validation

        Parameters:
            df_orginal (DataFrame): data to be split
            target (str): column name of the target variable
            n_s (int): number of outer splits
            n_grp (int): number of groups used for the stratification of the folds based on the target variable

        Returns:
            myCViterator (list): list containing (training,validation) pairs of lists of indexes for each fold
    """

    df = df_original.copy()
    df['Fold'] = -1

    skf = StratifiedKFold(n_splits=n_s)
    df['grp'] = pd.qcut(df[target], n_grp, labels=False)
    target = df.grp

    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        df.loc[v, 'Fold'] = fold_no

    myCViterator = []
    for i in np.unique(df.Fold.values):
        trainIndices = df[ df['Fold']!=i ].index.values.astype(int)
        testIndices =  df[ df['Fold']==i ].index.values.astype(int)
        myCViterator.append( (trainIndices, testIndices) )
    
    return myCViterator

def center_scale_X(X, y, train_idx, test_idx):
    """
    Divides the data in training and validation set while mean centering and std scaling the data

        Parameters:
            X (np.array): input data
            y (np.array): target variable
            train_idx (list): indexes of the training data
            test_idx (list): indexes of the validation data

        Returns:
            X_train (np.array): training input data
            y_train (np.array): traingin target data
            X_val (np.array): validation input data
            y_val (np.array): validation target data
    """

    X_train = np.array(X[train_idx],float)
    X_val = np.array(X[test_idx],float)
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    X_train = (X_train - mean)/std
    X_val = (X_val - mean)/std
    y_train = y[train_idx]
    y_val = y[test_idx]

    return X_train, y_train, X_val, y_val


def nested_cv(X,y,max_comp,indices,cv,preprocessing):
    """
    Performs nested cross-valdiation

        Parameters:
            X (np.array): independent data
            y (np.array): target data
            max_comp (int): maximum number of components of the PLSR model
            indices (list): iterable object containing the training and validation indexes for each outer fold
            cv (int): number of folds of the inner CV loop           

        Returns:
            r2_mean, r2_std (float,float): mean and standard deviation of the coefficient of determination in the outer folds
            mse_mean, mse_std (float,float): mean and standard deviation of the mean squared error in the outer folds
            rmse_mean, rmse_std (float,float): mean and standard deviation of the root mean squared error in the outer folds
            n_components_cv (list): number of optimized PLSR components in the outer folds 
            true_ys_cv (np.array): true target values
            pred_ys_cv (np.array): predicted target values
            data_per_fold (list(list)): list containing the information about the single folds 
            datum_total (list): list containing the final nested CV evaluations
    """

    # lists for storing the results in the outer folds
    mse_cv = []
    rmse_cv = []
    r2_cv = []
    n_components_cv = []
    true_ys_cv = []
    pred_ys_cv = []
    data_per_fold = []

    # for each outer fold
    for train_idx,test_idx in indices:
        # divide the data in training set and validation set
        X_train, y_train, X_test, y_test = center_scale_X(X, y, train_idx, test_idx)  

        # optimize the number of components of the PLSR model on the training set (inner CV loop)
        ncomp, _ = pls_optimise_components(X=X_train, y=y_train, npc=max_comp, cv=cv)
        pls_fin = PLSRegression(n_components=ncomp)

        # fit the optimized PLSR model on the training set and predict the target values of the validation set
        pls_fin.fit(X_train,y_train)
        y_pred = pls_fin.predict(X_test)

        # compute and store the quantities of interest
        mse = mean_squared_error(y_test,y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test,y_pred)
        for i in range(len(test_idx)):
            true_ys_cv.append(y_test[i])
            pred_ys_cv.append(y_pred[i])
        mse_cv.append(mse)
        rmse_cv.append(rmse)
        r2_cv.append(r2)
        n_components_cv.append(ncomp)
        datum = [mse,None,rmse,None,r2,None,ncomp]
        data_per_fold.append(datum)

    # compute the mean and standard deviation of the quantities on the outer folds 
    r2_mean = np.mean(r2_cv)
    r2_std = np.std(r2_cv)
    mse_mean = np.mean(mse_cv)
    mse_std = np.std(mse_cv)
    rmse_mean = np.mean(rmse_cv)
    rmse_std = np.std(rmse_cv)
    datum = [mse_mean,mse_std,rmse_mean,rmse_std,r2_mean,r2_std,None]
    data_per_fold.append(datum)
    datum_total = preprocessing.copy()
    for tt in range(len(datum)-1):
        datum_total.append(datum[tt])

    return r2_mean, r2_std, mse_mean, mse_std, rmse_mean, rmse_std, n_components_cv, true_ys_cv, pred_ys_cv, data_per_fold, datum_total

def pred_vs_true_plot(pred_ys_cv,true_ys_cv,r2_mean,r2_std,rmse_mean,rmse_std,chemical_target):
    """
    Performs the predictive plots of the nested CV procedure

        Parameters:
            pred_ys_cv (np.array): true target values
            true_ys_cv (np.array): predicted target values
            r2_mean, r2_std (float,float): mean and standard deviation of the coefficient of determination in the outer folds
            rmse_mean, rmse_std (float,float): mean and standard deviation of the root mean squared error in the outer folds
    """
    true_ys_cv = np.array(true_ys_cv)
    pred_ys_cv = np.array(pred_ys_cv)
    z = np.polyfit(true_ys_cv, pred_ys_cv, 1)
    plt.figure()
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(pred_ys_cv, true_ys_cv, c='red', edgecolors='k')
        ax.plot(z[1]+z[0]*true_ys_cv, true_ys_cv, c='blue', linewidth=1)
        ax.plot(true_ys_cv, true_ys_cv, color='green', linewidth=1)
        plt.title('$R^{2}$:'+str(round(r2_mean,2))+'+'+str(round(r2_std,2))+', RMSE:'+str(round(rmse_mean,2))+'+'+str(round(rmse_std,2)))
        if chemical_target == 'Brix':
            plt.xlim([7.8,22.2])
            plt.ylim([7.8,22.2])
            plt.xlabel('Predicted TSS (°Brix)')
            plt.ylabel('Measured TSS (°Brix)')
        else:
            plt.xlim([-0.05,0.29])
            plt.ylim([-0.05,0.29])
            plt.xlabel('Predicted Anthocyanins (mg/g)')
            plt.ylabel('Measured Anthocyanins (mg/g)')

    
        

def single_cv(X,y,max_comp,indices,preprocessing):
    """
    Performs nested cross-valdiation

        Parameters:
            X (np.array): independent data
            y (np.array): target data
            max_comp (int): maximum number of components of the PLSR model
            indices (list): iterable object containing the training and validation indexes for each outer fold
            preprocessing (list): list containing information about the preprocessing (for saving the results)      

        Returns:
            best_comp (int): number of PLSR components optimized by single loop CV
            r2_mean, r2_std (float,float): mean and standard deviation of the coefficient of determination in the outer folds
            mse_mean, mse_std (float,float): mean and standard deviation of the mean squared error in the outer folds
            rmse_mean, rmse_std (float,float): mean and standard deviation of the mean squared error in the outer folds
            datum (list): list containing the final single loop CV evaluations
    """

    # arrays for storing the results
    mse_cv = np.zeros(max_comp)
    mse_cv_std = np.zeros(max_comp)
    rmse_cv = np.zeros(max_comp)
    rmse_cv_std = np.zeros(max_comp)
    r2_cv = np.zeros(max_comp)
    r2_cv_std = np.zeros(max_comp)

    # for every component
    for a in range(max_comp):

        # lists for storing the results
        mse_a = []
        rmse_a = []
        r2_a = []

        # for each fold
        for train_idx,test_idx in indices:
            # divide the data in training set and validation set
            X_train, y_train, X_test, y_test = center_scale_X(X, y, train_idx, test_idx)  

            # fit a PLSR model using the training set and predict the target value of the validation set
            pls = PLSRegression(a+1)
            pls.fit(X_train,y_train)
            y_pred = pls.predict(X_test)

            # compute the quantities of interes
            mse = mean_squared_error(y_test,y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test,y_pred)
            mse_a.append(mse)
            rmse_a.append(rmse)
            r2_a.append(r2)
        
        # compute the mean of the quantities on the folds
        mse_cv[a] = np.mean(mse_a)
        mse_cv_std[a] = np.std(mse_a)
        rmse_cv[a] = np.mean(rmse_a)
        rmse_cv_std[a] = np.std(rmse_a)
        r2_cv[a] = np.mean(r2_a)
        r2_cv_std[a] = np.std(r2_a)
        
    # select the number of components yielding the lowest mean squared error
    best_a = np.argmin(mse_cv)

    # save the results
    datum = []
    for p in preprocessing:
        datum.append(p)
    datum.append(best_a+1)
    for t in [mse_cv[best_a], mse_cv_std[best_a],rmse_cv[best_a], rmse_cv_std[best_a],r2_cv[best_a], r2_cv_std[best_a]]:
        datum.append(t)

    return best_a+1, r2_cv[best_a], r2_cv_std[best_a], mse_cv[best_a], mse_cv_std[best_a], rmse_cv[best_a], rmse_cv_std[best_a], datum
        

@hydra.main(version_base=None, config_path="/workspace/conf", config_name="config")
def prediction(cfg: DictConfig):
    
    seed = 7111999
    np.random.seed(seed)

    analysis = cfg.db.prediction.analysis

    chemical_target = cfg.db.prediction.chemical_target
    if (analysis != 'bunches') and (analysis != 'plants'):
        raise(Exception('The specfied analysis is invalid'))

    cv = cfg.db.prediction.inner_cv
    test_cv = cfg.db.prediction.outer_cv

    max_comp = cfg.db.prediction.max_comp

    derivatives=list(cfg.db.prediction.derivatives)
    orders=list(range(cfg.db.prediction.orders_start,cfg.db.prediction.orders_stop))
    window_lengths=list(range(cfg.db.prediction.wl_start,cfg.db.prediction.wl_stop))
    selected_range_start = cfg.db.prediction.selected_range_start
    selected_range_stop = cfg.db.prediction.selected_range_stop
    selected_range = list(range(selected_range_start,selected_range_stop))

    # load dataset
    if analysis == 'bunches':
        folder_path = '/home/user/data/results/bunches/'+chemical_target
        dataset_path = '/home/user/data/processed/2021-09-06-bunches/dataset.csv'
        data = pd.read_csv(dataset_path)
    elif analysis == 'plants':
        folder_path = '/home/user/data/results/plants/'+chemical_target
        dataset_path = '/home/user/data/processed/plants/plants_dataset.xlsx'
        data = pd.read_excel(dataset_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plots_folder = folder_path+'/plots'
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        

    # shuffle the rows
    data = data.sample(frac=1,random_state=seed).reset_index().iloc[:,1:]
    data.head()
    if chemical_target == 'Antociani':
        data = data.dropna()
        data.reset_index(drop=True, inplace=True)
    print('Chemical target: ', chemical_target)
    y = data[chemical_target].values
    X = data.values[:, 1:(len(selected_range)+1)]
    X_snv = snv(np.array(X,dtype = float))
    X_msc = msc(np.array(X,dtype = float))
    X = snv(np.array(X,dtype = float))

    if analysis != 'plants':
        indices = create_folds(data, chemical_target ,n_s = test_cv)
    else:
        indices = create_folds(data, chemical_target ,n_s = test_cv,n_grp=3)
    print('Dataset: ',analysis)
    print('CV: ', str(cv))


    data_single_loop = []
    data_nested_loop = []


    print('')
    print('Standard Normal Variate:')
    print('')

    preprocessing = ['snv',None,None]
    r2_mean, r2_std, mse_mean, mse_std, rmse_mean, rmse_std, n_components_cv, true_ys_cv, pred_ys_cv, data_per_fold, datum_total = nested_cv(X_snv,y,max_comp,indices,cv,preprocessing)

    df = pd.DataFrame(data=data_per_fold, columns=['mse','mse_std','rmse','rmse_std','r2','r2_std','components'])
    df.to_excel(folder_path+'/SNV_cv.xlsx', index=False)

    data_nested_loop.append(datum_total)


    print('R2: mean ', r2_mean ,' , std ', r2_std)
    print('MSE: mean ', mse_mean ,' , std ', mse_std)
    print('RMSE: mean ', rmse_mean ,' , std ', rmse_std)

    pred_vs_true_plot(pred_ys_cv,true_ys_cv,r2_mean,r2_std,rmse_mean,rmse_std,chemical_target)
    plt.savefig(plots_folder+'/SNV_predicted_vs_true.png')
    plt.close()

    best_comp, r2_mean, r2_std, mse_mean, mse_std, rmse_mean, rmse_std, datum = single_cv(X_snv,y,max_comp,indices,preprocessing)
    data_single_loop.append(datum)
    print('Best #LV: ',best_comp)

    print('')
    print('Multi Scattering Correction:')
    print('')

    preprocessing = ['msc',None,None]
    r2_mean, r2_std, mse_mean, mse_std, rmse_mean, rmse_std, n_components_cv, true_ys_cv, pred_ys_cv, data_per_fold, datum_total = nested_cv(X_msc,y,max_comp,indices,cv,preprocessing)

    df = pd.DataFrame(data=data_per_fold, columns=['mse','mse_std','rmse','rmse_std','r2','r2_std','components'])
    df.to_excel(folder_path+'/MSC_cv.xlsx', index=False)

    data_nested_loop.append(datum_total)

    print('R2: mean ', r2_mean ,' , std ', r2_std)
    print('MSE: mean ', mse_mean ,' , std ', mse_std)
    print('RMSE: mean ', rmse_mean ,' , std ', rmse_std)

    pred_vs_true_plot(pred_ys_cv,true_ys_cv,r2_mean,r2_std,rmse_mean,rmse_std,chemical_target)
    plt.savefig(plots_folder+'/MSC_predicted_vs_true.png')
    plt.close()

    best_comp, r2_mean, r2_std, mse_mean, mse_std, rmse_mean, rmse_std, datum = single_cv(X_msc,y,max_comp,indices,preprocessing)
    data_single_loop.append(datum)
    print('Best #LV: ',best_comp)


    print('')
    print('Savitzky-Golay:')
    print('')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        der, ord, wl = SG_cross_val(X=X,y=y,indices=indices,derivative=derivatives,order=orders,window_length=window_lengths,max_comp=max_comp,cv=cv,folder_path=folder_path,data_nested_loop=data_nested_loop,data_single_loop=data_single_loop)
    XX = savgol_filter(X, wl, polyorder=ord, deriv=der)
    to_exclude = wl//2
    XX = XX[:,to_exclude:-to_exclude]

    preprocessing = [der,ord,wl]
    r2_mean, r2_std, mse_mean, mse_std, rmse_mean, rmse_std, n_components, true_ys_cv, pred_ys_cv, data_per_fold, datum_total = nested_cv(XX,y,max_comp,indices,cv,preprocessing)

    df = pd.DataFrame(data=data_per_fold, columns=['mse','mse_std','rmse','rmse_std','r2','r2_std','components'])
    df.to_excel(folder_path+'/SG_cv.xlsx', index=False)

    print('R2: mean ', r2_mean ,' , std ', r2_std)
    print('MSE: mean ', mse_mean ,' , std ', mse_std)
    print('RMSE: mean ', rmse_mean ,' , std ', rmse_std)

    pred_vs_true_plot(pred_ys_cv,true_ys_cv,r2_mean,r2_std,rmse_mean,rmse_std,chemical_target)
    plt.savefig(plots_folder+'/SG_predicted_vs_true.png')
    plt.close()

    best_comp, r2_mean, r2_std, mse_mean, mse_std, rmse_mean, rmse_std, datum = single_cv(XX,y,max_comp,indices,preprocessing)
    print('Best #LV: ',best_comp)


    nested_colnames = ['der','ord','wl']
    for tt in [ 'mse_mean', 'mse_std', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']:
        nested_colnames.append(tt)
    df = pd.DataFrame(data=data_nested_loop, columns=nested_colnames)
    df.to_excel(folder_path+'/nested_loop.xlsx', index=False)

    single_colnames = ['der','ord','wl','Best']
    for tt in ['mse_mean', 'mse_std','rmse_mean', 'rmse_std','r2_mean', 'r2_std']:
        single_colnames.append(tt)
    df = pd.DataFrame(data=data_single_loop, columns=single_colnames)
    df.to_excel(folder_path+'/single_loop.xlsx', index=False)


if __name__ == '__main__':
    prediction()