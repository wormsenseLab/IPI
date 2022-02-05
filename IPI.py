#import basic libraries for plotting, data structures and signal processing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas
from scipy.signal import find_peaks, peak_widths, butter, filtfilt
from scipy.ndimage import gaussian_filter, minimum_filter
import imageio
import os
import fnmatch
import datetime

#%%
strain_key=pandas.DataFrame({('x82', 'GN965', 'WT', 'LAM-2::mNG', 0),
                             ('x175', 'GN1060', 'WT', 'LAM-1::wSc', 0),
                             ('x87a', 'GN973', 'WT', 'NID-1::wSc', 0)
                             }, columns=['Strain_code','Strain', 'Allele','Label', 'n'])


sigma_bgf=10
sigma_nf_2=20
sigma_nf_3=5

# Design the Buterworth filter
N  = 2    # Filter order
Wn = .2 # Cutoff frequency
B, A = butter(N, Wn, output='ba')

#define a custom function for calculating the neurite fluorescence from straightened neurite image
def neurite_fluorescence(img):
    n = img[7:13, :]                       #extract rows to use for neurite
    bg = np.concatenate((img[0:5, :], img[15: , :]))   #extract rows to use for background
    rawf = np.mean(n, axis=0)               #calculate average raw neurite fluorescence
    bgf = gaussian_filter(np.mean(bg, axis=0), sigma=sigma_bgf)             #calculate average background fluorescence
    nf = rawf - bgf                         #calculate background subtracted neurite fluorescence
    for i in range(0,len(nf)): 
        if nf[i]<0: nf[i]=0
    
    fnf_1 = filtfilt(B,A, nf)                   #apply Butterworth filter to smooth the data
    fnf_2 = minimum_filter(nf, sigma_nf_2)      #calculating the bottom edge of the signal
    fnf_3 = gaussian_filter(fnf_2, sigma_nf_3)  #smoothing the bottom edge of the signal
    fnf = fnf_1-fnf_3                           #subtracting the diffuse fluorescence from total fluorescence signal to obtain the the peak fluorescence that will be used for peak finding
    return(fnf_1,fnf_3,fnf)

# define a custom function to calculate the minimum height cutoff for finding peaks
def height_cutoff(fnf):
    avnoise = np.mean(fnf[fnf < np.percentile(fnf, 75)])
    stdnoise = np.std(fnf[fnf < np.percentile(fnf, 75)])
    height = avnoise + 3*stdnoise
    return(height)

#%%
fpath = 'G:/My Drive/ECM manuscript/github codes/IPI/sample_data/input_files/'          #filepath to the input file location
dfpath = 'G:/My Drive/ECM manuscript/github codes/IPI/sample_data/output_files/'        #destination filepath where output files will be created
imgfiles = fnmatch.filter(os.listdir(fpath), '*.tif')
toa = str(datetime.datetime.today()).split()
today = toa[0]
now = toa[1]
timestamp = today.replace('-','')+'-'+now.replace(':','')[:6]

#PARAMETERS
mu_per_px = 0.126     #pixels to microns conversion factor

#specify columns of the pandas dataframe and excel sheets
cols_Data =     ['Date', 'Strain', 'Allele', 'Label', 'Neuron', 'ImageID', 'Distance', 'Normalized distance', 'Neurite intensity']
cols_Peaks =    ['Date', 'Strain', 'Allele', 'Label', 'Neuron', 'ImageID', 'Distance', 'Normalized distance', 'Punctum max intensity', 'Punctum width']
cols_IPDs =     ['Date', 'Strain', 'Allele', 'Label', 'Neuron', 'ImageID', 'Distance', 'Normalized distance', 'Inter-punctum interval']
cols_Analysis = ['Date', 'Strain', 'Allele', 'Label', 'Neuron', 'ImageID', 'Image size', 'Max neurite length', 'Average neurite intensity','Total peaks', 'Average peak intensity', 'Average peak width', 'Average ipd', 'Median ipd']

#initialize Pandas DataFrames
df_Data = pandas.DataFrame()
df_Peaks = pandas.DataFrame()
df_IPDs = pandas.DataFrame()
df_Analysis = pandas.DataFrame()

#%%
for x in imgfiles:                          #create loop for number of images in folder
    img = imageio.imread(fpath+x)    #import image and store it in a list of lists
    
    
    #extract info from filename
    date=x.split('_')[0]
    strain = x.split('_')[1].split('-')[0]
    row_index=strain_key[(strain_key['Strain']==strain)|(strain_key['Strain_code']==strain)].index[0]
    allele = strain_key.loc[row_index,'Allele']
    label =  strain_key.loc[row_index,'Label']
    count = strain_key.loc[row_index,'n'] + 1
    strain_key.at[row_index,'n']=count
    neuron = 'ALM'

    imsize = np.shape(img)                  #calculate image size
    d=np.arange(imsize[1])                  #create list of integers from 0 to length of image for x-axis
    dist = d*mu_per_px                      #pixel to microns conversion
    normdist=d/d[-1]

    
    nf, bsf, gf_nf = neurite_fluorescence(img)
    
    height = height_cutoff(gf_nf)

    #find peaks
    peaks = find_peaks(gf_nf, height=height, prominence=0.5*height)

    pd = (peaks[0])*mu_per_px
    pnd = pd/dist[-1]
    ph = [gf_nf[i] for i in peaks[0]]
    ipd = np.diff(pd)
    ipdd = [pd[i]+ipd[i]/2 for i in range(0,len(ipd))]
    ipdnd = ipdd/dist[-1]
    pw=peak_widths(gf_nf, peaks[0], rel_height=0.5)
         
    plt.figure(1, figsize=(0.010*imsize[1],15))
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.title(x+' peaks')
    plt.xlabel('Distance (um)')
    plt.ylabel('Intensity (AU)')
    plt.plot(normdist, gf_nf,'y-')
    plt.plot(pnd, ph, 'go')
    plt.hlines(height, 0, normdist[-1])
    plt.axis([0, 1, 0, max(gf_nf)])

    sns.set_style('white')
    sns.set_style('ticks', {'xtick.direction': 'in', 'ytick.direction': 'in'})
    sns.despine(offset=5, trim=False)
    
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['svg.fonttype'] = 'none'

    plt.savefig(dfpath+timestamp+'_'+x[:-4]+'_analysis.png')
    
    plt.show()
    plt.close()
    
        
    #add data to pandas dataframe
    all_data1 = pandas.DataFrame({'Date':[date]*imsize[1], 'Strain':[strain]*imsize[1], 'Allele':[allele]*imsize[1], 'Label':[label]*imsize[1], 'Neuron':[neuron]*imsize[1], 'ImageID':[x]*imsize[1], 'Distance':dist, 'Normalized distance':normdist, 'Neurite intensity':gf_nf}, columns=cols_Data)
    df_Data=df_Data.append(all_data1)
    all_data2 = pandas.DataFrame({'Date':[date]*len(pd), 'Strain':[strain]*len(pd), 'Allele':[allele]*len(pd), 'Label':[label]*len(pd), 'Neuron':[neuron]*len(pd), 'ImageID':[x]*len(pd), 'Distance':pd, 'Normalized distance':pnd, 'Punctum max intensity':ph, 'Punctum width':pw[0]*mu_per_px}, columns=cols_Peaks)
    df_Peaks=df_Peaks.append(all_data2)
    all_data3 = pandas.DataFrame({'Date':[date]*len(ipd), 'Strain':[strain]*len(ipd), 'Allele':[allele]*len(ipd), 'Label':[label]*len(ipd), 'Neuron':[neuron]*len(ipd), 'ImageID':[x]*len(ipd), 'Distance':ipdd, 'Normalized distance':ipdnd, 'Inter-punctum interval':ipd}, columns=cols_IPDs)
    df_IPDs=df_IPDs.append(all_data3)    
    frame = pandas.DataFrame([[date, strain, allele, label, neuron, x, imsize[1], dist[-1], np.mean(gf_nf), len(pd), np.mean(ph), np.mean(pw[0]*mu_per_px), np.mean(ipd), np.median(ipd)]], columns=cols_Analysis)
    df_Analysis = df_Analysis.append(frame)
#%%
df_temp=pandas.DataFrame()
for item in strain_key['Label']:
    mean_ipd = np.mean(df_IPDs['Inter-punctum interval'][(df_IPDs['Label']==item)])
    median_ipd = np.median(df_IPDs['Inter-punctum interval'][(df_IPDs['Label']==item)])
    std_ipd = np.std(df_IPDs['Inter-punctum interval'][(df_IPDs['Label']==item)])
    mean_nl = np.mean(df_Analysis['Max neurite length'][(df_Analysis['Label']==item)])
    mean_pw = np.mean(df_Peaks['Punctum width'][(df_Peaks['Label']==item)])
    frame = pandas.DataFrame([[item, mean_ipd, median_ipd, std_ipd, mean_nl, mean_pw]], columns=['Label', 'Mean ipd', 'Median ipd', 'Stdev ipd', 'Mean neurite length', 'Mean puncta width'])
    df_temp = df_temp.append(frame)
df_new=strain_key.merge(df_temp, how='outer', on='Label')

#%%
# save data to excel file
wb = pandas.ExcelWriter(dfpath+timestamp+'_Analysis.xlsx', engine='xlsxwriter')
df_Analysis.to_excel(wb, sheet_name='Analysis')
df_new.to_excel(wb, sheet_name='Summarized strain info')
wb.save()

df_Data.to_pickle(dfpath+timestamp+'_Data.pkl')
df_Peaks.to_pickle(dfpath+timestamp+'_Peaks.pkl')
df_IPDs.to_pickle(dfpath+timestamp+'_IPDs.pkl')
df_Analysis.to_pickle(dfpath+timestamp+'_Analysis.pkl')
