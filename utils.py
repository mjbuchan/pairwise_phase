from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):

    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):

    from scipy.signal import butter, lfilter

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)

    return y

def analytic_signal(x):
    
    from scipy.fftpack import fft,ifft
    import numpy as np

    N = len(x)
    X = fft(x,N)
    h = np.zeros(N)
    h[0] = 1
    h[1:N//2] = 2*np.ones(N//2-1)
    h[N//2] = 1
    Z = X*h
    z = ifft(Z,N)

    return z

def transform_lfp(lfp):
    
    import numpy as np

    hilbert_lfp_real = []
    hilbert_lfp_imag = []
    
    for trial in lfp:
    
        x = butter_bandpass_filter(trial, 19,21 , fs = 1000, order = 1)

        z = analytic_signal(x)

        t = np.arange(0,5.001,1/1000)
        
        hilbert_lfp_real.append(z.real)
        hilbert_lfp_imag.append(z.imag)
        
    return hilbert_lfp_real, hilbert_lfp_imag

def whisk_phase(pw_ID, whisk_1, lfp_real_1, lfp_imag_1, whisk_2, lfp_real_2, lfp_imag_2):

    import numpy as np 
    
####################### first whisker ###########################        
####################### first whisker ###########################   
####################### first whisker ###########################   
####################### first whisker ###########################   
    
    w1_ppc = []

    for neuron in range(len(whisk_1)):
        
        print('starting neuron', neuron, 'PW')

        spike_times = whisk_1[neuron]
        
        neuron_ppc = []

        for trial in range(len(whisk_1[neuron])):

            hist, bins = np.histogram(spike_times[trial], 5000, range = (0,5))
            
            trial_vectors = []
            
            trial_vectors = [[lfp_real_1[trial][time], lfp_imag_1[trial][time]] for time in np.argwhere(hist==1)]
            
            trial_vectors = np.squeeze(trial_vectors)
            
            ppc_matrix = np.zeros((len(trial_vectors),len(trial_vectors)))
        
            for i in range(len(trial_vectors)):
            
                for j in range(len(trial_vectors)):

                    ppc_matrix[i][j] = np.dot(trial_vectors[i]/np.linalg.norm(trial_vectors[i]), trial_vectors[j]/np.linalg.norm(trial_vectors[j]))
                
            xu, yu = np.triu_indices_from(ppc_matrix, k=1)
        
            out = np.ones((len(trial_vectors),len(trial_vectors)), dtype=bool)
            
            out[(xu, yu)] = False
            
            neuron_ppc.append(np.mean(ppc_matrix[~out],0))
            
        w1_ppc.append(np.nanmean(neuron_ppc,0))
        
        
####################### second whisker ###########################        
####################### second whisker ########################### 
####################### second whisker ########################### 
####################### second whisker ########################### 
        
    w2_ppc = []

    for neuron in range(len(whisk_2)):

        print('starting neuron', neuron, 'AW')

        spike_times = whisk_2[neuron]
        
        neuron_ppc = []

        for trial in range(len(whisk_2[neuron])):

            hist, bins = np.histogram(spike_times[trial], 5000, range = (0,5))
            
            trial_vectors = []
            
            trial_vectors = [[lfp_real_2[trial][time], lfp_imag_2[trial][time]] for time in np.argwhere(hist==1)]
            
            trial_vectors = np.squeeze(trial_vectors)
            
            ppc_matrix = np.zeros((len(trial_vectors),len(trial_vectors)))
        
            for i in range(len(trial_vectors)):
            
                for j in range(len(trial_vectors)):

                    ppc_matrix[i][j] = np.dot(trial_vectors[i]/np.linalg.norm(trial_vectors[i]), trial_vectors[j]/np.linalg.norm(trial_vectors[j]))
                
            xu, yu = np.triu_indices_from(ppc_matrix, k=1)
        
            out = np.ones((len(trial_vectors),len(trial_vectors)), dtype=bool)
            
            out[(xu, yu)] = False
            
            neuron_ppc.append(np.mean(ppc_matrix[~out],0))
            
        w2_ppc.append(np.nanmean(neuron_ppc,0))
        
    
    if pw_ID == 1:
        
        pw_ppc = w1_ppc
        aw_ppc = w2_ppc
        
    if pw_ID == 2:
        
        pw_ppc = w2_ppc
        aw_ppc = w1_ppc
        
    return w1_ppc, w2_ppc