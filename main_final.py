#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:36:24 2022

@author: pierrehouzelstein
"""
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from Brian_functions_final import  Brian_run
from MF_functions_final import MF_run, read_P_data

def main():
    startTime = datetime.now()
    
    #Define all parameters
    #Integration time parameters (s)
    TotTime = 2; timestep = 0.0001; T= 0.005
    #Amount of integration time we erase to keep only steady state when averaging
    percentage = 0.9
    
    #Network parameters
    N1 = 2000; N2 = 8000
    prbC = 0.05
    Ki = N1*prbC; Ke = N2*prbC
    
    #Izhikevich neurons parameters
    gizi = 0.04 ; Eizi = -60
    gize = 0.01; Eize = -65
    tauize=1; tauizi=1
    Tve = 1; Tvi = 1
    
    #adaptation parameters
    aFS = 1; bFS = 0.5; cFS = -60; dFS = 0
    aRS = 1; bRS = 0.2; cRS = -65; dRS = 15
    Tue = 1; Tui = 1
    
    #Synaptic current terms
    Ee = 0; Ei = -80
    Qe = 1.5; Qi = 5.0
    Tsyne = 5e-3; Tsyni = 5e-3
    tause=5e-3; tausi=5e-3
    
    #External population frequency
    nu_ext = 10
    
    #Input current
    Ie = 0; Ii = 0
    
    #TF fit
    PRS, PFS = read_P_data()
    
    #test
    #Brian
    Br_excitatory_params = [N2, Qe, Tve, Ee, Ei, Ie, gize, Eize, aRS, bRS, cRS, dRS, Tue, Tsyne]
    Br_inhibitory_params = [N1, Qi, Tvi, Ee, Ei, Ii, gizi, Eizi, aFS, bFS, cFS, dFS, Tui, Tsyni]

    trajectories, mean_results = Brian_run(prbC, nu_ext, Br_excitatory_params, Br_inhibitory_params, TotTime, timestep, T, percentage)
    
    time_Brian = trajectories[0]
    BrPe = trajectories[1]; BrPi = trajectories[2]
    BrpopRateG_exc= trajectories[3]; BrpopRateG_inh = trajectories[4]
    
    #MF
    muve_params = [gize, Eize, Qe, Qi, Ke, Ki, tause, tausi, Ee, Ei, bRS]
    muvi_params = [gizi, Eizi, Qe, Qi, Ke, Ki, tause, tausi, Ee, Ei, bFS]
    sv_params = [gize, Eize, gizi, Eizi, Qe, Qi, Ee, Ei, tause, tausi, Ke, Ki, tauize, tauizi]
    #Make MF start where Brian integration starts
    #nueIni=BrpopRateG_exc[1]; nuiIni=BrpopRateG_inh[1]
    nueIni=5; nuiIni=10
    
    MFtrajectories, MFmean_results = MF_run(nueIni, nuiIni, nu_ext, PRS, PFS, aRS, dRS, sv_params, muvi_params, muve_params, T, TotTime, timestep, percentage)
    
    MF_t = MFtrajectories[0]; MFLSfe = MFtrajectories[2]; MFLSfi = MFtrajectories[3]
    MFPe = MFtrajectories[4]; MFPi = MFtrajectories[5]
    
    #see if everything's working fine
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(time_Brian, BrPe, "b-")
    plt.plot(time_Brian, BrPi, "r-")
    plt.plot(MF_t, MFPe, "b--")
    plt.plot(MF_t, MFPi, "r--")
    plt.title("Test run: membrane potential evolution across time")
    plt.xlabel("t (s)")
    plt.ylabel(r"$\mu_V$ (mV)")
    plt.legend(["Brian - Excitatory", "Brian - Inhibitory", "MF - Excitatory", "MF - Inhibitory"])
    plt.savefig("./test_mu.png")
    plt.show()

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(time_Brian, BrpopRateG_exc, "b-")
    plt.plot(time_Brian, BrpopRateG_inh, "r-")
    plt.plot(MF_t, MFLSfe, "b--")
    plt.plot(MF_t, MFLSfi, "r--")
    plt.title("Test run: mean firing rate evolution across time")
    plt.xlabel("t (s)")
    plt.ylabel(r"$\nu$ (Hz)")
    plt.legend(["Brian - Excitatory", "Brian - Inhibitory", "MF - Excitatory", "MF - Inhibitory"])
    plt.savefig("./test_nu.png")
    plt.show()
    
    print("Checking adaptation effects...")
    d_adaplist = np.linspace(0, 30, 20)
    #prepare lists
    mean_pot_list_E = np.zeros(len(d_adaplist)); mean_pot_list_I = np.zeros(len(d_adaplist))
    mean_nue_list = np.zeros(len(d_adaplist)); mean_nui_list = np.zeros(len(d_adaplist))
    
    MF_mean_pot_list_E = np.zeros(len(d_adaplist)); MF_mean_pot_list_I = np.zeros(len(d_adaplist))
    MF_mean_nue_list = np.zeros(len(d_adaplist)); MF_mean_nui_list = np.zeros(len(d_adaplist))
    
    for i in tqdm(range(len(d_adaplist))):
        #Change adaptations
        d_test = d_adaplist[i]
        #Put everything cleanly for Brian
        excitatory_params = [N2, Qe, Tve, Ee, Ei, Ie, gize, Eize, aRS, bRS, cRS, d_test, Tue, Tsyne]
        inhibitory_params = [N1, Qi, Tvi, Ee, Ei, Ii, gizi, Eizi, aFS, bFS, cFS, d_test, Tui, Tsyni]
        #Run Brian
        trajectories, mean_results = Brian_run(prbC, nu_ext, excitatory_params, inhibitory_params, TotTime, timestep, T, percentage)
        #Store
        mean_pot_list_E[i]= mean_results[0]; mean_pot_list_I[i] = mean_results[1]
        mean_nue_list[i] = mean_results[2]; mean_nui_list[i] = mean_results[3]
        #Run MF
        MFtrajectories, MFmean_results = MF_run(nueIni, nuiIni, nu_ext, PRS, PFS, aRS, d_test, sv_params, muvi_params, muve_params, T, TotTime, timestep, percentage)
        #Store
        MF_mean_pot_list_E[i] = MFmean_results[0]; MF_mean_pot_list_I[i] = MFmean_results[1]
        MF_mean_nue_list[i] = MFmean_results[2]; MF_mean_nui_list[i] = MFmean_results[3]
        
    
    plt.figure(figsize=(8, 6), dpi=80)
    
    plt.plot(d_adaplist, mean_pot_list_E, "b-")
    plt.plot(d_adaplist, mean_pot_list_I, "r-")
    
    plt.plot(d_adaplist, MF_mean_pot_list_E, "b--")
    plt.plot(d_adaplist, MF_mean_pot_list_I, "r--")
    
    plt.plot(d_adaplist, mean_pot_list_E, "b+")
    plt.plot(d_adaplist, mean_pot_list_I, "r+")
    
    plt.plot(d_adaplist, MF_mean_pot_list_E, "bo")
    plt.plot(d_adaplist, MF_mean_pot_list_I, "ro")
    
    plt.xlabel("d value")
    plt.ylabel(r"$\mu_V$ (mV)")
    plt.title("Adaptation parameter effect on the mean membrane potential of the populations")
    plt.legend(["Brian - Excitatory", "Brian - Inhibitory", "MF - Excitatory", "MF - Inhibitory"])
    plt.savefig("./mu_vs_d.png")
    plt.show()
    
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(d_adaplist, mean_nue_list, "b-")
    plt.plot(d_adaplist, mean_nui_list, "r-")
    
    plt.plot(d_adaplist, MF_mean_nue_list, "b--")
    plt.plot(d_adaplist, MF_mean_nui_list, "r--")
    
    plt.plot(d_adaplist, mean_nue_list, "b+")
    plt.plot(d_adaplist, mean_nui_list, "r+")
    
    plt.plot(d_adaplist, MF_mean_nue_list, "bo")
    plt.plot(d_adaplist, MF_mean_nui_list, "ro")
    
    plt.xlabel("d value")
    plt.ylabel(r"$\nu$ (Hz)")
    plt.title("Adaptation parameter effect on the mean firing rates of the populations")
    plt.legend(["Brian - Excitatory", "Brian - Inhibitory", "MF - Excitatory", "MF - Inhibitory"])
    plt.savefig("./nu_vs_d.png")
    plt.show()
    
    
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))
    
if __name__ == '__main__':
    main()