# MOGABA_pipeline
The Python scripts calibrate polarization data observed by the Korean VLBI Network (KVN) in single-dish mode.

  - This script is built based on python3-pyclass of GILDAS/CLASS.
    Any problems potentially arose in python2-pyclass are not tested.

  - You need to know the gain curves of the KVN at each frequency to run this code properly.

  - All the results containing the calibrated data are saved in '.xlsx' format.
    If you want to use another format, you may want to change it manually by yourself.

  - Depending on your preference, you may need to handle several toggle options in the 'mogaba_pipe_run.py' file.
    (Run_CSFit, Auto_Flag, SaveCSLog, SaveCSFit, SavePSLog, SaveACPlot, LR_Swap, UnwrapPhs, FlagChannel)

    : If you set "True" on "Run_CSFit", then the cross-scan (CS) fitting 
      with a 1-D Gaussian model will be performed.
    
    : CS fitting is useful to obtain the flux density of a source.
    
    : The positional offset is considered and used to correct in computing the total flux density.

    : If you set "True" on "Auto_Flag", then bad scans are flagged out 
      based on the observed auto-/cross-power spectrum.
      * However, it is recommended to set it "False" when the weather condition was not so good during the observation,
        and to flag bas scans manually by yourself until you are happy.

    : "SaveCSLog" and "SavePSLog" are the toggling options whether you want to save plots
      showing system temperature (Tsys), optical depth (tau), and elevation during the observation.
      If you do not want to save log plots, set it to "False".

    : If you set "True" on "SaveCSFit", then fitting results on every cross-scan 
      in every azimuth and elevation scan and corresponding corner plots generated from the emcee package.

    : If you want to check the auto-correlated (LL, RR) power spectrum 
      obtained from position-switching (POS) observation, then set "True" the "SaveACPlot".

    : If the signal streams into LCP and RCP signal is interchanged, set "LR_Swap" as "True". Please note that LCP-RCP swapping is forced at 129 GHz (@ L:229).
      * You can remove the line if you do not want to force the swapping at 129 GHz.

    : "UnwrapPhs" is used to unwrap the phase information on cross-power spectrum (i.e., to resolve n-pi ambiguity).
      Note that the unwrapped phase is just for visually checking the consistency of the phase among the scans.
      Therefore, this option does not affect the calibrated polarization angle.

    : "FlagChannel" is used to flag out a specific range of channels.
      For example, if you set [1900,2100], the channels between 1900 and 2100 will not be used
      in further calibration process.
      * This option might be useful if there is an unwanted artificial (or RFI) signal in a specific channel.
        * RFI : Radio Frequency Interference // potentially from a satellite

  - Basically, only the three planets (Venus, Mars, Jupiter) are used as d-term calibrators (i.e., un-polarized source),
    and selected into the "Unpols" python list though you can add "3C84" if you want to use it as a d-term calibrator
    at low (22 and 43 GHz) frequencies.
      * It is not recommended to use "3C84" at high frequencies (e.g., 86 and 129 GHz).

  - We use "CRAB" as an angle reference target source though you can also use "CRAB2" or "3C286" as the reference.

  - There are two lists, "flag_scan1" and "flag_scan2", which are used to flag out bad scans.
    : The two lists are applied differently by rx_pol.
      * (for example, scan1 -> flagging scan(s) at K-band for KQ data)

  - The paths "path_p" and "path_c" should be the same but used differently for the Python and pyclass, respectively.
    The paths should be assigned to the directory where the ".sdd" file is located.

  - All the calibration procedures are recorded into the "pipelog".

  - The antenna aperture efficiency is estimated based on the estimated antenna temperature of the planets
    and their known brightness temperature.
    If the aperture efficiency is obtained properly, it is used for calculating flux density.
    This scheme uses the antenna temperature of the planets obtained from the CS fitting.

  - The "emcee" package is used in running CS fitting.
    * emcee : https://emcee.readthedocs.io/en/stable/

    Basically, the number of walkers and steps is set as 10 and 2000, respectively.
    If you want to use another setting, you can change them to your preference.

  - Finally, you can easily run the pipeline by running the "mogaba_pipe_run.py".
    * Recommendation : choose which scans you will flag, then turn on "Run_CSFit" and run the pipeline

