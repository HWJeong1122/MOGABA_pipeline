
from __main__ import *
import pyclass as p

from mogaba_pipe_imports import cal_gain_curve

c_ps = p.comm
g_ps = p.gdict

c_ps("set variable general")
c_ps("set variable calibration")
c_ps("set variable position")

# pps = PosPolScan()    : temporary class to read scan infomation
# ppd = PosPolData()    : fianl class to assign polarization data
# Main stream           : scan-by-scan data                       (dsmCal & read_wps)
#                           -> subset of scans                    (pps)
#                           -> overall data for individual source (ppd)
#                           -> load to PoSwitch

class PoSwitch:
    def __init__(self,
                 path_p   =None, path_c   =None, path_dir=None , saveplot=False,
                 file     =None, station  =None, telname =None , unpol   =None ,
                 unpol_lst=None, unpol_all=None, polnum  =None , aref_n  =None ,
                 mode     =None, lr_swap  =None, autoflag=False, pipe_log=None
                 ):
        self.path_p   = path_p          # path used in python
        self.path_c   = path_c          # path used in class
        self.path_dir = path_dir        # path to load Tb excel file
        self.file     = file            # sdd file name
        self.station  = station         # station name (e.g., KYS)
        self.saveplot = saveplot        # path to save plots (Tsys, Tau, AzEl, ..)
        self.unpol    = unpol           # unpol source name
        self.unpol_lst= unpol_lst       # list of unpol sources in given data
        self.unpol_all= unpol_all       # list of available sources as unpol
        self.aref_n   = aref_n          # name of angle reference source
        self.polnum   = polnum          # polnum (e.g., 1:22 GHz , 2:43 GHz for KQ data)
        self.mode     = mode            # data reduction mode ('unpol', 'aref', 'target')
        self.binnum   = 128
        self.nsetup   = 4 * 2           # N(Stokes | RR, LL, Q, U) * Nfreq (e.g., 22, 43)
        self.nswitch  = 16              # number of switching (off-on-on- ... | refer to Kang et al., 2015, JKAS, 48, 257)
        self.nonoff   = 2               # on- or off-source position
        self.pos_sour = None            # source name for data processing
        self.lr_swap  = lr_swap         # toggle whether to swap L-R pol into R-L order
        self.autoflag = autoflag
        self.pipe_log = pipe_log
        self.errmsg   = None
        self.out_pang = pd.DataFrame([])

    def set_init(self):
        if self.station is None : self.station=self.file.split('.sdd')[0].split('_')[-1]
        self.proj = self.file.split('_%s'%(self.station))[0]

        path_fig = self.path_dir + 'Figures/'
        path_dat = self.path_dir + 'data_pos/'
        mkdir('%s'         %(path_fig))
        mkdir('%s/pos/'    %(path_fig))
        mkdir('%s/pos/22/' %(path_fig))
        mkdir('%s/pos/43/' %(path_fig))
        mkdir('%s/pos/86/' %(path_fig))
        mkdir('%s/pos/94/' %(path_fig))
        mkdir('%s/pos/129/'%(path_fig))
        mkdir('%s/pos/141/'%(path_fig))

        mkdir('%s'     %(path_dat))
        mkdir('%s/22/' %(path_dat))
        mkdir('%s/43/' %(path_dat))
        mkdir('%s/86/' %(path_dat))
        mkdir('%s/94/'%(path_dat))
        mkdir('%s/129/'%(path_dat))
        mkdir('%s/141/'%(path_dat))

        try:
            c_ps("sic directory %s"%(self.path_c))
        except:
            self.errmsg = '    !!! Given path "%s" does not exist.'%(self.path_c)
            print(self.errmsg)
            abort()
        try:
            c_ps("file in %s"%(self.file))
        except:
            self.errmsg = '    !!! Given sdd file "%s" does not exist.'%(self.file)
            print(self.errmsg)
            abort()
        c_ps("set default")
        c_ps("set variable general")
        c_ps("set ty l")
        c_ps("find")

    def load_source_info(self):
        Nscans  = int(g_ps.found)
        sources = np.array([]).astype(str)
        scans   = np.array([]).astype(int)
        dtime   = np.array([]).astype(str)
        tsys    = np.array([]).astype(float)
        tau     = np.array([]).astype(float)
        az      = np.array([]).astype(float)
        el      = np.array([]).astype(float)
        freq    = np.array([]).astype(int)
        stokes  = np.array([]).astype(str)
        for n in range(Nscans):
            if n==0: c_ps("get first")
            else   : c_ps("get next")
            c_ps("define character*12 sour")
            c_ps("define character*12 sourdate")
            c_ps("let sour     'r%head%pos%sourc'")
            c_ps("let sourdate 'r%head%gen%cdobs'")
            date     = format_date(str(g_ps.sourdate.__sicdata__))
            datetime = format_time(date, g_ps.ut.astype(float))
            source   = str(g_ps.sour.__sicdata__ ).replace('b','').replace("'",'').replace(' ','')
            teles    = str(g_ps.teles.astype(str)).replace('b','').replace("'",'').replace(' ','')
            S = teles[-1]   # Stokes parameter(LL, RR, Q, U)
            f = int(g_ps.line.astype(float)/1000)
            if source in ["IK_TAU", "V1111_OPH", "R_LEO", "U_ORI", "ORION_KL", "TX_CAM", "OMI_CET", "VY_CMA", "R_HYA", "W_HYA", "U_HER", "VX_SGR", "R_AQL", "R_AQR", "R_CAS", "T_CEP"]:
                delattr(g_ps, "sour")
                delattr(g_ps, "sourdate")
                continue

            sources = np.append(sources, source)
            scans   = np.append(scans  , g_ps.scan.astype(int))
            dtime   = np.append(dtime  , str(datetime))
            freq    = np.append(freq   , f)
            tsys    = np.append(tsys   , g_ps.tsys.astype(float))
            tau     = np.append(tau    , g_ps.tau .astype(float))
            az      = np.append(az     , g_ps.az  .astype(float)*u.rad.to(u.deg))
            el      = np.append(el     , g_ps.el  .astype(float)*u.rad.to(u.deg))
            stokes  = np.append(stokes , S)
            delattr(g_ps, "sour")
            delattr(g_ps, "sourdate")
        npol = len(np.unique(freq))
        pos_log_all = pd.DataFrame([sources, scans, dtime, tsys, tau, az, el, freq, stokes],
                                  index=['Source', 'ScanNum', 'Date', 'Tsys', 'Tau', 'Az', 'El', 'Freq', 'Stokes']).transpose()
        if not pos_log_all.shape[0]<2112:
            pos_log_all['Year'] = Ati(np.array(pos_log_all['Date']).astype(str), format='iso').byear
            pos_log_all['MJD']  = Ati(np.array(pos_log_all['Date']).astype(str), format='iso').mjd
            pos_log_all = pos_log_all[['Date', 'Year', 'MJD', 'Source', 'ScanNum', 'Tsys', 'Tau', 'Az', 'El', 'Freq', 'Stokes']]

            self.pos_log_all  = pos_log_all
            mjd0  = int(np.min(self.pos_log_all['MJD']))
            nchan = int(npol*4)
            self.pos_log_all['Time']         =np.round((self.pos_log_all['MJD']-mjd0) * u.day.to(u.s),0).astype(int)
            self.pos_log_all['Time'][nchan:] =np.array(self.pos_log_all['Time'][nchan:]) -np.array(self.pos_log_all['Time'][0:-nchan])
            self.pos_log_all['Time'][0:nchan]=np.array(self.pos_log_all['Time'][0:nchan])-self.pos_log_all['Time'][0]
            self.pos_log_all = correct_time(self.pos_log_all, nchan)

            time_error = self.pos_log_all[np.abs(self.pos_log_all.Time)<120]['Time']
            thresh_avg, thresh_err = np.mean(time_error), np.std(time_error)

            gap_ = np.where(self.pos_log_all['Time']>thresh_avg+10*thresh_err)[0][0::nchan]
            gap_ = np.append(np.array([0]), gap_)
            pos_log_sour = self.pos_log_all.iloc[gap_].reset_index()

            check_sour = lambda x, series : x in series
            list_avg   = np.array(pos_log_sour['Source']).astype(str)
            list_all   = np.array(self.pos_log_all['Source']).astype(str)
            CRABs    = ['CRAB', 'CRAB1', 'CRAB2']
            row_crab = []
            for crab in CRABs:
                if not np.logical_and(check_sour(crab, list_avg), check_sour(crab, list_all)):
                    row_ = np.where(self.pos_log_all.Source==crab)[0]
                    if row_.shape[0]!=0:
                        row_crab.append(row_[0])
            if len(row_crab)!=0:
                crab_dat = pd.DataFrame(self.pos_log_all.iloc[row_crab]).reset_index()
                pos_log_sour = pd.concat([pos_log_sour, crab_dat], axis=0).sort_values(by='index').reset_index(drop=True)
            index1 = np.append(np.array(pos_log_sour['index'][1:]), np.array([self.pos_log_all.shape[0]]))
            index2 = np.array(pos_log_sour['index'])
            pos_log_sour['Nscan'] = np.array(index1-index2).astype(int)
            pos_log_sour = pos_log_sour[['Source', 'Date', 'Year', 'MJD', 'ScanNum', 'Nscan', 'Az', 'El']]
            pos_log_sour["Tsys_1"] = np.full(pos_log_sour.shape[0], np.nan)
            pos_log_sour["Tsys_2"] = np.full(pos_log_sour.shape[0], np.nan)
            pos_log_sour["Tau_1" ] = np.full(pos_log_sour.shape[0], np.nan)
            pos_log_sour["Tau_2" ] = np.full(pos_log_sour.shape[0], np.nan)
            ufreq = np.unique(freq)
            for nsour, source in enumerate(pos_log_sour["Source"]):
                scannum1, scannum2 = pos_log_sour["ScanNum"][nsour], pos_log_sour["ScanNum"][nsour]+pos_log_sour["Nscan"][nsour]
                pla  = pos_log_all[pos_log_all.Source==source].reset_index(drop=True)
                pla  = pla[np.logical_and(scannum1<=pla.ScanNum, pla.ScanNum<scannum2)].reset_index(drop=True)
                pla1 = pla[pla.Freq==ufreq[0]].reset_index(drop=True)
                pos_log_sour["Tsys_1"][nsour] = np.median(pla1["Tsys"])
                pos_log_sour["Tau_1" ][nsour] = np.median(pla1["Tau" ])
                if len(ufreq)==2:
                    pla2 = pla[pla.Freq==ufreq[1]].reset_index(drop=True)
                    pos_log_sour["Tsys_2"][nsour] = np.median(pla2["Tsys"])
                    pos_log_sour["Tau_2" ][nsour] = np.median(pla2["Tau" ])

            pos_log_sour['Nrep'] = np.array(pos_log_sour['Nscan']//(self.nsetup * (self.nswitch*self.nonoff+1))).astype(int)
            lowrep_      = np.where(pos_log_sour['Nrep'] < 4)[0]
            pos_log_sour = pos_log_sour.drop(lowrep_, axis=0).reset_index(drop=True)
            pos_log_sour['Nswitch'] = [self.nswitch for i in range(pos_log_sour.shape[0])]

            self.pos_log_sour = pos_log_sour

            log_all = self.file.split('.sdd')[0] + '_All.xlsx'
            self.pos_log_all .to_excel(self.path_p + log_all )
            self.pos_log_sour.to_excel(self.path_p + self.log)
        else:
            self.pos_log_all = pos_log_all

    def remake_log_target(self):
        log_target = self.pos_log_sour.copy()
        log_target = log_target[np.logical_and(log_target.ScanNum!=self.scan_aref,
                                               log_target.ScanNum!=self.scan_unpol)].reset_index(drop=True)
        self.log_target = log_target

    def get_info_calib(self):
        self.date = self.pos_log_all['Date'][0].split(' ')[0]
        Nsour     = self.pos_log_sour.shape[0]
        Sources   = np.array(self.pos_log_sour['Source']).astype(str)
        Sources   = np.char.upper(Sources)
        Unpol_all = np.char.upper(np.array(self.unpol_all))
        unpols_n  = []  # unpol sources name
        for N in range(Nsour):
            if Sources[N] in Unpol_all:
                unpols_n.append(self.pos_log_sour['Source' ][N])
        self.unpols_n = list(np.unique(unpols_n))

        if self.aref_n in Sources:
            self.aref_s = self.pos_log_sour[self.pos_log_sour==self.aref_n]['ScanNum']
            self.aref_r = self.pos_log_sour[self.pos_log_sour==self.aref_n]['Nrep'   ]
        else:
            self.aref_n = 'CRAB2'
            if self.aref_n in Sources:
                self.aref_s = self.pos_log_sour[self.pos_log_sour==self.aref_n]['ScanNum']
                self.aref_r = self.pos_log_sour[self.pos_log_sour==self.aref_n]['Nrep'   ]
            else:
                self.aref_n = 'CRAB1'
                if self.aref_n in Sources:
                    self.aref_s = self.pos_log_sour[self.pos_log_sour==self.aref_n]['ScanNum']
                    self.aref_r = self.pos_log_sour[self.pos_log_sour==self.aref_n]['Nrep'   ]
                else:
                    self.errmsg = 'There is no Pol. Angle reference source (CRAB, CRAB1, CRAB2).'
                    print(self.errmsg)
                    print('File name : %s'%(self.file))
                    print('End Process')
                    writelog(self.path_dir, self.pipe_log, "No Pol. Angle reference source in %s"%(self.file), 'a')

    def get_info_scan(self):
        source  = self.pos_sour
        nswitch = self.nswitch
        nsetup  = self.npol*4 ; self.nsetup = nsetup
        nonoff  = self.nonoff
        scan1r  = nsetup*(nswitch*nonoff+1)
        polnum  = self.polnum
        if self.npol>2:
            print("File Name: %s"%(self.file))
            raise ValueError("Number of frequency exceeds two.")
        if self.mode !='target':
            if self.mode=='unpol':idx_=self.idx_unpol
            if self.mode=='aref' :idx_=self.idx_aref
            scan_ = self.pos_log_sour.copy()
            drop_ = np.where(scan_['Nrep']<4)[0]
            scan_ = scan_.drop(drop_, axis=0).reset_index(drop=True)
            scani = scan_[scan_.Source==source].reset_index(drop=True)
            nrep = scani['Nrep'][idx_]
            self.scannums  = [scani['ScanNum'][idx_]+scan1r*n+4*(polnum-1) for n in range(nrep)]
        elif self.mode =='target':
            scan_ = self.log_target.copy()
            scani = scan_[scan_.Source==source].reset_index(drop=True)
            scani = scani.iloc[self.pos_nseq]
            nrep  = scani['Nrep']
            self.scannums  = [scani['ScanNum']+scan1r*n+4*(polnum-1) for n in range(nrep)]
        self.scan_info = scani
        self.nrepeat   = nrep


    def get_bad_scans(self):
        bad_scans = self.bad_chans
        nflag = len(bad_scans)

        scannums = []
        chans = []
        subchans = []
        for Nscan, flag_info in enumerate(bad_scans):
            scannum = flag_info[0]
            chan = list(flag_info[1].keys())
            subchan = list(flag_info[1].values())
            scannums.append(scannum)
            chans.append(chan)
            subchans.append(subchan)
        self.flag_info = pd.DataFrame(
            [scannums, chans, subchans],
            index=["scan", "chan", "subchan"]
        ).transpose()

    def get_poldata(self):
        ppd = self.ppd
        mode        = self.mode
        scannums    = self.scannums
        nscans      = len(scannums)
        sqrtn       = sqrt(nscans)
        ch1, ch2    = 500, 3500
        stat_method = np.nanmean
        elevation   = self.elevation
        year        = int(self.date[:4])
        self.gain   = cal_gain_curve(self.station, year, self.freq, elevation)

        if mode != 'unpol':
            cc = ppd.cc
            si = ppd.si
            sv = ppd.sv

            Tvfc1, Tvfc2 = self.data_unpol.Tvfc1, self.data_unpol.Tvfc2
            v1, v2 = stat_method(Tvfc1), stat_method(Tvfc2)
            v_mean = (v1*v2)**0.5
            sa     = stat_method(si[:, ch1:ch2], axis=1)*v_mean # si  : I/I_0
            va     = stat_method(sv[:, ch1:ch2], axis=1)        # sv  : V/I

            sa_mean = stat_method(sa)
            si_mean = sa_mean/v_mean

            cc_angle   = angle(cc)
            cc_phs     = stat_method(cc[:, ch1:ch2], axis=1)

            angle_mean = angle(stat_method(cc_phs))
            angle_std  = np.nanstd(angle(cc_phs*exp(-1j*angle_mean)))
            if ppd.data_aref:
                data_aref    = self.data_aref
                data_aref_cc = data_aref.cc
                aref_phs     = exp(1.j*angle(stat_method(data_aref_cc, axis=0)))*exp(-1.j*angle(stat_method(data_aref_cc[:,ch1:ch2])))

                cc_phs_mean = exp(1.j*angle_mean)

                cca   = cc/cc_phs_mean/aref_phs
                cca_r = real(cca)[:, ch1:ch2]
                cca_i = imag(cca)[:, ch1:ch2]
                pm    = np.abs(stat_method(cca[:, ch1:ch2]))/si_mean
                dpm   = sqrt(np.nanstd(stat_method(cca_r, axis=1))**2+np.nanstd(stat_method(cca_i, axis=1))**2)/sqrtn/si_mean/sqrt(2)
                tp    = np.abs(stat_method(stat_method(cca[:, ch1:ch2]*v_mean, axis=1)))
                dtp   = np.nanstd(stat_method(cca[:, ch1:ch2]*v_mean, axis=1))/sqrtn
            else:
                ca  = stat_method(np.abs(cc)[:,ch1:ch2], axis=1)
                pm  = stat_method(ca/si_mean)
                dpm = np.nanstd(ca/si_mean)/sqrtn
                tp  = np.abs(stat_method(ca*v_mean))
                dtp = np.nanstd(ca*v_mean)/sqrtn
            vm = va/sa_mean*100
            angle_mean *= 180/pi/2
            angle_std  *= 180/pi/2

            self.getp_ti, self.getp_dti = sa_mean/self.gain   , np.nanstd(sa)/sqrtn/self.gain
            self.getp_tp, self.getp_dtp = tp/self.gain        , dtp/self.gain
            self.getp_tv, self.getp_dtv = stat_method(vm), np.nanstd(vm)/sqrtn
            self.getp_pm, self.getp_dpm = pm             , dpm
            self.getp_pa, self.getp_dpa = angle_mean     , angle_std/sqrtn
        self.getp_t1, self.getp_dt1 = stat_method(ppd.Tvfc1)/self.gain, np.nanstd(ppd.Tvfc1)/self.gain
        self.getp_t2, self.getp_dt2 = stat_method(ppd.Tvfc2)/self.gain, np.nanstd(ppd.Tvfc2)/self.gain

    def rd_polscans(self):
        self.get_info_scan()
        polnum       = self.polnum
        binnum       = self.binnum
        mode         = self.mode
        nswitch      = self.nswitch
        npol         = self.npol
        scani        = self.scan_info
        scannums     = self.scannums
        sour_scannum = scani['ScanNum']

        ppd = PosPolData(station=self.station)
        ppd.scannums = scannums
        ppd.c_p      = c_ps
        ppd.g_p      = g_ps
        ppd.station  = self.station
        ppd.binnum   = binnum
        ppd.polnum   = polnum
        ppd.mode     = mode
        ppd.nswitch  = nswitch
        ppd.npol     = npol

        tsys1, tsys2 = [],[]

        self.get_bad_scans()
        flag_info = self.flag_info.copy()
        flag_info = flag_info[flag_info.scan == scannums[0]].reset_index(drop=True)

        for iscan, scannum in enumerate(ppd.scannums):
            if self.autoflag:
                flag_subchan = "none"
            else:
                if flag_info.empty:
                    flag_subchan = "none"
                else:
                    flag_chan_ = flag_info["chan"][0]
                    if iscan in flag_chan_:
                        loc = np.where(np.array(flag_chan_)==iscan)[0][0]
                        flag_subchan = flag_info["subchan"][0][loc]
                    else:
                        flag_subchan = "none"
            if np.logical_and(mode=='unpol', iscan==0):
                pps = PosPolScan(mode=mode, scannum=scannum, npol=npol, delay_fit=True, delay=0, station=self.station)
                pps.c_p        = c_ps
                pps.g_p        = g_ps
                pps.read_scan(flag_subchan=flag_subchan)
                ppd.delay      = pps.delay
                ppd.sideband   = pps.sideband
            elif np.logical_and(mode=='unpol', iscan!=0):
                pps.scannum    = scannum
                pps.delay      = ppd.delay
                pps.delay_fit  = False
                pps.read_scan(flag_subchan=flag_subchan)

            if np.logical_and(mode!='unpol', iscan==0):
                pps = PosPolScan(mode=mode, scannum=scannum, npol=npol, delay_fit=False, delay=self.ppd.delay, station=self.station)
                pps.c_p        = c_ps
                pps.g_p        = g_ps
                pps.read_scan(flag_subchan=flag_subchan)
                ppd.sideband   = pps.sideband
            elif np.logical_and(mode!='unpol', iscan!=0):
                pps.scannum    = scannum
                pps.read_scan(flag_subchan=flag_subchan)

            ppd.az   .append(pps.az)
            ppd.el   .append(pps.el)
            ppd.d    .append(pps.d)
            ppd.c    .append(pps.c)
            ppd.v    .append(pps.v)
            ppd.vc   .append(pps.vc)    # convolved vane information
            ppd.d_vfc.append(pps.d_vfc)
            ppd.tsys1.append(pps.tsys1)
            ppd.tsys2.append(pps.tsys2)

        ppd.ra, ppd.dec = pps.ra, pps.dec
        ppd.az    = np.array(ppd.az)
        ppd.el    = np.array(ppd.el)
        ppd.d     = np.array(ppd.d)
        ppd.c     = np.array(ppd.c)
        ppd.v     = np.array(ppd.v)
        ppd.vc    = np.array(ppd.vc)
        ppd.d_vfc = np.array(ppd.d_vfc)
        ppd.Tvfc1 = ppd.d_vfc[:,:,0,0]
        ppd.Tvfc2 = ppd.d_vfc[:,:,1,0]
        ppd.tsys1 = np.array(ppd.tsys1)
        ppd.tsys2 = np.array(ppd.tsys2)

        if self.autoflag:
            Tvfc1_90l = np.percentile(ppd.Tvfc1,  5)
            Tvfc1_90h = np.percentile(ppd.Tvfc1, 95)
            Tvfc2_90l = np.percentile(ppd.Tvfc2,  5)
            Tvfc2_90h = np.percentile(ppd.Tvfc2, 95)
            mask1_l = ppd.Tvfc1 < Tvfc1_90l
            mask1_h = ppd.Tvfc1 > Tvfc1_90h
            mask2_l = ppd.Tvfc2 < Tvfc2_90l
            mask2_h = ppd.Tvfc2 > Tvfc2_90h
            mask1 = np.logical_or(mask1_l, mask1_h)
            mask2 = np.logical_or(mask2_l, mask2_h)

            ppd.az    = []
            ppd.el    = []
            ppd.d     = []
            ppd.c     = []
            ppd.v     = []
            ppd.vc    = []
            ppd.d_vfc = []
            ppd.Tvfc1 = None
            ppd.Tvfc2 = None
            ppd.tsys1 = []
            ppd.tsys2 = []
            chans     = []
            subchans  = []

            for iscan, scannum in enumerate(ppd.scannums):
                mask = np.logical_or(mask1[iscan], mask2[iscan])
                flag_subchan = np.where(mask)[0].tolist()

                if not flag_subchan:
                    flag_subchan = "none"

                if np.logical_and(mode=='unpol', iscan==0):
                    pps = PosPolScan(mode=mode, scannum=scannum, npol=npol, delay_fit=True, delay=0, station=self.station)
                    pps.c_p        = c_ps
                    pps.g_p        = g_ps
                    pps.read_scan(flag_subchan=flag_subchan)
                    ppd.delay      = pps.delay
                    ppd.sideband   = pps.sideband
                elif np.logical_and(mode=='unpol', iscan!=0):
                    pps.scannum    = scannum
                    pps.delay      = ppd.delay
                    pps.delay_fit  = False
                    pps.read_scan(flag_subchan=flag_subchan)

                if np.logical_and(mode!='unpol', iscan==0):
                    pps = PosPolScan(mode=mode, scannum=scannum, npol=npol, delay_fit=False, delay=self.ppd.delay, station=self.station)
                    pps.c_p        = c_ps
                    pps.g_p        = g_ps
                    pps.read_scan(flag_subchan=flag_subchan)
                    ppd.sideband   = pps.sideband
                elif np.logical_and(mode!='unpol', iscan!=0):
                    pps.scannum    = scannum
                    pps.read_scan(flag_subchan=flag_subchan)

                ppd.az   .append(pps.az)
                ppd.el   .append(pps.el)
                ppd.d    .append(pps.d)
                ppd.c    .append(pps.c)
                ppd.v    .append(pps.v)
                ppd.vc   .append(pps.vc)    # convolved vane information
                ppd.d_vfc.append(pps.d_vfc)
                ppd.tsys1.append(pps.tsys1)
                ppd.tsys2.append(pps.tsys2)

            ppd.ra, ppd.dec = pps.ra, pps.dec
            ppd.az    = np.array(ppd.az)
            ppd.el    = np.array(ppd.el)
            ppd.d     = np.array(ppd.d)
            ppd.c     = np.array(ppd.c)
            ppd.v     = np.array(ppd.v)
            ppd.vc    = np.array(ppd.vc)
            ppd.d_vfc = np.array(ppd.d_vfc)
            ppd.Tvfc1 = ppd.d_vfc[:,:,0,0]
            ppd.Tvfc2 = ppd.d_vfc[:,:,1,0]
            ppd.tsys1 = np.array(ppd.tsys1)
            ppd.tsys2 = np.array(ppd.tsys2)

        if mode == 'unpol':
            self.data_unpol = ppd
        else:
            ppd.unpol_data = self.data_unpol
            self.pol_data = ppd
            if mode == 'aref': self.data_aref=ppd
            else             : ppd.data_aref =self.data_aref

        if pps.delay != ppd.delay:
            ppd.delay = pps.delay

        self.pps = pps
        self.ppd = ppd

    def run_pos(self):
        self.mode       = 'unpol'
        self.pos_sour   = self.unpol
        log_unpol       = self.pos_log_sour[self.pos_log_sour.Source==self.pos_sour].reset_index(drop=True)
        if self.polnum==1:
            idx_unpol = np.where(log_unpol['Tsys_1']==np.min(log_unpol['Tsys_1']))[0][0] ; self.idx_unpol=idx_unpol
            idx_unpol = np.where(log_unpol['Tau_1' ]==np.min(log_unpol['Tau_1' ]))[0][0] ; self.idx_unpol=idx_unpol
        if self.polnum==2:
            idx_unpol = np.where(log_unpol['Tsys_2']==np.min(log_unpol['Tsys_2']))[0][0] ; self.idx_unpol=idx_unpol
            idx_unpol = np.where(log_unpol['Tau_2' ]==np.min(log_unpol['Tau_2' ]))[0][0] ; self.idx_unpol=idx_unpol

        self.mjd        = log_unpol['MJD'    ][idx_unpol]
        self.nswitch    = log_unpol['Nswitch'][idx_unpol]
        self.scannum    = log_unpol['ScanNum'][idx_unpol]
        self.elevation  = log_unpol['El'     ][idx_unpol]
        self.scan_unpol = self.scannum
        self.rd_polscans()
        self.ppd.mode     = self.mode
        self.ppd.pos_sour = self.pos_sour
        self.ppd.cal_pangle()
        self.out_pang = pd.concat([self.out_pang, self.ppd.out_pang], axis=0).reset_index(drop=True)
        self.ppd.cal_leak()
        self.get_poldata()
        self.PlotPolDat()
        self.SavePolDat()

        self.mode      = 'aref'
        self.pos_sour  = self.aref_n
        log_aref       = self.pos_log_sour[self.pos_log_sour.Source==self.pos_sour].reset_index(drop=True)
        if self.polnum==1:
            idx_aref = np.where(log_aref['Tsys_1']==np.min(log_aref['Tsys_1']))[0][0] ; self.idx_aref=idx_aref
            idx_aref = np.where(log_aref['Tau_1' ]==np.min(log_aref['Tau_1' ]))[0][0] ; self.idx_aref=idx_aref
        if self.polnum==2:
            idx_aref = np.where(log_aref['Tsys_2']==np.min(log_aref['Tsys_2']))[0][0] ; self.idx_aref=idx_aref
            idx_aref = np.where(log_aref['Tau_2' ]==np.min(log_aref['Tau_2' ]))[0][0] ; self.idx_aref=idx_aref

        self.mjd       = log_aref['MJD'    ][idx_aref]
        self.nswitch   = log_aref['Nswitch'][idx_aref]
        self.scannum   = log_aref['ScanNum'][idx_aref]
        self.elevation = log_aref['El'     ][idx_aref]
        self.scan_aref = self.scannum
        self.rd_polscans()
        self.ppd.data_unpol = self.data_unpol
        self.ppd.mode       = self.mode
        self.ppd.pos_sour   = self.pos_sour
        self.ppd.cal_pangle()
        self.out_pang = pd.concat([self.out_pang, self.ppd.out_pang], axis=0).reset_index(drop=True)
        self.ppd.cal_stokes()
        sign = 1
        if self.lr_swap : sign = -1
        self.ppd.sv *= sign
        sideband = self.ppd.sideband
        if sign*sideband==-1 : self.ppd.cc = np.conj(self.ppd.cc)
        self.ppd.correct_rot()
        self.get_poldata()
        self.PlotPolDat()
        self.SavePolDat()

        self.remake_log_target()
        count = np.array([])
        for ntarget in range(self.log_target.shape[0]):
            self.mode      = 'target'
            self.pos_sour  = self.log_target['Source' ][ntarget]
            self.mjd       = self.log_target['MJD'    ][ntarget]
            self.nswitch   = self.log_target['Nswitch'][ntarget]
            self.scannum   = self.log_target['ScanNum'][ntarget]
            self.elevation = self.log_target['El'     ][ntarget]
            self.pos_nseq  = count[count==self.pos_sour].shape[0]
            count = np.append(count, self.pos_sour)
            self.rd_polscans()
            self.ppd.data_unpol = self.data_unpol
            self.ppd.mode       = self.mode
            self.ppd.pos_sour   = self.pos_sour
            self.ppd.cal_pangle()
            self.out_pang = pd.concat([self.out_pang, self.ppd.out_pang], axis=0).reset_index(drop=True)
            self.ppd.cal_stokes()
            sign = 1
            if self.lr_swap : sign = -1
            self.ppd.sv *= sign
            sideband = self.ppd.sideband
            if sign*sideband==-1 : self.ppd.cc = np.conj(self.ppd.cc)
            self.ppd.correct_rot()
            self.get_poldata()
            self.PlotPolDat()
            self.SavePolDat()

        freq = self.freq
        path_dat = self.path_dir + 'data_pos/%s/'%(self.freq)
        mkdir(path_dat)
        self.poldat.to_excel(path_dat + '%s_%s_%s_%s_%s.xlsx'%(self.station, self.date, self.freq, self.unpol, self.proj))


    def PlotPolDat(self):
        if   self.mode !='target': log_sour = self.pos_log_sour
        elif self.mode =='target': log_sour = self.log_target
        mode     = self.mode
        ppd      = self.ppd
        scannums = ppd.scannums
        log_scan = log_sour[log_sour.ScanNum==self.scannum].reset_index(drop=True)
        date_time= log_scan['Date'][0]
        nrep     = log_scan['Nrep'][0]
        az       = log_scan['Az'][0]
        el       = log_scan['El'][0]
        nswitch  = self.nswitch
        Tintg    = 3.0
        ch1, ch2 = 500, 3500
        if mode!='unpol' : si_mean =np.nanmean(ppd.si[:, ch1:ch2])

        if self.SaveACPlot:
            mkdir(self.path_dir + "Figures/pos_ac/")
            mkdir(self.path_dir + "Figures/pos_ac/%s/"      %(self.freq))
            mkdir(self.path_dir + "Figures/pos_ac/%s/%s/"   %(self.freq, self.date))
            mkdir(self.path_dir + "Figures/pos_ac/%s/%s/%s/"%(self.freq, self.date, self.unpol))
            fig_ph, ax_ph = plt.subplots(1,2, figsize=(14,7))
            for i in range(nrep):
                ax_ph[0].plot(ppd.d[i,0])
                ax_ph[1].plot(ppd.d[i,1])
            fig_ph.suptitle("Auto-correlation Power Spectrum\n%s (%s GHz)"%(self.pos_sour, self.freq), fontsize=20, fontweight="bold")
            ax_ph[0].set_title("%sR"%(self.freq), fontsize=16, fontweight="bold")
            ax_ph[1].set_title("%sL"%(self.freq), fontsize=16, fontweight="bold")
            fig_ph.savefig(self.path_dir + "Figures/pos_ac/%s/%s/%s/%s_%s.png"%(self.freq, self.date, self.unpol, self.freq, self.pos_sour))
            close_figure(fig_ph)

        fsize  = 14
        fs, fw = 10, 'bold'
        lines  = ['-', '--', ':']
        marks  = ['D', 'o' , 's']
        fig_pol, ax_pol = plt.subplots(5, 2, figsize=(fsize, fsize*11/16))
        cpwr_amp, cpwr_phs = ax_pol[0,0], ax_pol[0,1]   # cross power spectrum
        lpol_amp, lpol_phs = ax_pol[1,0], ax_pol[1,1]   # (leakage for 'unpol' | cross-correlation  for 'pol')
        ccpl_1  , ccpl_2   = ax_pol[2,0], ax_pol[2,1]   # (LL, RR  for 'unpol' | Stokes I, Stokes V for 'pol')
        tvfc_1  , tvfc_2   = ax_pol[3,0], ax_pol[3,1]   # VFC temperature      at LL and RR
        tvfc_1_ , tvfc_2_  = ax_pol[4,0], ax_pol[4,1]   # mean VFC temperature at LL and RR

        label_cpwr       = r'$Re({\rm LR}) + Im({\rm LR})$'
        label_lpol_unpol = r'$d_{\rm L} - d_{\rm R}$'
        label_lpol_pol   = r'$(Q+iU)/I$'
        label_ccpl_unpol_1, label_ccpl_unpol_2 = r'$v_{\rm LL}$', r'$v_{\rm RR}$'
        label_ccpl_pol_1  , label_ccpl_pol_2   = r'$I/I_{0}$'   , r'$V/I$'
        label_tvfc_1      , label_tvfc_2       = r'$T_{\rm VFC, LL}$'  , r'$T_{\rm VFC, RR}$'
        label_tvfc_1_     , label_tvfc_2_      = r'<$T_{\rm VFC, LL}>$', r'$<T_{\rm VFC, RR}>$'

        if self.lr_swap : LRpol = ['R', 'L']
        else            : LRpol = ['L', 'R']
        title1 = '%s'%(date_time) + ' '*40 \
                 +'%s %s%s %s%s %s %s %s %s %.2f %.2f %.2f'%(self.pos_sour, self.freq, LRpol[0], self.freq, LRpol[1], int(scannums[0]), nrep, nswitch, Tintg, az, el, self.gain)
        title2 = '%.3f (%.3f) | %.3f (%.3f)'%(self.getp_t1, self.getp_dt1, self.getp_t2, self.getp_dt2)
        if mode=='unpol':
            label_lpol = label_lpol_unpol
            label_ccpl_1, label_ccpl_2 = label_ccpl_unpol_1, label_ccpl_unpol_2
        else:
            s = ' | %.3f (%.3f) | %.3f (%.3f) | %.3f (%.3f) | %.3f (%.3f) | %.3f (%.3f)'%(
                 self.getp_ti, self.getp_dti, self.getp_tp, self.getp_dtp, self.getp_tv, self.getp_dtv, self.getp_pm*100, self.getp_dpm*100, self.getp_pa, self.getp_dpa)
            title2 += s
            label_lpol = label_lpol_pol
            label_ccpl_1, label_ccpl_2 = label_ccpl_pol_1, label_ccpl_pol_2

        cpwr_amp.set_ylabel(label_cpwr   , fontsize=fs, fontweight=fw) ; cpwr_phs.set_ylabel(label_cpwr   , fontsize=fs, fontweight=fw)
        lpol_amp.set_ylabel(label_lpol   , fontsize=fs, fontweight=fw) ; lpol_phs.set_ylabel(label_lpol   , fontsize=fs, fontweight=fw)
        ccpl_1  .set_ylabel(label_ccpl_1 , fontsize=fs, fontweight=fw) ; ccpl_2  .set_ylabel(label_ccpl_2 , fontsize=fs, fontweight=fw)
        tvfc_1  .set_ylabel(label_tvfc_1 , fontsize=fs, fontweight=fw) ; tvfc_2  .set_ylabel(label_tvfc_2 , fontsize=fs, fontweight=fw)
        tvfc_1_ .set_ylabel(label_tvfc_1_, fontsize=fs, fontweight=fw) ; tvfc_2_ .set_ylabel(label_tvfc_2_, fontsize=fs, fontweight=fw)

        index_list = np.arange(ppd.c.shape[0])
        n=0
        for nidx, idx in enumerate(index_list):
            o = n // 10
            ls, mk = lines[o], marks[o]
            n+=1

            cpow_dat_amp = np.abs(ppd.c[idx])
            cpow_dat_phs = angle(ppd.c[idx], deg=True)
            if mode=='unpol':
                lpol_dat_amp, lpol_dat_phs = np.abs(ppd.leak[idx]), angle(ppd.leak[idx], deg=True)
                crspol_1    , crspol_2     = ppd.d[idx,0]         , ppd.d[idx,1]
            else:
                lpol_dat_amp, lpol_dat_phs = np.abs(ppd.cc[idx])/si_mean, angle(ppd.cc[idx], deg=True)
                crspol_1    , crspol_2     = ppd.si[idx]                , ppd.sv[idx]

            Tvfc1 = ppd.Tvfc1[idx]
            Tvfc2 = ppd.Tvfc2[idx]
            Tx1 = np.arange(len(Tvfc1))
            Tx2 = np.arange(len(Tvfc2))
            mask_nan1 = ~np.isnan(Tvfc1)
            mask_nan2 = ~np.isnan(Tvfc2)
            Tvfc1 = Tvfc1[mask_nan1]
            Tvfc2 = Tvfc2[mask_nan2]
            Tx1 = Tx1[mask_nan1]
            Tx2 = Tx2[mask_nan2]
            cpwr_amp.plot(cpow_dat_amp                       , ls=ls)
            cpwr_phs.plot(cpow_dat_phs                       , ls=ls)
            lpol_amp.plot(lpol_dat_amp                       , ls=ls)
            lpol_phs.plot(lpol_dat_phs                       , ls=ls)
            ccpl_1  .plot(crspol_1                           , ls=ls)
            ccpl_2  .plot(crspol_2                           , ls=ls)
            tvfc_1  .plot(Tx1, Tvfc1                         , ls=ls, marker=".")
            tvfc_2  .plot(Tx2, Tvfc2                         , ls=ls, marker=".")
            tvfc_1_ .plot([idx], [np.nanmean(ppd.Tvfc1[idx])], marker=mk)
            tvfc_2_ .plot([idx], [np.nanmean(ppd.Tvfc2[idx])], marker=mk)

        fig_pol.suptitle('%s \n %s'%(title1, title2), fontweight='bold')

        freq = self.freq
        if freq == 21:
            freq = 22

        path_fig = self.path_dir + 'Figures/pos/%s/%s_%s_%s/'%(freq, self.station, self.date, self.proj)
        mkdir(path_fig)
        path_fig += '%s/'%(self.unpol)
        mkdir(path_fig)

        fig_pol.savefig(path_fig + '%s_%s_%s_%s.png'%(self.mode, int(self.scannums[0]), self.pos_sour, freq))
        close_figure(fig_pol)

    def SavePolDat(self):
        t1, dt1 = self.getp_t1, self.getp_dt1
        t2, dt2 = self.getp_t2, self.getp_dt2
        el = self.pos_log_sour[np.logical_and(self.pos_log_sour.Source==self.pos_sour, self.pos_log_sour.ScanNum==self.scannum)].reset_index(drop=True)['El'][0]
        if self.polnum==1: eta = self.eta1
        if self.polnum==2: eta = self.eta2
        if self.mode == 'unpol':
            t1, t2= ufloat(t1, dt1), ufloat(t2, dt2)
            vt1t2 = unp.nominal_values(t1*t2)
            if   np.logical_or(vt1t2<0, np.isnan(vt1t2)) : uti = ufloat(np.nan, np.nan)
            elif vt1t2>=0                             : uti = unp.sqrt(t1*t2)
            usi = 8*C.k_B.value/(pi*21**2*eta)*uti * (u.J/u.m**2).to(u.Jy)
            eta = np.round(unp.nominal_values(eta), 5)
            ti, dti = np.round(unp.nominal_values(uti), 5), np.round(unp.std_devs(uti), 5)
            si, dsi = np.round(unp.nominal_values(usi), 5), np.round(unp.std_devs(usi), 5)
            poldat = pd.DataFrame([self.pos_sour, self.mjd, el, ti ,dti, 0 ,0, 0, 0, 0, 0, 0, 0, si ,dsi, 0 ,0, eta],
                                  index=['Source', 'MJD', 'El', 'Ti', 'dTi', 'Tp', 'dTp', 'PM', 'dPM', 'PA', 'dPA', 'PA_c', 'dPA_c', 'Si', 'dSi', 'Sp', 'dSp', 'eta']).transpose()
            self.poldat = poldat
        else:
            ti, dti = self.getp_ti, self.getp_dti
            tp, dtp = self.getp_tp, self.getp_dtp
            tv, dtv = self.getp_tv, self.getp_dtv
            pm, dpm = self.getp_pm, self.getp_dpm
            pa, dpa = self.getp_pa, self.getp_dpa
            uti, utp = ufloat(ti, dti), ufloat(tp, dtp)
            usi = 8*C.k_B.value/(pi*21**2*eta)*uti * (u.J/u.m**2).to(u.Jy)
            usp = 8*C.k_B.value/(pi*21**2*eta)*utp * (u.J/u.m**2).to(u.Jy)
            eta = np.round(unp.nominal_values(eta), 5)
            ti, dti = np.round(unp.nominal_values(uti), 5), np.round(unp.std_devs(uti), 5)
            tp, dtp = np.round(unp.nominal_values(utp), 5), np.round(unp.std_devs(utp), 5)
            si, dsi = np.round(unp.nominal_values(usi), 5), np.round(unp.std_devs(usi), 5)
            sp, dsp = np.round(unp.nominal_values(usp), 5), np.round(unp.std_devs(usp), 5)
            pm, dpm = np.round(pm, 5), np.round(dpm, 5)
            if self.mode == 'aref':
                self.pa_aref = ufloat(pa, dpa)
                if self.pos_sour in ["CRAB", "CRAB1", "CRABP", "CRAB2"]:
                    pa  , dpa    = np.round(pa, 5), np.round(dpa, 5)
                    pa_c, dpa_c  = 152, 0
                    self.pa_corr = 152
                if self.pos_sour == "3C286":
                    pa  , dpa    = np.round(pa, 5), np.round(dpa, 5)
                    pa_c, dpa_c  = 32, 0
                    self.pa_corr = 32
            else:
                pa_source = ufloat(pa, dpa)
                pa_source = pa_source-self.pa_aref+self.pa_corr
                pa  , dpa   = np.round(pa, 5), np.round(dpa, 5)
                pa_c, dpa_c = np.round(unp.nominal_values(pa_source), 5), np.round(unp.std_devs(pa_source), 5)
                if pa_c>180 : pa_c-=180
                if pa_c<0   : pa_c+=180

            sourdat = pd.DataFrame([self.pos_sour, self.mjd, el, ti, dti, tp, dtp, pm*100, dpm*100, pa, dpa, pa_c, dpa_c, si, dsi, sp, dsp, eta],
                                   index=['Source', 'MJD', 'El', 'Ti', 'dTi', 'Tp', 'dTp', 'PM', 'dPM', 'PA', 'dPA', 'PA_c', 'dPA_c', 'Si', 'dSi', 'Sp', 'dSp', 'eta']).transpose()
            self.poldat = pd.concat([self.poldat, sourdat], axis=0)

    def SavePSLog(self):
        pos_log_all = self.pos_log_all
        date  = self.date
        freqs = np.sort(np.unique(pos_log_all['Freq']).astype(int))
        LRs   = ['L' , 'R' ]
        self.freqs = freqs

        fsize = 16
        fig_pslog, ax_pslog = plt.subplots(3, 4, figsize=(fsize, fsize*9/16), sharex=True)
        tsys_1_l, tsys_1_r, tsys_2_l, tsys_2_r = ax_pslog[0,0], ax_pslog[0,1], ax_pslog[0,2], ax_pslog[0,3]
        tau_1_l , tau_1_r , tau_2_l , tau_2_r  = ax_pslog[1,0], ax_pslog[1,1], ax_pslog[1,2], ax_pslog[1,3]
        el_1_l  , el_1_r  , el_2_l  , el_2_r   = ax_pslog[2,0], ax_pslog[2,1], ax_pslog[2,2], ax_pslog[2,3]

        tsys_1_l.set_title('%sL'%(freqs[0]), fontweight='bold') ; tsys_1_r.set_title('%sR'%(freqs[0]), fontweight='bold')
        tsys_1_l.set_ylabel(r'$T_{\rm sys}$'+r'$\rm \,(K)$'    , fontsize=15, fontweight='bold')
        tau_1_l .set_ylabel(r'$\tau$'                          , fontsize=15, fontweight='bold')
        el_1_l  .set_ylabel(r'$\rm Elevation$'+r'$\rm \,(deg)$', fontsize=12, fontweight='bold')
        if self.npol==2:
            tsys_2_l.set_title('%sL'%(freqs[1]), fontweight='bold') ; tsys_2_r.set_title('%sR'%(freqs[1]), fontweight='bold')

        axes_xlabel = [tsys_1_l, tsys_1_r, tsys_2_l, tsys_2_r, tau_1_l, tau_1_r, tau_2_l, tau_2_r]
        axes_ylabel = [tsys_1_r, tsys_2_r, tau_1_r , tau_2_l, tau_2_r, el_1_r , el_2_l , el_2_r]
        axes_ylim   = [el_1_l  , el_1_r  , el_2_l  , el_2_r]
        axes_all    = [tsys_1_l, tsys_1_r, tsys_2_l, tsys_2_r,
                       tau_1_l , tau_1_r , tau_2_l , tau_2_r ,
                       el_1_l  , el_1_r  , el_2_l  , el_2_r  ]

        for n in range(len(axes_xlabel)) : axes_xlabel[n].tick_params(labelbottom=False)
        for n in range(len(axes_ylabel)) : axes_ylabel[n].tick_params(labelleft=False)
        for n in range(len(axes_ylim  )) : axes_ylim[n].set_ylim(0,90)
        for n in range(len(axes_all   )) :
            axes_all[n].xaxis.set_major_locator(MultipleLocator(3))
            axes_all[n].xaxis.set_minor_locator(MultipleLocator(1))

        project = self.proj
        fig_pslog.text(0.5, 0.02, '(%s, %s) | UTC (hour)'%(project, date), ha='center', fontsize=15, fontweight='bold')

        colors = ['pink'     , 'coral', 'red' , 'lime'   , 'green' ,
                  'lightblue', 'aqua' , 'blue', 'magenta', 'purple']
        for Nfreq, freq in enumerate(freqs):
            time_thresh=120
            log_all  = pos_log_all.copy()
            log_all  = log_all[log_all.Freq==freq].reset_index(drop=True)
            log_cp   = log_all.copy()
            nrow_cp  = [0]
            nrep_cp  = []
            sour_cp  = log_cp['Source'][0]
            for n in range(log_cp.shape[0]):
                if sour_cp != log_cp['Source'][n]:
                    nrep_cp.append(int((n-nrow_cp[-1])/132))
                    nrow_cp.append(n)
                    sour_cp = log_cp['Source'][n]
                if log_cp['Source'][n] == np.array(log_cp['Source'])[-1]:
                    nrep_cp.append(int((log_cp.shape[0]-n)/132))
                    break

            chop_idx = []
            for i in range(len(nrow_cp)):
                scanN=nrow_cp[i]
                for n in range(nrep_cp[i]):
                    for m in range(4):
                        chop_idx.append(132*n+scanN+m)
            log_cp = log_cp.drop(chop_idx, axis=0).reset_index(drop=True)

            time_error = self.pos_log_all[np.abs(self.pos_log_all.Time)<time_thresh]['Time']
            thresh_avg, thresh_err = np.mean(time_error), np.std(time_error)
            nchan = int(self.npol*4)
            sep = np.append( np.array([0]), np.where(log_all['Time'] > thresh_avg+10*thresh_err)[0][0::nchan] )
            log_sour = log_all.iloc[sep].reset_index(drop=True)
            if Nfreq==0:
                ax_tsys = [tsys_1_l, tsys_1_r]
                ax_tau  = [tau_1_l , tau_1_r ]
                ax_el   = [el_1_l  , el_1_r  ]
            elif Nfreq==1:
                ax_tsys = [tsys_2_l, tsys_2_r]
                ax_tau  = [tau_2_l , tau_2_r ]
                ax_el   = [el_2_l  , el_2_r  ]

            if np.max(log_cp['Tsys'])-np.min(log_cp['Tsys'])<1000:
                tsys_1_l.set_yscale('linear') ; tsys_1_r.set_yscale('linear') ; tsys_2_l.set_yscale('linear') ; tsys_2_r.set_yscale('linear')
                tau_1_l .set_yscale('log')    ; tau_1_r .set_yscale('log')    ; tau_2_l .set_yscale('log')    ; tau_2_r .set_yscale('log')
                tau_1_l .set_ylim(1e-2, 3e0)  ; tau_1_r .set_ylim(1e-2, 3e0)  ; tau_2_l .set_ylim(1e-2, 3e0)  ; tau_2_r .set_ylim(1e-2, 3e0)
            else:
                tsys_1_l.set_yscale('log')   ; tsys_1_r.set_yscale('log')   ; tsys_2_l.set_yscale('log')   ; tsys_2_r.set_yscale('log')
                tau_1_l .set_yscale('log')   ; tau_1_r .set_yscale('log')   ; tau_2_l .set_yscale('log')   ; tau_2_r .set_yscale('log')
                tau_1_l .set_ylim(1e-2, 3e0) ; tau_1_r .set_ylim(1e-2, 3e0) ; tau_2_l .set_ylim(1e-2, 3e0) ; tau_2_r .set_ylim(1e-2, 3e0)
            log_all['Time'] = (log_all['MJD']-Ati(date, format='iso').mjd) * u.day.to(u.hr)
            tmin, tmax = 0.9*np.min(log_all['Time']), 1.1*np.max(log_all['Time'])

            nrow = [0]
            sour = log_all['Source'][0]
            for n in range(log_all.shape[0]):
                if sour != log_all['Source'][n]:
                    nrow.append(n)
                    sour = log_all['Source'][n]

            for Nsour in range(len(nrow)):
                if   Nsour//10 < 1:mtype = 'o'
                elif Nsour//10 >=1:mtype = '^'
                if   (Nsour//10)%2 !=1:mface = colors[Nsour%10]
                elif (Nsour//10)%2 ==1:mface = 'none'
                color = colors[Nsour%10]
                source = log_all.iloc[nrow[Nsour]]['Source']
                for NLR, LR in enumerate(LRs):
                    all_ = log_all.copy()
                    if Nsour!=len(nrow)-1:
                        row1, row2 = nrow[Nsour], nrow[Nsour+1]
                        all_ = all_.iloc[row1:row2].reset_index(drop=True)
                    else:
                        all_ = all_.iloc[nrow[-1]:].reset_index(drop=True)
                    all_ = all_[all_.Stokes==LR].reset_index(drop=True)
                    chop_idx = np.arange(nrep_cp[Nsour]) * 33
                    all_ = all_.drop(chop_idx, axis=0).reset_index(drop=True)

                    if np.logical_and(NLR==0, Nfreq==0):
                        ax_tsys[NLR].scatter(all_['Time'], all_['Tsys'], marker=mtype, edgecolor=color, facecolor=mface, label='%s'%(source))
                        ax_tsys[NLR].legend(ncol=10, fontsize=10, bbox_to_anchor=(4.75, 1.50), fancybox=True)
                    else:
                        ax_tsys[NLR].scatter(all_['Time'], all_['Tsys'], marker=mtype, edgecolor=color, facecolor=mface)
                    ax_tau [NLR].scatter(all_['Time'], all_['Tau' ], marker=mtype, edgecolor=color, facecolor=mface)
                    ax_el  [NLR].scatter(all_['Time'], all_['El'  ], marker=mtype, edgecolor=color, facecolor=mface)
            ax_tsys[NLR].set_xlim(tmin, tmax)
            ax_tau [NLR].set_xlim(tmin, tmax)
            ax_el  [NLR].set_xlim(tmin, tmax)

        path_fig = self.path_dir + 'Figures/pos/PS_Logs/'
        mkdir(path_fig)
        fig_pslog.savefig(path_fig + '/%s_%s_%s_PSLog.png'%(self.station, self.date, self.proj))
        close_figure(fig_pslog)


def correct_time(data, nchan):
    time_err = data[data.Time > 3600*20].reset_index()
    daysec   = 3600*24
    for nrow in range(time_err.shape[0]):
        time = time_err['MJD'][nrow]-int(time_err['MJD'][nrow]) * u.day.to(u.s)
        if np.logical_or(time<60, time>daysec-60):
           data['MJD'][time_err['index'][nrow]] -= 1
    data['Date'] = Ati(data['MJD'], format='mjd').iso
    data['Year'] = Ati(data['MJD'], format='mjd').byear
    mjd0 = int(np.min(data['MJD']))
    data['Time']     =np.round((data['MJD']-mjd0) * u.day.to(u.s),0).astype(int)
    data['Time'][nchan:] =np.array(data['Time'][nchan:]) -np.array(data['Time'][0:-nchan])
    data['Time'][0:nchan]=np.array(data['Time'][0:nchan])-data['Time'][0]
    return data
