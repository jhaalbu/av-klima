import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.dates import DateFormatter
from windrose import WindroseAxes
import requests
import datetime
import numpy as np
import folium
from pyproj import CRS
from pyproj import Transformer

import streamlit as st
from streamlit_folium import folium_static
import folium
import extreme as e

def nve_api(lat, lon, startdato, sluttdato, para):
    """Henter data frå NVE api GridTimeSeries

    Args:
        lat (str): øst-vest koordinat (i UTM33)
        output er verdien i ei liste, men verdi per dag, typ ne
        lon (str): nord-sør koordinat (i UTM33)
        startdato (str): startdato for dataserien som hentes ned
        sluttdato (str): sluttdato for dataserien som hentes ned
        para (str): kva parameter som skal hentes ned f.eks rr for nedbør
        
    Returns:
        verdier (liste) : returnerer i liste med klimaverdier
        
    """

    api = 'http://h-web02.nve.no:8080/api/'
    url = api + '/GridTimeSeries/' + str(lat) + '/' + str(lon) + '/' + str(startdato) + '/' + str(sluttdato) + '/' + para + '.json'
    r = requests.get(url)

    verdier = r.json()
    return verdier

def transformer(lat, lon):
    transformer = Transformer.from_crs(5973, 4326)
    trans_x, trans_y =  transformer.transform(lat, lon)
    return trans_x, trans_y

@st.cache
def klima_dataframe(lat, lon, startdato, sluttdato):
    """Funksjonen tar inn ei liste  med lister med klimaparameter og lager dataframe

    Args:
        lat (str): øst-vest koordinat (i UTM33)
        output er verdien i ei liste, men verdi per dag, typ ne
        lon (str): nord-sør koordinat (i UTM33)
        startdato (str): startdato for dataserien som hentes ned
        sluttdato (str): sluttdato for dataserien som hentes ned
        para (str): kva parameter som skal hentes ned f.eks rr for nedbør

    Returns
        df (dataframe): Returnerer ei pandas dataframe
    """
    #bar = st.progress(0)
    rr = nve_api(lat, lon, startdato, sluttdato, 'rr') #Nedbør døgn
    #bar.progress(20)
    fsw = nve_api(lat, lon, startdato, sluttdato, 'fsw') #Nysnø døgn
    #bar.progress(40)
    sdfsw3d = nve_api(lat, lon, startdato, sluttdato, 'sdfsw3d') #Nynsø 3 døgn
    #bar.progress(60)
    sd = nve_api(lat, lon, startdato, sluttdato, 'sd') #Snødybde
    #bar.progress(80)
    tm = nve_api(lat, lon, startdato, sluttdato, 'tm') #Døgntemperatur
    #bar.progress(100)
    start = datetime.datetime(int(startdato[0:4]), int(startdato[5:7]), int(startdato[8:10]))#'1960-01-01' 
    end = datetime.datetime(int(sluttdato[0:4]), int(sluttdato[5:7]), int(sluttdato[8:10]))
    #Etablerer pandas dataframe frå rr liste
    df = pd.DataFrame(rr['Data'])
    #Lager kolonne med datoer
    df['dato'] = pd.date_range(start, end)
    #Gir nytt navn til rr kolonna
    df.rename({0 : rr['Theme']}, axis=1, inplace=True)
    #Setter datokolonna til indexkolonna, slik at det går an å bruke grouperfunksjon for å sortere på tidsserie
    df.set_index('dato', inplace=True)
    #Etablerer kolonner i dataframefor andre parameter
    df[fsw['Theme']] = fsw['Data']
    df[sd['Theme']] = sd['Data']
    df[sdfsw3d['Theme']] = sdfsw3d['Data']
    df[tm['Theme']] = tm['Data']
    df['rr3'] = df.rr.rolling(3).sum() #Summerer siste 3 døgn 
    df[df > 60000] = 0
    return df

def max_df(df):
    maxrrsd3 = df['sdfsw3d'].groupby(pd.Grouper(freq='Y')).max()  #3 døgns snømengde
    maxrr = df['rr'].groupby(pd.Grouper(freq='Y')).max()
    maxrr3 = df['rr3'].groupby(pd.Grouper(freq='Y')).max()
    maxsd = df['sd'].groupby(pd.Grouper(freq='Y')).max()
    maxrr_df = pd.concat ([maxrr, maxrr3, maxrrsd3, maxsd], axis=1)

    return maxrr_df



def vind_dataframe(lat, lon, startdato, sluttdato):
    #Vinddata - finnes kun i ein begrensa tidsperiode som grid data!
    windDirection = nve_api(lat, lon, startdato, sluttdato, 'windDirection10m24h06') #Vindretning for døgnet
    windSpeed = nve_api(lat, lon, startdato, sluttdato, 'windSpeed10m24h06')
    rr_vind = nve_api(lat, lon, startdato, sluttdato, 'rr')
    tm_vind = nve_api(lat, lon, startdato, sluttdato, 'tm')

    #3 timersdata mot vind, fra 2018-03-01 til 2019-12-31
    windDirection3h = nve_api(lat, lon, startdato, sluttdato, 'windDirection10m3h')
    windSpeed3h = nve_api(lat, lon, startdato, sluttdato, 'windSpeed10m3h')
    rr3h_vind = nve_api(lat, lon, startdato, sluttdato, 'rr3h')
    tm3h_vind = nve_api(lat, lon, startdato, sluttdato, 'tm3h')

    startwind = datetime.datetime(2018, 3, 1)
    endwind = datetime.datetime(2019, 12, 31)

    
    #Lager dataframe for daglig vindretning og styrke, sammen med nedbør?
    dfw = pd.DataFrame(windDirection['Data']) #Henter inn verdiar for vindretning
    dfw['dato'] = pd.date_range(startwind, endwind) #Lager til datoer som ikkje kjem automatisk frå NVE
    dfw.rename({0 : windDirection['Theme']}, axis=1, inplace=True) #Gir nytt navn til kolonne med windretning
    dfw.set_index('dato', inplace=True) #Setter dato som index i dataframe
    dfw[windSpeed['Theme']] = windSpeed['Data']
    dfw[rr_vind['Theme']] = rr_vind['Data']
    dfw[tm_vind['Theme']] = tm_vind['Data']

    dfwx = dfw.copy()
    indexNames = dfw[dfw['windSpeed10m24h06'] <= 3].index
    dfw.drop(indexNames , inplace=True)
    indexNames = dfw[dfw['rr'] <= 5].index
    dfw.drop(indexNames , inplace=True)
    indexNames = dfw[dfw['tm'] >= 1].index
    dfw.drop(indexNames , inplace=True)

    #dfw[dfw > 60000] = 0
    indexNames = dfwx[dfwx['windDirection10m24h06'] >= 1000].index
    dfwx.drop(indexNames , inplace=True)
    indexNames = dfwx[dfwx['windSpeed10m24h06'] >= 1000].index
    dfwx.drop(indexNames , inplace=True)

    #Lager dataframe med verdier for nedbør over 1 mm
    dfwxrr = dfwx.copy()
    indexNames = dfwxrr[dfwxrr['rr'] <= 5].index
    dfwxrr.drop(indexNames , inplace=True)


def plot_normaler(df, ax1=None):
    #Lager dataframe med månedsvise mengder av temperatur og nedbør GRAF1
    #df['month'] = pd.DatetimeIndex(df.index).month #Lager ei kolonne med månedsnummer

    mon_rr = df['rr'].groupby(pd.Grouper(freq='M')).sum() #Grupperer nedbør etter måneder per år og summerer
    mon_tm = df['tm'].groupby(pd.Grouper(freq='M')).mean() #Grupperer temperatur etter måneder per år og tar snitt
    month_rr_temp = mon_rr.to_frame() #Lager dataframe, unødvendig? Eklere å plotte?
    month_tm = mon_tm.to_frame() #Lager dataframe, unødvendig? Eklere å plotte?
    startaar = int(str(month_rr_temp.index[0])[0:4])
    sluttaar = int(str(month_rr_temp.index[-1])[0:4])
    month_rr = month_rr_temp['rr'].groupby(month_rr.index.month).sum()/(sluttaar-startaar)
    month_rr['m'] = pd.DatetimeIndex(month_rr.index).month #lager kolonne for månedsnummer
    month_tm['m'] = pd.DatetimeIndex(month_tm.index).month #Lager kolonne for månedsnummer
    month_mean_tm = month_tm.groupby(['m']).mean()
    if ax1 is None:
        ax1 = plt.gca()
    ax1.set_title('Gjennomsnittlig månedsnedbør og temperatur ' + startdato[0:4] + ' til ' + sluttdato[0:4])
    ax1.bar(month_rr['m'], month_rr['rr'], width=0.5, snap=False)
    ax1.set_xlabel('Måned')
    ax1.set_ylabel('Nedbør (mm)')
    ax1.set_ylim(0, month_rr['rr'].max()+50)
    #ax1.text('1960', aar_df['rr'].max()+20, "Gjennomsnittlig månedsnedbør:  " + str(int(snitt)) + ' mm')

    ax2 = ax1.twinx()#Setter ny akse på høgre side 
    ax2.plot(month_mean_tm.index, month_mean_tm['tm'], 'r', label='Gjennomsnittstemperatur', linewidth=3.5)
    ax2.set_ylim(month_mean_tm['tm'].min()-2, month_mean_tm['tm'].max()+5)
    ax2.set_ylabel(u'Temperatur (\u00B0C)')
    ax2.yaxis.set_tick_params(length=0)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.get_yaxis().set_visible(True)
    ax2.legend()

    return ax1, ax2

def plot_snomengde(df, ax1=None):
    dag = df['sd'].groupby(df.index.strftime('%m-%d')).mean()
    dag_sd_df = dag.to_frame()
    dagtm = df['tm'].groupby(df.index.strftime('%m-%d')).mean()
    dag_tm_df = dagtm.to_frame()
    dag_tm_df['tm_min'] = df['tm'].groupby(df.index.strftime('%m-%d')).min()
    dag_tm_df['tm_max'] = df['tm'].groupby(df.index.strftime('%m-%d')).max()
    dag_sd_df['sd_max'] = df['sd'].groupby(df.index.strftime('%m-%d')).max()
    dag_sd_df['sd_min'] = df['sd'].groupby(df.index.strftime('%m-%d')).min()

    if ax1 is None:
        ax1 = plt.gca()
    
    ax1.plot(dag_sd_df.index, dag_sd_df['sd'], label='Snitt snømengde')
    ax1.plot(dag_sd_df.index, dag_sd_df['sd_max'], label='Max snømengde')
    ax1.plot(dag_sd_df.index, dag_sd_df['sd_min'], label='Min snømengde')
    ax1.xaxis.set_major_locator(MultipleLocator(32))
    #ax1.xaxis.set_major_formatter(FormatStrFormatter('%m'))
    ax1.set_title('Snømengde  ' + startdato[0:4] + ' til ' + sluttdato[0:4])
    ax1.set_xlabel('Dag i året (måned-dag)')
    ax1.set_ylabel('Snøhøgde (cm)')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(dag_tm_df.index, dag_tm_df['tm'], 'r--', label='Gjennomsnittstemperatur')
    ax2.xaxis.set_major_locator(MultipleLocator(32))
    ax2.legend(loc='lower left')
    ax2.set_ylim(dag_tm_df['tm'].min()-5, dag_tm_df['tm'].max()+5)
    ax2.axhline(0, linestyle='--', color='grey', linewidth=0.5)
    ax2.set_ylabel(u'Temperatur (\u00B0C)')

    return ax1, ax2

def plot_maks_snodjupne(df, ax1=None):
    sno = df['sd'].groupby(pd.Grouper(freq='Y')).max() #Finner maksimal snødjupne per år
    sno_df = sno.to_frame() #Lager pandas series om til dataframe
    maxaar = sno_df.idxmax() #Finner året med maksimal snømengde
    snomax = sno_df.max() #Finner maksimal snømengde
   
    sno_df_6090 = sno_df.loc['1961':'1990']
    snosnitt_6090 = sno_df_6090 ['sd'].mean() #Finner snitt snømengd

    sno_df_9020 = sno_df.loc['1991':'2020']
    snosnitt_9020 = sno_df_9020['sd'].mean() #Finner snitt snømengd
    snosnitt = sno_df['sd'].mean() #Finner snitt snømengd
    sno_df['snitt'] = snosnitt #Lager kolonne med snittmengde for å plotte strek i graf
    maxstr = str(maxaar)[5:15] #Klipper til streng for å vise makspunkt i graf
    maxsno = str(snomax)[5:12] #Klipper til streng for å vise makspunkt i graf
    maksimal_snodato = df['sd'].idxmax().date() 
    maksimal_sno = df['sd'].max()
    anotstring = 'Maks snøhøgde: ' +str(maksimal_snodato) + ' | ' + str(maksimal_sno) + 'cm'
    
    sd_df_trend = sno_df.copy()
    sd_df_trend.index = sd_df_trend.index.map(datetime.date.toordinal)
    slope, y0, r, p, stderr = stats.linregress(sd_df_trend.index, sd_df_trend['sd'])
    x_endpoints = pd.DataFrame([sd_df_trend.index[0], sd_df_trend.index[-1]])
    y_endpoints = y0 + slope * x_endpoints
    if sno_df['sd'].min() - 60 < 0:
        snomin = 0
    else:
        snomin = sno_df['sd'].min() - 60

    if ax1 is None:
        ax1 = plt.gca()
    ax1.set_title('Maksimal snødjupe fra ' + startdato[0:4] + ' til ' + sluttdato[0:4])
    ax1.bar(sno_df.index, sno_df['sd'], width=320, snap=False, color='powderblue') 
    ax1.set_xlabel('Årstall')
    ax1.set_ylabel('Snødjupne (cm)')
    ax1.set_ylim(snomin, sno_df['sd'].max()+60)
    # ax1.annotate(anotstring, xy=(maksimal_snodato, maksimal_sno),  xycoords='data',
    #             xytext=(0.80, 0.95), textcoords='axes fraction',
    #             arrowprops=dict(arrowstyle="->"))

    ax2 = ax1.twinx()
    ax2.plot([sno_df.index[0], sno_df.index[-1]], [y_endpoints[0][0], y_endpoints[0][1]], linestyle='dashed', linewidth=1, color='y', label='Trend')
    ax2.plot(sno_df.index, sno_df['snitt'], linewidth=1, linestyle='dashed', color='b', label='Snitt')
    ax2.hlines(y=snosnitt_6090, xmin='1961-01-01', xmax='1990-12-31', linestyle='dashed', linewidth=2, color='g', label='Snitt 1961-1990')
    ax2.hlines(y=snosnitt_9020, xmin='1990-01-01', xmax='2020-12-31', linestyle='dashed', linewidth=2, color='r', label='Snitt 1991-2020')
    ax2.set_ylim(snomin, sno_df['sd'].max()+60)
    ax2.get_yaxis().set_visible(False)
    ax1.text(startdato[0:4], df['sd'].max()+20, "Gjennomsnittlig maksimal snødjupne (1991-2020):  " + str(int(snosnitt_9020)) + ' cm')
    ax2.legend(loc='best')
    
    return ax1, ax2

def plot_aarsnedbor(df, ax1=None):
    aar = df['rr'].groupby(pd.Grouper(freq='Y')).sum() #Summerer all nedbør iløpet av eit år
    aar_df = aar.to_frame() #Lager dataframe
    snitt = aar_df['rr'].mean() #Lager snitt av nedbør
    aar_df['snitt'] = snitt #Lager kolonne med snittverdi for plotting
    aar_df_6090 = aar_df.loc['1961':'1990']
    aarsnitt_6090 = aar_df_6090 ['rr'].mean() 
    aar_df_9020 = aar_df.loc['1991':'2020']
    aarsnitt_9020 = aar_df_9020['rr'].mean() 
    aar_df_trend = aar_df.copy()
    aar_df_trend.index = aar_df_trend.index.map(datetime.date.toordinal)
    slope, y0, r, p, stderr = stats.linregress(aar_df_trend.index, aar_df_trend['rr'])
    x_endpoints = pd.DataFrame([aar_df_trend.index[0], aar_df_trend.index[-1]])
    y_endpoints = y0 + slope * x_endpoints

    if ax1 is None:
        ax1 = plt.gca()

    ax1.set_title('Årsnedbør fra ' + startdato[0:4] + ' til ' + sluttdato[0:4])
    ax1.bar(aar_df.index,  aar_df['rr'], width=320, snap=False) #Width er litt vanseklig, dersom ei søyle har bredde 365 dekker den "heile bredden", det er det samme som eit år, da x-aksen er delt opp i år..
    ax1.set_xlabel('Årstall')
    ax1.set_ylabel('Nedbør (mm)')
    ax1.set_ylim(aar_df['rr'].min()-200, aar_df['rr'].max()+500)
    ax1.text(startdato[0:4], aar_df['rr'].max()+200, "Gjennomsnittlig årsnedbør (1991-2020):  " + str(int(aarsnitt_9020)) + ' mm')

    ax2 = ax1.twinx()
    ax2.plot([aar_df.index[0], aar_df.index[-1]], [y_endpoints[0][0], y_endpoints[0][1]], linestyle='dashed', linewidth=1, color='r', label='Trend')
    ax2.hlines(y=aarsnitt_6090, xmin='1961-01-01', xmax='1990-12-31', linestyle='dashed', linewidth=2, color='g', label='Snitt 1961-1990')
    ax2.hlines(y=aarsnitt_9020, xmin='1990-01-01', xmax='2020-12-31', linestyle='dashed', linewidth=2, color='r', label='Snitt 1991-2020')
    ax2.plot(aar_df.index, aar_df['snitt'], linestyle='dotted', linewidth=1, color='r', label='Snitt 1958-2020')
    ax2.set_ylim(aar_df['rr'].min()-200, aar_df['rr'].max()+500)
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc='lower right')

    return ax1, ax2

def plot_3dsno(df, ax1=None):
    maxrrsd3 = df['sdfsw3d'].groupby(pd.Grouper(freq='Y')).max()  #3 døgns snømengde
    maxrr = df['rr'].groupby(pd.Grouper(freq='Y')).max()
    maxrr3 = df['rr3'].groupby(pd.Grouper(freq='Y')).max()
    maxrr_df = pd.concat ([maxrr, maxrr3, maxrrsd3], axis=1)
    maksimal_sdfsw3ddato = df['sdfsw3d'].idxmax().date()
    maksimal_sdfsw3d = df['sdfsw3d'].max()
    if ax1 is None:
        ax1 = plt.gca()

    ax1.set_title('Maksimal 3 døgns snømengde')
    ax1.bar(maxrr_df.index, maxrr_df['sdfsw3d'], width=320, snap=False, color='powderblue')
    ax1.set_xlabel('Årstall')
    ax1.set_ylabel('Maksimal årlig 3 døgns snømengde (mm)')
    ax1.set_ylim(0, maxrr_df['sdfsw3d'].max()+10)
    # ax1.annotate((str(maksimal_sdfsw3ddato) + ' | ' + str(maksimal_sdfsw3d) + ' mm'), xy=(maksimal_sdfsw3ddato, maksimal_sdfsw3d),  xycoords='data',
    #             xytext=(50, 0), textcoords='offset points',
    #             arrowprops=dict(arrowstyle="->"))
    ax1.text(startdato[0:4], df['sdfsw3d'].max(), 'Maksimalverdi: ' + str(maksimal_sdfsw3ddato) + ' | ' + str(maksimal_sdfsw3d) + ' mm' )
    return ax1

def plot_maks_dognnedbor(df, ax1=None):
    maxrrsd3 = df['sdfsw3d'].groupby(pd.Grouper(freq='Y')).max()  #3 døgns snømengde
    maxrr = df['rr'].groupby(pd.Grouper(freq='Y')).max()
    maxrr3 = df['rr3'].groupby(pd.Grouper(freq='Y')).max()
    maxrr_df = pd.concat ([maxrr, maxrr3, maxrrsd3], axis=1)
    maksimal_rrdato = df['rr'].idxmax().date()
    maksimal_rr = df['rr'].max()

    if ax1 is None:
        ax1 = plt.gca()

    ax1.set_title('Maksimal døgnnedbør')
    ax1.bar(maxrr_df.index, maxrr_df['rr'], width=320, snap=False)
    ax1.set_xlabel('Årstall')
    ax1.set_ylabel('Maksimal årlig døgnnedbør (mm)')
    ax1.set_ylim(0, maxrr_df['rr'].max()+10)
    ax1.text(startdato[0:4], df['rr'].max(), 'Maksimalverdi: ' + str(maksimal_rrdato) + ' | ' + str(maksimal_rr) + ' mm')
    # ax1.annotate((str(maksimal_rrdato) + ' | ' + str(maksimal_rr) + ' mm'), xy=(maksimal_rrdato, maksimal_rr),  xycoords='data',
    #             xytext=(-150, 0), textcoords='offset points',
    #             arrowprops=dict(arrowstyle="->"))

    return ax1

def plot_vind(ax1=None):
    windDirection = nve_api(lat, lon, '2018-03-01', '2021-03-01', 'windDirection10m24h06') #Vindretning for døgnet
    windSpeed = nve_api(lat, lon, '2018-03-01', '2021-03-01', 'windSpeed10m24h06')
    rr_vind = nve_api(lat, lon, '2018-03-01', '2021-03-01', 'rr')
    tm_vind = nve_api(lat, lon, '2018-03-01', '2021-03-01', 'tm')

    startwind = datetime.datetime(2018, 3, 1)
    endwind = datetime.datetime(2021, 3, 1)

    #Lager dataframe for daglig vindretning og styrke, sammen med nedbør?
    dfw = pd.DataFrame(windDirection['Data']) #Henter inn verdiar for vindretning
    dfw['dato'] = pd.date_range(startwind, endwind) #Lager til datoer som ikkje kjem automatisk frå NVE
    dfw.rename({0 : windDirection['Theme']}, axis=1, inplace=True) #Gir nytt navn til kolonne med windretning
    dfw.set_index('dato', inplace=True) #Setter dato som index i dataframe
    dfw[windSpeed['Theme']] = windSpeed['Data']
    dfw[rr_vind['Theme']] = rr_vind['Data']
    dfw[tm_vind['Theme']] = tm_vind['Data']


    dfwx = dfw.copy()
    indexNames = dfw[dfw['windSpeed10m24h06'] <= 3].index
    dfw.drop(indexNames , inplace=True)
    indexNames = dfw[dfw['rr'] <= 1].index
    dfw.drop(indexNames , inplace=True)
    indexNames = dfw[dfw['tm'] >= 1].index
    dfw.drop(indexNames , inplace=True)

    #dfw[dfw > 60000] = 0
    indexNames = dfwx[dfwx['windDirection10m24h06'] >= 1000].index
    dfwx.drop(indexNames , inplace=True)
    indexNames = dfwx[dfwx['windSpeed10m24h06'] >= 1000].index
    dfwx.drop(indexNames , inplace=True)
    indexNames = dfwx[dfwx['windSpeed10m24h06'] <= 5].index
    dfwx.drop(indexNames , inplace=True)
    


    #Lager dataframe med verdier for nedbør over 1 mm
    dfwxrr = dfwx.copy()
    indexNames = dfwxrr[dfwxrr['rr'] <= 1].index
    dfwxrr.drop(indexNames , inplace=True)
    indexNames = dfwxrr[dfwxrr['tm'] < 1].index
    dfwxrr.drop(indexNames , inplace=True)

    dfw['retning'] = dfw['windDirection10m24h06']*45
    dfwx['retning'] = dfwx['windDirection10m24h06']*45
    dfwxrr['retning'] = dfwxrr['windDirection10m24h06']*45
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw=dict(projection='windrose'), figsize=(20,20))
    
    ax1.bar(dfwx['retning'], dfwx['windSpeed10m24h06'], normed=True, opening=1.8)
    ax1.set_title('%-vis dager med gitt vindretning')
    #ax1.legend(title='Vindstyrke (m/s')
    ax1.set_legend(title='Vindstyrke (m/s)')

    ax2.bar(dfwxrr['retning'], dfwxrr['rr'], normed=True, opening=1.8)
    ax2.set_title('%-vis dager med gitt vindretning og nedbør')
    ax2.set_legend(title='Nedbør (mm)')

    ax3.bar(dfw['retning'], dfw['rr'], normed=True,opening=1.8)
    ax3.set_title('%-vis dager med gitt vindretning og snø', )
    ax3.set_legend(title='Nedbør, temp < 1 grad (mm)')

    return fig

def plot_ekstremverdier_3dsno(maxrr_df, ax1=None):
       
    liste =  maxrr_df['sdfsw3d'].tolist()
    array = np.array(liste)
    model = e.Gumbel(array, fit_method = 'mle', ci = 0.05, ci_method = 'delta')
    
    if ax1 is None:
        ax1 = plt.gca()

    return model.plot_return_values('3ds')

def plot_ekstremverdier_1drr(maxrr_df, ax1=None):
       
    liste =  maxrr_df['rr'].tolist()
    array = np.array(liste)
    model = e.Gumbel(array, fit_method = 'mle', ci = 0.05, ci_method = 'delta')
    
    if ax1 is None:
        ax1 = plt.gca()

    return model.plot_return_values('1drr')

def plot_ekstremverdier_sd(maxrr_df, ax1=None):
    liste = maxrr_df['sd'].tolist()
    array = np.array(liste)
    model = e.Gumbel(array, fit_method = 'mle', ci = 0.05, ci_method = 'delta')

    if ax1 is None:
        ax1 = plt.gca()

    return model.plot_return_values('sd')

st.sidebar.title('AV-Klima')


#Gi in kordinater for posisjon og start og sluttdato for dataserien.
lokalitet = st.sidebar.text_input("Gi navn til lokalitet", 'Blåfjellet')
lon = st.sidebar.text_input("Gi NORD koordinat (UTM 33)", 6822565)
#lon = 6822565  #Y
lat = st.sidebar.text_input("Gi ØST koordinat (UTM 33)", 67070)

#lat = 67070      #X
#startdato = st.text_input('Gi startdato', '1958-01-01')
startdato = '1958-01-01'
#sluttdato = st.text_input('Gi sluttdato', '2019-12-31')
sluttdato = '2020-12-31'
knapp = st.sidebar.button('Vis plott')
#lon = int(lon)
#lat = int(lat)
st.sidebar.write("Nettsida henter ut gridda klimadata frå senorge.no og presenterer disse på plott. Det blir også rekna ut returverdier for 3 døgns snømengde. Ved spørsmål eller feil ta kontakt på jan.aalbu@asplanviak.no")
st.sidebar.write('')
st.sidebar.write('Vinddata må brukast med forsiktigheit. Vinddata finnes kunn fra mars 2018 - mars 2021. Vinddata bør hentes fra høgaste punkt i området, og ikkje nede i fjord/dalstrøk.')
st.sidebar.write('')
st.sidebar.write('Scriptet bruker litt tid på å hente ned data og rekne ut ekstremverdier. Det er ferdigkjørt når blå statusbar er fyllt.')



if knapp:
    lon = int(float(lon.strip()))
    lat = int(float(lat.strip()))
    bar = st.sidebar.progress(10)
    transformer = Transformer.from_crs(5973, 4326)
    trans_x, trans_y =  transformer.transform(lat, lon)
    bar.progress(5)
    m = folium.Map(location=[trans_x, trans_y], zoom_start=10) #Bruker transformerte koordinater (ikkej funne ut korleis bruke UTM med folium)
    folium.Circle(
        radius=1000,
        location=[trans_x, trans_y],
        fill=False,
    ).add_to(m)
    #Legger til Norgeskart WMS
    folium.raster_layers.WmsTileLayer(
        url='https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=topo4&zoom={z}&x={x}&y={y}',
        name='Norgeskart',
        fmt='image/png',
        layers='topo4',
        attr=u'<a href="http://www.kartverket.no/">Kartverket</a>',
        transparent=True,
        overlay=True,
        control=True,
        
    ).add_to(m)
    st.write('Posisjonen plottes på kart for å verifisere riktig koordinater.')
    folium_static(m)
    bar.progress(10)
    
    #Lager dataframe fra klimadata
    df = klima_dataframe(lat, lon, startdato, sluttdato) 

    #Plotter figur
    #logo = plt.imread('logo_av_asplanviak.png')
    fig = plt.figure(figsize=(20, 18))
    #fig.suptitle('Klimasammendrag for' + lokalitet, fontsize=16)
    #fig.suptitle(f'Klimaoversikt for {lokalitet}', fontsize=16, verticalalignment='bottom')
    ax1 = fig.add_subplot(321)
    bar.progress(15)
    ax1, ax2 = plot_normaler(df)
    ax3 = fig.add_subplot(322)
    bar.progress(20)
    ax3, ax4 = plot_snomengde(df)
    ax5 = fig.add_subplot(323)
    bar.progress(25)
    ax5, ax6 = plot_maks_snodjupne(df)
    ax7 = fig.add_subplot(324)
    bar.progress(30)
    ax7, ax8 = plot_aarsnedbor(df)
    ax9 = fig.add_subplot(325)
    bar.progress(40)
    ax9 = plot_3dsno(df)
    ax10 = fig.add_subplot(326)
    bar.progress(60)
    #ax10 = plot_maks_dognnedbor(df)
    ax10, values_3ds = plot_ekstremverdier_3dsno(max_df(df))
    bar.progress(70)
    #plt.savefig('samle_figur1.png')
    st.write('Figur med klimadata, trykk på piler i høgre hjørne av figuren for å vise fullskjerm')
    #ax9.figure.figimage(logo, 20, 20, alpha=.30, zorder=1)
    fig.suptitle(f'Klimaoversikt for {lokalitet}', fontsize=30, y=0.9, va='bottom')
    st.pyplot(fig)
    

    st.write('Plot med vindroser for ' + lokalitet)
    st.write('OBS! Bruk med varsemd. Kortvarig datasett!')
    fig = plot_vind()
    st.pyplot(fig)
 
    st.write('Returverdier figur med returverdier')
    fig = plt.figure(figsize=(8, 4))
    ax21 = fig.add_subplot(111)
    ax21 = plot_ekstremverdier_3dsno(max_df(df))
    bar.progress(75)
    st.pyplot(fig)
    fig = plt.figure(figsize=(8, 4))
    ax22 = fig.add_subplot(111)
    ax22, values_1drr = plot_ekstremverdier_1drr(max_df(df))
    st.pyplot(fig)
    fig = plt.figure(figsize=(8, 4))
    ax23 = fig.add_subplot(111)
    bar.progress(80)
    ax22, values_sd = plot_ekstremverdier_sd(max_df(df))
    st.pyplot(fig)


    #ax, values = plot_ekstremverdier(df)
    st.write('Returverdier 3 døgn snømengde ' + lokalitet)
    st.write('Basert på snøkart')
    st.write('100 år: ' + str(round(values_3ds[0], 0)))
    st.write('1000 år: ' + str(round(values_3ds[1],0)))
    st.write('5000 år: ' + str(round(values_3ds[2],0)))

    st.write('Returverdier 1 døgn nedbør')
    st.write('100 år: ' + str(round(values_1drr[0], 0)))
    st.write('1000 år: ' + str(round(values_1drr[1],0)))
    st.write('5000 år: ' + str(round(values_1drr[2],0)))

#     st.write('Returverdier snøhøgde')
#     st.write('100 år: ' + str(round(values_sd[0], 0)))
#     st.write('1000 år: ' + str(round(values_sd[1],0)))
#     st.write('5000 år: ' + str(round(values_sd[2],0)))
    bar.progress(90)  

    
    bar.progress(100)
#ax1 = fig.add_subplot(111)
#ax2 = fig.add_subplot(112)

#plot_something(data1, ax1, color='blue')
#plot_something(data2, ax2, color='red')
