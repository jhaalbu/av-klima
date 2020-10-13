import pandas as pd
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
    maxrr_df = pd.concat ([maxrr, maxrr3, maxrrsd3], axis=1)

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
    month_rr = mon_rr.to_frame() #Lager dataframe, unødvendig? Eklere å plotte?
    month_tm = mon_tm.to_frame() #Lager dataframe, unødvendig? Eklere å plotte?
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
    snosnitt = sno_df['sd'].mean() #Finner snitt snømengd
    sno_df['snitt'] = snosnitt #Lager kolonne med snittmengde for å plotte strek i graf
    maxstr = str(maxaar)[5:15] #Klipper til streng for å vise makspunkt i graf
    maxsno = str(snomax)[5:12] #Klipper til streng for å vise makspunkt i graf
    maksimal_snodato = df['sd'].idxmax().date() 
    maksimal_sno = df['sd'].max()
    anotstring = str(maksimal_snodato) + ' | ' + str(maksimal_sno) + 'cm'

    if ax1 is None:
        ax1 = plt.gca()
    ax1.set_title('Maksimal snødjupe fra ' + startdato[0:4] + ' til ' + sluttdato[0:4])
    ax1.bar(sno_df.index, sno_df['sd'], width=320, snap=False, color='powderblue') 
    ax1.set_xlabel('Årstall')
    ax1.set_ylabel('Snødjupne (cm)')
    ax1.set_ylim(0, sno_df['sd'].max()+60)
    #ax1.text(maxaar, snomax, "Maksimal snødjupne:  " + str(snomax) + ' cm')
    # ax1.annotate(anotstring, xy=(maksimal_snodato, maksimal_sno),  xycoords='data',
    #             xytext=(50, -10), textcoords='offset points',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))

    ax2 = ax1.twinx()
    ax2.plot(sno_df.index, sno_df['snitt'], 'r--', label='Snitt')
    ax2.set_ylim(0, sno_df['sd'].max()+60)
    ax2.get_yaxis().set_visible(False)
    ax1.text(startdato[0:4], df['sd'].max()+20, "Gjennomsnittlig maksimal snødjupne:  " + str(int(snosnitt)) + ' cm')
    #ax2.text(startdato[0:4], df['sd'].max(), anotstring )
    ax2.legend()
    return ax1, ax2

def plot_aarsnedbor(df, ax1=None):
    aar = df['rr'].groupby(pd.Grouper(freq='Y')).sum() #Summerer all nedbør iløpet av eit år
    aar_df = aar.to_frame() #Lager dataframe
    snitt = aar_df['rr'].mean() #Lager snitt av nedbør
    aar_df['snitt'] = snitt #Lager kolonne med snittverdi for plotting

    if ax1 is None:
        ax1 = plt.gca()

    ax1.set_title('Årsnedbør fra ' + startdato[0:4] + ' til ' + sluttdato[0:4])
    ax1.bar(aar_df.index,  aar_df['rr'], width=320, snap=False) #Width er litt vanseklig, dersom ei søyle har bredde 365 dekker den "heile bredden", det er det samme som eit år, da x-aksen er delt opp i år..
    ax1.set_xlabel('Årstall')
    ax1.set_ylabel('Nedbør (mm)')
    ax1.set_ylim(500, aar_df['rr'].max()+500)
    ax1.text(startdato[0:4], aar_df['rr'].max()+200, "Gjennomsnittlig årsnedbør:  " + str(int(snitt)) + ' mm')

    ax2 = ax1.twinx()
    ax2.plot(aar_df.index, aar_df['snitt'], 'r--', label='Snitt')
    ax2.set_ylim(500, aar_df['rr'].max()+500)
    ax2.get_yaxis().set_visible(False)
    ax2.legend()

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
    windDirection = nve_api(lat, lon, '2018-03-01', '2019-12-31', 'windDirection10m24h06') #Vindretning for døgnet
    windSpeed = nve_api(lat, lon, '2018-03-01', '2019-12-31', 'windSpeed10m24h06')
    rr_vind = nve_api(lat, lon, '2018-03-01', '2019-12-31', 'rr')
    tm_vind = nve_api(lat, lon, '2018-03-01', '2019-12-31', 'tm')

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

    dfw['retning'] = dfw['windDirection10m24h06']*45
    dfwx['retning'] = dfwx['windDirection10m24h06']*45
    dfwxrr['retning'] = dfwxrr['windDirection10m24h06']*45
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw=dict(projection='windrose'), figsize=(20,20))
    
    ax1.bar(dfwx['retning'], dfwx['windSpeed10m24h06'], normed=True, opening=1.8)
    ax1.set_title('Generell vindretning - Inndelt i i vindstyrke (m/s)')
    ax1.set_legend()
    ax2.bar(dfw['retning'], dfw['rr'], normed=True,opening=1.8)
    ax2.set_title('Vindretning ved nedbør som snø (mm)', )
    ax2.set_legend()
    ax3.bar(dfwxrr['retning'], dfwxrr['rr'], normed=True, opening=1.8)
    ax3.set_title('Vindretning ved nedbør (mm)')
    ax3.set_legend()

    return fig

def plot_ekstremverdier_3dsno(maxrr_df, ax1=None):
       
    liste =  maxrr_df['sdfsw3d'].tolist()
    array = np.array(liste)
    model = e.Gumbel(array, fit_method = 'mle', ci = 0.05, ci_method = 'delta')
    
    if ax1 is None:
        ax1 = plt.gca()

    return model.plot_return_values()

def plot_ekstremverdier_1drr(maxrr_df, ax1=None):
       
    liste =  maxrr_df['rr'].tolist()
    array = np.array(liste)
    model = e.Gumbel(array, fit_method = 'mle', ci = 0.05, ci_method = 'delta')
    
    if ax1 is None:
        ax1 = plt.gca()

    return model.plot_return_values()

st.sidebar.title('AV-Klima')


#Gi in kordinater for posisjon og start og sluttdato for dataserien.
lokalitet = st.sidebra.text_input("Gi navn til lokalitet", Blåfjellet)
lon = st.sidebar.text_input("Gi NORD koordinat (UTM 33)", 6822565)
#lon = 6822565  #Y
lat = st.sidebar.text_input("Gi ØST koordinat (UTM 33)", 67070)
lon = int(lon)
lat = int(lat)
#lat = 67070      #X
#startdato = st.text_input('Gi startdato', '1958-01-01')
startdato = '1958-01-01'
#sluttdato = st.text_input('Gi sluttdato', '2019-12-31')
sluttdato = '2019-12-31'
knapp = st.sidebar.button('Vis plott')

st.sidebar.write("Nettsida henter ut gridda klimadata frå senorge.no og presenterer disse på plott. Det blir også rekna ut returverdier for 3 døgns snømengde. Ved spørsmål eller feil ta kontakt på jan.aalbu@asplanviak.no")




if knapp:
    bar = st.sidebar.progress(10)
    transformer = Transformer.from_crs(5973, 4326)
    trans_x, trans_y =  transformer.transform(lat, lon)
    bar.progress(20)
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
    folium_static(m)
    bar.progress(25)
    
    #Lager dataframe fra klimadata
    df = klima_dataframe(lat, lon, startdato, sluttdato) 
    
    #Plotter figur

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f'Klimaoversikt for {lokalitet}', fontsize=16)
    ax1 = fig.add_subplot(321)
    bar.progress(30)
    ax1, ax2 = plot_normaler(df)
    ax3 = fig.add_subplot(322)
    bar.progress(40)
    ax3, ax4 = plot_snomengde(df)
    ax5 = fig.add_subplot(323)
    bar.progress(50)
    ax5, ax6 = plot_maks_snodjupne(df)
    ax7 = fig.add_subplot(324)
    bar.progress(60)
    ax7, ax8 = plot_aarsnedbor(df)
    ax9 = fig.add_subplot(325)
    bar.progress(70)
    ax9 = plot_3dsno(df)
    ax10 = fig.add_subplot(326)
    bar.progress(80)
    #ax10 = plot_maks_dognnedbor(df)
    ax10, values = plot_ekstremverdier_3dsno(df)
    bar.progress(90)
    #plt.savefig('samle_figur1.png')
    st.pyplot(fig)
    
    


    #ax, values = plot_ekstremverdier(df)
    st.write('Returverdier 3 døgn snømengde')
    st.write('100 år: ' + str(round(values[0], 0)))
    st.write('1000 år: ' + str(round(values[1],0)))
    st.write('5000 år: ' + str(round(values[2],0)))

    fig = plot_vind()
 
    
    st.pyplot(fig)
    bar.progress(100)
#ax1 = fig.add_subplot(111)
#ax2 = fig.add_subplot(112)

#plot_something(data1, ax1, color='blue')
#plot_something(data2, ax2, color='red')
