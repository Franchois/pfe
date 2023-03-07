
###### Importation des modules ######
import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import scipy.optimize as sc
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import openpyxl


########################################
# Importation et nettoyage des données #
########################################

carbon_int=pd.read_excel("data_pfe.xlsx",sheet_name=0)
returns=pd.read_excel("data_pfe.xlsx",sheet_name=1)
cov=pd.read_excel("data_pfe.xlsx",sheet_name=2)
price=pd.read_excel("data_pfe.xlsx",sheet_name=3)
price.fillna(value=price.mean(), inplace=True)

price["Asia Local Bond USD Unhedged"]= price["Asia Local Bond USD Unhedged"].pct_change()
price["Asia Pacific ex Japan Equity USD Unhedged"]= price["Asia Pacific ex Japan Equity USD Unhedged"].pct_change()
price["Asian High Yield Corporate Bond USD Unhedged"]= price["Asian High Yield Corporate Bond USD Unhedged"].pct_change()
price["Emerging Market Equity USD Unhedged"]= price["Emerging Market Equity USD Unhedged"].pct_change()
price["Europe Equity USD Unhedged"]= price["Europe Equity USD Unhedged"].pct_change()
price["GEM Debt - Hard Currency USD Unhedged"]= price["GEM Debt - Hard Currency USD Unhedged"].pct_change()
price["Global Corporate Bond USD Unhedged"]= price["Global Corporate Bond USD Unhedged"].pct_change()
price["Global Equity USD Unhedged"]= price["Global Equity USD Unhedged"].pct_change()
price["Global Government Bond USD Unhedged"]= price["Global Government Bond USD Unhedged"].pct_change()
price["Global High Yield Bond BB-B USD Unhedged"]= price["Global High Yield Bond BB-B USD Unhedged"].pct_change()
price["Global Inflation Linked Bond USD Unhedged"]= price["Global Inflation Linked Bond USD Unhedged"].pct_change()
price["Global Property USD Unhedged"]= price["Global Property USD Unhedged"].pct_change()
price["Private Equity USD Unhedged"]= price["Private Equity USD Unhedged"].pct_change()
price["US Equity USD Unhedged "]= price["US Equity USD Unhedged"].pct_change()

price.drop(0, axis=0, inplace=True)

Liste_Actif=['Asia Local Bond USD Unhedged',
       'Asia Pacific ex Japan Equity USD Unhedged',
       'Asian High Yield Corporate Bond USD Unhedged',
       'Emerging Market Equity USD Unhedged', 'Europe Equity USD Unhedged',
       'Global Corporate Bond USD Unhedged', 'Global Equity USD Unhedged',
       'Global High Yield Bond BB-B USD Unhedged',
       'Global Property USD Unhedged', 'Private Equity USD Unhedged',
       'US Equity USD Unhedged']



#####################################
########### Préliminaire ############
#####################################

def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
    return returns, std

def Total_Intensity_Carbon(weights,date):
    '''
    Renvoit l'intensity carbonne totale du portefeuille c'est à dire la somme ponderé par les poids des intensités
    carbone de chaque actifs
    '''
    Intensity_Carbon = carbon_int[Liste_Actif].loc[carbon_int["Date"]==date]
    Int_carb_matrix=Intensity_Carbon.to_numpy()
    Intensity_Carbon_Portfolio = np.sum(Int_carb_matrix*weights)
    return Intensity_Carbon_Portfolio

def get_return(date):
    returns_bis = returns[Liste_Actif].loc[returns["Date"]==date]
    return returns_bis

def get_cov(date) :
    Cov_Matrix=cov[cov.Assets.isin(Liste_Actif)][["Date"]+Liste_Actif]
    Cov_Matrix=Cov_Matrix[Cov_Matrix["Date"]==date]
    return Cov_Matrix[Liste_Actif]

## Etape 2 / Min Variance


def portfolioVariance(weights, meanReturns, covMatrix, benchmark=False):
    '''
    Cette fonction permet de calculer sqrt((w-b)*Cov*(w-b))
    '''
    if not isinstance(benchmark, np.ndarray) :
        benchmark = np.zeros(len(weights))
    return portfolioPerformance(weights - benchmark, meanReturns, covMatrix)[1]
    


def minimizeVariance(meanReturns, covMatrix, date, intensityTarget, benchmark=False, constraintSet=(0,1)):
    numAssets = np.shape(meanReturns)[1]
    args = (meanReturns, covMatrix, benchmark)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'ineq', 'fun' : lambda x: intensityTarget - Total_Intensity_Carbon(x, date)})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def minimizeVariance_year(intensityTarget):
    '''
    Cette fonction permet de nous donner pour chaque date l'allocation optimale
    '''
    
    Allocation_by_month={}
    for t in range (0, 20):
        '''
        Dans nos données, on a les covariances et intensité carbone le 29 pour certains mois et les returns le
        30 ou 31 pour d'autres
        '''
        if t not in [6, 11, 14]:
            date=str((returns["Date"].loc[t]))[0:10]
            Cov_Matrix=get_cov(date).to_numpy()
            Retours = get_return(date).to_numpy()
            result=minimizeVariance(Retours,Cov_Matrix,date,intensityTarget)
            Allocation_by_month[date]=result.x
        else :
            date=str((returns["Date"].loc[t]))[0:10]
            Retours = get_return(date).to_numpy()
            date_bis=date[0:8]+"29"
            Cov_Matrix=get_cov(date_bis).to_numpy()
            result=minimizeVariance(Retours,Cov_Matrix,date_bis,intensityTarget)
            Allocation_by_month[date]=result.x
    return Allocation_by_month

def Performance_by_date(intensityTarget):
    '''
    On aimerait voir les résultat du programme d'optimisation sous contrainte d'une intensité carbone à ne pas
    dépasser
    '''
    Allocation_by_month=minimizeVariance_year(intensityTarget)
    Tot_Carbon_Intensity = {}
    Retours = {}
    Volatilite = {}
    for date in Allocation_by_month.keys():
        if date not in ["2021-10-31", "2022-04-30", "2022-07-31"]:
            Tot_Carbon_Intensity[date]=Total_Intensity_Carbon(Allocation_by_month[date],date)
            Cov_Matrix=get_cov(date).to_numpy()
            Back = get_return(date).to_numpy()
            Retours[date]=portfolioPerformance(Allocation_by_month[date], Back, Cov_Matrix)[0]
            Volatilite[date]=portfolioPerformance(Allocation_by_month[date], Back, Cov_Matrix)[1]
        else :
            date_bis=date_bis=date[0:8]+"29"
            Tot_Carbon_Intensity[date]=Total_Intensity_Carbon(Allocation_by_month[date],date_bis)
            Cov_Matrix=get_cov(date_bis).to_numpy()
            Back = get_return(date).to_numpy()
            Retours[date]=portfolioPerformance(Allocation_by_month[date], Back, Cov_Matrix)[0]
            Volatilite[date]=portfolioPerformance(Allocation_by_month[date], Back, Cov_Matrix)[1]
    return Tot_Carbon_Intensity, Retours, Volatilite
        
def GraphPerformance(date):
    x_axis=np.linspace(200,600,50)
    Y_intens_carbon=[]
    Y_retour=[]
    Y_vol=[]
    for intensityTarget in x_axis:
        Performance = Performance_by_date(intensityTarget)
        Y_intens_carbon.append(Performance[0][date])
        Y_retour.append(Performance[1][date])
        Y_vol.append(Performance[2][date])
    Fig_vol = go.Scatter(
        name='Volatilité selon intensité carbone limite',
        mode='lines',
        x=x_axis,
        y=Y_vol,
        line=dict(color='black', width=4, dash='dashdot')
    )
    Fig_retour = go.Scatter(
        name='Retour selon intensité carbone limite',
        mode='lines',
        x=x_axis,
        y=Y_retour,
        line=dict(color='red', width=4, dash='dashdot')
    )
    Fig_intens_tot = go.Scatter(
        name='Intensité Carbon du portefeuille selon intensité carbone limite',
        mode='lines',
        x=x_axis,
        y=Y_intens_carbon,
        line=dict(color='blue', width=4, dash='dashdot')
    )
    
    data = [Fig_vol, Fig_retour, Fig_intens_tot]
    layout = go.Layout(
        title = 'Portfolio Optimisation with intensity carbon threshold',
        yaxis = dict(title=''),
        xaxis = dict(title='Intensity carbon threshold'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)
    
    fig = go.Figure(data=data, layout=layout)
    return fig.show()
    
    
## Etape 2 : Construction de la frontière efficiente

## Idée 1 : On va supposer que l'investisseur ne veuille pas dépasser un certain seuil d'intensité carbone.
## Alors pour un retour donnée on cherche à minimiser la variance sous une contrainte sur l'intensité carbone du
## portefeuille.

def portfolioReturn(weights, meanReturns, covMatrix):
        return portfolioPerformance(weights, meanReturns, covMatrix)[0]


def frontiere_efficiente (date, intensityTarget, constraintSet=(0,1)):
    '''
    On va faire varier la valeur d'un return et trouver le portefeuille optimal pour ce retour donné.
    Il faut donc voir de où à où nous faisons varier notre portefeuille. Une idée naïve consiste à prendre
    le plus petit rendement possible, et le plus grand possible en allouant tout le portefeuille sur un seul actif.
    '''
    Cov_Matrix=get_cov(date).to_numpy()
    Back = get_return(date).to_numpy()
    W = np.identity(11)
    borne_inf = min([portfolioPerformance(W[i], Back, Cov_Matrix)[0] for i in range(11)])
    borne_sup = max([portfolioPerformance(W[i], Back, Cov_Matrix)[0] for i in range(11)])
    returnTarget = np.linspace(borne_inf, borne_sup, 200)
    X_efficient = []
    for target in returnTarget:
        numAssets = np.shape(Back)[1]
        args = (Back, Cov_Matrix)
        constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x, Back, Cov_Matrix) - target},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'ineq', 'fun' : lambda x: intensityTarget - Total_Intensity_Carbon(x, date)})
        bound = constraintSet
        bounds = tuple(bound for asset in range(numAssets))
        effOpt = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
        X_efficient.append(effOpt['fun']) # ['fun'] permet de recuperer directement la variance associé à l'allocation optimale
    return X_efficient, returnTarget

def GraphEfficient(date, intensityTarget):
    X_efficient = frontiere_efficiente (date, intensityTarget)[0]
    returnTarget = frontiere_efficiente (date, intensityTarget)[1]
    #Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std, 2) for ef_std in X_efficient],
        y=[round(target, 2) for target in returnTarget],
        line=dict(color='black', width=4, dash='dashdot')
    )
    
    data = [EF_curve]
    layout = go.Layout(
        title = 'Portfolio Optimisation with the Efficient Frontier',
        yaxis = dict(title='Return (%)'),
        xaxis = dict(title='Volatility (%)'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)
    fig = go.Figure(data=data, layout=layout)
    return fig.show()


#################
## BackTesting ##
#################

