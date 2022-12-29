
'''
L'objectif de ce programme est d'utiliser la théorie de Markowitz pour allouer un portefeuille
'''

###### Importation des modules ######
import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import scipy.optimize as sc
import plotly.graph_objects as go

#####################################
###### Importation des données ######
#####################################
''' On utilise Yahoo finance, les symboles seront sur : https://finance.yahoo.com/ '''
def get_data(stocks, start, end):
    '''
    stocks := représente une liste de symbole des entreprises qui nous interessent
    start et end := permettent de déterminer la période sur laquelle on veut récupérer le prix des actions
    '''
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close'] #On peut récupérer plusieurs informations (High, Low, Open, Volume) on décide de garder le prix close
    returns = stockData.pct_change() #methode permettant d'avoir les retour quotidien
    meanReturns = returns.mean() #on s'interesse au rendement moyen de chaque action
    covMatrix = returns.cov() 
    return meanReturns, covMatrix

####################################################
###### Préliminaire : Fonction de performance ######
####################################################

def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std

############################################
###### Etape 1 : Maximum Sharpe Ratio ######
############################################

# Maximum Sharpe Ratio : Maximiser le retour à une volatilité fixé.
# Afin d'utiliser la fonction du package scipy, nous allons nous rammener à
# un problème de minimisation, donc il faut implémenter le Negative Sharp Ratio.

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns - riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate=0, constraintSet =(0,1)):
    "On veut minimiser la fonction negativeSR"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    # La contrainte à notre problème de minimisation est que la somme des poids doit être égale à 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    # w0 = numAssets*[1./numAssets]
    # La fonction minimize, minimise la fonction par rapport à son premier argument
    result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

#############################################################
###### Etape 2 : Minimiser la variance du portefeuille ######
#############################################################

# Minimization of Portfolio Variance
# On procède exactement de la même manière que précedemment, on commence par
# implémenter la fonction à minimiser, puis on utilise la fonction de scipy

def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

############################################
###### Etape 3 : Portefeuille optimale #####
############################################

# Pour un retour donnée, on cherche le portefeuille de variance minimale

def portfolioReturn(weights, meanReturns, covMatrix):
        return portfolioPerformance(weights, meanReturns, covMatrix)[0]
    
def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return effOpt

############################################
###### Etape 4 : Frontière Efficiente ######
############################################

# On va utiliser la fonction précedente pour construire la frontière efficiente.
# On va parcourir tout les retour possible entre celui associé à la plus basse variance, jusqu'a le retour associé au max sharp ratio

def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]
    
    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])
        
    minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)        
    maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)


    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns

############################################################
###### Etape 5 : Graphique de la Frontière Efficiente ######
############################################################

# On utilise le module plotly
# On crée un graphique pour chaque optimisation, puis on crée un layout pour fusionner les trois graphiques.

def EF_graph(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)
    #Max SR
    MaxSharpeRatio = go.Scatter(
        name='Maximium Sharpe Ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_returns],
        marker=dict(color='red',size=14,line=dict(width=3, color='black'))
    )
    #Min Vol
    MinVol = go.Scatter(
        name='Mininium Volatility',
        mode='markers',
        x=[minVol_std],
        y=[minVol_returns],
        marker=dict(color='green',size=14,line=dict(width=3, color='black'))
    )
    #Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100, 2) for ef_std in efficientList],
        y=[round(target*100, 2) for target in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )
    data = [MaxSharpeRatio, MinVol, EF_curve]
    layout = go.Layout(
        title = 'Portfolio Optimisation with the Efficient Frontier',
        yaxis = dict(title='Annualised Return (%)'),
        xaxis = dict(title='Annualised Volatility (%)'),
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


######################################################################################################
################################################ TEST ################################################
######################################################################################################

stockList = ['TTE', 'BNP.PA', 'GLE.PA']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)
test = yf.download(['^FCHI', '^GDAXI'], startDate, endDate)
meanReturns, covMatrix = get_data(stockList, startDate, endDate)

'''
efficientOpt(meanReturns, covMatrix, )
calculatedResults(meanReturns,covMatrix)


weights = np.array([0.3, 0.3, 0.4])

returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

result = maxSR(meanReturns, covMatrix)
maximum_SR, maxWeights = result['fun'], result['x']
print(maximum_SR, maxWeights)

minVarresult = minimizeVariance(meanReturns, covMatrix)
minVar, minVarWeights = minVarresult['fun'], minVarresult['x']
print(minVar, minVarWeights)
print(efficientOpt(meanReturns, covMatrix, 0.05))

print(calculatedResults(meanReturns, covMatrix))
'''

EF_graph(meanReturns,covMatrix)