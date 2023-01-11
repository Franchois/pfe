
### On récupère le code fournissant la frontière efficiente.
### La seule chose qui changera sera qu'on ajoutera une contrainte lié au risque climatique

###### Importation des modules ######
import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import scipy.optimize as sc
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#####################################
###### Importation des données ######
#####################################
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close'] #On peut récupérer plusieurs informations (High, Low, Open, Volume) on décide de garder le prix close
    returns = stockData.pct_change() #methode permettant d'avoir les retour quotidien
    meanReturns = returns.mean() #on s'interesse au rendement moyen de chaque action
    covMatrix = returns.cov() 
    return meanReturns, covMatrix

def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std

Emission_GES = pd.read_csv("ghg-emissions.csv", sep=",", encoding="utf-8")
PIB_Country = pd.read_csv("PIB_country.csv", sep=",", encoding="utf-8")

def get_intensity_carbon(weights, Countries):
    Intensity_Carbon = {}
    for i in range (0, len(Countries)):
        Intensity_Carbon[Countries[i]]=Emission_GES.loc[Emission_GES["Country/Region"] == Countries[i], '2018'].iloc[0] / PIB_Country.loc[PIB_Country["Country Name"]==Countries[i], '2018'].iloc[0]
    valeurs = np.array(list(Intensity_Carbon.values()))
    Intensity_Carbon_Portfolio = np.sum(valeurs*weights)
    return valeurs, Intensity_Carbon_Portfolio

## Etape 2 / Min Variance

def portfolioVariance(weights, meanReturns, covMatrix, benchmark=False):
    if not isinstance(benchmark, np.ndarray) :
        benchmark = np.zeros(len(weights))
    return portfolioPerformance(weights - benchmark, meanReturns, covMatrix)[1]

def Total_Intensity_Carbon(weights, Countries):
    return get_intensity_carbon(weights, Countries)[1]
    
def minimizeVariance(meanReturns, covMatrix, Countries, intensityTarget, benchmark=False, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, benchmark)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'ineq', 'fun' : lambda x: intensityTarget - Total_Intensity_Carbon(x, Countries)})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

## Graphique volatilité en fonction de intensityTarget

def GraphIC(meanReturns, covMatrix, Countries, benchmark): 
    result = minimizeVariance(meanReturns, covMatrix, Countries, intensityTarget=100000, constraintSet=(0,1))
    tot_intensite_carbon = get_intensity_carbon(result["x"], Countries)[1]
    R=[a/100 for a in range(101)]
    Allocations = [minimizeVariance(meanReturns, covMatrix, Countries, intensityTarget=(1-r)*tot_intensite_carbon, benchmark=benchmark, constraintSet=(0,1)) for r in R]
    Volatilite = [portfolioPerformance(x['x'], meanReturns, covMatrix)[1] for x in Allocations]
    plt.figure()
    plt.plot(R, Volatilite)
    plt.show()
    
    
    

######################
### TEST #####
######################
'''
Allemagne, France, Royaume Unis, Espagne, Pays Bas, Grece, Suede, Belgique, Danemark
DAX, CAC40, FTSE100, IBEX35, AEX, Athens General Composite, OMXS30, BEL20, OMXC20
'''

IndexList=['^GDAXI', '^FCHI', '^IBEX', '^AEX', '^BFX']
endDate = dt.date(2018, 12, 31)
startDate = endDate - dt.timedelta(days=252)

meanReturns, covMatrix = get_data(IndexList, startDate, endDate)

result = minimizeVariance(meanReturns, covMatrix, ['Germany', 'France', 'Spain', 'Netherlands','Belgium'], intensityTarget=100000, constraintSet=(0,1))
weights_intensity = result["x"]

retour, std = portfolioPerformance(weights_intensity, meanReturns, covMatrix)

each_intensite_carbon, tot_intensite_carbon = get_intensity_carbon(result["x"], ['Germany', 'France', 'Spain', 'Netherlands','Belgium'])

print(each_intensite_carbon)
print(tot_intensite_carbon)

#########

result_after = minimizeVariance(meanReturns, covMatrix, ['Germany', 'France', 'Spain', 'Netherlands','Belgium'], benchmark = weights_intensity, intensityTarget=1.7e-10, constraintSet=(0,1))

retour_after, std_after = portfolioPerformance(result_after['x'], meanReturns, covMatrix)

each_intensite_carbon_after, tot_intensite_carbon_after = get_intensity_carbon(result_after["x"], ['Germany', 'France', 'Spain', 'Netherlands','Belgium'])

print(retour_after, std_after)
print(each_intensite_carbon_after)
print(tot_intensite_carbon_after)

####

GraphIC(meanReturns, covMatrix, ['Germany', 'France', 'Spain', 'Netherlands','Belgium'], benchmark=weights_intensity)


'''

[0.48894644 0.22011436 0.         0.         0.29093919]

retour
-0.21480983519048688

std
0.12406074733910856

Intensite_carbon = 1.8184936856868873e-10
-----------------------------

Sans prendre en compte le risque carbonne :

La volatilité  
'''

