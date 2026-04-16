import numpy as np
import os
from pathlib import Path
referenceBox = 12

projectPath = Path(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


fluxesPath = 'FluxesAndContents/'
Prod_DB_FileName = 'ResultDB/ProductionSaves/'
Sim_DB_FileName = 'SimulationSaves/'
boxNameList = ['StratoS', 'TropoS', 'SurfaceWaterS',
                'SurfaceBiotaS','IntermedDeepWaterS',
                'SLBS','LLBS','LitterS','SoilS','PeatS','SedimentarySinkS',
                'StratoN','TropoN','SurfaceWaterN',
                'SurfaceBiotaN','IntermedDeepWaterN',
                'SLBN','LLBN','LitterN','SoilN','PeatN','SedimentarySinkN']
fullNameList = ['Stratosphere', 'Troposphere', 'Surface Water',
                'Surface Biota','Intermed&Deep Sea Water',
                'Short lived Biota','Long lived Biota','Litter','Soil','Peat','Sedimentary Sink',
                'Stratosphere','Troposphere','Surface Water',
                'Surface Biota','Intermed&Deep Sea Water',
                'Short lived Biota','Long lived Biota','Litter','Soil','Peat','Sedimentary Sink']

responseFunctionTime = 100000 #y
nBoxes = 22
massC12 = 12 #u
massC14 = 14.003241 #u
AEarth = 5.1e18 #cm^2$
Avogadro = 6.022140*10**23
secondsPerYear = 31622400
landa = np.log(2)/5740 #1/y
co2mass = 44.01
