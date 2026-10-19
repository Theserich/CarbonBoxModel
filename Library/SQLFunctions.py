from pandas import DataFrame
from Library.DBconnect import DBconnect
from numpy import array
import numpy as np
from Library.dfFunctions import datadict
from Library.GlobalPathsAndConstants import *


DB = DBconnect()
def getData(t0, t1, dbdata=True, bp=True,maxdate=None):
    if bp:
        t0_ = t0
        t1_ = t1
    if bp == False:
        t0_ = 1950-t0
        t1_ = 1950-t1
    t0 = min(t0_, t1_)
    t1 = max(t0_, t1_)
    if dbdata:
        cnx = DB.getConnection()
        cursor = cnx.cursor()
        dataSelectString = "SELECT sample_nr as sample_nr,target_nr as target_nr,prep_nr,Intcal,  project,project_nr ,magazine,user_label,last_name, target_pressed, cast(substring_index(user_label_nr,'|',1) as decimal(6,1)) as bp,  user_label_nr ,c14_age,c14_age_sig, fm as fm,fm_sig as fm_sig" \
                           + " , dc13, dc13_sig FROM db_ams.target_v WHERE research='calibration' and stop=0 and c14_age is not null and sample_nr != 107999 " \
                           + " and CONVERT(user_label_nr,UNSIGNED INTEGER) >= %i and CONVERT(user_label_nr,UNSIGNED INTEGER)<= %i AND user_label !=  'OXA 2' and user_label != 'K-19' and project != 'MWK-80-103A' and project != '2022, DISJ-e-10611'" % (
                               t0, t1) #and project !='MWK85-05B(2)'
        dataSelectString_save = "SELECT sample_nr as sample_nr,target_nr as target_nr,prep_nr, project ,magazine,user_label,last_name,  cast(substring_index(user_label_nr,'|',1) as decimal(6,1)) as bp,  user_label_nr ,c14_age,c14_age_sig, fm as fm,fm_sig as fm_sig" \
                           + " , dc13 FROM db_ams.target_v WHERE research='calibration' and stop=0 and c14_age is not null and sample_nr != 107999 " \
                           + " and CONVERT(user_label_nr,UNSIGNED INTEGER) >= %i and CONVERT(user_label_nr,UNSIGNED INTEGER)<= %i AND user_label !=  'OXA 2' and project != 'MWK-80-103A' and project != '2022, DISJ-e-10611'" % (
                                    t0, t1)  # and project !='MWK85-05B(2)'
        cursor.execute(dataSelectString)# project !='MWK85-05B(2)'
        df = DataFrame(cursor.fetchall())
        try:
            df.columns = [name[0] for name in cursor.description]
        except:
            keys = ['fm','bp','fm_sig','user_label','target_id','dc13', 'dc13_sig']
            retdf = {}
            for key in keys:
                retdf[key] = array([])
            cursor.close()
            cnx.close()
            return retdf
        #mask = df.loc[:, 'bp'] != '0.0'
        #df = df.loc[mask]
        cursor.close()
        cnx.close()
        retdf = {}
        for key in df.keys():
            retdf[key] = array(df[key])
        retdf['target_id'] = [str(i)+'.'+str(j)+'.'+str(k) for i,j,k in zip(df['sample_nr'],df['prep_nr'],df['target_nr'])]
        retdf['target_id'] = array(retdf['target_id'])
    else:
        import pandas as pd
        import os
        file_path = os.path.abspath(os.path.dirname(__file__))
        dfeth = pd.read_excel(file_path+r'\Data\DBsave\ETHALL2024-09-02.xlsx')
        retdf = {}
        for key in dfeth.keys():
            retdf[key]=np.array(dfeth[key])
        indexes = np.where((retdf['bp'] >= t0) & (retdf['bp'] <= t1))
        for key in retdf.keys():
            retdf[key] = retdf[key][indexes]
    if maxdate is not None:
        target_date = np.datetime64('-'.join(reversed(maxdate.split('.'))))
        indexes = np.where(retdf['target_pressed'] < target_date)[0]
        for key in retdf.keys():
            retdf[key] = retdf[key][indexes]
    return retdf

def getStoppedData(t0, t1, dbdata=True, bp=True,maxdate=None):
    if bp:
        t0_ = t0
        t1_ = t1
    if bp == False:
        t0_ = 1950-t0
        t1_ = 1950-t1
    t0 = min(t0_, t1_)
    t1 = max(t0_, t1_)
    cnx = DB.getConnection()
    cursor = cnx.cursor()
    dataSelectString = "SELECT sample_nr as sample_nr,target_nr as target_nr,prep_nr, project,project_nr ,magazine,user_label,last_name, target_pressed, cast(substring_index(user_label_nr,'|',1) as decimal(6,1)) as bp,  user_label_nr ,c14_age,c14_age_sig, fm as fm,fm_sig as fm_sig" \
                       + " , dc13, dc13_sig FROM db_ams.target_v WHERE research='calibration' and stop=1 and c14_age is not null and sample_nr != 107999 " \
                       + " and CONVERT(user_label_nr,UNSIGNED INTEGER) >= %i and CONVERT(user_label_nr,UNSIGNED INTEGER)<= %i AND user_label !=  'OXA 2'" % (
                           t0, t1)
    cursor.execute(dataSelectString)
    df = DataFrame(cursor.fetchall())
    try:
        df.columns = [name[0] for name in cursor.description]
    except:
        cursor.close()
        return None
    cursor.close()
    cnx.close()
    retdf = {}
    for key in df.keys():
        retdf[key] = array(df[key])
    retdf['target_id'] = [str(i)+'.'+str(j)+'.'+str(k) for i,j,k in zip(df['sample_nr'],df['prep_nr'],df['target_nr'])]
    retdf['target_id'] = array(retdf['target_id'])
    if maxdate is not None:
        target_date = np.datetime64('-'.join(reversed(maxdate.split('.'))))
        indexes = np.where(retdf['target_pressed'] < target_date)[0]
        for key in retdf.keys():
            retdf[key] = retdf[key][indexes]
    return retdf



DB = DBconnect()
def magazineData(magazines):
    cnx = DB.getConnection()
    cursor = cnx.cursor()
    dataSelectString = ("SELECT sample_nr as sample_nr,target_nr as target_nr, "
                        "project,user_label, magazine, cast(substring_index(user_label_nr,'|',1) "
                        "as decimal(6,1)) as bp,  user_label_nr, fm as fm,fm_sig as fm_sig , c14_age as c14_age, "
                        "c14_age_sig as c14_age_sig, dc13 FROM db_ams.target_v WHERE type = 'sample'"
                        " and magazine in (")
    for i,magazine in enumerate(magazines):
        dataSelectString+="'"+magazine +"'"
        if i<len(magazines)-1:
            dataSelectString+=" , "
    dataSelectString+=");"
    print(dataSelectString)
    cursor.execute(dataSelectString)
    df = DataFrame(cursor.fetchall())
    df.columns = [name[0] for name in cursor.description]
    #mask = df.loc[:, 'bp'] != '0.0'
    #df = df.loc[mask]
    cursor.close()
    cnx.close()
    retdf = {}
    for key in df.keys():
        retdf[key] = array(df[key])
    return retdf

def getProjectData(projectnr,db=True,save=False,stopped=True):
    if db:
        cnx = DB.getConnection()
        cursor = cnx.cursor()
        if stopped == True:
            query = f"SELECT * FROM db_ams.target_v where db_ams.target_v.project_nr = {projectnr};"
        else:
            query = f"SELECT * FROM db_ams.target_v where db_ams.target_v.project_nr = {projectnr} and stop = 0 and fm is not null;"
        cursor.execute(query)  # project !='MWK85-05B(2)'
        df = DataFrame(cursor.fetchall())
        df.columns = [name[0] for name in cursor.description]
        retdf = {}
        for key in df.keys():
            retdf[key] = array(df[key])
        if save:
            import pandas as pd
            import os
            file_path = os.path.abspath(os.path.dirname(__file__))
            file = file_path+ r'\Data\DBsave\ETH'+f'{projectnr}.xlsx'
            exceldata = pd.DataFrame(retdf)
            exceldata = exceldata.where(df.notnull(), None)
            import xlsxwriter
            with xlsxwriter.Workbook(file) as workbook:
                worksheet = workbook.add_worksheet('Sheet')
                for i, key in enumerate(exceldata.keys()):
                    worksheet.write(0, i, key)
                    for j, val in enumerate(exceldata[key]):
                        try:
                            worksheet.write(j + 1, i, val)
                        except:
                            pass
    else:
        import pandas as pd
        import os
        file_path = os.path.abspath(os.path.dirname(__file__))
        dfeth = pd.read_excel(projectPath + r'\Data\DBsave\ETH'+f'{projectnr}.xlsx')
        retdf = {}
        for key in dfeth.keys():
            retdf[key] = np.array(dfeth[key])
    return retdf


def getMultiProjectData(project_nrs, bpuserlabelnr=True, magazines=None):
    testdatadf = datadict()
    for i,project_nr in enumerate(project_nrs):
        data = getProjectData(project_nr,db=True,save=False,stopped=False)
        data['index'] = np.full(len(data['target_id']),i)
        testdatadf += data
    nonaninds = ~np.isnan(testdatadf['fm_sig'])
    for key in testdatadf:
        testdatadf[key] = testdatadf[key][nonaninds]
    #if maxdate is not None:
    #    target_date = np.datetime64('-'.join(reversed(maxdate.split('.'))))
    #    indexes = np.where(testdatadf['target_pressed'] > target_date)[0]
    #    for key in testdatadf.keys():
    #        testdatadf[key] = testdatadf[key][indexes]
    try:
        testdatadf['user_label_nr'] = testdatadf['user_label_nr'].astype(float)
    except:
        pass
    testdatadf['bp'] = testdatadf['user_label_nr']
    if bpuserlabelnr:
        testdatadf['year'] = 1950-testdatadf['bp']
    elif bpuserlabelnr == False:
        testdatadf['year'] = testdatadf['bp']
    elif bpuserlabelnr is  None:
        testdatadf['user_label_nr'] = np.array([int(s.split('|')[1]) for s in testdatadf['user_label']])
        testdatadf['year'] = testdatadf['user_label_nr']
    testdatadf['bp'] = 1950-testdatadf['year']
    if 8237 in project_nrs or 8216 in project_nrs:
        testdatadf['year'] = testdatadf['year'] +1
        testdatadf['bp'] = testdatadf['bp']  - 1
    if magazines is not None:
        magazines = np.array(magazines)
        idx = np.where(np.isin(testdatadf['magazine'], magazines))[0]
        for key in testdatadf:
            testdatadf[key] = testdatadf[key][idx]
    return testdatadf


def getProjectData3(projectnr):
    cnx = DB.getConnection()
    cursor = cnx.cursor()
    query = "SELECT sample_nr as sample_nr,target_nr as target_nr,prep_nr, project ,magazine,user_label,last_name,  cast(substring_index(user_label_nr,'|',1) as decimal(6,1)) as bp,  user_label_nr ,c14_age,c14_age_sig, fm as fm,fm_sig as fm_sig" \
                       + " , dc13 FROM db_ams.target_v where db_ams.target_v.project_nr = %i;" % projectnr
    cursor.execute(query)  # project !='MWK85-05B(2)'
    df = DataFrame(cursor.fetchall())
    df.columns = [name[0] for name in cursor.description]
    retdf = {}
    for key in df.keys():
        retdf[key] = array(df[key])
    return retdf

def getTargetRawData(target_id):
    (sample_nr, prep_nr, target_nr) = target_id.split('.')
    sample_nr = int(sample_nr)
    prep_nr = int(prep_nr)
    target_nr = int(target_nr)
    cnx = DB.getConnection()
    cursor = cnx.cursor()
    query = "SELECT * FROM db_mc14.workfinal where SAMPLE_NR = %i and prep_nr = %i and target_nr = %i;" % (sample_nr, prep_nr, target_nr)
    cursor.execute(query)  # project !='MWK85-05B(2)'
    df = DataFrame(cursor.fetchall())
    df.columns = [name[0] for name in cursor.description]
    retdf = {}
    for key in df.keys():
        retdf[key] = array(df[key])
    return retdf

def getCorrectionData(target_id):
    (sample_nr, prep_nr, target_nr) = target_id.split('.')
    sample_nr = int(sample_nr)
    prep_nr = int(prep_nr)
    target_nr = int(target_nr)
    cnx = DB.getConnection()
    cursor = cnx.cursor()
    query = "SELECT db_ams.calc_corr_t.*,db_ams.calc_set_t.* FROM (db_ams.calc_corr_t join db_ams.calc_sample_t join db_ams.calc_set_t on db_ams.calc_corr_t.calcset = db_ams.calc_sample_t.calcset and db_ams.calc_corr_t.calcset = db_ams.calc_set_t.calcset) WHERE db_ams.calc_sample_t.sample_nr = %i and db_ams.calc_sample_t.prep_nr = %i and db_ams.calc_sample_t.target_nr = %i;" % (sample_nr, prep_nr, target_nr)
    cursor.execute(query)  # project !='MWK85-05B(2)'
    df = DataFrame(cursor.fetchall())
    df.columns = [name[0] for name in cursor.description]
    retdf = {}
    for key in df.keys():
        retdf[key] = array(df[key])
    return retdf


def getProjectData2(projectnr):
    query = "SELECT \n        CONCAT(db_ams.target_t.sample_nr,\n                _UTF8'.',\n                db_ams.target_t.prep_nr,\n                _UTF8'.',\n                db_ams.target_t.target_nr) AS target_id,\n        db_ams.target_t.*,\n        db_mc14.workproto.TIMEDAT AS timedat,\n        db_ams.preparation_t.step1_method AS step1_method,\n        db_ams.preparation_t.step2_method AS step2_method,\n        db_ams.preparation_t.batch AS batch,\n        db_ams.sample_t.type AS type,\n        db_ams.sample_t.material AS material,\n        db_ams.sample_t.fraction AS fraction,\n        db_ams.sample_t.sampling_date AS sampling_date,\n        db_ams.sample_t.not_tobedated AS not_tobedated,\n        db_ams.sample_t.user_label AS user_label,\n        db_ams.sample_t.user_label_nr AS user_label_nr,\n        db_ams.sample_t.user_desc1 AS user_desc1,\n        db_ams.sample_t.user_desc2 AS user_desc2,\n        db_ams.project_t.project_nr AS project_nr,\n        db_ams.project_t.project AS project,\n        db_ams.project_t.priority AS priority,\n        db_ams.project_t.status AS status,\n        db_ams.project_t.advisor AS advisor,\n        db_ams.project_t.research AS research,\n        db_ams.user_t.user_nr AS user_nr,\n        db_ams.user_t.first_name AS first_name,\n        db_ams.user_t.last_name AS last_name\n    FROM\n        (((((db_ams.sample_t\n        JOIN db_ams.preparation_t)\n        JOIN db_ams.target_t)\n        JOIN db_mc14.workproto)\n        JOIN db_ams.project_t)\n        JOIN db_ams.user_t)\n    WHERE\n         db_ams.sample_t.sample_nr = db_ams.preparation_t.sample_nr\n            AND db_ams.sample_t.sample_nr = db_ams.target_t.sample_nr\n            AND db_ams.preparation_t.prep_nr = db_ams.target_t.prep_nr\n            AND db_ams.sample_t.project_nr = db_ams.project_t.project_nr\n            AND db_ams.project_t.user_nr = db_ams.user_t.user_nr\n            AND db_mc14.workproto.SAMPLE_NR = db_ams.target_t.sample_nr\n            AND db_mc14.workproto.TARGET_NR = db_ams.target_t.target_nr\n            AND db_mc14.workproto.PREP_NR = db_ams.target_t.prep_nr\n            AND db_ams.project_t.project_nr = %i \n\tGROUP BY CONCAT(db_ams.target_t.sample_nr,\n            _UTF8'.',\n            db_ams.target_t.prep_nr,\n            _UTF8'.',\n            db_ams.target_t.target_nr)" % projectnr
    cnx = DB.getConnection()
    cursor = cnx.cursor()
    cursor.execute(query)  # project !='MWK85-05B(2)'
    df = DataFrame(cursor.fetchall())
    df.columns = [name[0] for name in cursor.description]
    retdf = {}
    for key in df.keys():
        retdf[key] = array(df[key])
    return retdf

def appendData(df1,df2):
    df = {}
    for key in df1.keys():
        df[key] = np.append(df1[key],df2[key])
    return df


def getMillenniumData():
    minimum = 0
    maximum = 1000
    cnx = DB.getConnection()
    cursor = cnx.cursor()
    dataSelectString = "SELECT sample_nr as sample_nr,target_nr as target_nr,prep_nr, project ,magazine,user_label,last_name,  cast(substring_index(user_label_nr,'|',1) as decimal(6,1)) as bp,  user_label_nr ,c14_age,c14_age_sig, fm as fm,fm_sig as fm_sig" \
                       + " , dc13 FROM db_ams.target_v WHERE research='calibration' and stop=0 and c14_age is not null and sample_nr != 107999 " \
                       + " and CONVERT(user_label_nr,UNSIGNED INTEGER) >= %i and CONVERT(user_label_nr,UNSIGNED INTEGER)<= %i AND user_label !=  'OXA 2' and project != 'MWK-80-103A' and project != '2022, DISJ-e-10611'" % (
                       minimum, maximum) #and project !='MWK85-05B(2)'
    cursor.execute(dataSelectString)# project !='MWK85-05B(2)'

    df = DataFrame(cursor.fetchall())
    df.columns = [name[0] for name in cursor.description]
    #mask = df.loc[:, 'bp'] != '0.0'
    #df = df.loc[mask]
    cursor.close()
    cnx.close()
    retdf = {}
    for key in df.keys():
        retdf[key] = array(df[key])
    retdf['target_id'] = [str(i)+'.'+str(j)+'.'+str(k) for i,j,k in zip(df['sample_nr'],df['prep_nr'],df['target_nr'])]
    retdf['target_id'] = array(retdf['target_id'])

    projects = ['2013, ZH-644415','2016, ZH-600799','2017, HFD-C01','2017, COD-C38','2017, DRN-D11','2017, ALT-B05','2017, BLS-B10',
                '2018, LYB-H40','2018, SYD-M22','2018, ALT-B04','2017, TAT-C13','2018, STK-C10','2018, STA-C34']
    indexes = []
    for project in projects:
        inds = np.where(retdf['project']==project)[0]
        #print(indexes)
        for ind in inds:
            indexes.append(ind)
    indexes = array(indexes)
    for key in df.keys():
        retdf[key] = array(df[key][indexes])
    return retdf

def get_mazama_data(offset=0,offset_sig=0,project = 8639):
    mazamadata = getProjectData(project)
    nostopinds = np.where(mazamadata['stop'] == 0)[0]
    for key in mazamadata:
        mazamadata[key] = mazamadata[key][nostopinds]
    naninds = ~np.isnan(mazamadata['fm_sig'])
    for key in mazamadata.keys():
        mazamadata[key] = mazamadata[key][naninds]
    mazyears = np.array([1950 - int(s.split('|')[1]) for s in mazamadata['user_label']])
    mazamadata['bp'] = mazyears
    if offset !=0:
        mazamadata['age'] = -8033*np.log(mazamadata['fm'])
        mazamadata['age_sig'] = 8033/mazamadata['fm']*mazamadata['fm_sig']
        mazamadata['age'] = mazamadata['age']+offset
        mazamadata['age_sig'] = (mazamadata['age_sig']**2+offset_sig**2)**0.5
        mazamadata['fm'] = np.exp(-mazamadata['age']/8033)
        mazamadata['fm_sig'] = mazamadata['age_sig']/8033*mazamadata['fm']
    return mazamadata






