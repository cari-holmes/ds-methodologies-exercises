import pandas as pd
import numpy as np
import summarize
from env import host, user, password

def get_db_url(db_name):
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"

def get_zillow_data():
    query = '''
    select 
    svi.`COUNTY` county,
    p.`taxamount`/p.`taxvaluedollarcnt` tax_rate,
    p.`id`,
    p.`parcelid`,
    p.`airconditioningtypeid`,
    act.`airconditioningdesc`,
	p.`architecturalstyletypeid`,
	ast.`architecturalstyledesc`,
    p.`basementsqft`,
	p.`bathroomcnt`,
    p.`bedroomcnt`,
	p.`buildingclasstypeid`,
	bct.`buildingclassdesc`,
    p.`buildingqualitytypeid`,
    p.`calculatedbathnbr`,
    p.`calculatedfinishedsquarefeet`,
    p.`decktypeid`,
    p.`finishedfloor1squarefeet`,
    p.`finishedsquarefeet12`,
    p.`finishedsquarefeet13`,
    p.`finishedsquarefeet15`,
    p.`finishedsquarefeet50`,
    p.`finishedsquarefeet6`,
    p.`fips`,
    svi.`ST_ABBR` state,
    p.`fireplacecnt`,
    p.`fullbathcnt`,
    p.`garagecarcnt`,
    p.`garagetotalsqft`,
    p.`hashottuborspa`,
    p.`heatingorsystemtypeid`,
    hst.`heatingorsystemdesc`,
    p.`latitude`,
    p.`longitude`,
    p.`lotsizesquarefeet`,
    p.`poolcnt`,
    p.`poolsizesum`,
    p.`pooltypeid10`,
    p.`pooltypeid2`,
    p.`pooltypeid7`,
    p.`propertycountylandusecode`,
    p.`propertylandusetypeid`,
    plut.`propertylandusedesc`,
    p.`propertyzoningdesc`,
    p.`rawcensustractandblock`,
    p.`regionidcity`,
    p.`regionidcounty`,
    p.`regionidneighborhood`,
    p.`regionidzip`,
    p.`roomcnt`,
	p.`storytypeid`,
	st.`storydesc`,
    p.`taxvaluedollarcnt`,
    p.`threequarterbathnbr`,
    p.`unitcnt`,
    p.`yardbuildingsqft17`,
    p.`yardbuildingsqft26`,
    p.`yearbuilt`,
    p.`numberofstories`,
    p.`fireplaceflag`,
    p.`structuretaxvaluedollarcnt`,
    p.`assessmentyear`,
    p.`landtaxvaluedollarcnt`,
    p.`taxamount`,
    p.`taxdelinquencyflag`,
    p.`taxdelinquencyyear`, 
	p.`typeconstructiontypeid`,
	tct.`typeconstructiondesc`,
    p.`censustractandblock`,
    pred.`transactiondate`,
    pred.`logerror`,
    m.`transactions`
    from 
	`properties_2017` p
    inner join `predictions_2017`  pred
    on p.`parcelid` = pred.`parcelid` 
    inner join 
	(select 
		`parcelid`, 
		max(`transactiondate`) `lasttransactiondate`, 
		max(`id`) `maxid`, 
		count(*) `transactions`
	from 
		predictions_2017
	group by 
		`parcelid`
	) m
	on 
	pred.parcelid = m.parcelid
	and pred.transactiondate = m.lasttransactiondate
    left join `propertylandusetype` plut
        on p.`propertylandusetypeid` = plut.`propertylandusetypeid`
            
    left join svi_db.svi2016_us_county svi
        on p.`fips` = svi.`FIPS`
    left join `airconditioningtype` act
        using(`airconditioningtypeid`)
    left join heatingorsystemtype hst
        using(`heatingorsystemtypeid`)
    left join `architecturalstyletype` ast
        using(`architecturalstyletypeid`)
    left join `buildingclasstype` bct
        using(`buildingclasstypeid`)
    left join `storytype` st
        using(`storytypeid`)
    left join `typeconstructiontype` tct
        using(`typeconstructiontypeid`)
    where 
        p.`latitude` is not null
        and p.`longitude` is not null
        and p.bedroomcnt > 0 and p.bathroomcnt > 0
        and plut.propertylandusetypeid = '261';
    '''
    return pd.read_sql(query, get_db_url('zillow'))



