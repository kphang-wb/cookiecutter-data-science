# -*- coding: utf-8 -*-
"""
******************************************************************************
*
*
* 
* Program: WBMatch
*
*
*
* Definition: The WBMatch function will allow the user to enter in as little
* as a single string to search for a matching WayBase ID. The function
* currently supports name, postal code, bounding boxes, and generic location
* strings all searchable without using Google by utilizing a combination of 
* geopy and pgeocode. The function will only output results if there is a
* "confident" match i.e., the data top matching result is clearly a far better
* match than any other response hits.
*
#   Parameters:
#       client: the Elasticsearch client with login information must be passed
#               to the function in order to retrieve IDs. REQUIRED
#
#       namestring: a string variable with the charity name, REQUIRED.
#
#       postcode: a string containing the postal code of the charity. Must be
#               formatted approximately correct (missing space or changed case
#               is fine, but can't contain incorrect characters). If improperly
#               formatted the function will fall back to the name only. Useful
#               if only for the fact that it is done entirely offline. Can be
#               included with a boundaries parameter to even more closely limit
#               the results.
#
#       boundaries: Can be either a list of top_left and bottom_right coordinates
#                   for a bounding box, or a string containing an address/city/
#                   region etc. Searches only within Canada. This slows the
#                   function from 5it/sec to 1it/sec maximum as the OSM project
#                   is rate limited (but FREE!). Generic query is then converted
#                   to a tight bounding box by Nominatim/OSM.
#        
#       epsilon: The density/distance parameter for the clustering algorithm.
#                Default is 4. Increase to make more conservative. Should never
#                be reduced below ~1.5. Lowering increases false positives.
#                Raising increases false negatives (eventually will never 
#                return a match at all).
#
# DiagnosticDictionary: Below shows how the dictionary is laid out (the keys  #
# you can interact with to add columns). Will return BOOLEAN T/F values for   #
# exact matches between the WB hit value and the inputted data. Note that     #
# using the DD wil also trigger an additional output of Levenstein distance   #
# between the inputted namestring and it's best matching WB Alias. The        #
# example below lists the exact field name returned from the ElasticSearch    #
# query. Note that your input values may have to be reformatted to match the  #
# standard WB value.                                                          #
#
#   DiagnosticDictionary = {
#    'DenomBool': 'tags.denomination',
#    'PostBool' : 'postalCode',
#    'FaithBool' : 'tags.faith',
#    'AgeBool' : 'tags.age',
#    'CategoryBool': 'tags.category',
#    'CultureBool' : 'tags.culture',
#    'FaithstreamBool' : 'tags.faithstream',
#    'LanguageBool' : 'tags.language'
#    }
#
#       Funny note: If a postal code is provided, often the name and postal code
#               alone is the most accurate and fastest method. For files like
#               "Liens entre les églises et les paroisses", the address boundary
#               inclusion only aided in adding 1 more hit. The postal code
#               boundaries are pretty darn accurate. Use name + location (PC)
#               whenever possible. This is 5-15x faster, with maybe a 2% change
#               in accuracy. In some cases the boundary can actually return
#               fewer hits since it will have to increase the boundaries too
#               far before submitting the ES_Query, making the confidence in
#               separating scores lower.   
#
#       Another Note: As per Rhodri's information, the elasticsearch query
#       will choke on namestrings that have the location in the name, for
#       example "Pinegrove Fellowship Church - Port Carling". If at all possible
#       strip location out of the name itself. It can still say Pinegrove if
#       that is indeed the churches name, it is the addition of extraneous info
#       into the name field that the ES Query really struggles with. 
*
* Outputs: ID, Confidence, WB_Name, WB_AKA, WB_Locality, PC, DENOM, LEV, CC
* as a list. I prefer to then apply pd.Series to pull that DF column into 
* multiple.
*
*
*******************************************************************************
#---------------------------------Change log----------------------------------#
*
* Created on Tue Mar 31 12:15:13 2020
* Finalized on Thu Apr 16 2020
* Between April 16 and 22 there were numerous undocumented bug fixes based on
* real-data usage experience.
* 20200422 Bug fix: added a check permission step to get_boundaries to prevent 
* returning 'None' to the query.
* 20200423 Improvement: Postal code now removes the leading whitespace before 
* checking centre whitespace.
* 20200423 Improvement: Boundary will now at least once try to back out and get
* a more generic bounding box if the street address was too close. Doubles the
* algorithm runtime (if there's to small a box), but may prove more accurate.
* 20200423 General Bugfixes. Ensured that when a sole hit is returned from a 
* postal code, it is still compared against other hits for confidence values.
* Some will still return sole match since the initial query can match so well
* from elastic search! These are tagged 'SOle Return Before Clustering'
* 20200423 Improvement: Added the Diagnostic Dictionary, a
* way to output more information to check veracity of matches.
* 20200427 Improvement: Added a combined confidence score to make Nathan's life
* easier. If there is a normal confidence score and a lev, averages them
* together. If there is only lev, reports lev.
* 20200428 Improvement/Bugfix: After having a few ES_Query failures from
* TrasportErrors, added a "retry" to WB_Match to attempt the query a few more
* times before dumping out. This fixed some frustrating breaks where it had 
* read 3/4 through a large file. Also recommend increasing the client timeout
* setting to 30 seconds at time of creation if you have my crappy internet.
* 20200429 Cleanup: Output changed to list instead of pd.Series for greater
* flexability. Removed extraneous options from WB_Match since normal use-case
* will likely need as much information outputted as possible (old toggles let
* return as little as just ID or ID and CONF values).
*
* @author: Stephen J.C. Luehr
*
******************************************************************************
"""

#-----------------------------------IMPORTS-----------------------------------#
import pgeocode
import pandas as pd
from pandas import json_normalize
from sklearn.cluster import DBSCAN
from scipy import stats
from tqdm import tqdm
import time
from geopy import Nominatim #Can be changed to GoogleV3
import re
from fuzzywuzzy import fuzz #For diagnostic check
import elasticsearch

from web_search_template import web_search

#--------------------------INITIALIZATION PARAMETERS--------------------------#
#Set the OSM user_agent to any non-default profile.
locator = Nominatim(user_agent = "WB_Find_Bounds")

'''
# Alternatively, can be changed to Google maps service using the format:
# geopy.geocoders.GoogleV3(api_key=None, domain='maps.googleapis.com',        #
# scheme=None, client_id=None, secret_key=None, timeout=DEFAULT_SENTINEL,     #
# proxies=DEFAULT_SENTINEL, user_agent=None, format_string=None,              #
# ssl_context=DEFAULT_SENTINEL, channel='')                                   #
'''


#----------------------------------CONSTANTS----------------------------------#
# Postal code validator
# Set the standard for the Postal Codes to be compared against for Canada.
zipCode = re.compile(
    r"^(?!.*[DFIOQU])[A-VXY][0-9][A-Z]\s[0-9][A-Z][0-9]$"
)

#Open the mustache template for the ElasticSearch.
#Worth noting that small modifications had to be made to the formatting to 
#Accomodate the python elasticsearch limitations. For example had to add
#polygon into query directly since source is cumbersome to modify through API.
# Now opened as an import, no longer declared in script.

#Activate tqdm progress bar for pandas.
tqdm.pandas() #Currently throws a FutureWarning. Creator has said they will fix
# this shortly. When pandas does update to break this, docs say it will silently
# break and maintain functionality.


#----------------------------------FUNCTIONS----------------------------------#
#=============================================================================#
#   Function: ES_Query
#
#   Parameters:
#       client: the Elasticsearch client with login information must be passed
#               to the function in order to retrieve IDs. REQUIRED.
#
#       namestring: a string variable with the charity name, REQUIRED.
#
#       location: a string containing the postal code of the charity. Must be
#               formatted approximately correct (missing space or changed case
#               is fine, but can't contain incorrect characters). If improperly
#               formatted the function will fall back to the name only. Useful
#               if only for the fact that it is done entirely offline. Can be
#               included with a boundaries parameter to even more closely limit
#               the results.
#
#       boundaries: Can be either a list of top_left and bottom_right coordinates
#                   for a bounding box, or a string containing an address/city/
#                   region etc. Searches only within Canada. This slows the
#                   function from 5it/sec to 1it/sec maximum as the OSM project
#                   is rate limited (but FREE!). Generic query is then converted
#                   to a tight bounding box by Nominatim/OSM. String is
#                   processed left-to-right then right-to-left if that fails,
#                   so address may be input in any order i.e., "street, city,
#                   province" OR "province, city, street".
#                   Can also input a dictionary with keys are one of: 
#                   street, city, county, state, country, or postalcode. This
#                   structured query has less chance of being misinterpreted.
#                   Polygon type boundaries should be entered in polygon param
#                   since not all queries are capable of returning a polygon in
#                   OSM/Google. 
#
#         polygon: an array of coordinates that form a polygon. In the same   #
# format as the boundaries for the bounding box, only many more entries than  #
# the two corners! Uses 'should' boolean filter as in original function       #
#           Currently bugged: 'parse exception', geo_point expected. Seems to
#           be index related. Not working as I'd hoped about 50% of the time.
#
#
#       Funny note: If a postal code is provided, often the name and postal code
#               alone is the most accurate and fastest method. For files like
#               "Liens entre les églises et les paroisses", the address boundary
#               inclusion only aided in adding 1 more hit. The postal code
#               boundaries are pretty darn accurate. Use name + location (PC)
#               whenever possible. This is 5-15x faster, with maybe a 2% change
#               in accuracy.
#
#=============================================================================# 
def ES_Query(client, namestring=None, postcode=None, boundaries=None, polygon=None):

    if namestring == None:
        return "No name supplied"
    
    postcode = normalize_postalcode(postcode)  # Normalize the postal code here
    postcode = get_geocode(postcode)

    if polygon != None: #Takes priority over all other search boundary params.
        response = client.search_template(
            body={
                "inline": web_search,
                "params": {
                    "keywords": namestring,
                    "polygon" : polygon
                },
            },
            index="search_profiles",
        )
        
    elif boundaries != None:
        #At a rate limited 1/sec. OSM returns likely bounding box for query.
        if isinstance(boundaries, str) or isinstance(boundaries, dict):
            # If boundary is a generic string, query locator service for tight
            # bounding box coordinates.
            boundaries = get_bounds(boundaries)
        center = [
            (boundaries[0][0] + boundaries[1][0]) / 2,
            (boundaries[0][1] + boundaries[1][1]) / 2,
        ]
        response = client.search_template(
            body={
                "inline": web_search,
                "params": {
                    "keywords": namestring,
                    "lowerbounds": boundaries[0],
                    "upperbounds": boundaries[1],
                    "center": [center],
                },
            },
            index="search_profiles",
        )
        
    elif postcode != None:
        response = client.search_template(
            body={
                "inline": web_search,
                "params": {"keywords": namestring, "center": [postcode]},
            },
            index="search_profiles",
        )
        
    else:
        response = client.search_template(
            body={"inline": web_search, "params": {"keywords": namestring}},
            index="search_profiles",
        )
                
    results = response["hits"]["hits"]
    
    if len(results) == 0:
        return "No match found"
    
    results = pd.DataFrame(results)
    results = pd.concat(
        [results, json_normalize(results["_source"])], axis=1
    ).set_index("_id")

    # Clean output to remove the "source" columns that were normalized
    results = results.drop(["_index", "_type", "_source", "matched_queries"], axis=1)

    # Fix names so I can rewrite this, instead of the entire match function lol.
    results = results.rename(columns={"_score": "score"})
    results.index.names = ["id"]

    # Adding in a "alsoKnownAs" column if it doesn't exist. For some reason, some
    # Entries in ElasticSearch just don't return the field, rather than returning
    # it blank. Caused many headaches.

    if "alsoKnownAs" not in results.columns:
        results["alsoKnownAs"] = [""]

    return results

#=============================================================================#
#    Function: get_bounds
#
#         Definition: Takes any address string and queries whatever the       #
# current search provider is (default is OSM) for a bounding envelope.        #
# Defaults to a tight bounding box as in Rhodri's search algorithm, returning #
# the top_left and bottom_right coordinate values. Only searches in Canada.   #
#   
#   Parameters:
#
#       searchstring: Can be either a list of top_left and bottom_right 
#                   coordinates for a bounding box, or a string containing an 
#                   address/city/region etc. Searches only within Canada. 
#                   Generic query is then converted to a tight bounding box by 
#                   Nominatim/OSM or Google. Bounding envelope is VERY tight so
#                   be careful with very specific addresses as it is capable
#                   of bounding a single house or road.
#
#         RateLimiter: Default = 1. The number of seconds to wait between     #
# search queries. A minimum of 1 second is recommended to utilize the free    #
# Nominatim API to be polite to their servers (and not get kicked). They also #
# request caching results however we rarely duplicate the same search.        #
#
#=============================================================================# 
def get_bounds(searchstring, RateLimiter = 1):
    #Pull results from OSM Foundation/Google
    #Buffer step to dump if bounding box isn't found.
    try:
        location = locator.geocode(searchstring, country_codes = 'ca').raw['boundingbox']
    except AttributeError:
        #Try one more time with first portion of address removed for more broad
        #results. Simply check for comma, and pull after the first one.
        searchstring = searchstring[searchstring.index(",")+1:]
        searchstring = searchstring.lstrip(' ')
        try:
            location = locator.geocode(searchstring, country_codes = 'ca').raw['boundingbox']
        except AttributeError:
            location = None
    
    if location == None: #Sets search boundary to Canada at least.
        boundaries = [[None, None],[None, None]]
        boundaries[0][0] = -166.73
        boundaries[0][1] = 76.27
        boundaries[1][0] = -28.74
        boundaries[1][1] = 41.18
        return boundaries
    #Reformat to match order that remaining function expects (geopandas standard)
    #Boundaries[0] = top left, long, lat
    #Boundaries[1] = bottom right, long, lat
    boundaries = [[None, None],[None, None]]
    boundaries[0][0] = float(location[2])
    boundaries[0][1] = float(location[1])
    boundaries[1][0] = float(location[3])
    boundaries[1][1] = float(location[0])
    #What the prescribed RateLimiter time to submit the next search.
    time.sleep(RateLimiter)
    return boundaries

def normalize_postalcode(PC):
    # If not a string, throw it back.
    if PC == None or type(PC) == float:
        return
    # Remove leading space
    PC = PC.lstrip(' ')
    
    # Add the missing space.
    if len(PC) == 6:
        PC = PC[:3] + " " + PC[3:]
    
    # Capitalize since OSM requires it.
    PC = PC.upper()
    return PC

#=============================================================================#
#    Function: get_confidence
#
#         Definition: Returns a confidence score for all resulting hits in a  #
# dataframe. Calculates the number of standard deviations from the mean score #
# a result is. Returns the percentage of 3.5SD the result is i.e., if the     #
# score is 3.5 standard deviations from the mean score, the confidence will   #
# be 100%. Useful for quick checks on uncertain results. If there is only one #
# match, returns 'Sole Match' so it doesn't appear to be a zero confidence    #
# hit.                                                                        #
#   
#   Parameters:
#
#         points: a pandas dataframe with a column labelled 'score' at any    #
#               position. A 'z' column will be added with confidence values.  #
#
#         thresh: default 3.5. The number of SD's from the mean to be         #
# considered 100% confidence.                                                 #
#
#=============================================================================# 
def get_confidence(points, thresh=3.5):
    if len(points) == 1:
        points["z"] = "Sole Return Before Clustering"
        return points
    points["z"] = abs(stats.zscore(points.score)) / thresh * 100
    return points

#=============================================================================#
#    Function: get_cluster
#
#         Definition: Utilizes DBSCAN, a density based clustering algorithm   #
# to cluster the scores of search query hits. Clusters only need a single     #
# point to be clustered (so max score can be by itself, this is good).        #
# Returns the original dataframe with new column 'clusters' concatenated on.  #
#   
#   Parameters:
#
#         results: a pandas dataframe with a 'scores' column in position 0.
#                  Will be ingested, concatenated and returned.               #
#
#       epsilon: The density/distance parameter for the clustering algorithm.
#                Default is 4. Increase to make more conservative. Should never
#                be reduced below ~1.5. Lowering increases false positives.
#                Raising increases false negatives (eventually will never 
#                return a match at all).
#
#=============================================================================#   
def get_cluster(results, epsilon = 4):
    # Run clustering algorithm
    # Reshape the data for DBScan
    dfs = results.to_numpy()
    dfs = dfs[:, 0] #Assuming score is always the first column!
    dfs = dfs.reshape(-1, 1)
    outlier_detection = DBSCAN(
        min_samples=1, eps=epsilon
    )
    # Retrieve the cluster group tag
    clusters = outlier_detection.fit_predict(dfs)
    # Nicely format output and reattach results to input index.
    clusters = pd.DataFrame({"clusters": clusters}, index=results.index)
    results = pd.concat([results, clusters], axis=1)
    return results

#=============================================================================#
#    Function: get_geocode
#
#    Definition: Utilizes the offline pgeocode package to reverse lookup      #
# a postal code's center point, as provided by the Government of Canada. The  #
# entry must be formatted approximately correct, however it will be converted #
# to uppercase and a space may be added. If there is no location, or a severe #
# error, will return NoneType.                                                #
#   
#   Parameters:
#
#         location: a string containing the postal code of the charity. Must  #
# be formatted approximately correct (missing space or changed case is fine,  #
# but can't contain incorrect characters). If improperly formatted the        #
# WB_Match function will fall back to the name only. Useful if only for the   #
# fact that it is done entirely offline. Can be included with a boundaries    #
# parameter to even more closely limit the results.                           #
#
#=============================================================================# 
def get_geocode(location):
    # Immediate dump if None.
    if location == None or type(location) == float: 
        return None
    # Set search database to use the Nominatim offline database in pgeocode.
    # "ca" indicates canada only. Significantly speeds up search and quality.
    nomi = pgeocode.Nominatim("ca")
    # Clean input. Some postal codes lack a space, which causes error.
    location = normalize_postalcode(location)
    if zipCode.match(location): # Check if valid canadian postal code.
        nomi = nomi.query_postal_code(location)
        # Elasticsearch needs longitude as first number.
        locationstring = [None] * 2
        locationstring[0] = nomi["longitude"]
        locationstring[1] = nomi["latitude"]
        return locationstring


# Wrap it all in a function
#=============================================================================#
#   Function: WB_Match
#
# No longer described here. Please view function main definition at top of the
# file for more information.
#
#=============================================================================#        
    
def WB_Match(
    client,
    namestring=None,
    postcode=None,
    boundaries=None,
    polygon=None,
    DiagnosticDictionary = None,
    epsilon=4
):
    #Attempt an ES_Query. If there's an error, retry it a few times. This solved
    #Having very rare transport errors ruining long term scans of files.
    for attempts in range(0,3):
        try:
            results = ES_Query(client, namestring, postcode, boundaries, polygon)
        except elasticsearch.TransportError:
            continue
        else:
            break
    CONF = float("NaN")
    if isinstance(results, pd.DataFrame) == False:
            return

    # Normalize Postal Code
    postcode = normalize_postalcode(postcode)
    
    # Check if only one match in the postal code.
    if postcode != None:
        # Check postal code for singular match. If yes, return that sole match,
        # but note is distance from the remaining listings.
        results["postalCode"] = results.apply(
            lambda row: normalize_postalcode(row["postalCode"]), axis=1
        )

        # Now compare the input, only keep matching postal codes.
        results = results[results["postalCode"] == postcode]

        #Now returning nothing if only 1 postal code match. Use Lev for conf.
        # Find confidence
        if len(results.index) > 1:
            results = get_confidence(results)
        else:
            results['z'] = "Sole Postal Code"

    else:
        results = get_confidence(results) #If there's no postal code check, carry on with only confidence.
    
    if len(results.index) > 1: #If theres more than one hit, perform clustering.
        results = get_cluster(results, epsilon)
        # Retrieve the cluster of the top scored item.
        targetcluster = results[results["score"] == results["score"].max()][
            "clusters"
        ].tolist()
        if len(results[results["clusters"] == targetcluster[0]]) != 1:
            # If this doesn't equal 1, don't output anything.
            return

    # Okay, adding a really dumb check to see if there's still results at this
    # stage of the game.
    if len(results.index) == 0:
        return
    # Subset to only the match.
    results = results[results["score"] == results["score"].max()]

    # Convert index to nicely formatted string
    ID = results.index.tolist()
    CONF = results.z.tolist()
    NAME = results.name.tolist()
    ALSO = results.alsoKnownAs.tolist()
    LOC = results.locality.tolist()
    PC = results.postalCode.tolist()
    DENOM = results['tags.denomination'].tolist()

    ID = str(ID[0])
    CONF = CONF[0]
    NAME = str(NAME[0])
    ALSO = ','.join(map(str, ALSO))
    LOC = str(LOC[0])
    PC = str(PC[0])
    DENOM = str(DENOM[0])
    
    # Leven ratio for name
    # Check both name and AKA, take the higher of the two. Convert to lowercase
    LEV = max(fuzz.partial_ratio(namestring.lower(), str(NAME).lower()), fuzz.partial_ratio(namestring.lower(), str(ALSO).lower()))

    #Get Combined Confidence Score
    #If Confidence is missing, only do LEV. If there is both, take average.
    # Changed from multiplying since LEV is often 100% which means you just get
    # the CONF value anyways. Average might be more informative for scanning.
    if type(CONF) == float:
        CC = (LEV+CONF)/2
    else:
        CC = LEV
    
    OutputList = [ID, CONF, NAME, ALSO, LOC, PC, DENOM, LEV, CC]

    # Diagnostic Fields (optional)
    if DiagnosticDictionary != None:
        
        # ComparisonDictionary
        # Gather the values from the resulting listing for boolean testing.
        ComparisonDictionary = {
        'DenomBool': DENOM,
        'PostBool' : PC,
        'FaithBool' : results['faith'],
        'AgeBool' : results['tags.age'],
        'CategoryBool': results['tags.category'],
        'CultureBool' : results['tags.culture'],
        'FaithstreamBool' : results['tags.faithstream'],
        'LanguageBool' : results['tags.language']
        }
        

        for i in DiagnosticDictionary.keys(): #For each key in dict...
            if DiagnosticDictionary[i] != '':
                # We'll return the boolean to the dictionary itself since it's 
                # reinstantiated with each query.
                DiagnosticDictionary[i] = ComparisonDictionary[i] == DiagnosticDictionary[i]
            OutputList.append(DiagnosticDictionary[i])


    return OutputList