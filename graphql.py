import requests
import json
import pandas as pd


def listingsearch(keywords,near=""):
    gqlvar = locals()
    
    # search for boundary to define relevant listings if location info provided
    if near != "":
        locquery = """
        query SearchViewportQuery(
            $keywords: String            
            $near: String            
        ) {
            search(keywords: $keywords)                
                {                
                viewport(near: $near) {
                    bounds
                    coordinates
                }
                
            }
        }
    """

    locr = requests.post("https://waybase.com/graphql", 
        json= {'query': locquery, 'variables':gqlvar})    
    
    gqlvar["viewport"]=(json.loads(locr.text)["data"]["search"]["viewport"]["bounds"])
        
    
    query = """
        query SearchQuery(
            $keywords: String
            $viewport: BoundsInput!                        
            ){
            search(
                keywords:$keywords
                types:[listing]                
            ){                
                results(
                    #first: 10
                    #sort: {
                    #    "field": "score",
                    #    "direction": desc"
                    #}
                    viewport: $viewport
                ) {
                    edges {
                        node {
                            id
                            ... on Listing {
                                name                                                        
                                location {
                                    coordinates
                                }
                                primaryLink
                                email
                                tags {
                                    category
                                    denomination
                                    type
                                }
                            }
                        }
                    }
                }
            }
        }
    """    
    print(gqlvar)

    listr = requests.post("https://waybase.com/graphql", 
        json= {'query': query, 'variables':gqlvar})
    
    r2 = pd.json_normalize(pd.DataFrame(
                json.loads(listr.text)["data"]["search"]["results"]["edges"]
                )["node"])
        

    
    return r2

def locationsearch(near=""):
    
    query = """
        query SearchViewportQuery(            
            $near: String            
        ) {
            search(keywords: $keywords) {                
                viewport(near: $near) {
                    bounds
                    coordinates
                }
                
            }
        }

    """

    
    
    
    r = requests.post("https://waybase.com/graphql", 
        json= {'query': query, 'variables':{'near': near }})    
    r2 = json.loads(r.text)#["viewport"]["bounds"]
        

    return r2
    return dict(r2)

