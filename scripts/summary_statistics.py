
import urllib2 as urllib
import json
import pprint


query = {
  "aggs": {
    "group_by_state": {
      "terms": {
        "field": "state"
      }
    }
  }
}

query = json.dumps(query)
response = urllib.urlopen(
	'http://54.212.36.132:9200/state_bills/bill_document/_search?')

result = json.loads( response.read() )

pprint.pprint(result)



from elasticsearch import Elasticsearch
es = Elasticsearch(['54.212.36.132:9200', '54.212.36.132:9200'], timeout=300)

search_ga = es.search(index="state_bills", body={"query": {"match": {'state':'ga'}}})
count_ga = es.count(index="state_bills", body={"query": {"match": {'state':'ga'}}})

#Queries, by state, by chamber, by chamber and state
group_by_state = es.search(index="state_bills", body= {
  "size": 0,
  "aggs": {
    "group_by_state": {
      "terms": {
        "field": "state"
      }
    }
  }
}	)

group_by_chamber = es.search(index="state_bills", body={"size":0,"aggs":{"group_by_state":{"terms":{"field":"state"}}},
"aggs":{"group_by_chamber":{"terms":{"field":"chamber"}}}})

group_by_chamber_and_state = es.search(index="state_bills", body=
{"size":0,"aggs":{"group_by_state":{"terms":{"field":"state"},
"aggs":{"group_by_chamber":{"terms":{"field":"chamber"}}}}}})

#Significant terms are uncommonly common words: terms that appear with a 
#frequency that is statistically anomalous compared to the background data.
significant_terms_bystate = es.search(index="state_bills", body= {"size" : 0,
    "aggregations": {
        "states": {
            "terms": {"field": "state"},
            "aggregations": {
                "significantStateTerms": {
                    "significant_terms": {"field": "bill_document_first"}
                }
            }
        }
    }
})


filter_by_date_range = es.search(index="state_bills", body=({
    "filter" : {
        "range" : {
            "date_created" : {
                "from" : "2014051T000000",
                "to" : "20141201T235959"}
            }
        }}))

search_by_date = es.search(index="state_bills", body={
  "filter": {"range": {"date_created": {
        "from": "2014-01-01",
        "to": "2014-12-31"
      }
    }
  }
})

#Get term frequency using termvector
GET test_terms/bill_document/AU4hHfFfZCxOTjAd7QWY/_termvector?fields=bill_document_first

date_and_state = es.search(index="state_bills", body={
  "size": 0,
  "aggs": {
    "group_by_state": {
      "terms": {
        "field": "state"
      },
      "aggs": {
        "by_Date": {
          "date_histogram": {
            "field": "date_created",
            "interval": "month",
            "format": "yyyy-MM-dd"
          }
        }
      }
    }
  }
})
