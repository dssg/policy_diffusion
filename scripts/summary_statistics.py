
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
es = Elasticsearch(['54.212.36.132:9200', '54.212.36.132:9200'])

search_ga = es.search(index="state_bills", body={"query": {"match": {'state':'ga'}}})
count_ga = es.count(index="state_bills", body={"query": {"match": {'state':'ga'}}})

#Queries from elasticsearch
group_by_state = es.search(index="state_bills", body={"size":0,"aggs":{"group_by_state":{"terms":{"field":"state"}}}})

group_by_chamber = es.search(index="state_bills", body={"size":0,"aggs":{"group_by_state":{"terms":{"field":"state"}}},
"aggs":{"group_by_chamber":{"terms":{"field":"chamber"}}}})

group_by_chamber_and_state = es.search(index="state_bills", body=
{"size":0,"aggs":{"group_by_state":{"terms":{"field":"state"},
"aggs":{"group_by_chamber":{"terms":{"field":"chamber"}}}}}})

#Check exactly how they calculate significant terms
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






