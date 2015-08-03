

similar_doc_query_filter_state ="""
{
          "query": {
            "filtered": {
              "query": {
                "more_like_this": {
                  "fields": [
                    "bill_document_last.shingles"
                  ],
                  "like_text": "proof of identity",
                  "max_query_terms": 25,
                  "min_term_freq": 1,
                  "min_doc_freq": 2,
                  "minimum_should_match": 1
                }
              },
              "filter": {
                "bool": {
                  "must": [],
                  "must_not": [
                    {
                      "term": {
                        "bill_document.state": "'ca'"
                      }
                    }
                  ],
                  "should": [
                    {
                      "match_all": {}
                    }
                  ]
                }
              }
            }
          }
        }
        """




