curl http://0.0.0.0:5000/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {
      "data": [[10, 10, 10,10], [0,0, 0, 0]]
  }
}'

# curl http://35.247.231.146:5000/invocations -H 'Content-Type: application/json' -d '{
#   "dataframe_split": {
#       "data": [[10, 10, 10,10], [0,0, 0, 0]]
#   }
# }'
