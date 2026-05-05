for i in 1 2 3 4 5; do
  curl -sS -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}' >/dev/null
done
