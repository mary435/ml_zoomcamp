import bentoml
from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")


model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_homework", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(vector):
    prediction = await model_runner.predict.async_run(vector)

    return prediction