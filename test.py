from pr3d.de import ConditionalGammaMixtureEVM
from pathlib import Path

p = Path(__file__).parents[0]
predictor_path = str(p) + "/projects/qlen_benchmark/predictors/test.h5"
print(predictor_path)

predictor = ConditionalGammaMixtureEVM(
    centers=3,
    x_dim=['hey','gay'],
)

predictor._core_model._model.summary()

predictor.save(predictor_path)

del predictor

predictor = ConditionalGammaMixtureEVM(
    h5_addr=predictor_path,
)
