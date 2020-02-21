# TrackML with Graph Neural Net
This workflow uses [MLFlow](https://www.mlflow.org) to control the operation.

## To Prepare
```
mlflow run -e prepare -P output_dir=output -P n_workers=5 -P n_files=4 -P n-eta-sections=1 -P n-phi-sections=1 .
```

## To Stage
```
mlflow run -e stage -P input_dir=output -P stage_dir=xxx -Psector=g001 .
```

## To Train
```
mlflow run -e train -P stage_dir=xxx -P output_dir=xxx .
```
