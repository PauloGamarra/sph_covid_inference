# sph_covid_inference
Inference script for the spherical approach for CT COVID classification

Keras .h5 model can be downloaded [here](https://drive.google.com/file/d/10GTg2jPILvCbQTbgv-wHVAmbJsn1ztWz/view?usp=sharing)

For inference, run 

```bash
$ python pglpClassifySphericalSamples.py <pre-processed_data> ./results.csv
```

where pre-processed_data is a directory of .npy files containing spherical images pairs of lung CT's

