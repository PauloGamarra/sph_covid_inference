# sph_covid_inference
Inference script for the spherical approach for CT COVID classification

Keras .h5 model can be downloaded [here](https://drive.google.com/file/d/1uUltQDHip_cFHCofhQYaIz1Ns3vF3kqY/view?usp=sharing)

For inference, run 

```bash
$ python pglpClassifySphericalSamples.py <pre-processed_data> ./results.csv
```

where pre-processed_data is a directory of .npy files containing spherical images pairs of lung CT's

