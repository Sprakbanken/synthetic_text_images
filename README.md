# Synthetic OCR Data
This repository contains code to create synthetic text images for training OCR-models. Specifically, it contains functionality to create images of text with a random typeface and automatically adding simulated scanning artifacts with [Augraphy](https://augraphy.readthedocs.io/en/latest/).

## Installation
This repository uses [PDM](https://pdm-project.org/latest/) to manage dependencies and the easiest way to use the code is to run `pdm install`.

## Using the library
The code in this repository was used to create synthetic text images for for "Enstad T, Trosterud T, Røsok MI, Beyer Y, Roald M. Comparative analysis of optical character recognition methods for Sámi texts from the National Library of Norway. Accepted for publication in Proceedings of the 25th Nordic Conference on Computational Linguistics (NoDaLiDa) 2025." (see the paper repository [here](https://github.com/Sprakbanken/nodalida25_sami_ocr)) and if you are interested in running the data generation pipeline, then you can run the scripts in [`scripts/saami_ocr_nodalida25/`](scripts/saami_ocr_nodalida25/). If you use that dataset in your research, then please cite both the NoDaLiDa paper mentioned above and the SIKOR dataset where the Sámi text is from: "SIKOR UiT The Arctic University of Norway and the Norwegian Saami Parliament's Saami text collection, http://gtweb.uit.no/korp, Version 01.12.2021 [Data set]." Also note that the SIKOR dataset to get Sámi text for the images is (CC-BY 3.0) licensed. 
