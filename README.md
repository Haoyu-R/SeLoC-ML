# SeLoC-ML: Semantic Low-Code Engineering for Machine Learning Applications in Industrial IoT

Internet of Things (IoT) is transforming the industry by bridging the gap between Information Technology (IT) and Operational Technology (OT). Machines are being integrated with connected sensors and managed by intelligent analytics applications, accelerating digital transformation and business operations. Bringing Machine Learning (ML) to industrial devices is an advancement aiming to advance the convergence of IT and OT. However, developing an ML application in Industrial IoT (IIoT) presents various challenges, including hardware heterogeneity, non-standardized representations of ML models, devices and ML models compatibility issues, and slow app development. Successful deployment in this area requires a deep understanding of hardware, algorithms, software tools, and applications. Therefore, this paper presents a framework called Semantic Low-Code Engineering for ML Applications (SeLoC-ML), built on a low-code platform to support the rapid development of ML applications in IIoT by leveraging Semantic Web technologies. SeLoC-ML enables non-experts to easily model, discover, reuse, and matchmake ML models and devices at scale. The project code can be automatically generated for deployment on hardware based on the matching results. Developers can benefit from semantic application templates, called recipes, to fast prototype end-user applications. The evaluations confirm an engineering effort reduction by a factor of at least three compared to traditional approaches on an industrial ML classification case study, showing the efficiency and usefulness of SeLoC-ML. We share the code and welcome any contributions.

For more information on the project, please see our paper
[Link to be added](https://github.com/Haoyu-R/SeLoC-ML)

## Citation
If our work has been useful for your research and you would like to cite it in an scientific publication, please cite [Link to be updated](https://github.com/Haoyu-R/SeLoC-ML) as follows:
```
To be added
```

## Project Structure
* [collected_models](https://github.com/Haoyu-R/SeLoC-ML/tree/main/collected_models): some open-source neural network (NN) models that we trained or collected
* [estimate_tensor_arena_size](https://github.com/Haoyu-R/SeLoC-ML/tree/main/estimate_tensor_arena_size): the source file to estimate the memory consumption given a NN model in the .tflite (TensorFlow Lite) format
* [mendix_app](https://github.com/Haoyu-R/SeLoC-ML/tree/main/mendix_app): the end-user Mendix application with relevant programs that allow user to easily discover, reuse, and matchmake ML models and IIoT devices at scale
* [model_server](https://github.com/Haoyu-R/SeLoC-ML/tree/main/model_server): a folder to stimulate a server for hosting the parsed ML models
* [semantic_description](https://github.com/Haoyu-R/SeLoC-ML/tree/main/semantic_description): the semantic descriptions of the stored NN models and IIoT devices
* [semantic_schema](https://github.com/Haoyu-R/SeLoC-ML/tree/main/semantic_schema): the ontology for NN and IIoT devices, as well as supplementary schemas
* [sparql_query](https://github.com/Haoyu-R/SeLoC-ML/tree/main/sparql_query): some example SPARQL queries
* bin2tflite.py: convert a binary NN model to tflite format
* jsonld2rdf_things_description.py: Convert a JSONLD-TD file into RDF format
* models_information.xlsx: an excel sheet storing the information of collected NN models for easier parsing
* rdflib_push.py: push the semantic representations of NN models or devices to the GraphDB Knowledge Graph
* rdflib_read_ttl.py: pretty print a serialized RDF turtle file
* requirements.txt: use `pip install -r requirements.txt` to install required packages
* semantic_querying.py: use SPARQL to query the Knowledge Graph hosted in GraphDB
* semantic_utils.py
* tflite2semantic_parser_xlsx.py: generate semantic representations of the NN models stored in the folder [collected_models](https://github.com/Haoyu-R/SeLoC-ML/tree/main/collected_models) against the [proposed semantic schema](## Semantic Schema of Neural Network) combining the information provided in `models_information.xlsx`
* tflite2semantic_user_input.py: generate a semantic representation for each given NN model against the [proposed semantic schema](## Semantic Schema of Neural Network) by asking users a few questions

## Use

Our project is runnable in a Linux environment, as the binary executable is built on Linux environment.

Install the project:

```
git clone 'git@github.com:Haoyu-R/SeLoC-ML.git'
```

Install the dependency:
```
pip install -r requirement.txt
```

Run `tflite2semantic_parser_xlsx.py` to see how the collected models in the [model_repo](https://github.com/Haoyu-R/SeLoC-ML/tree/main/collected_models) can be parsed into semantic representations against the [proposed semantic schema](## Semantic Schema of Neural Network) combining the information provided in `models_information.xlsx` in one go. Please be aware that the order of the models listed in the the folder `collected_models` and in the information sheet `models_information.xlsx` should both be in alphabetic order and match with each other.

Run `tflite2semantic_user_input.py` to see how each model can be parsed into semantic representation by answering a few questions in the CMD.

To work with the semantic representations of neural networks and IoT devices, we recommend using [GraphDB free](https://graphdb.ontotext.com/). The scripts `rdflib_push.py`, `semantic_querying.py`, `sparql_queries.py` contain the code for interacting with GraphDB.

## Semantic Schema of Neural Network

![Capture2.PNG](/_resources/Capture2.PNG)

The ontology is also online [NN Ontology](https://tinyml-schema-collab.github.io/)

## To do
```
To be added
```

## Related Project
* [How to Manage TinyML at Scale](https://github.com/Haoyu-R/How-to-Manage-TinyML-at-Scale): the repo for hosting the code and examples of the paper "How to Manage TinyML at Scale" .

## Contributing to the project

We welcome contributions. Please contact us by email to get started!
