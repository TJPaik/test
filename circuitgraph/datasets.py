from circuitgraph.classification_builder import create_classification_dataset
from circuitgraph.regression_builder import create_regression_dataset

# allow: python -m circuitgraph.datasets
if __name__ == "__main__":
    dataset1_dir = "./AnalogGenie/Dataset"
    create_classification_dataset(
        dataset_dir=dataset1_dir,
        hypergraph_dataset_path='classification_hypergraph_dataset.pt',
        bipartite_dataset_path='classification_bipartite_dataset.pt'
    )

    dataset2_dir = "./AICircuit"
    create_regression_dataset(
        dataset_dir=dataset2_dir,
        hypergraph_dataset_path='regression_hypergraph_dataset.pt',
        bipartite_dataset_path='regression_bipartite_dataset.pt'
    )
