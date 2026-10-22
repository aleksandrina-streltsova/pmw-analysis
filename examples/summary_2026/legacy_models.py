"""
Load clustering models pickled with module paths that don't exist on main.
"""
import sys
import types

import joblib
import umap

from pmw_analysis.analysis import clustering

# (module, class) in the pickle -> class on main
LEGACY_CLASSES = {
    ("pmw_analysis.processing.model", "ClusterModel"): clustering.ClusterModel,
    ("pmw_analysis.processing.model", "CLusterIndexModel"): clustering.CLusterIndexModel,
    ("pmw_analysis.processing.model", "DimensionalityReductionIndexModel"):
        clustering.DimensionalityReductionIndexModel,
    ("pmw_analysis.dr.clusterization", "ClusterModel"): clustering.ClusterModel,
    ("pmw_analysis.dr.clusterization", "HDBSCANCLusterModel"): clustering.CLusterIndexModel,
    # partial/pca_kmeans.pkl was saved from an IPython console
    ("__main__", "ClusterModel"): clustering.ClusterModel,
}


def load_legacy_model(path):
    """
    Register the old module paths, then load the model with joblib.
    """
    for (module_name, class_name), cls in LEGACY_CLASSES.items():
        module = sys.modules.setdefault(module_name, types.ModuleType(module_name))
        setattr(module, class_name, cls)
    return joblib.load(path)


def describe(model):
    """
    Print the scaler, reducer and clusterer settings of a loaded model.
    """
    scaler, reducer, clusterer = model.scaler, model.reducer, model.clusterer
    print(f"scaler: feature_names_in_={list(getattr(scaler, 'feature_names_in_', []))}, "
          f"n_features_in_={scaler.n_features_in_}")

    print(f"reducer: {type(reducer).__module__}.{type(reducer).__name__}")
    if isinstance(reducer, umap.UMAP):
        print(f"  n_neighbors={reducer.n_neighbors}, min_dist={reducer.min_dist}, "
              f"n_epochs={reducer.n_epochs}, embedding_.shape={reducer.embedding_.shape}")
    elif isinstance(reducer, clustering.DimensionalityReductionIndexModel):
        print(f"  index ntotal={reducer.index_train.ntotal}, "
              f"embeddings_train.shape={reducer.embeddings_train.shape}")

    print(f"clusterer: {type(clusterer).__module__}.{type(clusterer).__name__}")
    if hasattr(clusterer, "n_clusters"):
        print(f"  n_clusters={clusterer.n_clusters}")
    elif isinstance(clusterer, clustering.CLusterIndexModel):
        print(f"  index ntotal={clusterer.index_train.ntotal}, "
              f"distinct labels={len(set(clusterer.labels_train.tolist()))}")


if __name__ == "__main__":
    for model_path in sys.argv[1:]:
        print(f"== {model_path}")
        describe(load_legacy_model(model_path))
