from data_utils import load_datasets


DATA_WITH_TYPES = "./data_with_type"
DATA_WITHOUT_TYPES = "./data_without_type"

if __name__ == "__main__":
    corpus_with_types, corpus_without_types = load_datasets(DATA_WITH_TYPES, DATA_WITHOUT_TYPES)
