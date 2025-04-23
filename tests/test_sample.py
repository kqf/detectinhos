from detectinhos.sample import read_dataset


def test_reads_dataset(annotations):
    dataset = read_dataset(annotations)
    assert len(dataset) > 0
    assert len(dataset[0].annotations) > 0
