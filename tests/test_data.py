from detectinhos.data import Annotation, Sample, read_dataset


def test_reads_dataset(annotations):
    dataset = read_dataset(annotations, sample_type=Sample[Annotation])
    assert len(dataset) > 0
    assert len(dataset[0].annotations) > 0
