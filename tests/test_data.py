import pytest

from detectinhos.sample import Annotation, Sample, read_dataset


@pytest.mark.parametrize(
    "resolution",
    [
        (480, 640),
    ],
)
def test_reads_dataset(annotations):
    dataset = read_dataset(annotations, sample_type=Sample[Annotation])
    assert len(dataset) > 0
    assert len(dataset[0].annotations) > 0
