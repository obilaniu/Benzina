import numpy as np

import benzina.torch.operations as ops


def test_similarity_transform():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform()

    assert similarity(0, (256, 256), (256, 256), rng) == (1, 0, 0,
                                                          0, 1, 0,
                                                          0, 0, 1)
    assert similarity(0, (256, 256), (224, 224), rng) == (1, 0, 16,
                                                          0, 1, 16,
                                                          0, 0, 1)

    similarity = ops.SimilarityTransform(scale=(0.5, 0.5))

    assert similarity(0, (256, 256), (128, 128), rng) == (2, 0, 0.5,
                                                          0, 2, 0.5,
                                                          0, 0, 1)
    assert similarity(0, (256, 128), (128, 64), rng) == (2, 0, 0.5,
                                                         0, 2, 0.5,
                                                         0, 0, 1)
    assert similarity(0, (256, 128), (64, 64), rng) == (2, 0, 64.5,
                                                        0, 2, 0.5,
                                                        0, 0, 1)

    similarity = ops.SimilarityTransform(translation_x=(64, 64))

    assert similarity(0, (256, 256), (256, 256), rng) == (1, 0, 64,
                                                          0, 1, 0,
                                                          0, 0, 1)
    assert similarity(0, (256, 256), (64, 64), rng) == (1, 0, 160,
                                                        0, 1, 96,
                                                        0, 0, 1)

    similarity = ops.SimilarityTransform(translation_y=(64, 64))

    assert similarity(0, (256, 256), (256, 256), rng) == (1, 0, 0,
                                                          0, 1, 64,
                                                          0, 0, 1)
    assert similarity(0, (256, 256), (64, 64), rng) == (1, 0, 96,
                                                        0, 1, 160,
                                                        0, 0, 1)

    similarity = ops.SimilarityTransform(translation_x=(-96, -96),
                                         translation_y=(-96, -96))

    assert similarity(0, (256, 256), (64, 64), rng) == (1, 0, 0,
                                                        0, 1, 0,
                                                        0, 0, 1)

    similarity = ops.SimilarityTransform(flip_h=1,
                                         flip_v=1)

    assert similarity(0, (256, 256), (256, 256), rng) == (-1, 0, 255,
                                                          0, -1, 255,
                                                          0, 0, 1)

    similarity = ops.SimilarityTransform(autoscale=True)

    assert similarity(0, (256, 256), (128, 128), rng) == (2, 0, 0.5,
                                                          0, 2, 0.5,
                                                          0, 0, 1)
    assert similarity(0, (256, 256), (64, 64), rng) == (4, 0, 1.5,
                                                        0, 4, 1.5,
                                                        0, 0, 1)
    assert similarity(0, (512, 256), (64, 64), rng) == (4, 0, 129.5,
                                                        0, 4, 1.5,
                                                        0, 0, 1)

    similarity = ops.SimilarityTransform(scale=(0.5, 0.5), autoscale=True)

    assert similarity(0, (256, 256), (128, 128), rng) == (4, 0, -126.5,
                                                          0, 4, -126.5,
                                                          0, 0, 1)
    assert similarity(0, (512, 256), (64, 64), rng) == (8, 0, 3.5,
                                                        0, 8, -124.5,
                                                        0, 0, 1)

    similarity = ops.SimilarityTransform(random_crop=True)

    rng = np.random.RandomState(0)
    assert similarity(0, (256, 256), (256, 256), rng) == (1, 0, 0,
                                                          0, 1, 0,
                                                          0, 0, 1)
    assert similarity(0, (256, 256), (128, 128), rng) == (1, 0, 9.092615449329529,
                                                          0, 1, 11.15255036179721,
                                                          0, 0, 1)
    assert similarity(0, (256, 256), (128, 128), rng) == (1, 0, 59.069358368375276,
                                                          0, 1, 99.9077345646663,
                                                          0, 0, 1)
    assert similarity(0, (256, 256), (64, 256), rng) == (1, 0, 50.79467752408837,
                                                         0, 1, 0,
                                                         0, 0, 1)
    assert similarity(0, (256, 256), (256, 64), rng) == (1, 0, 0,
                                                         0, 1, 130.9094974278688,
                                                         0, 0, 1)
